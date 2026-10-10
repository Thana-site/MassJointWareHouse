"""
sap_mass_core.py
----------------
อ่านไฟล์ Excel ที่ export จาก SAP2000 (File > Export > Excel) แล้วสร้างโมเดล node/frame
สำหรับแสดงผลมวล + ชุด back-check

ขอบเขตเวอร์ชันนี้: "แสดงผลและตรวจสอบ" เท่านั้น ยังไม่คำนวณ Cs / Fi

รูปแบบชีทของ SAP (ตามไฟล์ตัวอย่าง v26.1.0):
    แถว 1 = ชื่อตาราง (TABLE:  ...)
    แถว 2 = ชื่อ field
    แถว 3 = หน่วย / ชนิดข้อมูล (Text, Yes/No, Unitless, mm, Tonf, ...)
    แถว 4+ = ข้อมูล
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# หน่วย
# --------------------------------------------------------------------------
LENGTH_TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254, "ft": 0.3048}
G_SI = 9.80665  # m/s2


def g_in_length_unit(length_unit: str) -> float:
    """ความเร่งโน้มถ่วงในหน่วย length_unit/s2 (mm -> 9806.65)"""
    return G_SI / LENGTH_TO_M[length_unit]


class SapFileError(Exception):
    """ไฟล์ไม่ตรงรูปแบบที่คาดไว้"""


# --------------------------------------------------------------------------
# อ่านตาราง
# --------------------------------------------------------------------------
_TEXT_TYPES = {"text", "yes/no"}


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _clean_text(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _parse_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if raw.shape[0] < 3:
        return pd.DataFrame(), {}
    cols = [str(c).strip() if not pd.isna(c) else f"col{i}" for i, c in enumerate(raw.iloc[1])]
    units = {c: ("" if pd.isna(u) else str(u).strip()) for c, u in zip(cols, raw.iloc[2])}
    df = raw.iloc[3:].copy()
    df.columns = cols
    df = df.dropna(how="all").reset_index(drop=True)
    for c in cols:
        if units[c].lower() in _TEXT_TYPES:
            df[c] = df[c].map(_clean_text)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, units


class SapWorkbook:
    """ตัวอ่านชีทตามชื่อตาราง (ชื่อชีทอาจถูก Excel ตัดเหลือ 31 ตัวอักษร จึงจับแบบ prefix ด้วย)"""

    def __init__(self, src):
        self.xl = pd.ExcelFile(src)
        self._names = {_norm(n): n for n in self.xl.sheet_names}

    def find(self, name: str):
        n = _norm(name)
        if n in self._names:
            return self._names[n]
        for k, v in self._names.items():
            if k.startswith(n) or n.startswith(k):
                return v
        return None

    def table(self, name: str, required: bool = False):
        sh = self.find(name)
        if sh is None:
            if required:
                raise SapFileError(
                    f"ไม่พบตาราง '{name}' ในไฟล์ — ตอน Export ให้เลือกตารางนี้ด้วย "
                    f"(Display > Show Tables หรือ File > Export > Excel เลือก All Tables)"
                )
            return None, {}
        raw = pd.read_excel(self.xl, sheet_name=sh, header=None)
        return _parse_table(raw)


# --------------------------------------------------------------------------
# โมเดล
# --------------------------------------------------------------------------
@dataclass
class SapModel:
    source: str
    curr_units: str
    length_unit: str
    force_unit: str
    mass_unit: str
    g: float
    nodes: pd.DataFrame          # Joint,X,Y,Z,U1,U2,U3,W,is_support,is_orphan
    sum_rows: pd.DataFrame       # แถวสรุปของ SAP เช่น SumAccelUX (ไม่ใช่ node จริง)
    frames: pd.DataFrame         # Frame,JointI,JointJ,Length,Section,Groups,XI..ZJ
    groups: list[str]
    tables: dict = field(default_factory=dict)  # ตารางเสริม (optional)
    merge_tol: float = 1.0
    multi_mass_source: bool = False


def _parse_mass_unit(mass_unit: str, curr_units: str, length_unit_hint: str):
    """'Tonf-s2/mm' -> ('Tonf','mm')"""
    m = re.match(r"^(.+?)-s2/(\w+)$", mass_unit or "")
    if m:
        return m.group(1), m.group(2)
    parts = [p.strip() for p in (curr_units or "").split(",")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "Force", length_unit_hint or "mm"


def load_model(src, source_name: str = "upload") -> SapModel:
    wb = SapWorkbook(src)

    pc, _ = wb.table("Program Control")
    curr_units = str(pc.iloc[0]["CurrUnits"]) if pc is not None and len(pc) and "CurrUnits" in pc else ""

    jc, jc_u = wb.table("Joint Coordinates", required=True)
    am, am_u = wb.table("Assembled Joint Masses", required=True)
    for c in ("Joint", "GlobalX", "GlobalY", "GlobalZ"):
        if c not in jc:
            raise SapFileError(f"Joint Coordinates ไม่มีคอลัมน์ {c}")
    for c in ("Joint", "U1"):
        if c not in am:
            raise SapFileError(f"Assembled Joint Masses ไม่มีคอลัมน์ {c}")

    length_hint = jc_u.get("GlobalX", "")
    force_unit, length_unit = _parse_mass_unit(am_u.get("U1", ""), curr_units, length_hint)
    if length_unit not in LENGTH_TO_M:
        raise SapFileError(f"ไม่รู้จักหน่วยความยาว '{length_unit}' (รองรับ {list(LENGTH_TO_M)})")
    g = g_in_length_unit(length_unit)

    jc = jc[["Joint", "GlobalX", "GlobalY", "GlobalZ"]].rename(columns={"GlobalX": "X", "GlobalY": "Y", "GlobalZ": "Z"})
    jc = jc.drop_duplicates("Joint")

    # แยกแถวสรุป (SumAccelUX/UY/UZ) ออกจาก node จริง
    is_real = am["Joint"].isin(jc["Joint"])
    sum_rows = am[~is_real].reset_index(drop=True)
    real = am[is_real]
    multi_src = "MassSource" in real and real["MassSource"].nunique() > 1
    mass_cols = [c for c in ("U1", "U2", "U3") if c in real]
    agg = real.groupby("Joint", as_index=False)[mass_cols].sum()

    nodes = jc.merge(agg, on="Joint", how="left")
    for c in ("U1", "U2", "U3"):
        if c not in nodes:
            nodes[c] = 0.0
        nodes[c] = nodes[c].fillna(0.0)
    nodes["W"] = nodes["U1"] * g  # น้ำหนัก = มวล(ทิศ X) x g  (หน่วยแรง)

    # restraint
    rs, _ = wb.table("Joint Restraint Assignments")
    sup = set()
    if rs is not None and len(rs):
        m = np.zeros(len(rs), dtype=bool)
        for c in ("U1", "U2", "U3"):
            if c in rs:
                m |= (rs[c] == "Yes").to_numpy()
        sup = set(rs.loc[m, "Joint"])
    nodes["is_support"] = nodes["Joint"].isin(sup)

    # frames
    cf, _ = wb.table("Connectivity - Frame", required=True)
    for c in ("Frame", "JointI", "JointJ"):
        if c not in cf:
            raise SapFileError(f"Connectivity - Frame ไม่มีคอลัมน์ {c}")
    frames = cf[[c for c in ("Frame", "JointI", "JointJ", "Length") if c in cf]].copy()
    sa, _ = wb.table("Frame Section Assignments")
    if sa is not None and len(sa) and "AnalSect" in sa:
        frames = frames.merge(sa[["Frame", "AnalSect"]].rename(columns={"AnalSect": "Section"}),
                              on="Frame", how="left")
    else:
        frames["Section"] = "-"
    frames["Section"] = frames["Section"].fillna("-")

    g2, _ = wb.table("Groups 2 - Assignments")
    groups: list[str] = []
    if g2 is not None and len(g2):
        gf = g2[g2["ObjectType"] == "Frame"] if "ObjectType" in g2 else g2
        gmap = gf.groupby("ObjectLabel")["GroupName"].apply(lambda s: ", ".join(sorted(set(s))))
        frames["Groups"] = frames["Frame"].map(gmap).fillna("")
        groups = sorted(set(gf["GroupName"]))
    else:
        frames["Groups"] = ""

    coords = nodes.set_index("Joint")[["X", "Y", "Z"]]
    for end in ("I", "J"):
        j = frames[f"Joint{end}"]
        for ax in ("X", "Y", "Z"):
            frames[f"{ax}{end}"] = j.map(coords[ax])
    if "Length" not in frames or frames["Length"].isna().all():
        d = frames[["XJ", "YJ", "ZJ"]].to_numpy() - frames[["XI", "YI", "ZI"]].to_numpy()
        frames["Length"] = np.sqrt((d ** 2).sum(axis=1))

    used = set(frames["JointI"]) | set(frames["JointJ"])
    nodes["is_orphan"] = ~nodes["Joint"].isin(used)

    # ตารางเสริม
    tables = {}
    for key, name in {
        "mass_source": "Mass Source",
        "load_patterns": "Load Pattern Definitions",
        "frame_loads": "Frame Loads - Distributed",
        "releases": "Frame Releases 1 - General",
        "groups3": "Groups 3 - Masses and Weights",
        "matlist1": "Material List 1 - By Obj Type",
        "dim_pref": "Preferences - Dimensional",
        "base_reactions": "Base Reactions",
    }.items():
        t, _ = wb.table(name)
        if t is not None and len(t):
            tables[key] = t

    merge_tol = 1.0
    if "dim_pref" in tables and "MergeTol" in tables["dim_pref"]:
        v = tables["dim_pref"]["MergeTol"].iloc[0]
        if pd.notna(v) and v > 0:
            merge_tol = float(v)

    return SapModel(
        source=source_name, curr_units=curr_units, length_unit=length_unit,
        force_unit=force_unit, mass_unit=am_u.get("U1", ""), g=g,
        nodes=nodes, sum_rows=sum_rows, frames=frames, groups=groups,
        tables=tables, merge_tol=merge_tol, multi_mass_source=bool(multi_src),
    )


# --------------------------------------------------------------------------
# สรุปค่า
# --------------------------------------------------------------------------
def selection_mask(model: SapModel, exclude_supports=True, exclude_orphans=True, extra_exclude=()):
    n = model.nodes
    mask = pd.Series(True, index=n.index)
    if exclude_supports:
        mask &= ~n["is_support"]
    if exclude_orphans:
        mask &= ~n["is_orphan"]
    if extra_exclude:
        mask &= ~n["Joint"].isin(set(extra_exclude))
    return mask


def center_of_mass(nodes: pd.DataFrame, mask):
    sub = nodes[mask]
    tot = sub["W"].sum()
    if tot <= 0:
        return None
    return tuple(float((sub[c] * sub["W"]).sum() / tot) for c in ("X", "Y", "Z"))


def sap_reported(model: SapModel):
    """ค่ารวมที่ SAP รายงานเองในแถว SumAccelUX (ใช้เป็น checksum)"""
    s = model.sum_rows
    if s is None or not len(s) or "Joint" not in s:
        return None
    r = s[s["Joint"] == "SumAccelUX"]
    if not len(r):
        return None
    r = r.iloc[0]
    out = {"mass": float(r["U1"]), "W": float(r["U1"]) * model.g}
    if all(c in r for c in ("CenterX", "CenterY", "CenterZ")):
        out["com"] = (float(r["CenterX"]), float(r["CenterY"]), float(r["CenterZ"]))
    return out


# --------------------------------------------------------------------------
# Back-check
# --------------------------------------------------------------------------
STATUS_ORDER = {"FAIL": 3, "WARN": 2, "INFO": 1, "OK": 0}


@dataclass
class Check:
    id: str
    title: str
    status: str
    summary: str
    table: pd.DataFrame | None = None


def _fmt(x, nd=3):
    return f"{x:,.{nd}f}"


def _gravity_pattern_totals(model: SapModel) -> dict:
    fl = model.tables.get("frame_loads")
    out: dict = {}
    if fl is None:
        return out
    need = {"LoadPat", "AbsDistA", "AbsDistB", "FOverLA", "FOverLB"}
    if not need.issubset(fl.columns):
        return out
    d = fl
    if "Dir" in d:
        d = d[d["Dir"] == "Gravity"]
    if "Type" in d:
        d = d[d["Type"] == "Force"]
    tot = ((d["FOverLA"] + d["FOverLB"]) / 2 * (d["AbsDistB"] - d["AbsDistA"]))
    return tot.groupby(d["LoadPat"]).sum().to_dict()


def _self_weight_total(model: SapModel):
    g3 = model.tables.get("groups3")
    if g3 is not None and {"GroupName", "SelfWeight"}.issubset(g3.columns):
        r = g3[g3["GroupName"].str.lower() == "all"]
        if len(r):
            return float(r["SelfWeight"].iloc[0])
    ml = model.tables.get("matlist1")
    if ml is not None and "TotalWeight" in ml:
        return float(ml["TotalWeight"].sum())
    return None


def check_mass_source(model: SapModel) -> Check:
    title = "Mass Source: ผลรวมมวลอธิบายได้ครบและไม่นับซ้ำ"
    ms = model.tables.get("mass_source")
    actual = float(model.nodes["W"].sum())
    if ms is None or "LoadPat" not in ms:
        return Check("mass_source", title, "INFO",
                     "ไม่พบตาราง Mass Source ในไฟล์ จึงตรวจการนับซ้ำไม่ได้", None)

    first = ms.iloc[0]
    elements = first.get("Elements") == "Yes"
    loads_on = first.get("Loads") == "Yes"
    pats = ms[ms["LoadPat"].notna()][["LoadPat", "Multiplier"]].copy()
    lp = model.tables.get("load_patterns")
    swm = {}
    if lp is not None and {"LoadPat", "SelfWtMult"}.issubset(lp.columns):
        swm = dict(zip(lp["LoadPat"], lp["SelfWtMult"].fillna(0)))
    selfw = _self_weight_total(model)
    fl_tot = _gravity_pattern_totals(model)

    rows, expected = [], 0.0
    if elements and selfw is not None:
        rows.append({"แหล่งมวล": "Element self mass (Elements = Yes)", "ตัวคูณ": 1.0,
                     "น้ำหนักเต็ม": selfw, "นับเข้ามวล": selfw})
        expected += selfw
    double = []
    if loads_on:
        for _, p in pats.iterrows():
            name, mult = p["LoadPat"], float(p["Multiplier"]) if pd.notna(p["Multiplier"]) else 0.0
            sw_part = (swm.get(name, 0.0) * selfw) if selfw is not None else 0.0
            fr_part = fl_tot.get(name, 0.0)
            full = sw_part + fr_part
            rows.append({"แหล่งมวล": f"Load pattern: {name}", "ตัวคูณ": mult,
                         "น้ำหนักเต็ม": full, "นับเข้ามวล": full * mult})
            expected += full * mult
            if elements and sw_part * mult > 0:
                double.append((name, sw_part * mult))
    tbl = pd.DataFrame(rows)
    if len(tbl):
        tbl.loc[len(tbl)] = {"แหล่งมวล": "รวมที่คำนวณจากตาราง", "ตัวคูณ": np.nan,
                             "น้ำหนักเต็ม": np.nan, "นับเข้ามวล": expected}
        tbl.loc[len(tbl)] = {"แหล่งมวล": "รวมจาก node ในไฟล์ (ΣU1 × g)", "ตัวคูณ": np.nan,
                             "น้ำหนักเต็ม": np.nan, "นับเข้ามวล": actual}
    diff = actual - expected
    rel = abs(diff) / actual if actual else 0.0
    fu = model.force_unit

    if double:
        names = ", ".join(n for n, _ in double)
        amt = sum(a for _, a in double)
        confirmed = rel < 0.01
        msg = (f"เปิด 'Element self mass' และใส่ load pattern ที่มี self-weight ({names}) ใน Mass Source ด้วย "
               f"→ น้ำหนักตัวเองถูกนับ 2 ครั้ง ≈ {_fmt(amt)} {fu} "
               f"({amt / actual * 100:.0f}% ของมวลทั้งหมด {_fmt(actual)} {fu})")
        if confirmed:
            msg += f" ผลรวมจาก node ตรงกับตารางที่นับซ้ำ (ต่าง {rel * 100:.2f}%) ยืนยันว่าเกิดจริงในไฟล์นี้"
        else:
            msg += f" แต่ผลรวมจาก node ไม่ตรงกับตาราง (ต่าง {rel * 100:.1f}%) ควรตรวจใน SAP"
        return Check("mass_source", title, "FAIL" if confirmed else "WARN", msg, tbl)
    if rel > 0.01:
        return Check("mass_source", title, "WARN",
                     f"ผลรวมมวลจาก node ({_fmt(actual)} {fu}) ต่างจากที่คำนวณจากตาราง ({_fmt(expected)} {fu}) "
                     f"{rel * 100:.1f}% — อาจมี joint load / area load / additional mass ที่ไม่ได้ export",
                     tbl)
    return Check("mass_source", title, "OK",
                 f"ผลรวมมวลจาก node ({_fmt(actual)} {fu}) ตรงกับ Mass Source ({rel * 100:.2f}%)", tbl)


def run_checks(model: SapModel, release_group: str | None = None,
               expect_axial_release: bool = False, short_frame_tol: float | None = None) -> list[Check]:
    n, f = model.nodes, model.frames
    fu, lu = model.force_unit, model.length_unit
    out: list[Check] = []

    # 1 หน่วย
    out.append(Check("units", "หน่วยและค่า g ที่ใช้แปลงมวลเป็นน้ำหนัก", "INFO",
                     f"CurrUnits = {model.curr_units or '-'}; หน่วยมวล = {model.mass_unit}; "
                     f"g = {model.g:,.2f} {lu}/s² → น้ำหนัก ({fu}) = มวล × g", None))

    # 2 checksum กับแถวสรุปของ SAP
    rep = sap_reported(model)
    tot_mass = float(n["U1"].sum())
    if rep is None:
        out.append(Check("checksum", "ผลรวมมวล node เทียบกับแถวสรุปของ SAP (SumAccelUX)", "INFO",
                         "ไม่พบแถว SumAccelUX ในไฟล์", None))
    else:
        rel = abs(tot_mass - rep["mass"]) / rep["mass"] if rep["mass"] else 0
        st = "OK" if rel < 0.005 else "FAIL"
        out.append(Check("checksum", "ผลรวมมวล node เทียบกับแถวสรุปของ SAP (SumAccelUX)", st,
                         f"Σ node = {tot_mass:.6g}, SAP = {rep['mass']:.6g} ({model.mass_unit}) ต่าง {rel * 100:.3f}% "
                         f"(ค่าใน SAP ปัด 4 หลักนัยสำคัญ จึงต่างได้เล็กน้อย) "
                         f"แถวสรุป {len(model.sum_rows)} แถวถูกตัดออกจากรายการ node แล้ว",
                         model.sum_rows[[c for c in ("Joint", "U1", "U2", "U3", "CenterX", "CenterY", "CenterZ")
                                         if c in model.sum_rows]]))

    # 3 Mass source
    out.append(check_mass_source(model))
    if model.multi_mass_source:
        out.append(Check("multi_src", "มี Mass Source มากกว่า 1 ชุดในตาราง", "WARN",
                         "รวมมวลทุกชุดเข้าด้วยกัน ตรวจว่าตั้งใจหรือไม่", None))

    # 4 orphan
    orph = n[n["is_orphan"]]
    if len(orph):
        t = orph[["Joint", "X", "Y", "Z", "W"]].rename(columns={"W": f"W ({fu})"})
        massy = int((orph["W"] > 0).sum())
        out.append(Check("orphan", "Joint ที่ไม่ต่อกับ frame ใดเลย (ลอย)", "WARN",
                         f"พบ {len(orph)} joint, มีมวล > 0 จำนวน {massy} "
                         f"— อาจเป็นเศษจากการ divide/ลบ หรือ member ที่ควรมีแต่หายไป", t))
    else:
        out.append(Check("orphan", "Joint ที่ไม่ต่อกับ frame ใดเลย (ลอย)", "OK", "ไม่พบ", None))

    # 5 joint ซ้อน
    xyz = n[["X", "Y", "Z"]].to_numpy(dtype=float)
    pairs = sorted(cKDTree(xyz).query_pairs(model.merge_tol)) if len(n) > 1 else []
    if pairs:
        t = pd.DataFrame([{"Joint A": n.Joint.iloc[a], "Joint B": n.Joint.iloc[b],
                           f"ระยะ ({lu})": float(np.linalg.norm(xyz[a] - xyz[b]))} for a, b in pairs])
        out.append(Check("dup_joint", f"Joint ที่พิกัดซ้อนกัน (ระยะ ≤ {model.merge_tol:g} {lu})", "FAIL",
                         f"พบ {len(pairs)} คู่ — ไม่ได้ merge จึงไม่ต่อกันทางโครงสร้าง", t))
    else:
        out.append(Check("dup_joint", f"Joint ที่พิกัดซ้อนกัน (ระยะ ≤ {model.merge_tol:g} {lu})", "OK",
                         "ไม่พบ", None))

    # 6 frame อ้าง joint ที่ไม่มี / ซ้ำ / สั้น
    known = set(n["Joint"])
    miss = f[~f["JointI"].isin(known) | ~f["JointJ"].isin(known)]
    if len(miss):
        out.append(Check("frame_missing_joint", "Frame ที่อ้าง joint ซึ่งไม่อยู่ใน Joint Coordinates", "FAIL",
                         f"พบ {len(miss)} frame", miss[["Frame", "JointI", "JointJ"]]))
    lo = f[["JointI", "JointJ"]].astype(str).min(axis=1)
    hi = f[["JointI", "JointJ"]].astype(str).max(axis=1)
    dup_mask = pd.DataFrame({"a": lo, "b": hi}).duplicated(keep=False)
    dup = f[dup_mask].assign(_a=lo[dup_mask], _b=hi[dup_mask]).sort_values(["_a", "_b"])
    if len(dup):
        out.append(Check("dup_frame", "Frame ซ้อนกัน (ปลายทั้งสองเป็นคู่ joint เดียวกัน)", "FAIL",
                         f"พบ {len(dup)} frame — น้ำหนักและความแข็งถูกนับซ้ำ",
                         dup[["Frame", "JointI", "JointJ", "Section"]]))
    else:
        out.append(Check("dup_frame", "Frame ซ้อนกัน (ปลายทั้งสองเป็นคู่ joint เดียวกัน)", "OK", "ไม่พบ", None))
    tol = short_frame_tol if short_frame_tol is not None else 0.05 / LENGTH_TO_M[lu]
    short = f[f["Length"] < tol]
    if len(short):
        out.append(Check("short_frame", f"Frame สั้นผิดปกติ (ยาว < {tol:g} {lu})", "WARN",
                         f"พบ {len(short)} frame — อาจ divide เกิน",
                         short[["Frame", "JointI", "JointJ", "Length", "Section"]]))
    else:
        out.append(Check("short_frame", f"Frame สั้นผิดปกติ (ยาว < {tol:g} {lu})", "OK", "ไม่พบ", None))

    # 7 ความต่อเนื่องของโครงสร้าง (union-find)
    parent = {j: j for j in set(f["JointI"]) | set(f["JointJ"])}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(f["JointI"], f["JointJ"]):
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    comps: dict = {}
    for j in parent:
        comps.setdefault(find(j), []).append(j)
    sizes = sorted((len(v) for v in comps.values()), reverse=True)
    if len(comps) > 1:
        big = max(comps.values(), key=len)
        rest = [j for v in comps.values() if v is not big for j in v]
        t = n[n["Joint"].isin(rest)][["Joint", "X", "Y", "Z"]]
        out.append(Check("components", "โครงสร้างต่อเนื่องเป็นก้อนเดียว", "WARN",
                         f"พบ {len(comps)} ก้อนที่ไม่ต่อกัน (จำนวน joint: {sizes}) "
                         f"— ก้อนเล็กไม่ถ่ายแรงถึงก้อนหลัก", t))
    else:
        out.append(Check("components", "โครงสร้างต่อเนื่องเป็นก้อนเดียว", "OK",
                         f"ต่อกันเป็นก้อนเดียว ({sizes[0] if sizes else 0} joint)", None))

    # 8 joint ที่ต่อ frame แต่มวล 0 (ไม่ใช่ฐาน)
    z = n[(~n["is_orphan"]) & (~n["is_support"]) & (n["W"] <= 0)]
    if len(z):
        out.append(Check("zero_mass", "Joint ที่ต่อ frame (ไม่ใช่ฐาน) แต่มวล = 0", "INFO",
                         f"พบ {len(z)} joint", z[["Joint", "X", "Y", "Z"]]))

    # 9 มวลสามทิศเท่ากัน
    if {"U1", "U2", "U3"}.issubset(n.columns):
        scale = max(float(n["U1"].abs().max()), 1e-30)
        bad = n[((n["U1"] - n["U2"]).abs() > 1e-6 * scale) | ((n["U1"] - n["U3"]).abs() > 1e-6 * scale)]
        if len(bad):
            out.append(Check("isotropy", "มวลทิศ U1 = U2 = U3", "WARN",
                             f"พบ {len(bad)} joint ที่มวลต่างทิศ — แอปนี้ใช้ U1 (ทิศ X) เป็นฐานน้ำหนัก",
                             bad[["Joint", "U1", "U2", "U3"]]))
        else:
            out.append(Check("isotropy", "มวลทิศ U1 = U2 = U3", "OK", "เท่ากันทุก joint", None))

    # 10 ฐาน
    sup = n[n["is_support"]]
    out.append(Check("supports", "Joint ฐาน (restrained) ที่ตัดออกจากแรงเฉื่อย",
                     "INFO",
                     f"{len(sup)} joint, มวลรวม {_fmt(float(sup['W'].sum()))} {fu} "
                     f"({sup['W'].sum() / max(n['W'].sum(), 1e-30) * 100:.1f}% ของทั้งหมด) "
                     f"— มวลที่ฐานลงดินโดยตรง ไม่สร้างแรงเฉื่อยที่ระดับหลังคา",
                     sup[["Joint", "X", "Y", "Z", "W"]].rename(columns={"W": f"W ({fu})"}) if len(sup) else None))

    # 11 release ของกลุ่มที่เลือก (เช่น Purlin)
    rel = model.tables.get("releases")
    if release_group and release_group in model.groups:
        members = f[f["Groups"].str.split(", ").apply(lambda g: release_group in g)]["Frame"]
        mset = set(members)
        if rel is not None and {"Frame", "PI", "PJ"}.issubset(rel.columns):
            r = rel[rel["Frame"].isin(mset)]
            p_rel = r[(r["PI"] == "Yes") | (r["PJ"] == "Yes")]
            any_rel = len(r)
            msg = (f"กลุ่ม '{release_group}' มี {len(mset)} frame: มี release ใดๆ {any_rel}, "
                   f"มี axial release (P) {len(p_rel)}, ไม่มี release เลย {len(mset) - any_rel}")
            st = "INFO"
            if expect_axial_release and len(p_rel) < len(mset):
                st = "WARN"
                msg += f" — ตั้งไว้ว่าคาดหวัง axial release ทุกตัว แต่ยังขาด {len(mset) - len(p_rel)} frame"
            cols = [c for c in ("Frame", "PI", "V2I", "V3I", "TI", "M2I", "M3I", "PJ", "V2J", "V3J", "TJ", "M2J", "M3J")
                    if c in r]
            out.append(Check("releases", f"Release ของกลุ่ม '{release_group}'", st, msg,
                             r[cols] if len(r) else None))
        else:
            out.append(Check("releases", f"Release ของกลุ่ม '{release_group}'", "INFO",
                             "ไม่พบตาราง Frame Releases ในไฟล์", None))
    return out


def worst_status(checks: list[Check]) -> str:
    return max((c.status for c in checks), key=lambda s: STATUS_ORDER[s], default="OK")


# --------------------------------------------------------------------------
# รายงาน (markdown)
# --------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.4g}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.to_numpy()]
    extra = f"\n\n(แสดง {max_rows} แถวแรกจาก {len(df)})" if len(df) > max_rows else ""
    return "\n".join([head, sep, *body]) + extra


def report_markdown(model: SapModel, checks: list[Check], mask) -> str:
    n = model.nodes
    fu, lu = model.force_unit, model.length_unit
    sel = n[mask]
    com = center_of_mass(n, mask)
    rep = sap_reported(model)
    icon = {"OK": "✅", "INFO": "ℹ️", "WARN": "⚠️", "FAIL": "❌"}
    lines = [
        "# Back-check report — SAP2000 node mass",
        "",
        f"- ไฟล์: `{model.source}`",
        f"- หน่วย: {model.curr_units or '-'} (มวล {model.mass_unit}, g = {model.g:,.2f} {lu}/s²)",
        f"- Joint ทั้งหมด {len(n)}, Frame {len(model.frames)}, ฐาน {int(n.is_support.sum())}, ลอย {int(n.is_orphan.sum())}",
        f"- ΣW ทุก joint = {_fmt(float(n['W'].sum()))} {fu}",
        f"- ΣW ที่ใช้เป็นแรงเฉื่อย (หลังกรอง) = {_fmt(float(sel['W'].sum()))} {fu} จาก {len(sel)} joint",
    ]
    if com:
        lines.append(f"- COM ของ joint ที่เลือก (X, Y, Z) = ({com[0]:,.0f}, {com[1]:,.0f}, {com[2]:,.0f}) {lu}")
    if rep and "com" in rep:
        c = rep["com"]
        lines.append(f"- COM ที่ SAP รายงาน (รวมฐาน) = ({c[0]:,.0f}, {c[1]:,.0f}, {c[2]:,.0f}) {lu}")
    lines += ["", f"## ผลสรุป: {worst_status(checks)}", ""]
    for c in checks:
        lines.append(f"### {icon[c.status]} [{c.status}] {c.title}")
        lines.append("")
        lines.append(c.summary)
        if c.table is not None and len(c.table):
            lines += ["", _md_table(c.table)]
        lines.append("")
    lines += ["---", "ตัวเลขทั้งหมดอ่านจากไฟล์ที่ export เท่านั้น ข้อความ 'อาจ/น่าจะ' เป็นการตีความ ต้องยืนยันใน SAP2000 และมาตรฐานจริง"]
    return "\n".join(lines)
