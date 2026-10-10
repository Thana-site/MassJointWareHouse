"""
SAP2000 Mass Viewer  (เวอร์ชัน 0.1 — แสดงผล + back-check)

รัน:  streamlit run app.py
อัปโหลดไฟล์ Excel ที่ export จาก SAP2000 (File > Export > Excel, เลือก All Tables)
ยังไม่คำนวณ Cs / Fi — เวอร์ชันนี้แสดงตำแหน่งและมวลของ node และตรวจความผิดปกติของโมเดล
"""
from __future__ import annotations

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sap_mass_core as core

st.set_page_config(page_title="SAP2000 Mass Viewer", layout="wide")

STATUS_ICON = {"OK": "✅", "INFO": "ℹ️", "WARN": "⚠️", "FAIL": "❌"}
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
           "#ff7f0e", "#7f7f7f"]


# --------------------------------------------------------------------------
# โหลดไฟล์ (cache ตามเนื้อหาไฟล์)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="กำลังอ่านไฟล์ SAP2000 ...")
def _load_from_bytes(data: bytes, name: str):
    return core.load_model(io.BytesIO(data), name)


@st.cache_data(show_spinner="กำลังอ่านไฟล์ SAP2000 ...")
def _load_from_path(path: str, mtime: float):
    return core.load_model(path, os.path.basename(path))


# --------------------------------------------------------------------------
# Figure builders
# --------------------------------------------------------------------------
def _sizes(w: pd.Series, lo=5.0, hi=22.0):
    wmax = float(w.max()) if len(w) else 0.0
    if wmax <= 0:
        return np.full(len(w), lo)
    return lo + (hi - lo) * np.sqrt(w.clip(lower=0) / wmax)


def _frame_segments(frames: pd.DataFrame, axes):
    """คืนพิกัดเส้นแบบคั่นด้วย None สำหรับ plotly (axes = ('X','Y','Z') หรือ 2 แกน)"""
    cols = {}
    for ax in axes:
        a = frames[[f"{ax}I", f"{ax}J"]].to_numpy(dtype=float)
        v = np.full((len(frames), 3), np.nan)
        v[:, 0], v[:, 1] = a[:, 0], a[:, 1]
        cols[ax] = v.ravel()
    return cols


def build_figure(model: core.SapModel, mask: pd.Series, view: str, color_by: str, show_frames: bool,
                 frame_color: str, show_labels: bool, show_supports: bool, show_orphans: bool,
                 show_com: bool, size_scale: float):
    n, f = model.nodes, model.frames
    lu, fu = model.length_unit, model.force_unit
    is3d = view == "3D"
    axes = {"3D": ("X", "Y", "Z"), "Plan (X-Y)": ("X", "Y"), "Elevation (X-Z)": ("X", "Z"),
            "Elevation (Y-Z)": ("Y", "Z")}[view]
    Scat = go.Scatter3d if is3d else go.Scatter
    fig = go.Figure()
    key = dict(zip(axes, ("x", "y", "z")))  # 2D: แกนที่เลือกคือ x,y ของกราฟ

    def xy(df, suffix=""):
        return {key[ax]: df[f"{ax}{suffix}"] if suffix else df[ax] for ax in axes}

    # frames
    if show_frames and len(f):
        fv = f.dropna(subset=["XI", "XJ"])
        groups = [(None, fv)]
        if frame_color == "ตามหน้าตัด (Section)":
            groups = list(fv.groupby("Section"))
        elif frame_color == "ตามกลุ่ม (Group)":
            groups = list(fv.groupby(fv["Groups"].replace("", "(ไม่อยู่ในกลุ่ม)")))
        for i, (name, g) in enumerate(groups):
            seg = _frame_segments(g, axes)
            kw = {key[a]: seg[a] for a in axes}
            fig.add_trace(Scat(mode="lines", name=str(name) if name is not None else "Frames",
                               line=dict(color=PALETTE[i % len(PALETTE)] if name is not None else "#9aa0a6",
                                         width=4 if is3d else 2),
                               hoverinfo="skip", legendgroup="frames", **kw))
        mid = pd.DataFrame({ax: (fv[f"{ax}I"] + fv[f"{ax}J"]) / 2 for ax in axes})
        fig.add_trace(Scat(mode="markers", name="Frame (hover)", showlegend=False,
                           marker=dict(size=2.5 if is3d else 4, color="rgba(120,120,120,0.35)"),
                           text=[f"Frame {a}<br>{b}<br>{c}<br>L = {d:,.0f} {lu}" for a, b, c, d in
                                 zip(fv["Frame"], fv["Section"], fv["Groups"], fv["Length"])],
                           hoverinfo="text", **{key[ax]: mid[ax] for ax in axes}))

    # node ที่ใช้เป็นแรงเฉื่อย
    sel = n[mask]
    hover = [f"Joint {j}<br>W = {w:,.3f} {fu}<br>({x:,.0f}, {y:,.0f}, {z:,.0f}) {lu}"
             for j, w, x, y, z in zip(sel["Joint"], sel["W"], sel["X"], sel["Y"], sel["Z"])]
    cvals = sel["Z"] if color_by == "ระดับ Z" else sel["W"]
    fig.add_trace(Scat(
        mode="markers+text" if show_labels else "markers", name="Node (นับเป็นแรงเฉื่อย)",
        marker=dict(size=_sizes(sel["W"]) * size_scale, color=cvals, colorscale="Viridis", showscale=True,
                    colorbar=dict(title=("Z (" + lu + ")") if color_by == "ระดับ Z" else f"W ({fu})", len=0.6),
                    line=dict(width=0.5, color="black")),
        text=sel["Joint"] if show_labels else None, textposition="top center", textfont=dict(size=10),
        hovertext=hover, hoverinfo="text", **xy(sel)))

    if show_supports:
        sup = n[n["is_support"] & ~mask]
        if len(sup):
            fig.add_trace(Scat(mode="markers", name="ฐาน (ตัดออก)",
                               marker=dict(size=5 * size_scale, color="#9aa0a6", symbol="diamond"),
                               hovertext=[f"Joint {j} (ฐาน)<br>W = {w:,.3f} {fu}" for j, w in zip(sup.Joint, sup.W)],
                               hoverinfo="text", **xy(sup)))
    if show_orphans:
        orp = n[n["is_orphan"]]
        if len(orp):
            fig.add_trace(Scat(mode="markers+text" if show_labels else "markers", name="Joint ลอย (ไม่ต่อ frame)",
                               marker=dict(size=7 * size_scale, color="#d62728", symbol="x"),
                               text=orp["Joint"] if show_labels else None,
                               hovertext=[f"Joint {j} (ลอย)<br>W = {w:,.3f} {fu}<br>({x:,.0f}, {y:,.0f}, {z:,.0f}) {lu}"
                                          for j, w, x, y, z in zip(orp.Joint, orp.W, orp.X, orp.Y, orp.Z)],
                               hoverinfo="text", **xy(orp)))
    if show_com:
        com = core.center_of_mass(n, mask)
        if com:
            d = dict(zip(("X", "Y", "Z"), com))
            fig.add_trace(Scat(mode="markers", name="COM (node ที่เลือก)",
                               marker=dict(size=11 * size_scale, color="black", symbol="diamond"),
                               hovertext=f"COM ({com[0]:,.0f}, {com[1]:,.0f}, {com[2]:,.0f}) {lu}", hoverinfo="text",
                               **{key[ax]: [d[ax]] for ax in axes}))
        rep = core.sap_reported(model)
        if rep and "com" in rep:
            d = dict(zip(("X", "Y", "Z"), rep["com"]))
            fig.add_trace(Scat(mode="markers", name="COM ที่ SAP รายงาน (รวมฐาน)",
                               marker=dict(size=10 * size_scale, color="#e91e63",
                                           symbol="diamond-open" if is3d else "diamond-open"),
                               hovertext=f"SAP COM ({rep['com'][0]:,.0f}, {rep['com'][1]:,.0f}, {rep['com'][2]:,.0f}) {lu}",
                               hoverinfo="text", **{key[ax]: [d[ax]] for ax in axes}))

    if is3d:
        fig.update_layout(scene=dict(aspectmode="data", xaxis_title=f"X ({lu})", yaxis_title=f"Y ({lu})",
                                     zaxis_title=f"Z ({lu})"), height=720)
    else:
        fig.update_layout(xaxis_title=f"{axes[0]} ({lu})", yaxis_title=f"{axes[1]} ({lu})", height=640,
                          yaxis=dict(scaleanchor="x", scaleratio=1))
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=-0.02))
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("SAP2000 Mass Viewer")
up = st.sidebar.file_uploader("ไฟล์ Excel จาก SAP2000", type=["xlsx"])
path_in = st.sidebar.text_input("หรือระบุ path ไฟล์ (รันบนเครื่องตัวเอง)", value="")

model = None
err = None
try:
    if up is not None:
        model = _load_from_bytes(up.getvalue(), up.name)
    elif path_in.strip():
        p = path_in.strip().strip('"')
        model = _load_from_path(p, os.path.getmtime(p))
except core.SapFileError as e:
    err = str(e)
except Exception as e:  # noqa: BLE001
    err = f"อ่านไฟล์ไม่ได้: {e}"

st.title("SAP2000 Mass Viewer")
st.caption("v0.1 — แสดงตำแหน่ง/มวลของ node และ back-check โมเดล · ยังไม่คำนวณ Cs / Fi")

if err:
    st.error(err)
    st.stop()
if model is None:
    st.info("อัปโหลดไฟล์ Excel ที่ export จาก SAP2000 ทางซ้ายเพื่อเริ่ม")
    st.markdown(
        "**ตารางที่ต้องมีในไฟล์:** `Joint Coordinates`, `Assembled Joint Masses`, `Connectivity - Frame`  \n"
        "**ตารางที่ช่วยให้ตรวจได้ละเอียดขึ้น:** `Program Control`, `Joint Restraint Assignments`, "
        "`Frame Section Assignments`, `Groups 2 - Assignments`, `Groups 3 - Masses and Weights`, `Mass Source`, "
        "`Load Pattern Definitions`, `Frame Loads - Distributed`, `Frame Releases 1 - General`, "
        "`Preferences - Dimensional`")
    st.stop()

nodes = model.nodes
lu, fu = model.length_unit, model.force_unit

st.sidebar.markdown("---")
st.sidebar.subheader("เลือก node ที่นับเป็นแรงเฉื่อย")
ex_sup = st.sidebar.checkbox("ตัด joint ฐาน (restrained)", value=True,
                             help="มวลที่ฐานลงดินโดยตรง ไม่สร้างแรงเฉื่อยที่ระดับหลังคา")
ex_orp = st.sidebar.checkbox("ตัด joint ลอย (ไม่ต่อ frame)", value=True)
extra = st.sidebar.text_input("ตัด joint เพิ่ม (คั่นด้วย ,)", value="")
extra_ids = [s.strip() for s in extra.split(",") if s.strip()]
mask = core.selection_mask(model, ex_sup, ex_orp, extra_ids)

st.sidebar.subheader("การแสดงผล")
view = st.sidebar.radio("มุมมอง", ["3D", "Plan (X-Y)", "Elevation (X-Z)", "Elevation (Y-Z)"])
color_by = st.sidebar.radio("สีของ node", ["ระดับ Z", "น้ำหนัก W"])
show_frames = st.sidebar.checkbox("แสดง member", value=True)
frame_color = st.sidebar.selectbox("สี member", ["เทา", "ตามหน้าตัด (Section)", "ตามกลุ่ม (Group)"])
show_labels = st.sidebar.checkbox("แสดงเลข joint", value=False)
show_supports = st.sidebar.checkbox("แสดง joint ฐานที่ถูกตัด", value=True)
show_orphans = st.sidebar.checkbox("แสดง joint ลอย", value=True)
show_com = st.sidebar.checkbox("แสดง COM", value=True)
size_scale = st.sidebar.slider("ขนาดจุด", 0.5, 2.0, 1.0, 0.1)

st.sidebar.subheader("Back-check")
default_grp = "Purlin" if "Purlin" in model.groups else (model.groups[0] if model.groups else None)
rel_group = st.sidebar.selectbox("ตรวจ release ของกลุ่ม", ["(ไม่ตรวจ)"] + model.groups,
                                 index=(model.groups.index(default_grp) + 1) if default_grp else 0)
expect_axial = st.sidebar.checkbox("คาดหวัง axial release ทุก frame ในกลุ่มนี้", value=False,
                                   help="เช่น purlin ที่ออกแบบเป็น pin + release axial")
st.sidebar.markdown("---")
st.sidebar.caption("ไม่คำนวณ Cs / Fi ในเวอร์ชันนี้")

checks = core.run_checks(model, release_group=None if rel_group == "(ไม่ตรวจ)" else rel_group,
                         expect_axial_release=expect_axial)
worst = core.worst_status(checks)

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
sel = nodes[mask]
com = core.center_of_mass(nodes, mask)
rep = core.sap_reported(model)
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"ΣW ทุก joint ({fu})", f"{nodes['W'].sum():,.2f}")
c2.metric(f"ΣW ที่ฐาน ({fu})", f"{nodes.loc[nodes.is_support, 'W'].sum():,.2f}")
c3.metric(f"ΣW node ที่เลือก ({fu})", f"{sel['W'].sum():,.2f}", help=f"{len(sel)} joint")
c4.metric("Back-check", f"{STATUS_ICON[worst]} {worst}",
          help=f"FAIL {sum(c.status == 'FAIL' for c in checks)} · WARN {sum(c.status == 'WARN' for c in checks)}")
if com:
    msg = f"COM ของ node ที่เลือก (X, Y, Z) = ({com[0]:,.0f}, {com[1]:,.0f}, {com[2]:,.0f}) {lu}"
    if rep and "com" in rep:
        msg += f"  ·  SAP รายงาน (รวมฐาน) = ({rep['com'][0]:,.0f}, {rep['com'][1]:,.0f}, {rep['com'][2]:,.0f})"
    st.caption(msg)
nf = sum(c.status == "FAIL" for c in checks)
if nf:
    st.warning(f"พบ {nf} รายการ FAIL ใน back-check — ดูแท็บ 'Back-check report' ก่อนนำมวลไปใช้")

tab_view, tab_table, tab_check, tab_notes = st.tabs(["แสดงผล", "ตาราง node", "Back-check report", "สรุปแนวทาง"])

with tab_view:
    fig = build_figure(model, mask, view, color_by, show_frames, frame_color, show_labels, show_supports,
                       show_orphans, show_com, size_scale)
    st.plotly_chart(fig, width="stretch")
    st.caption("ขนาดจุด ∝ √W · จุดเทา = ฐานที่ถูกตัด · × แดง = joint ที่ไม่ต่อ frame · ◆ ดำ = COM ของ node ที่เลือก "
               "(COM เป็นค่าไว้เช็กมือ ไม่ใช่จุดสำหรับสร้าง node หรือใส่แรงในโมเดล)")

with tab_table:
    t = nodes.copy()
    t["สถานะ"] = np.where(t["is_support"], "ฐาน", np.where(t["is_orphan"], "ลอย", ""))
    t["ถูกเลือก"] = mask
    t = t[["Joint", "X", "Y", "Z", "U1", "W", "สถานะ", "ถูกเลือก"]].rename(
        columns={"U1": f"มวล U1 ({model.mass_unit})", "W": f"W ({fu})"})
    fc1, fc2 = st.columns([1, 3])
    only_sel = fc1.checkbox("เฉพาะ node ที่เลือก", value=False)
    if only_sel:
        t = t[t["ถูกเลือก"]]
    st.dataframe(t, width="stretch", hide_index=True, height=520)
    st.download_button("ดาวน์โหลด CSV", t.to_csv(index=False).encode("utf-8-sig"),
                       file_name="node_weights.csv", mime="text/csv")
    z_summary = (sel.groupby("Z")["W"].agg(["count", "sum"]).reset_index()
                 .rename(columns={"Z": f"ระดับ Z ({lu})", "count": "จำนวน joint", "sum": f"ΣW ({fu})"}))
    st.markdown("**สรุปตามระดับ Z (node ที่เลือก)**")
    st.dataframe(z_summary, width="stretch", hide_index=True)

with tab_check:
    counts = {s: sum(c.status == s for c in checks) for s in ("FAIL", "WARN", "INFO", "OK")}
    st.markdown(f"**ผลรวม:** {STATUS_ICON['FAIL']} FAIL {counts['FAIL']} · {STATUS_ICON['WARN']} WARN {counts['WARN']} · "
                f"{STATUS_ICON['INFO']} INFO {counts['INFO']} · {STATUS_ICON['OK']} OK {counts['OK']}")
    for c in sorted(checks, key=lambda c: -core.STATUS_ORDER[c.status]):
        with st.expander(f"{STATUS_ICON[c.status]} [{c.status}] {c.title}", expanded=c.status in ("FAIL", "WARN")):
            st.write(c.summary)
            if c.table is not None and len(c.table):
                st.dataframe(c.table, width="stretch", hide_index=True)
    md = core.report_markdown(model, checks, mask)
    st.download_button("ดาวน์โหลดรายงาน (.md)", md.encode("utf-8"),
                       file_name=f"backcheck_{datetime.now():%Y%m%d_%H%M}.md", mime="text/markdown")
    st.caption("ตัวเลขทั้งหมดอ่านจากไฟล์ที่ export ส่วนข้อความที่เป็นการคาดเดาต้องยืนยันใน SAP2000 และมาตรฐานจริง")

with tab_notes:
    st.markdown(
        """
### สรุปแนวทางที่คุยกัน (หลังคาลาดเดียว, ไม่นับ diaphragm)

**ระบบโครงสร้าง**
- เฟรมหลัก = เสา + rafter ต่อแบบ moment, ฐานเสา fix, มีคานหลักตามยาวที่หัวเสาทั้งสองแนว ต่อเป็น moment
- purlin ออกแบบเป็น pin และ release axial (ให้ตรวจจากแท็บ Back-check)
- ไม่มี bracing และไม่นับ metal sheet เป็น diaphragm

**หลักการ ESL**
- F = Cs × W เป็นแรง **แนวนอน** ใช้น้ำหนักจริงตามความลาด ไม่ต้องคูณ cosθ (cosθ/sinθ ใช้ตอนแตกแรงเข้าแกน rafter เท่านั้น)
- โครงสร้างชั้นเดียวไม่ต้องกระจายตามความสูง (Cvx = 1) ความต่างของ elevation แสดงผลผ่านโมเมนต์และ drift ที่โปรแกรมคำนวณเอง
- Fi = Cs × Wi ที่แต่ละ node ตามมวลจริง ตรวจ ΣFi = V และโมเมนต์ฐาน ≈ V × z_cm

**การจำลองมวล**
- ใช้ lumped / tributary mass: ใส่แรงเป็น joint load ที่ node purlin–rafter และหัวเสา
- ตัด node ฐานเสาออก (มวลลงดินโดยตรง)
- **ไม่** สร้าง COM node แล้วผูก rigid link ไปหลายเฟรม/หัวเสา (จะกลายเป็น rigid diaphragm โดยไม่ตั้งใจ) COM ใช้คิดมือเท่านั้น
- ถ้าจะตัดระยะเยื้อง purlin–rafter ทิ้ง ให้ตรวจแรงบิด Fy·e ที่ rafter ภายหลัง
- เก็บโมเดล rigid diaphragm แยกเป็นอีกโมเดลเพื่อเป็นขอบเขตด้านแข็ง ห้ามปนกับโมเดลหลัก

**แผนทดสอบ**
1. ใส่แรงหน่วย 1 ที่ node ต่างๆ ทั้ง X และ Y ดู load path และสัดส่วนเข้าเสาแต่ละต้น
2. ลดทอนโมเดลเหลือชิ้นส่วนถ่ายแรงข้างจริง (เสา + คานหลัก + rafter)
3. ใส่แรงจากมวลจริง แล้วดูแรงตามแกนใน rafter/คานหลัก, drift ทั้งสองทิศ, accidental torsion

**ยังไม่ได้ยืนยัน (ต้องเปิดมาตรฐานจริง)**
- ASCE 7-16 §12.3.1.1/§12.3.1.3 (flexible diaphragm กับ moment frame), §12.8.2.1 (hn ของหลังคาเอียง), §12.8.4.2 (accidental torsion เมื่อไม่มี diaphragm), Table 12.2-1 (R, Ω0, Cd)
- ข้อที่เทียบเท่าใน มยผ.1301/1302
- วิธี tributary และข้อเสนอทั้งหมดข้างบนเป็นการตีความ ไม่ใช่ข้อบังคับของมาตรฐาน
        """
    )
