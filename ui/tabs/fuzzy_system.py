import numpy as np
import plotly.graph_objects as go
import streamlit as st
from core.features import extract_features
from core.models   import predict_single
from core.fuzzy    import (membership_curves, classify, hybrid_predict,
                            CLASS_COLORS, FILL_COLORS, FuzzyInferenceEngine)

FUZZY_RULES = [
    ("High LogP + High MolWt",        "Insoluble",       "#c92a2a"),
    ("Low LogP + High H-Bond Donors",  "Highly Soluble",  "#1971c2"),
    ("Medium LogP + Medium MolWt",     "Soluble",         "#2f9e44"),
    ("High LogP + Low H-Bond Donors",  "Poorly Soluble",  "#e67700"),
    ("Low LogP",                       "Soluble",         "#2f9e44"),
    ("High TPSA",                      "Soluble",         "#2f9e44"),
]

def render_fuzzy_system_tab(cfg, trained_models, scaler):
    query_X, _ = extract_features([cfg["smiles"]])
    logS_mark = xgb_logS = hybrid_logS = logP = mol_wt = num_hbd = None

    if len(query_X) > 0:
        model    = trained_models[cfg["model_name"]]
        xgb_logS = predict_single(model, scaler, query_X)
        logP     = float(query_X[0,1])
        mol_wt   = float(query_X[0,0])
        num_hbd  = float(query_X[0,2])
        hybrid_logS, fuzzy_result = hybrid_predict(
            xgb_logS, logP, mol_wt, num_hbd, cfg["alpha"])
        logS_mark = hybrid_logS

    st.subheader("Fuzzy Membership Functions")
    st.caption("Each curve shows how a logS value maps to a class. "
               "Dashed line = your molecule.")

    curves = membership_curves()
    xs = curves["x"]
    fig = go.Figure()
    for cls_name, color in CLASS_COLORS.items():
        fig.add_trace(go.Scatter(x=xs, y=curves[cls_name], name=cls_name,
            mode="lines", line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=FILL_COLORS[cls_name]))
    if logS_mark is not None:
        fig.add_vline(x=logS_mark, line=dict(color="#555", dash="dot", width=2))
        fig.add_annotation(x=logS_mark, y=1.1,
            text=f"logS = {logS_mark:.3f}", font=dict(size=11, color="#333"),
            showarrow=False, bgcolor="#fff", bordercolor="#aaa", borderwidth=1)
    fig.update_layout(height=380, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#212529"),
        xaxis=dict(title="logS (log mol/L)", gridcolor="#e9ecef", zeroline=False),
        yaxis=dict(title="Membership μ(x)", range=[0,1.22], gridcolor="#e9ecef"),
        margin=dict(t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explore Any logS Value")
    slider_val = st.slider("logS", -8.0, 2.0,
        float(logS_mark) if logS_mark else -2.0, 0.05)
    result = classify(slider_val)
    cols = st.columns(4)
    for i,(cls_name,mu) in enumerate(result.memberships.items()):
        with cols[i]:
            c = CLASS_COLORS[cls_name]
            st.markdown(f"""<div class="metric-card" style="border-left-color:{c}">
                <span class="val" style="color:{c}">{mu:.3f}</span>
                <span class="lbl">{cls_name}</span>
            </div>""", unsafe_allow_html=True)
    st.markdown(f"**Dominant:** {result.dominant} — {result.confidence:.1%} confidence")

    if cfg.get("show_fuzzy_rules"):
        st.subheader("Fuzzy Rule Base")
        st.caption("IF-THEN rules encoding Lipinski chemistry knowledge")
        rc = st.columns(3)
        for i,(cond,res,color) in enumerate(FUZZY_RULES):
            with rc[i%3]:
                st.markdown(f"""<div class="rule-card" style="border-left-color:{color}">
                    <div class="cond">IF &nbsp;{cond}</div>
                    <div class="res" style="color:{color}">&#8594; {res}</div>
                </div>""", unsafe_allow_html=True)

    if logS_mark is not None:
        st.subheader("Inference Breakdown")
        engine = FuzzyInferenceEngine()
        delta  = engine.infer(logP, mol_wt, num_hbd)
        st.markdown(f"""<div class="info-box">
            <b>Inputs</b> &nbsp; LogP = {logP:.3f} | MolWt = {mol_wt:.1f} | HBD = {num_hbd:.0f}<br><br>
            <b>Fuzzy delta (δ)</b> = {delta:+.4f}<br>
            <b>Applied correction</b> = {cfg["alpha"]} × {delta:.4f} = {cfg["alpha"]*delta:+.4f}<br><br>
            <b>XGBoost logS</b> = {xgb_logS:.4f}<br>
            <b>Hybrid logS</b> = {hybrid_logS:.4f}
        </div>""", unsafe_allow_html=True)
