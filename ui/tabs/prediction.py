import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.fuzzy  import hybrid_predict
from core.data     import get_name_from_smiles
from core.features import extract_features, smiles_is_valid, check_lipinski
from core.models   import predict_single, get_feature_importances
from core.fuzzy    import hybrid_predict, CLASS_COLORS

def render_prediction_tab(cfg, trained_models, scaler, X, y, feature_names, df):
    smiles    = cfg["smiles"]
    mol_name  = get_name_from_smiles(smiles)

    if not smiles_is_valid(smiles):
        st.error("❌ Invalid SMILES string.")
        return

    query_X, _ = extract_features([smiles])
    if len(query_X) == 0:
        st.error("❌ Could not compute descriptors.")
        return

    model       = trained_models[cfg["model_name"]]
    xgb_logS    = predict_single(model, scaler, query_X)
    logP        = float(query_X[0, 1])
    mol_wt      = float(query_X[0, 0])
    num_hbd     = float(query_X[0, 2])
    hybrid_logS, fuzzy_result = hybrid_predict(
        xgb_logS, logP, mol_wt, num_hbd, cfg["alpha"]
    )
    dom_color = fuzzy_result.color

    
    if "history" not in st.session_state:
        st.session_state.history = []
    entry = {
        "Molecule":   mol_name,
        "SMILES":     smiles,
        "XGB logS":   round(xgb_logS,    3),
        "Hybrid logS":round(hybrid_logS,  3),
        "Class":      fuzzy_result.dominant,
        "Confidence": f"{fuzzy_result.confidence:.0%}",
    }
    if not st.session_state.history or st.session_state.history[-1]["SMILES"] != smiles:
        st.session_state.history.append(entry)

    
    st.subheader(f"Results — {mol_name}")
    st.caption(f"`{smiles}`  ·  Model: {cfg['model_name']}  ·  α = {cfg['alpha']}")
    st.divider()

    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <span class="val">{xgb_logS:.3f}</span>
            <span class="lbl">XGBoost LogS</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#7950f2">
            <span class="val" style="color:#5f3dc4">{hybrid_logS:.3f}</span>
            <span class="lbl">Hybrid LogS</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card" style="border-left-color:{dom_color}">
            <span class="val" style="color:{dom_color};font-size:1rem">
            {fuzzy_result.dominant}</span>
            <span class="lbl">Solubility Class</span>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#e67700">
            <span class="val" style="color:#e67700">{fuzzy_result.confidence:.0%}</span>
            <span class="lbl">Confidence</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

   # application for the lipinski's rule of 5
    if cfg.get("show_lipinski"):
        st.markdown("**Lipinski Rule of Five**")
        st.caption("Standard drug-likeness filter — oral drugs typically pass all 4 rules")
        rules = check_lipinski(query_X[0])
        passed = sum(1 for _, (val, ok) in rules.items() if ok)
        lip_cols = st.columns(4)
        for i, (rule, (value, ok)) in enumerate(rules.items()):
            with lip_cols[i]:
                icon = "✅" if ok else "❌"
                css_class = "rule-pass" if ok else "rule-fail"
                st.markdown(f"""<div class="{css_class}">
                    {icon} <b>{rule}</b><br>
                    <span style="font-size:0.85rem">Value: {value}</span>
                </div>""", unsafe_allow_html=True)
        if passed == 4:
            st.success(f"✅ Passes all 4 Lipinski rules — good oral drug candidate")
        elif passed == 3:
            st.warning(f"⚠️ Passes {passed}/4 rules — borderline drug-likeness")
        else:
            st.error(f"❌ Passes only {passed}/4 rules — poor oral drug-likeness")

        st.markdown("<br>", unsafe_allow_html=True)

   
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Fuzzy Membership Degrees**")
        st.caption("How much does this molecule belong to each class?")
        classes = list(fuzzy_result.memberships.keys())
        values  = list(fuzzy_result.memberships.values())
        colors  = [CLASS_COLORS[c] for c in classes]
        fig = go.Figure(go.Bar(
            x=classes, y=values, marker_color=colors,
            text=[f"{v:.3f}" for v in values], textposition="outside",
            textfont=dict(color="#212529", size=12),
        ))
        fig.update_layout(
            height=300, margin=dict(t=20, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(color="#212529"),
            yaxis=dict(range=[0, 1.3], gridcolor="#e9ecef",
                       title="Membership (0 to 1)"),
            xaxis=dict(gridcolor="#e9ecef"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Molecular Descriptor Profile**")
        st.caption("Radar of the 10 physicochemical properties")
        arr = query_X[0]
        feat_norm = (arr - arr.min()) / (np.ptp(arr) + 1e-9)
        short = ["MolWt","LogP","HBD","HBA","TPSA",
                 "RotBonds","ArRings","CSP3","HeavyAt","Rings"]
        fig2 = go.Figure(go.Scatterpolar(
            r=feat_norm.tolist() + [feat_norm[0]],
            theta=short + [short[0]],
            fill="toself", fillcolor="rgba(51,154,240,0.12)",
            line=dict(color="#339af0", width=2),
            marker=dict(size=4, color="#339af0"),
        ))
        fig2.update_layout(
            polar=dict(
                bgcolor="white",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#dee2e6"),
                angularaxis=dict(gridcolor="#dee2e6"),
            ),
            paper_bgcolor="white", font=dict(color="#212529"),
            height=300, margin=dict(t=20, b=10), showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

   
    if cfg.get("show_compare"):
        st.divider()
        st.markdown("**Side-by-Side Molecule Comparison**")
        st.caption("Compare two molecules and their predictions")
        ca, cb = st.columns(2)
        with ca:
            smi2 = st.text_input("Second molecule SMILES", value="CC(=O)Oc1ccccc1C(=O)O",
                                  key="compare_smi")
        with cb:
            st.markdown("<br>", unsafe_allow_html=True)

        if smi2:
            q2, _ = extract_features([smi2])
            if len(q2) > 0:
                xgb2 = predict_single(model, scaler, q2)
                h2, fr2 = hybrid_predict(xgb2, float(q2[0,1]),
                                         float(q2[0,0]), float(q2[0,2]), cfg["alpha"])
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{mol_name}**")
                    st.metric("Hybrid logS", f"{hybrid_logS:.3f}")
                    st.metric("Class", fuzzy_result.dominant)
                    st.metric("Confidence", f"{fuzzy_result.confidence:.0%}")
                with col2:
                    st.markdown(f"**{get_name_from_smiles(smi2)}**")
                    st.metric("Hybrid logS", f"{h2:.3f}")
                    st.metric("Class", fr2.dominant)
                    st.metric("Confidence", f"{fr2.confidence:.0%}")

    
    importances = get_feature_importances(model, feature_names)
    if importances:
        st.markdown(f"**Feature Importances — {cfg['model_name']}**")
        st.caption("Which properties drove this prediction the most?")
        items = sorted(importances.items(), key=lambda x: x[1])
        fig3 = go.Figure(go.Bar(
            x=[v for _, v in items], y=[k for k, _ in items],
            orientation="h", marker_color="#339af0",
            text=[f"{v:.3f}" for _, v in items], textposition="outside",
            textfont=dict(color="#212529", size=11),
        ))
        fig3.update_layout(
            height=300, paper_bgcolor="white", plot_bgcolor="white",
            font=dict(color="#212529"),
            xaxis=dict(title="Importance", gridcolor="#e9ecef"),
            yaxis=dict(gridcolor="#e9ecef"),
            margin=dict(t=10, b=10, r=60),
        )
        st.plotly_chart(fig3, use_container_width=True)

    
    if cfg.get("show_fuzzy_rules"):
        delta = hybrid_logS - xgb_logS
        st.markdown("**How was this prediction made?**")
        st.markdown(f"""<div class="info-box">
            <b>Step 1 — XGBoost prediction</b><br>
            &nbsp;&nbsp;logS = {xgb_logS:.4f} log(mol/L)<br><br>
            <b>Step 2 — Fuzzy correction</b><br>
            &nbsp;&nbsp;δ = {delta:+.4f} &nbsp; (α = {cfg["alpha"]})<br><br>
            <b>Step 3 — Hybrid result</b><br>
            &nbsp;&nbsp;{xgb_logS:.4f} {delta:+.4f} = <b>{hybrid_logS:.4f}</b><br><br>
            <b>Step 4 — Fuzzy classification</b><br>
            &nbsp;&nbsp;<span style="color:{dom_color};font-weight:600">
            {fuzzy_result.dominant}</span>
            &nbsp; with {fuzzy_result.confidence:.1%} confidence
        </div>""", unsafe_allow_html=True)

    # table for molecular descriptors as they are very vital for the entire process
    st.markdown("**Computed Molecular Descriptors**")
    desc_df = pd.DataFrame({
        "Descriptor": feature_names,
        "Value": [round(float(v), 4) for v in query_X[0]],
    })
    st.dataframe(desc_df.set_index("Descriptor").T, use_container_width=True)

    
    if cfg.get("show_history") and len(st.session_state.history) > 0:
        st.divider()
        st.markdown("**Session History**")
        st.caption("All molecules predicted in this session")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

    # SHAP stands for SHapley Additive exPlanations. It’s a method used to explain machine learning predictions
    # for example how much did each feature contributed to the entire system - positively or negatively
    if cfg.get("show_shap"):
        try:
            import shap, matplotlib.pyplot as plt
            st.markdown("**SHAP Feature Explanation**")
            st.caption("Which features pushed the logS prediction up or down?")
            X_scaled = scaler.transform(X)
            q_scaled = scaler.transform(query_X)
            explainer = shap.Explainer(model, X_scaled)
            shap_vals = explainer(q_scaled)
            fig_s, _ = plt.subplots(figsize=(8, 3))
            shap.plots.waterfall(shap_vals[0], show=False)
            st.pyplot(fig_s, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP not available for this model. ({e})")
