import io
import numpy as np
import pandas as pd
import streamlit as st
from core.features import extract_features, smiles_is_valid
from core.models   import predict_single
from core.fuzzy    import hybrid_predict

def render_batch_tab(trained_models, scaler, feature_names):
    st.subheader("Batch Prediction")
    st.markdown(
        "Upload a CSV file with a **SMILES** column and get solubility "
        "predictions for all molecules at once. Download the results as CSV."
    )

    
    st.markdown("**Step 1 — Download the template CSV**")
    template = pd.DataFrame({
        "SMILES": ["c1ccccc1", "CCO", "CC(=O)O", "c1ccc(cc1)O"],
        "Name":   ["Benzene",  "Ethanol", "Acetic Acid", "Phenol"],
    })
    st.download_button(
        label="⬇️ Download template CSV",
        data=template.to_csv(index=False),
        file_name="solubility_template.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # this section is mainly for uploading the csv file in batch section
    st.markdown("**Step 2 — Upload your CSV**")
    uploaded = st.file_uploader(
        "Upload CSV (must have a SMILES column)",
        type="csv",
        help="The file must have at least one column named SMILES"
    )

    if uploaded is None:
        st.info("Upload a CSV file above to run batch predictions.")
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    if "SMILES" not in batch_df.columns:
        st.error("CSV must have a column named **SMILES**")
        return

    st.success(f"Loaded {len(batch_df)} molecules")
    st.dataframe(batch_df.head(5), use_container_width=True)

    # this is one is for model selection
    st.markdown("**Step 3 — Choose model and settings**")
    bc1, bc2 = st.columns(2)
    with bc1:
        model_name = st.selectbox("Model", list(trained_models.keys()),
                                   index=len(trained_models)-1, key="batch_model")
    with bc2:
        alpha = st.slider("Fuzzy weight (α)", 0.0, 1.0, 0.25, 0.05, key="batch_alpha")

   
    st.markdown("**Step 4 — Run predictions**")
    if st.button("▶️ Run Batch Prediction", type="primary"):
        model = trained_models[model_name]
        results = []
        progress = st.progress(0)
        status   = st.empty()

        for i, row in batch_df.iterrows():
            smi = str(row["SMILES"]).strip()
            status.text(f"Processing {i+1}/{len(batch_df)}: {smi[:30]}...")
            progress.progress((i+1) / len(batch_df))

            if not smiles_is_valid(smi):
                results.append({
                    "SMILES":       smi,
                    "XGB_logS":     "Invalid",
                    "Hybrid_logS":  "Invalid",
                    "Class":        "Invalid",
                    "Confidence":   "—",
                })
                continue

            feats, _ = extract_features([smi])
            if len(feats) == 0:
                results.append({
                    "SMILES":       smi,
                    "XGB_logS":     "Error",
                    "Hybrid_logS":  "Error",
                    "Class":        "Error",
                    "Confidence":   "—",
                })
                continue

            xgb_logS = predict_single(model, scaler, feats)
            logP     = float(feats[0,1])
            mol_wt   = float(feats[0,0])
            num_hbd  = float(feats[0,2])
            h_logS, fr = hybrid_predict(xgb_logS, logP, mol_wt, num_hbd, alpha)

            row_result = {"SMILES": smi,
                          "XGB_logS":    round(xgb_logS, 3),
                          "Hybrid_logS": round(h_logS,   3),
                          "Class":       fr.dominant,
                          "Confidence":  f"{fr.confidence:.0%}"}
            if "Name" in batch_df.columns:
                row_result["Name"] = row.get("Name", "")
            results.append(row_result)

        progress.empty()
        status.empty()

        result_df = pd.DataFrame(results)
        st.success(f"✅ Done! Predicted {len(result_df)} molecules.")
        st.dataframe(result_df, use_container_width=True)

        
        valid = result_df[result_df["Class"] != "Invalid"]
        if len(valid) > 0:
            st.markdown("**Summary**")
            counts = valid["Class"].value_counts()
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, cls in zip([sc1,sc2,sc3,sc4],
                                 ["Highly Soluble","Soluble","Poorly Soluble","Insoluble"]):
                with col:
                    n = counts.get(cls, 0)
                    pct = f"{n/len(valid)*100:.0f}%"
                    st.metric(cls, f"{n} ({pct})")

        
        csv_out = result_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Results CSV",
            data=csv_out,
            file_name="solubility_predictions.csv",
            mime="text/csv",
        )
