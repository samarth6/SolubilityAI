import streamlit as st
from core.data import get_molecule_options, get_smiles_from_selection, pubchem_name_to_smiles



def render_sidebar(trained_models, df):
    with st.sidebar:
        st.markdown("## 🧪 SolubilityAI")
        st.caption("XGBoost + Fuzzy Logic Engine")
        st.divider()

        st.markdown("### 🔍 Molecule Input")
        input_mode = st.radio(
    "How to enter molecule",
    ["Pick from list", "Search by name", "Type SMILES"],
    horizontal=False,
    key="molecule_input_mode"
)
        

        smiles = "c1ccc(cc1)O"

        if input_mode == "Pick from list":
            options = get_molecule_options()
            selected = st.selectbox(
                "Select a molecule",
                options=options,
                index=3,
                help="30 built-in molecules with known solubility"
            )
            smiles = get_smiles_from_selection(selected)
            st.caption(f"SMILES: `{smiles}`")

        elif input_mode == "Search by name":
            mol_name = st.text_input(
                "Molecule name",
                placeholder="e.g. Caffeine, Ibuprofen, Glucose",
                help="Searches PubChem database"
            )
            if mol_name:
                with st.spinner("Searching PubChem..."):
                    result = pubchem_name_to_smiles(mol_name)
                if result:
                    smiles = result
                    st.success(f"Found!")
                    st.caption(f"SMILES: `{smiles}`")
                else:
                    st.error("Not found in PubChem. Try a different name.")

        else:
            smiles = st.text_input(
                "SMILES String",
                value="c1ccc(cc1)O",
                help="e.g. CCO = Ethanol, c1ccccc1 = Benzene"
            )

        st.divider()

        
        st.markdown("### 🤖 Model")
        model_name = st.selectbox(
            "Algorithm",
            options=list(trained_models.keys()),
            index=len(trained_models) - 1,
        )

        st.divider()

        
        st.markdown("### ⚙️ Fuzzy Settings")
        alpha = st.slider(
            "Fuzzy correction weight (α)",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05,
            help="0 = pure XGBoost. 0.25 = recommended blend."
        )

        st.divider()

        
        st.markdown("### 🖥️ Options")
        show_lipinski     = st.toggle("Show Lipinski Rule Checker", value=True)
        show_compare      = st.toggle("Show molecule comparison",   value=False)
        show_fuzzy_rules  = st.toggle("Show fuzzy rule details",    value=True)
        show_history      = st.toggle("Show session history",       value=True)
        show_shap         = st.toggle("Show SHAP explanation",      value=False)
        
        n_samples = len(df)

        max_perp = max(2, min(30, (n_samples - 1) // 3))
        perplexity = st.slider(
            "t-SNE perplexity",
            min_value = 2,
            max_value = max_perp,
            value = min(10, max_perp),
            help =f"Must be < dataset size ({n_samples})"
        )

        st.divider()
        predict = st.button("🔬 Run Prediction", use_container_width=True, type="primary")
        st.divider()
        st.caption("Stack: XGBoost · scikit-learn · scikit-fuzzy · RDKit · Streamlit")

    return {
        "smiles":           smiles,
        "model_name":       model_name,
        "alpha":            alpha,
        "show_lipinski":    show_lipinski,
        "show_compare":     show_compare,
        "show_fuzzy_rules": show_fuzzy_rules,
        "show_history":     show_history,
        "show_shap":        show_shap,
        "perplexity":       perplexity,
        "predict":          predict,
    }
