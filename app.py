import streamlit as st

st.set_page_config(
    page_title="SolubilityAI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>

.header-container {
    display: flex;
    align-items: center;
    gap: 4px;   
    margin-bottom: 0;
}


.header-icon {
    font-size: 32px;
    margin: 0;
    padding: 0;
}


.header-title {
    font-size: 34px;
    font-weight: 700;
    margin: 0;
    padding: 0;
}


.header-subtitle {
    margin-top: -8px;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="header-container">
    <span class="header-icon">⬡</span>
    <div class="header-title">SolubilityAI</div>
</div>
<div class="header-subtitle">
    Molecular Solubility Prediction — XGBoost + Fuzzy Logic
</div>
""", unsafe_allow_html=True)


st.markdown("---")

from ui.styles  import inject_styles
from ui.sidebar import render_sidebar
from ui.tabs.about          import render_about_tab
from ui.tabs.prediction     import render_prediction_tab
from ui.tabs.comparison     import render_comparison_tab
from ui.tabs.chemical_space import render_chemical_space_tab
from ui.tabs.fuzzy_system   import render_fuzzy_system_tab
from ui.tabs.batch          import render_batch_tab
from core.data     import get_demo_dataset
from core.features import extract_features
from core.models   import train_all_models

inject_styles()



df = get_demo_dataset()
X, feature_names = extract_features(df["smiles"].tolist())
y = df["logS"].values
trained_models, scaler, metrics = train_all_models(X, y)


cfg = render_sidebar(trained_models, df)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 About",
    "🔬 Prediction",
    "📊 Model Comparison",
    "🗺️ Chemical Space",
    "📐 Fuzzy System",
    "📁 Batch Prediction",
])

with tab0: render_about_tab()
with tab1: render_prediction_tab(cfg, trained_models, scaler, X, y, feature_names, df)
with tab2: render_comparison_tab(cfg, trained_models, scaler, metrics, X, y)
with tab3: render_chemical_space_tab(cfg, scaler, X, y, df)
with tab4: render_fuzzy_system_tab(cfg, trained_models, scaler)
with tab5: render_batch_tab(trained_models, scaler, feature_names)
