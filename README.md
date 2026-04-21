# ⬡ SolubilityAI — Molecular Solubility Predictor

A hybrid **XGBoost + Fuzzy Logic** bioinformatics web app built with Streamlit.

## Features
- 🤖 **5 Models** — Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- 🔮 **Fuzzy Classification** — Membership degrees across Highly Soluble / Soluble / Poorly Soluble / Insoluble
- 🗺️ **Chemical Space** — PCA & t-SNE projections with query molecule highlighted
- 📡 **Molecular Radar** — Descriptor profile visualization
- 📊 **Model Comparison** — R², RMSE, CV scores, parity plots
- 📐 **Fuzzy System Viewer** — Full membership function visualization

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Usage
1. Enter a SMILES string in the sidebar (e.g. `c1ccccc1` for benzene)
2. Select your model
3. Click **PREDICT SOLUBILITY**
4. Explore tabs for deeper analysis

## Tech Stack
- `streamlit` — UI framework
- `xgboost` / `scikit-learn` — ML models
- `scikit-fuzzy` — Fuzzy logic system
- `rdkit-pypi` — Molecular descriptor extraction
- `plotly` — Interactive visualizations
- `shap` — Model explainability
