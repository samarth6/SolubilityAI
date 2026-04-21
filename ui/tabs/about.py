import streamlit as st

def render_about_tab():
    st.header("What is SolubilityAI?")
    st.markdown(
        "SolubilityAI predicts how well a molecule dissolves in water — "
        "a critical property in drug discovery, environmental science, and chemistry. "
        "It combines **XGBoost machine learning** with a **Fuzzy Logic inference engine** "
        "to give you both an accurate prediction and a confidence-weighted classification."
    )
    st.divider()

    st.subheader("❓ Why does solubility matter?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="about-card">
            <h4>💊 Drug Discovery</h4>
            A drug must dissolve in water to travel through blood and reach its target.
            Testing in a lab costs money and days. SolubilityAI predicts it in under
            1 second — before anyone enters a lab. Pharma companies screen thousands
            of molecules this way to cut costs.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="about-card">
            <h4>🌿 Environmental Science</h4>
            If a chemical spills into a river, its solubility determines how far it spreads.
            SolubilityAI helps assess environmental risk instantly from just the
            molecule's structure — no lab equipment needed.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="about-card">
            <h4>🧪 Academic Research</h4>
            Researchers can screen hundreds of compounds without expensive commercial
            software like Schrödinger or ACD/Labs which cost thousands of dollars per year.
            SolubilityAI is completely free.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="about-card">
            <h4>🌾 Agrochemicals</h4>
            Pesticides and herbicides must dissolve properly to be sprayed on crops.
            SolubilityAI filters candidates before committing to synthesis,
            saving time and money.
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("⚙️ How it works — 5 Steps")

    steps = [
        ("1", "Provide a molecule",
         "Select from the built-in list, search by common name (PubChem), or type a SMILES string. "
         "SMILES is text notation — CCO = Ethanol, c1ccccc1 = Benzene."),
        ("2", "RDKit extracts 10 descriptors",
         "Molecular Weight, LogP, H-Bond Donors/Acceptors, TPSA, Rotatable Bonds, "
         "Aromatic Rings, CSP3 Fraction, Heavy Atom Count, Ring Count."),
        ("3", "XGBoost predicts logS",
         "Our trained XGBoost model predicts the log solubility value. "
         "Higher logS = more soluble. Range: +2 (very soluble) to -8 (nearly insoluble)."),
        ("4", "Fuzzy correction applied",
         "A Mamdani fuzzy inference engine applies Lipinski chemistry rules on top. "
         "Formula: Hybrid logS = XGBoost logS + α × fuzzy correction."),
        ("5", "Confidence-weighted classification",
         "Instead of a hard Soluble/Insoluble answer, you get membership degrees across "
         "4 classes. e.g. Soluble: 83%, Poorly Soluble: 17%. More realistic and useful."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""<div class="about-card">
            <h4>Step {num} — {title}</h4>
            {desc}
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("📏 logS Scale")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""<div class="about-card">
            <h4>Solubility Classes</h4>
            <table style="width:100%;font-size:0.85rem;border-collapse:collapse">
                <tr style="border-bottom:1px solid #dee2e6">
                    <th style="text-align:left;padding:6px;color:#1864ab">logS</th>
                    <th style="text-align:left;padding:6px;color:#1864ab">Class</th>
                    <th style="text-align:left;padding:6px;color:#1864ab">Example</th>
                </tr>
                <tr style="border-bottom:1px solid #dee2e6">
                    <td style="padding:6px;color:#1971c2">≥ 0</td>
                    <td style="padding:6px;color:#1971c2">Highly Soluble</td>
                    <td style="padding:6px;color:#555">Ethanol, Sorbitol</td>
                </tr>
                <tr style="border-bottom:1px solid #dee2e6">
                    <td style="padding:6px;color:#2f9e44">-2 to 0</td>
                    <td style="padding:6px;color:#2f9e44">Soluble</td>
                    <td style="padding:6px;color:#555">Phenol, Pyridine</td>
                </tr>
                <tr style="border-bottom:1px solid #dee2e6">
                    <td style="padding:6px;color:#e67700">-4 to -2</td>
                    <td style="padding:6px;color:#e67700">Poorly Soluble</td>
                    <td style="padding:6px;color:#555">Aspirin, Naphthalene</td>
                </tr>
                <tr>
                    <td style="padding:6px;color:#c92a2a">< -4</td>
                    <td style="padding:6px;color:#c92a2a">Insoluble</td>
                    <td style="padding:6px;color:#555">Decane, Testosterone</td>
                </tr>
            </table>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class="about-card">
            <h4>App Features</h4>
            <p><b style="color:#1864ab">🔬 Prediction</b> — logS, fuzzy class, confidence,
            Lipinski checker, descriptor radar, feature importances.</p>
            <p><b style="color:#1864ab">📊 Model Comparison</b> — R², RMSE, MAE for all
            5 models with parity plots.</p>
            <p><b style="color:#1864ab">🗺️ Chemical Space</b> — PCA/t-SNE scatter showing
            where your molecule sits among training data.</p>
            <p><b style="color:#1864ab">📐 Fuzzy System</b> — Membership curves, interactive
            explorer, rule base, inference breakdown.</p>
            <p><b style="color:#1864ab">📁 Batch Prediction</b> — Upload a CSV of molecules
            and download all predictions at once.</p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("🛠️ Tech Stack")
    cols = st.columns(5)
    stack = [
        ("XGBoost",      "Primary ML model"),
        ("scikit-learn", "Other models + PCA/t-SNE"),
        ("scikit-fuzzy", "Fuzzy logic engine"),
        ("RDKit",        "Molecular descriptors"),
        ("Streamlit",    "Web application"),
    ]
    for i, (tech, desc) in enumerate(stack):
        with cols[i]:
            st.markdown(f"""<div class="about-card" style="text-align:center;padding:12px">
                <b style="color:#1864ab;font-size:0.88rem">{tech}</b><br>
                <span style="font-size:0.73rem;color:#868e96">{desc}</span>
            </div>""", unsafe_allow_html=True)
