# ⬡ SolubilityAI — Molecular Solubility Predictor

A hybrid **Machine Learning + Fuzzy Logic** web application that predicts and interprets molecular solubility from SMILES strings. Built using **Streamlit**, this project combines predictive performance with human-readable insights for better understanding of chemical properties.

---

## 🚀 Overview

SolubilityAI is designed to:

* Predict molecular solubility using multiple ML models
* Provide **interpretable results** using fuzzy logic
* Visualize molecular data in an intuitive and interactive way

It is ideal for **students, researchers, and developers** working in cheminformatics, bioinformatics, or applied machine learning.

---

## ✨ Key Features

### 🤖 Machine Learning Models

* Linear Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

### 🔮 Fuzzy Logic Classification

Converts numeric predictions into human-friendly categories:

* Highly Soluble
* Soluble
* Poorly Soluble
* Insoluble

### 🧪 Flexible Input Options

* Select molecules from dataset
* Search by chemical name (via PubChem API)
* Enter custom SMILES strings

### 📊 Visualization & Analysis

* **Chemical Space Mapping** (PCA & t-SNE)
* **Molecular Radar Charts**
* **Model Comparison Dashboard** (R², RMSE, CV scores)
* **Parity Plots**

### 📦 Batch Processing

* Upload CSV files with SMILES
* Predict solubility for multiple molecules

### 📐 Fuzzy System Viewer

* Visualize membership functions
* Understand decision-making logic

---

## 🌍 Real-World Impact

* 🏥 **Drug Discovery**
  Helps identify poorly soluble compounds early

* 🔬 **Research Efficiency**
  Reduces costly experimental trials

* 🎓 **Educational Tool**
  Demonstrates ML + fuzzy logic integration

* ♻️ **Sustainable Science**
  Minimizes unnecessary lab experimentation

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Machine Learning:** scikit-learn, XGBoost
* **Fuzzy Logic:** scikit-fuzzy
* **Cheminformatics:** RDKit
* **Visualization:** Plotly
* **Explainability:** SHAP
* **API Integration:** PubChem (requests)

---

## ⚙️ Virtual Environment Setup

This project uses a **Python virtual environment** for dependency isolation.

### 1️⃣ Create Environment

```bash
python -m venv venv
```

---

### 2️⃣ Activate Environment

#### ▶️ Windows (PowerShell)

```bash
venv\Scripts\activate
```

#### ▶️ macOS / Linux

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Upgrade pip (Recommended)

```bash
python -m pip install --upgrade pip
```

---

### 5️⃣ Deactivate (Optional)

```bash
deactivate
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧪 Usage Guide

1. Launch the app
2. Choose input method (list / search / SMILES)
3. Select ML model
4. Click **Run Prediction**
5. Explore insights across tabs

---

## 📊 Model Insights

* Ensemble models like **Random Forest** and **XGBoost** perform best
* Fuzzy logic improves interpretability
* Visualization reveals molecular relationships

---

## ⚠️ Common Issues

### t-SNE Error

Ensure perplexity is smaller than dataset size.

### RDKit Issues

```bash
pip install rdkit-pypi
```

### Streamlit Not Running

```bash
pip install streamlit
```

---

## 📁 Project Structure (Simplified)

```
core/              # ML models, features, projections
ui/                # Streamlit UI components
app.py             # Main application
requirements.txt   # Dependencies
```


