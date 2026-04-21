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

* Converts numeric predictions into human-friendly categories:

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
* **Molecular Radar Charts** (descriptor visualization)
* **Model Comparison Dashboard** (R², RMSE, CV scores)
* **Parity Plots** for prediction accuracy

### 📦 Batch Processing

* Upload CSV files with SMILES
* Predict solubility for multiple molecules at once

### 📐 Fuzzy System Viewer

* Visualize membership functions
* Understand how classification decisions are made

---

## 🌍 Real-World Impact

* 🏥 **Drug Discovery**
  Helps identify poorly soluble compounds early in the pipeline

* 🔬 **Research Efficiency**
  Reduces reliance on costly experimental testing

* 🎓 **Educational Tool**
  Demonstrates ML + fuzzy logic integration in chemistry

* ♻️ **Sustainable Science**
  Minimizes unnecessary lab work and resource usage

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Machine Learning:** scikit-learn, XGBoost
* **Fuzzy Logic:** scikit-fuzzy
* **Cheminformatics:** RDKit
* **Visualization:** Plotly
* **Explainability:** SHAP
* **API Integration:** PubChem (via requests)

---


---

## ⚙️ Installation

```bash
git clone https://github.com/samarth6/SolubilityAI.git
cd SolubilityAI

pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Usage Guide

1. Launch the app
2. Select input method (list / search / SMILES)
3. Choose a machine learning model
4. Click **Run Prediction**
5. Explore results across different tabs

---

## 📊 Model Insights

* Ensemble models like **Random Forest** and **XGBoost** typically perform best
* Fuzzy logic enhances interpretability of predictions
* Visualizations help understand molecular relationships and distribution

---


