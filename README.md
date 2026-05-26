# 🫀 XAI Heart Disease Dashboard

![CI](https://github.com/SrajanS7/xai-dashboard/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An interactive **Explainable AI (XAI) dashboard** that trains ML models on the
Heart Disease UCI dataset and explains every prediction using SHAP — with global
feature importance, individual case analysis, and a live what-if simulator.

---

## 📸 Dashboard Panels

| Panel | Description |
|---|---|
| 📊 Dataset Explorer | Feature distributions split by disease class |
| 🤖 Model Comparison | Random Forest vs Logistic Regression — Accuracy, AUC, F1 |
| 🌍 Global SHAP | Feature importance bar chart + beeswarm plot |
| 🔍 Individual Prediction | Per-patient waterfall explanation |
| 🎛️ What-If Simulator | Adjust sliders, watch prediction change in real time |

---

## 🧠 What is SHAP?

SHAP (SHapley Additive exPlanations) is a game-theory-based method for
explaining ML model predictions. For each prediction it assigns every feature
a contribution value — showing exactly how much each clinical measurement
pushed the model toward or away from a disease diagnosis.

This matters in healthcare: a model that can't explain its decisions can't be
trusted in practice.

---

## 🗂️ Project Structure

```
xai-dashboard/
├── app/
│   ├── data_loader.py   # Load + validate Heart Disease UCI dataset
│   ├── trainer.py       # Train Random Forest + Logistic Regression
│   ├── explainer.py     # Compute SHAP values (global + individual)
│   ├── visualiser.py    # Build Plotly charts for all dashboard panels
│   └── ui.py            # Streamlit dashboard — 5 interactive panels
├── models/              # Serialised .pkl files (git-ignored)
├── data/                # heart.csv (git-ignored, downloaded via Kaggle API)
├── tests/               # pytest test suite
├── .github/workflows/   # GitHub Actions CI
├── requirements.txt
└── README.md
```

---

## ⚙️ Stack

| Tool | Purpose |
|---|---|
| `scikit-learn` | Random Forest + Logistic Regression |
| `SHAP` | TreeExplainer for global + individual explanations |
| `Plotly` | Interactive charts |
| `Streamlit` | Dashboard UI |
| `GitHub Actions` | CI — runs full test suite on every push |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/SrajanS7/xai-dashboard.git
cd xai-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Requires a free [Kaggle](https://www.kaggle.com) account:
```bash
pip install kaggle
kaggle datasets download -d johnsmith88/heart-disease-dataset -p data --unzip
```

**4. Launch the dashboard**
```bash
streamlit run app/ui.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

25 tests across data loading, model training, SHAP explanation, and
visualisation layers. CI runs automatically on every push to `main`.

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Random Forest | 0.7705 | 0.8539 | 0.7879 |
| Logistic Regression | 0.8033 | 0.8712 | 0.8235 |

Logistic Regression slightly outperforms Random Forest on this small tabular
dataset — a good reminder that simpler models often win when data is limited.
Random Forest is used for SHAP explanations as TreeExplainer gives exact
(not approximate) values for tree-based models.

---

## 🎓 Academic Context

This project applies XAI concepts studied at **Paderborn University**, including
SHAP-based feature attribution, model-agnostic vs model-specific explanation
methods, and the trade-off between predictive performance and interpretability
in high-stakes domains.

---

## 👤 Author

**Srajan Sharma** — Paderborn University  
[GitHub](https://github.com/SrajanS7)

---

## 📄 License

MIT
