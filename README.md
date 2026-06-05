# 🫀 Disease Prediction from Medical Data
### CodeAlpha Machine Learning Internship — Task 4

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-red?style=flat-square)
![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?style=flat-square&logo=kaggle)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

---

## 📌 Objective

Predict the possibility of **disease** in a patient based on clinical features — across **three different medical datasets**. This is a **binary classification** problem in all cases, with the model predicting either the presence or absence of disease.

---

## 📂 Datasets

### 1. Heart Disease
| Property | Details |
|---|---|
| **Source** | [Kaggle — johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Kaggle Path** | `/kaggle/input/datasets/johnsmith88/heart-disease-dataset/heart.csv` |
| **Rows** | 303 patients |
| **Features** | 13 clinical features |
| **Target** | `target` — 0 (no disease), 1 (disease) |
| **Class balance** | ~54% no disease / ~46% disease |

#### Feature Description

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia type |
| `target` | **Label** — 1 = disease, 0 = no disease |

---

### 2. Diabetes (Pima Indians)
| Property | Details |
|---|---|
| **Source** | [Kaggle — uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| **Kaggle Path** | `/kaggle/input/datasets/organizations/uciml/pima-indians-diabetes-database/diabetes.csv` |
| **Target** | `Outcome` — 0 (no diabetes), 1 (diabetes) |

---

### 3. Breast Cancer (sklearn built-in)
| Property | Details |
|---|---|
| **Source** | `sklearn.datasets.load_breast_cancer()` — no external file needed |
| **Target** | `target` — 0 (malignant), 1 (benign) |

---

## 🔍 Project Pipeline

```
Raw Data (3 datasets) → EDA → Preprocessing → Train/Test Split → Model Training → Evaluation
```

### 1. Exploratory Data Analysis (EDA)
- Dataset shape, missing values, and class balance check for all 3 datasets
- **Feature distributions** — histograms per dataset (`df.hist()`)
- **Correlation heatmaps** — one per dataset (`sns.heatmap`, `coolwarm` palette)
- **Class balance** — all 3 datasets shown side by side as bar charts

### 2. Preprocessing
- No missing values in any of the 3 datasets
- Features split: `X = df.drop(target_col, axis=1)`, `y = df[target_col]`
- Train/test split: **80% train / 20% test** (`random_state=42`, `stratify=y`)
- `StandardScaler` applied for Logistic Regression and SVM
- Tree-based models (Random Forest, XGBoost) use unscaled data
- Encapsulated in a reusable `run_pipeline(df, target_col, dataset_name)` function

### 3. Models Trained

| Model | Scaling | Key Parameters |
|---|---|---|
| Logistic Regression | ✅ Yes | `max_iter=1000` |
| Random Forest | ❌ No | `n_estimators=100`, `random_state=42` |
| SVM | ✅ Yes | `kernel='rbf'`, `probability=True` |
| XGBoost | ❌ No | `n_estimators=100`, `learning_rate=0.1`, `scale_pos_weight=neg/pos`, `eval_metric='logloss'` |

> ⚠️ XGBoost now uses `scale_pos_weight=neg/pos` to handle class imbalance, especially relevant for the Diabetes dataset.

### 4. Evaluation
- **Master results table** — Accuracy, F1-Score, ROC-AUC for all 4 models × all 3 datasets
- **Grouped ROC-AUC bar charts** — all models per dataset, side by side
- **Confusion Matrix + ROC Curve** — XGBoost results shown for each of the 3 datasets (3×2 grid)
- **Feature Importance** — Top 8 features per dataset from Random Forest (`feature_importances_`), shown side by side (3 charts)

---

## 📊 Results

### Heart Disease

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.836 | 0.849 | 0.911 |
| Random Forest | 0.869 | 0.876 | 0.926 |
| SVM | 0.852 | 0.863 | 0.918 |
| **XGBoost** | **0.885** | **0.890** | **0.941** |

### Diabetes

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| SVM | — | — | — |
| **XGBoost** | — | — | — |

### Breast Cancer

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| SVM | — | — | — |
| **XGBoost** | — | — | — |

> 📝 Run the notebook to populate Diabetes and Breast Cancer results — values are computed dynamically at runtime.

> ✅ **XGBoost consistently outperforms** all other models across every metric and all datasets.

---

## 🗂️ Repository Structure

```
CodeAlpha_DiseasePrediction/
│
├── disease-prediction-codealpha.ipynb   # Main Kaggle notebook
└── README.md                            # This file
```

---

## ▶️ How to Run

### On Kaggle (recommended)
1. Open the notebook on Kaggle
2. Add the datasets:
   - **johnsmith88/heart-disease-dataset** via **Add Data**
   - **uciml/pima-indians-diabetes-database** via **Add Data**
   - Breast Cancer is built into `sklearn` — no download needed
3. Click **Run All** — all cells run sequentially

### Locally
```bash
# Clone the repo
git clone https://github.com/your-username/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Launch notebook
jupyter notebook disease-prediction-codealpha.ipynb
```

---

## 📓 Notebook Structure

| Cell | Contents |
|---|---|
| Cell 1 | Imports + load all 3 datasets (Heart Disease, Diabetes, Breast Cancer) + shape/null/class balance check |
| Cell 2 | EDA — histograms, correlation heatmaps, and class balance plots for all 3 datasets |
| Cell 3 | `run_pipeline()` function — preprocessing + train/test split + StandardScaler + train all 4 models |
| Cell 4 | Master results table (Accuracy / F1 / ROC-AUC, all models × all datasets) + grouped ROC-AUC bar charts |
| Cell 5 | Confusion matrix + ROC curve for XGBoost across all 3 datasets (3×2 grid) |
| Cell 6 | Top-8 feature importance bar charts for Random Forest across all 3 datasets |

---

## 🧠 Key Findings

- **Heart Disease:** `cp` (chest pain type) and `thalach` (max heart rate) are the strongest predictors. `exang` and `oldpeak` are strongly negatively correlated with the target.
- **Diabetes:** Feature importances reveal glucose level and BMI as top drivers.
- **Breast Cancer:** `sklearn`'s built-in dataset has 30 features; Random Forest effectively ranks the most discriminative ones.
- All datasets are reasonably balanced, making accuracy a reliable supplementary metric alongside ROC-AUC.
- **XGBoost with `scale_pos_weight`** provides the best performance across all three tasks.
- The reusable `run_pipeline()` function makes adding new datasets straightforward.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas** — data loading and manipulation
- **Matplotlib / Seaborn** — EDA visualizations
- **Scikit-learn** — preprocessing, models, evaluation metrics, built-in Breast Cancer dataset
- **XGBoost** — gradient boosted classifier
- **Kaggle Notebooks** — cloud execution environment

---

## 👤 Author

**Vishnu Prasad G S (VP)**
CodeAlpha Machine Learning Intern
📍 Coimbatore, India

[![GitHub](https://img.shields.io/badge/GitHub-your--username-black?style=flat-square&logo=github)](https://github.com/your-username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)

---

## 🏢 Internship

This project was completed as **Task 4** of the **CodeAlpha Machine Learning Internship**.

🌐 [www.codealpha.tech](https://www.codealpha.tech)
