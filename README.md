# 🫀 Disease Prediction from Medical Data
### CodeAlpha Machine Learning Internship — Task 4

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-red?style=flat-square)
![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?style=flat-square&logo=kaggle)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

---

## 📌 Objective

Predict the possibility of **heart disease** in a patient based on clinical features such as age, cholesterol level, chest pain type, resting blood pressure, and more. This is a **binary classification** problem — the model outputs either `0` (no disease) or `1` (disease present).

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Name** | Heart Disease Dataset |
| **Source** | [Kaggle — johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Kaggle Path** | `/kaggle/input/datasets/johnsmith88/heart-disease-dataset/heart.csv` |
| **Rows** | 303 patients |
| **Features** | 13 clinical features |
| **Target** | `target` — 0 (no disease), 1 (disease) |
| **Class balance** | 54% no disease / 46% disease |

### Feature description

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

## 🔍 Project Pipeline

```
Raw Data → EDA → Preprocessing → Train/Test Split → Model Training → Evaluation
```

### 1. Exploratory Data Analysis (EDA)
- Dataset shape, first 5 rows (`df.head()`)
- Missing value check (`df.isnull().sum()`)
- Class balance check (`df['target'].value_counts()`)
- Feature distributions — histograms (`df.hist()`)
- Correlation heatmap (`sns.heatmap` with `coolwarm` palette)
- Class balance bar chart (`sns.countplot`)

### 2. Preprocessing
- No missing values in this dataset
- Features split: `X = df.drop('target', axis=1)`, `y = df['target']`
- Train/test split: **80% train / 20% test** (`random_state=42`)
- `StandardScaler` applied for Logistic Regression and SVM
- Tree-based models (Random Forest, XGBoost) use unscaled data

### 3. Models Trained

| Model | Scaling | Key Parameters |
|---|---|---|
| Logistic Regression | ✅ Yes | `max_iter=1000` |
| Random Forest | ❌ No | `n_estimators=100`, `random_state=42` |
| SVM | ✅ Yes | `kernel='rbf'`, `probability=True` |
| XGBoost | ❌ No | `n_estimators=100`, `learning_rate=0.1`, `eval_metric='logloss'` |

### 4. Evaluation
- **Metrics:** Accuracy, F1-Score, ROC-AUC (all 4 models compared)
- **Confusion Matrix** — XGBoost best model (`ConfusionMatrixDisplay`)
- **ROC Curve** — XGBoost best model (`RocCurveDisplay`)
- **Feature Importance** — Random Forest (`feature_importances_`, horizontal bar chart)

---

## 📊 Results

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.836 | 0.849 | 0.911 |
| Random Forest | 0.869 | 0.876 | 0.926 |
| SVM | 0.852 | 0.863 | 0.918 |
| **XGBoost** | **0.885** | **0.890** | **0.941** |

> ✅ **Best model: XGBoost** with 88.5% accuracy and 0.941 ROC-AUC.

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
2. Add the dataset: **johnsmith88/heart-disease-dataset** via **Add Data**
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
| Cell 1 | Imports + load dataset + shape/head/null/class balance check |
| Cell 2 | EDA — histograms, correlation heatmap, class balance plot |
| Cell 3 | Preprocessing — train/test split + StandardScaler |
| Cell 4 | Train all 4 models + print Accuracy / F1 / ROC-AUC comparison table |
| Cell 5 | Confusion matrix + ROC curve (XGBoost) |
| Cell 6 | Feature importance bar chart (Random Forest) |

---

## 🧠 Key Findings

- **`cp` (chest pain type)** and **`thalach` (max heart rate)** are the strongest predictors of heart disease.
- **`exang` (exercise-induced angina)** and **`oldpeak` (ST depression)** are strongly negatively correlated with disease.
- The dataset is well balanced (54/46), making accuracy a reliable metric.
- **XGBoost outperforms** all other models across every metric.
- Random Forest's `feature_importances_` provides interpretable insight into which clinical features drive the prediction.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas** — data loading and manipulation
- **Matplotlib / Seaborn** — EDA visualizations
- **Scikit-learn** — preprocessing, models, evaluation metrics
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
