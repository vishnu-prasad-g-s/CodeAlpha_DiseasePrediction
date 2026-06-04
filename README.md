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
- Shape, dtypes, missing value check
- Feature distributions (histograms)
- Correlation heatmap
- Class balance check

### 2. Preprocessing
- No missing values in this dataset
- `StandardScaler` applied for Logistic Regression and SVM
- Train/test split: 80% train, 20% test (`random_state=42`)

### 3. Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble of 100 decision trees |
| Support Vector Machine | RBF kernel |
| XGBoost | Gradient boosted trees — best performer |

### 4. Evaluation Metrics
- Accuracy
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Feature Importance (Random Forest)

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
├── disease_prediction.ipynb   # Main Kaggle notebook
├── README.md                  # This file
└── heart.csv                  # Dataset (or linked from Kaggle)
```

---

## ▶️ How to Run

### On Kaggle (recommended)
1. Open the notebook on Kaggle
2. Add the Heart Disease dataset via **Add Data**
3. Click **Run All**

### Locally
```bash
# Clone the repo
git clone https://github.com/your-username/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Run the notebook
jupyter notebook disease_prediction.ipynb
```

---

## 🧠 Key Findings

- **`cp` (chest pain type)** and **`thalach` (max heart rate)** are the strongest predictors of heart disease.
- **`exang` (exercise-induced angina)** and **`oldpeak` (ST depression)** are strongly negatively correlated — higher values indicate lower disease probability.
- The dataset is well balanced (54/46), so accuracy is a reliable metric here.
- XGBoost outperforms all other models across every metric.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas** — data manipulation
- **Matplotlib / Seaborn** — visualization
- **Scikit-learn** — ML models and evaluation
- **XGBoost** — gradient boosting
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
