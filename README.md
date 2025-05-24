# Breast Cancer classification using Random Forest
# 🧠 Breast Cancer Classification with Random Forest

## 📖 Project Overview

This project applies a machine learning model to classify breast cancer tumors as **benign** or **malignant** using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. A **Random Forest Classifier** is employed to build a reliable, interpretable model with strong performance on a real-world biomedical dataset.

---

## 📂 Dataset

* **Source**: UCI Machine Learning Repository
* **Access**: `/kaggle/input/breast-cancer-wisconsin-data`
* **Target Variable**: `diagnosis` (M = malignant, B = benign)
* **Features**: 30 numerical features derived from digitized images of fine needle aspirate (FNA) of breast masses

---

## 🛠️ Tools & Libraries

* **Language**: Python
* **Libraries**:

  * `pandas`, `numpy` for data manipulation
  * `matplotlib`, `seaborn` for data visualization
  * `scikit-learn` for machine learning modeling and evaluation

---

## 🔍 Workflow

### 1. Data Preprocessing

* Dropped `id` and unnamed columns
* Encoded `diagnosis`: `M` → 1, `B` → 0
* Checked for nulls and confirmed clean data
* Split dataset (80% training / 20% test)
* Standardized features using `StandardScaler`

### 2. Model Building

* Used **RandomForestClassifier** from Scikit-learn
* Set random state for reproducibility
* Fit model on training data

### 3. Evaluation Metrics

| Metric        | Value |
| ------------- | ----- |
| Accuracy      | 96.5% |
| Precision (M) | 0.98  |
| Recall (M)    | 0.93  |
| F1-Score (M)  | 0.95  |
| ROC-AUC Score | 0.91  |

### 4. Visualizations

* Confusion matrix heatmap
* ROC curve
* Feature importance bar chart

Top contributing features:

* `area_worst`
* `concave points_worst`
* `concave points_mean`

---

## 🔎 Insights

The model performs highly in distinguishing malignant tumors, with strong precision and recall values. Feature importance analysis adds interpretability, aiding in understanding which clinical indicators most affect prediction outcomes.

---

## ✅ How to Run

1. Load dataset from: `/kaggle/input/breast-cancer-wisconsin-data`
2. Run the notebook sequentially from data import to evaluation
3. Visualizations are automatically generated

---

## 🚀 Future Improvements

* Perform hyperparameter tuning using GridSearchCV
* Compare with other models (SVM, XGBoost)
* Deploy using Streamlit or Flask for interactive predictions

---

## 💼 Disclaimer

This project is for educational purposes only. It is not intended for clinical or diagnostic use.

