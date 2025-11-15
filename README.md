# Customer Churn

📘 Customer Churn Prediction — Machine Learning Project
## 📌 Overview

This project builds a predictive model for customer churn using the Telco Customer Churn dataset.
The goal is to identify which customers are likely to leave a service provider and understand the drivers behind churn.

* This project demonstrates:
* Data cleaning & preprocessing
* Exploratory data analysis (EDA)
* Feature engineering
* Handling imbalanced data
* Training & evaluating multiple ML models
* Comparing performance with/without SMOTE
* Visualizing ROC curves & confusion matrices
* Drawing business-level insights

It complements my CO₂ vs GDP project by adding a predictive machine-learning example to my portfolio.

## 📂 Dataset

**Source:** Telco Customer Churn (Kaggle)
**Rows:** ~7,000
**Features:** mix of numerical, categorical, behavioral, and account-level details
**Target:**

* Churn (Yes / No)

The dataset is imbalanced (~73% non-churn, ~27% churn), which influences model behavior.

## 🔧 Workflow
1. Data Preprocessing

* Loaded CSV and cleaned missing values
* Converted categorical variables using OneHotEncoding
* Scaled numeric features
* Split data into train/test sets

2. Handling Class Imbalance

* Since churners are the minority class, we evaluated models with and without SMOTE:
* No SMOTE: natural distribution
* With SMOTE: synthetic oversampling of minority class
* Compared performance in both conditions

SMOTE helps recall but can reduce accuracy — important to discuss in business terms.

3. Models Trained

We trained and compared:

* Logistic Regression
* Random Forest
* XGBoost

Each model was evaluated twice:

* Using original training data
* Using SMOTE-balanced training data

## 📊 Model Performance

| Model               | Accuracy (No SMOTE) | Accuracy (SMOTE) | F1 (No SMOTE) | F1 (SMOTE) | ROC-AUC (No SMOTE) | ROC-AUC (SMOTE) |
| ------------------- | ------------------- | ---------------- | ------------- | ---------- | ------------------ | --------------- |
| Logistic Regression | 0.805               | 0.737            | 0.603         | 0.589      | 0.842              | 0.819           |
| Random Forest       | 0.790               | 0.760            | 0.556         | 0.585      | 0.826              | 0.819           |
| XGBoost             | 0.785               | 0.756            | 0.569         | 0.586      | 0.821              | 0.806           |



## 🧠 Insights & Interpretation
**SMOTE Effects**

* Accuracy dropped for all models — expected when balancing classes
* F1-score (recall precision balance) improved for Random Forest & XGBoost
* Logistic Regression performance slightly worsened under SMOTE (sensitive to synthetic noise)
* ROC-AUC remained consistently high (~0.80–0.84)

**Best Performing Model**

* **XGBoost with SMOTE**

    * Best F1-score (important for catching churners)
    * Strong ROC-AUC
    * Robust to class imbalance
