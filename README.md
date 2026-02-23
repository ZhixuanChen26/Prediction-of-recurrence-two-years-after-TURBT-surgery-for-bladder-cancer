# Prediction of 2-Year Recurrence After TURBT Surgery for Bladder Cancer

A machine learning pipeline for predicting post-surgical recurrence risk in non-muscle-invasive bladder cancer (NMIBC) patients, developed in collaboration with Sir Run Run Shaw Hospital, Zhejiang University School of Medicine.

## Overview

This project develops and compares 6 ML models to predict whether bladder cancer patients will experience recurrence within 2 years of TURBT (transurethral resection of bladder tumor) surgery, using multimodal clinical data.

- **Dataset:** 454 patients (316 non-recurrence, 138 recurrence)
- **Data sources:** Patient demographics (EMR), blood test results, pathology reports, CT imaging
- **Best model:** Random Forest — AUC **0.7136**

## Pipeline

```
Data Collection → Missing Value Imputation → Feature Selection → Model Training → SHAP Analysis
```

1. **Preprocessing** (`imputation.py`, `distribution.py`): Missing value imputation, one-hot encoding, log transformation for skewed variables
2. **Feature Selection** (`preliminary_screening.py`, `univariate_analysis_selection.py`, `Lasso.py`, `lasso_selection.py`, `elastic.py`): Spearman correlation → univariate analysis → union of LASSO and Elastic Net → 6 final features
3. **Modeling** (`rf.py`, `xgb.py`, `catb.py`, `logis.py`, `knn.py`, `svm.py`): 6 models with nested 5-fold cross-validation
4. **Prediction** (`prediction.py`): Final model inference
5. **SHAP Analysis**: Feature importance and interpretability for the best model

## Final Features Selected

| Feature | Description |
|---|---|
| 恶性值 | Tumor malignancy grade |
| 坏死值 | Tumor necrosis |
| 肾盂积水值 | Hydronephrosis |
| 浸润值 | Invasion depth |
| 形状值 | Tumor shape |
| 红细胞计数(RBC#)-尿液 | Urine red blood cell count |

## Model Performance Comparison

| Model | Validation AUC | 95% CI | Stability CV | Train-Val Gap | PR AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.7136** | [0.6507, 0.7765] | 0.1005 | 0.0447 | 0.5145 |
| K-NN | 0.7091 | [0.6400, 0.7782] | 0.1112 | 0.0188 | 0.5195 |
| CatBoost | 0.7025 | [0.6577, 0.7472] | 0.0727 | 0.0316 | 0.5156 |
| SVM | 0.7009 | [0.6514, 0.7504] | 0.0806 | 0.0131 | 0.5037 |
| XGBoost | 0.7007 | [0.6489, 0.7525] | 0.0844 | 0.0528 | 0.4922 |
| Logistic Regression | 0.6931 | [0.6505, 0.7357] | — | 0.0137 | — |

## Top SHAP Features (Random Forest)

| Feature | Mean Absolute SHAP | Influence |
|---|---|---|
| Urine RBC count | 0.0578 | High |
| Tumor malignancy grade | 0.0479 | Medium |
| Invasion depth | 0.0391 | Low |
| Tumor shape | 0.0200 | Low |
| Hydronephrosis | 0.0119 | Low |
| Tumor necrosis | 0.0105 | Low |

## Requirements

```
pandas
numpy
scikit-learn
xgboost
catboost
shap
matplotlib
seaborn
```

## Note on Data

Patient data is not included in this repository due to privacy and ethics requirements. The dataset was approved by the Ethics Committee of Sir Run Run Shaw Hospital (No. 20250399).
