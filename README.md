# Diabetes Disease Diagnosis — ML Case Study

A comprehensive machine learning case study on the PIMA Indian Diabetes dataset, covering the full pipeline from exploratory data analysis to cost-benefit deployment analysis.

---

## Overview

| Item | Details |
|------|---------|
| Dataset | PIMA Indian Diabetes (768 samples, 8 features) |
| Models | Logistic Regression, Decision Tree, Random Forest, SVM, Neural Network |
| Best Model | Neural Network (MLP) — AUC-ROC: 0.8315, Recall: 83.33% |
| Optimal Threshold | 0.30 (ROI: 657.7%, Net Benefit: $1.49M per 154 patients) |

---

## Project Structure

```
├── Case study/diabetes_ml_project/
│   ├── code/                   # Python scripts
│   ├── data_exports/           # CSV results
│   ├── visualizations/         # 31 PNG charts (EDA, evaluation, etc.)
│   ├── final_outputs/          # Key summary charts
│   └── reports/                # Detailed text reports
└── ML CASE STUDY REPORT.pdf    # Full written report
```

---

## Pipeline

1. **Preprocessing** — Replace biological zeros with NaN, median imputation (experimentally validated), feature engineering (Glucose × BMI, Age_Group)
2. **Splitting** — Stratified 80/20 train-test split
3. **Balancing** — Random oversampling on training set only (50-50)
4. **Scaling** — MinMaxScaler fitted on balanced training data
5. **Training** — 5 algorithms trained and compared
6. **Evaluation** — Accuracy, Recall, AUC-ROC, F1, cross-validation
7. **Analysis** — Feature importance, hyperparameter tuning, cost-benefit

---

## Model Results

| Rank | Model | AUC-ROC | Accuracy | Recall | F1 |
|------|-------|---------|----------|--------|----|
| 1 | Neural Network (MLP) | 0.8315 | 79.22% | 83.33% | 0.7377 |
| 2 | Random Forest | 0.8272 | 73.38% | 70.37% | 0.6944 |
| 3 | SVM (RBF) | 0.8176 | 75.32% | 72.22% | 0.7027 |
| 4 | Logistic Regression | 0.8126 | 74.03% | 72.22% | 0.7027 |
| 5 | Decision Tree | 0.7771 | 73.38% | 64.81% | 0.6471 |

---

## Key Findings

- **Feature Engineering**: The engineered `Glucose × BMI` interaction became the single most important feature (score: 0.986), outperforming all original features.
- **Preprocessing Matters**: Median imputation outperformed mean and mode by +1.56% AUC — validated experimentally, not assumed.
- **Hyperparameter Tuning Risk**: Tuning Random Forest with Grid Search *degraded* recall by 10.53% due to overfitting (`min_samples_leaf=1`). Default regularisation was actually optimal.
- **Economic Optimum ≠ Statistical Optimum**: Lowering the classification threshold from 0.50 → 0.30 increased net benefit by $304.9K (+25.8%) by catching 4 more diabetics, since a false negative costs 142× more than a false positive.

---

## Cost-Benefit Analysis

| Model | Net Benefit | ROI |
|-------|------------|-----|
| Neural Network | $1,183.8K (default) / $1,488.7K (optimal) | 302.6% / 657.7% |
| SVM | $950.4K | 182.9% |
| Random Forest | $640.7K | 92.9% |
| Logistic Regression | $563.2K | 76.9% |
| Decision Tree | $407.0K | 49.7% |

**At scale** (10,000 annual screenings): projected **$76.87 million** net benefit.

---

## Top Predictive Features

1. `Glucose_BMI_Interaction` (engineered) — 0.986
2. `Glucose` — 0.648
3. `BMI` — 0.345

---

## Code

| Script | Description |
|--------|-------------|
| `diabetes_eda.py` | Exploratory data analysis (10 visualizations) |
| `preprocessing_comparison.py` | Mean vs Median vs Mode imputation comparison |
| `model_training.py` | Train all 5 models with full pipeline |
| `model_evaluation.py` | Metrics, ROC curves, confusion matrices, CV |
| `feature_importance.py` | Cross-model feature importance analysis |
| `hyperparameter_tuning.py` | Grid search with 96 combinations |
| `cost_benefit_analysis.py` | Economic analysis and threshold optimization |
| `project_summary.py` | Generate summary report |

---

## Deployment Configuration

```
Model      : MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh', solver='adam')
Threshold  : 0.30  (use predict_proba[:, 1] >= 0.30)
Scaler     : MinMaxScaler (fit on balanced training data only)
Monitor    : Recall >= 80%, FN rate < 20%
Retrain    : Quarterly, or immediately if FN rate > 20%
```

---

## Dataset

[PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — 768 female patients of Pima Indian heritage, 8 clinical features, binary outcome (diabetic/non-diabetic).
