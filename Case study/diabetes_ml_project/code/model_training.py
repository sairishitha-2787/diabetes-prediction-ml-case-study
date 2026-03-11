# =============================================================================
# PIMA DIABETES — FINAL MODEL TRAINING (5 ML ALGORITHMS)
# Preprocessing: MEDIAN imputation (winner from preprocessing_comparison.py)
# =============================================================================
# Pipeline summary:
#   1. Median imputation  ->  feature engineering  ->  80/20 stratified split
#   ->  RandomOverSampler (train only)  ->  MinMaxScaler  ->  5 ML models
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for PNG output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time
import warnings
from datetime import datetime

# --- Preprocessing & modelling ---
from sklearn.impute            import SimpleImputer
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import MinMaxScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.svm               import SVC
from sklearn.neural_network    import MLPClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from imblearn.over_sampling    import RandomOverSampler
from preprocessing             import build_pipeline, COLUMN_NAMES, ZERO_COLS, ENG_FEATURES

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# PART 1: FINAL PREPROCESSING PIPELINE  (median imputation — experiment winner)
# =============================================================================

print("=" * 70)
print("FINAL PREPROCESSING PIPELINE")
print("Using MEDIAN imputation (Winner from preprocessing experiment)")
print("=" * 70)

# ------------------------------------------------------------------
# 1-A  Run shared preprocessing pipeline
# ------------------------------------------------------------------
data        = build_pipeline()
df          = data.df
X_train_raw = data.X_train_raw
y_train_raw = data.y_train_raw
X_train_res = data.X_train_res
y_train_res = data.y_train_res
X_train_sc  = data.X_train_sc
X_test      = data.X_test_raw
X_test_sc   = data.X_test_sc
y_test      = data.y_test
scaler      = data.scaler

print(f"\n  Raw dataset loaded : {data.df_raw.shape[0]} rows x {data.df_raw.shape[1]} cols")
print("  Median imputation  : applied to", ZERO_COLS)
print("  Feature engineering: Glucose_BMI_Interaction, Age_Group added")
print(f"\n  Train samples (raw)     : {len(X_train_raw)}"
      f"  (Non-diab={sum(y_train_raw==0)}, Diab={sum(y_train_raw==1)})")
print(f"  Test  samples           : {len(X_test)}"
      f"  (Non-diab={sum(y_test==0)}, Diab={sum(y_test==1)})")
print(f"  Train samples (balanced): {len(X_train_res)}"
      f"  (Non-diab={sum(y_train_res==0)}, Diab={sum(y_train_res==1)})")
print("  Scaling                 : MinMaxScaler applied")
print(f"  Final feature count     : {X_train_sc.shape[1]}"
      " (8 original + 2 engineered)")

# ------------------------------------------------------------------
# 1-H  Save preprocessing summary
# ------------------------------------------------------------------
prep_summary_lines = [
    "=" * 62,
    "  FINAL PREPROCESSING SUMMARY",
    f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 62,
    "",
    "  METHOD CHOSEN",
    "  -------------",
    "  Imputation   : Median",
    "  Justification: Experiment showed +1.56% AUC-ROC over worst method",
    "                 (median best for skewed/outlier-heavy features like",
    "                  Insulin and SkinThickness)",
    "",
    "  DATASET SPLIT",
    "  -------------",
    f"  Total samples          : {len(df)}",
    f"  Training (before ROS)  : {len(X_train_raw)}"
    f"  [{sum(y_train_raw==0)} non-diab | {sum(y_train_raw==1)} diab]",
    f"  Training (after  ROS)  : {len(X_train_res)}"
    f"  [{sum(y_train_res==0)} non-diab | {sum(y_train_res==1)} diab]",
    f"  Test set               : {len(X_test)}"
    f"  [{sum(y_test==0)} non-diab | {sum(y_test==1)} diab]",
    "",
    "  FEATURES",
    "  --------",
    "  Original (8) : Pregnancies, Glucose, BloodPressure, SkinThickness,",
    "                  Insulin, BMI, DiabetesPedigreeFunction, Age",
    "  Engineered(2): Glucose_BMI_Interaction, Age_Group",
    "  Total        : 10",
    "",
    "  SCALING",
    "  -------",
    "  Method : MinMaxScaler  (range [0, 1])",
    "  Fit on : balanced training set only (no leakage)",
    "",
    "=" * 62,
]
with open("final_preprocessing_summary.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(prep_summary_lines))
print("\n  Saved -> final_preprocessing_summary.txt")

# =============================================================================
# PART 2: TRAIN 5 ML MODELS
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: MODEL TRAINING")
print("=" * 70)

# We will collect everything into these shared containers
models       = {}    # name -> fitted estimator
predictions  = {}    # name -> y_pred  (hard labels)
probabilities = {}   # name -> y_prob  (class-1 probability)
train_times  = {}    # name -> float (seconds)

# --------------------------------------------------------------------------
# Helper: fit, predict, and record results in one call
# --------------------------------------------------------------------------
def fit_and_store(name: str, clf) -> None:
    """Fit clf, store model + predictions + proba + timing."""
    t0     = time.perf_counter()
    clf.fit(X_train_sc, y_train_res)
    elapsed = time.perf_counter() - t0

    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]

    models[name]        = clf
    predictions[name]   = y_pred
    probabilities[name] = y_prob
    train_times[name]   = round(elapsed, 4)


# ============================================================
# MODEL 1 — Logistic Regression
# ============================================================
# Logistic Regression: Linear model using sigmoid function
# Best for: Baseline performance, interpretable coefficients
# Formula: P(y=1) = 1 / (1 + e^(-wx+b))
# The model learns one weight per feature; large positive weights
# increase the predicted probability of diabetes, large negative
# weights decrease it.  We use lbfgs (quasi-Newton) for stability.
# ============================================================

print("\n  [1/5] Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)
fit_and_store("Logistic Regression", lr)

# Print top-3 positive and top-3 negative coefficients
coef_series = pd.Series(lr.coef_[0], index=ENG_FEATURES).sort_values()
print(f"        Training time : {train_times['Logistic Regression']:.4f} s")
print("        Coefficients (top 3 positive / top 3 negative):")
for feat, val in coef_series.head(3).items():          # most negative
    print(f"          {feat:<30}  {val:+.4f}")
for feat, val in coef_series.tail(3).items():          # most positive
    print(f"          {feat:<30}  {val:+.4f}")

# ============================================================
# MODEL 2 — Decision Tree
# ============================================================
# Decision Tree: Recursive binary splits using Gini impurity
# Best for: Interpretable rules, non-linear patterns
# Creates hierarchical if-then rules (e.g. "if Glucose > 127 and
# BMI > 29.9 then predict Diabetic").
# We cap max_depth=5 and require min 10 samples per leaf to prevent
# overfitting while keeping the tree human-readable.
# ============================================================

print("\n  [2/5] Training Decision Tree...")
dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    criterion='gini'
)
fit_and_store("Decision Tree", dt)

print(f"        Training time  : {train_times['Decision Tree']:.4f} s")
print(f"        Actual depth   : {dt.get_depth()}")
print(f"        Number of leaves: {dt.get_n_leaves()}")

# ============================================================
# MODEL 3 — Random Forest
# ============================================================
# Random Forest: Ensemble of decision trees with bagging
# Best for: Robust predictions, handles overfitting
# Combines multiple trees through majority voting.
# Each tree is trained on a bootstrap sample of the data and a
# random subset of features, reducing correlation between trees.
# 100 estimators with max_depth=10 balances bias vs variance.
# ============================================================

print("\n  [3/5] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
fit_and_store("Random Forest", rf)

print(f"        Training time  : {train_times['Random Forest']:.4f} s")

# ============================================================
# MODEL 4 — Support Vector Machine (SVM with RBF kernel)
# ============================================================
# SVM: Finds optimal hyperplane maximizing margin
# RBF kernel enables non-linear classification by mapping data
# into a higher-dimensional space via the radial basis function.
# Best for: Clear margin of separation, high-dimensional data.
# C=1 controls regularisation (higher C = tighter fit to training).
# probability=True enables Platt scaling so we can extract AUC-ROC.
# ============================================================

print("\n  [4/5] Training SVM with RBF kernel...")
svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)
fit_and_store("SVM", svm)

print(f"        Training time        : {train_times['SVM']:.4f} s")
print(f"        Support vector count : {svm.n_support_.sum()}"
      f"  (class 0: {svm.n_support_[0]}, class 1: {svm.n_support_[1]})")

# ============================================================
# MODEL 5 — Neural Network (MLP)
# ============================================================
# Multi-Layer Perceptron: Neural network with 2 hidden layers
# Architecture: Input(10) -> Hidden1(64) -> Hidden2(32) -> Output(1)
# Activation='tanh' maps each neuron to [-1, 1]; works well for
# balanced/scaled inputs.  The Adam optimiser adapts learning rates
# per parameter, converging faster than vanilla SGD.
# Best for: Complex non-linear patterns, large datasets.
# ============================================================

print("\n  [5/5] Training Neural Network (MLP)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='tanh',
    solver='adam',
    max_iter=200,
    random_state=42,
    early_stopping=False
)
fit_and_store("Neural Network", mlp)

print(f"        Training time    : {train_times['Neural Network']:.4f} s")
print(f"        Iterations run   : {mlp.n_iter_}")
print(f"        Final loss       : {mlp.loss_:.6f}")

# =============================================================================
# PART 3: STORE & COMPUTE METRICS
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: COMPUTING EVALUATION METRICS")
print("=" * 70)

MODEL_NAMES  = list(models.keys())
METRIC_KEYS  = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']

rows = []
for name in MODEL_NAMES:
    y_pred = predictions[name]
    y_prob = probabilities[name]
    rows.append({
        'Model'    : name,
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall'   : recall_score(y_test, y_pred, zero_division=0),
        'F1'       : f1_score(y_test, y_pred, zero_division=0),
        'AUC-ROC'  : roc_auc_score(y_test, y_prob),
        'Time(s)'  : train_times[name],
    })

metrics_df = pd.DataFrame(rows)

# Rank by F1 (balanced metric for imbalanced data)
metrics_df['Rank'] = metrics_df['F1'].rank(ascending=False).astype(int)
metrics_df = metrics_df.sort_values('Rank').reset_index(drop=True)

# Print formatted table
header = (f"\n  {'Model':<22} {'Acc':>7} {'Prec':>7} "
          f"{'Rec':>7} {'F1':>7} {'AUC':>7} {'Time':>7} {'Rank':>5}")
sep    = "  " + "-" * 70
print(header)
print(sep)
for _, row in metrics_df.iterrows():
    print(f"  {row['Model']:<22} "
          f"{row['Accuracy']:>7.4f} "
          f"{row['Precision']:>7.4f} "
          f"{row['Recall']:>7.4f} "
          f"{row['F1']:>7.4f} "
          f"{row['AUC-ROC']:>7.4f} "
          f"{row['Time(s)']:>7.3f}s "
          f"{int(row['Rank']):>5}")
print(sep)

best_model_name = metrics_df.iloc[0]['Model']
print(f"\n  Best model (by F1): {best_model_name}")

# =============================================================================
# PART 4: PRINT SUMMARY
# =============================================================================

print("\n")
print("=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(f"""
  Models trained : 5
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - SVM (RBF)
    - Neural Network (MLP)

  Preprocessing  : Median imputation (selected via experiment)
  Training samples (balanced) : {len(X_train_res)}
  Test samples                : {len(X_test)}
  Features                    : {X_train_sc.shape[1]} (8 original + 2 engineered)

  All models ready for evaluation.
""")

total_time = sum(train_times.values())
print(f"  Total training time : {total_time:.4f} s")
print(f"  Fastest model       : "
      f"{min(train_times, key=train_times.get)}"
      f"  ({min(train_times.values()):.4f}s)")
print(f"  Slowest model       : "
      f"{max(train_times, key=train_times.get)}"
      f"  ({max(train_times.values()):.4f}s)")

print("\n" + "=" * 70)
print("[OK] 5 models trained successfully")
print("[OK] Preprocessing summary saved")
print("=" * 70)

# =============================================================================
# APPENDIX: Classification Reports (full detail per model)
# =============================================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORTS (per model)")
print("=" * 70)

for name in MODEL_NAMES:
    print(f"\n  --- {name} ---")
    print(classification_report(
        y_test, predictions[name],
        target_names=['Non-Diabetic', 'Diabetic'],
        digits=4
    ))
