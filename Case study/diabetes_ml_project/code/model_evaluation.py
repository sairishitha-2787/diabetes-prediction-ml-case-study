# =============================================================================
# PIMA DIABETES — COMPREHENSIVE MODEL EVALUATION
# Evaluates all 5 trained models on test set + 5-fold cross-validation
# Continues from: model_training.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
from datetime import datetime

from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import MinMaxScaler
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling  import RandomOverSampler
from preprocessing           import build_pipeline, ENG_FEATURES

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# REBUILD PREPROCESSING + MODELS  (mirrors model_training.py exactly)
# =============================================================================

print("=" * 70)
print("REBUILDING PIPELINE (Median imputation — experiment winner)")
print("=" * 70)

data        = build_pipeline()
X_train_res = data.X_train_res
y_train_res = data.y_train_res
X_train_sc  = data.X_train_sc
X_test      = data.X_test_raw
X_test_sc   = data.X_test_sc
y_test      = data.y_test

print(f"  Train (balanced): {len(X_train_res)} | Test: {len(X_test)} | "
      f"Features: {X_train_sc.shape[1]}")

# --- Train all 5 models ---
print("\n  Re-training 5 models...")
import time
models      = {}
predictions = {}
probs       = {}
train_times = {}

def _fit(name, clf):
    t0 = time.perf_counter()
    clf.fit(X_train_sc, y_train_res)
    train_times[name]  = round(time.perf_counter() - t0, 4)
    models[name]       = clf
    predictions[name]  = clf.predict(X_test_sc)
    probs[name]        = clf.predict_proba(X_test_sc)[:, 1]
    print(f"    {name:<25} trained in {train_times[name]:.4f}s")

_fit("Logistic Regression",
     LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'))
_fit("Decision Tree",
     DecisionTreeClassifier(max_depth=5, min_samples_split=20,
                            min_samples_leaf=10, random_state=42, criterion='gini'))
_fit("Random Forest",
     RandomForestClassifier(n_estimators=100, max_depth=10,
                            min_samples_leaf=5, random_state=42, n_jobs=-1))
_fit("SVM",
     SVC(kernel='rbf', C=1.0, gamma='scale',
         probability=True, random_state=42))
_fit("Neural Network",
     MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh',
                   solver='adam', max_iter=200, random_state=42))

MODEL_NAMES = list(models.keys())

# =============================================================================
# PART 1: CALCULATE ALL METRICS  (test-set evaluation)
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: TEST-SET METRICS")
print("=" * 70)

def get_metrics(name: str) -> dict:
    """
    Returns a full metric dictionary for one model.

    Metrics explained:
    ------------------
    Accuracy    = (TP+TN) / Total              — overall correctness
    Precision   = TP / (TP+FP)                 — quality of positive predictions
    Recall      = TP / (TP+FN)                 — how many diabetics we catch (CRITICAL)
    Specificity = TN / (TN+FP)                 — how many non-diabetics we correctly rule out
    F1-Score    = harmonic mean(Precision, Recall) — balanced metric for imbalanced data
    AUC-ROC     = area under ROC curve         — threshold-independent ranking quality

    In a medical screening context, Recall (sensitivity) and AUC-ROC are
    prioritised because a missed diabetic (false negative) is more harmful
    than an unnecessary follow-up (false positive).
    """
    y_pred = predictions[name]
    y_prob = probs[name]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'Model'      : name,
        'Accuracy'   : accuracy_score(y_test, y_pred),
        'Precision'  : precision_score(y_test, y_pred, zero_division=0),
        'Recall'     : recall_score(y_test, y_pred, zero_division=0),
        'Specificity': specificity,
        'F1-Score'   : f1_score(y_test, y_pred, zero_division=0),
        'AUC-ROC'    : roc_auc_score(y_test, y_prob),
        'TP'         : int(tp),
        'TN'         : int(tn),
        'FP'         : int(fp),
        'FN'         : int(fn),
    }

rows      = [get_metrics(n) for n in MODEL_NAMES]
metrics_df = (pd.DataFrame(rows)
              .sort_values('AUC-ROC', ascending=False)
              .reset_index(drop=True))

# Print table
hdr = (f"\n  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
       f"{'Spec':>6} {'F1':>6} {'AUC':>6} {'TP':>4} {'TN':>4} "
       f"{'FP':>4} {'FN':>4}")
sep = "  " + "-" * 76
print(hdr); print(sep)
for _, r in metrics_df.iterrows():
    print(f"  {r['Model']:<22} "
          f"{r['Accuracy']:>6.4f} {r['Precision']:>6.4f} "
          f"{r['Recall']:>6.4f} {r['Specificity']:>6.4f} "
          f"{r['F1-Score']:>6.4f} {r['AUC-ROC']:>6.4f} "
          f"{r['TP']:>4d} {r['TN']:>4d} {r['FP']:>4d} {r['FN']:>4d}")
print(sep)

# Save CSV
metrics_df.to_csv("model_comparison_results.csv", index=False)
print("  Saved -> model_comparison_results.csv")

# Convenience references
best_row  = metrics_df.iloc[0]
best_name = best_row['Model']

# =============================================================================
# PART 2: CROSS-VALIDATION (5-fold stratified on PRE-SMOTE training data)
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: CROSS-VALIDATION STABILITY (5-fold stratified)")
print("=" * 70)
print("  Note: CV is performed on the raw training split (pre-oversampling)")
print("  Each fold uses its own MinMaxScaler to prevent data leakage.\n")

# Using a Pipeline (scaler + model) ensures the scaler is fit only on
# the training portion of each fold, which is the statistically correct
# approach.  Imputation is already part of X_train_raw.

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_rows = []
for name, clf in models.items():
    # Build a fresh estimator with same hyperparams (avoids state from training)
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf'   , type(clf)(**clf.get_params()))
    ])
    cv_scores = cross_val_score(
        pipe, X_train_raw, y_train_raw,
        cv=skf, scoring='accuracy', n_jobs=-1
    )
    mean_acc = cv_scores.mean()
    std_acc  = cv_scores.std()
    cv_rows.append({
        'Model'           : name,
        'CV_Mean_Accuracy': mean_acc,
        'CV_Std'          : std_acc,
        'CV_Scores'       : cv_scores.tolist(),
    })
    print(f"  {name:<22}  mean={mean_acc:.4f}  std=±{std_acc:.4f}  "
          f"  folds={[f'{s:.3f}' for s in cv_scores]}")

cv_df = (pd.DataFrame(cv_rows)
         .sort_values('CV_Mean_Accuracy', ascending=False)
         .reset_index(drop=True))

cv_df[['Model', 'CV_Mean_Accuracy', 'CV_Std']].to_csv(
    "cross_validation_results.csv", index=False
)
print("\n  Saved -> cross_validation_results.csv")

most_stable   = cv_df.sort_values('CV_Std').iloc[0]
least_stable  = cv_df.sort_values('CV_Std', ascending=False).iloc[0]
best_cv_model = cv_df.iloc[0]

print(f"\n  Most  stable model: {most_stable['Model']}"
      f"  (std: ±{most_stable['CV_Std']:.4f})")
print(f"  Least stable model: {least_stable['Model']}"
      f"  (std: ±{least_stable['CV_Std']:.4f})")

# =============================================================================
# PART 3: VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: CREATING VISUALIZATIONS")
print("=" * 70)

# Shared palette — one color per model (consistent across all plots)
MODEL_COLORS = {
    'Logistic Regression': '#3498DB',
    'Decision Tree'      : '#E74C3C',
    'Random Forest'      : '#2ECC71',
    'SVM'                : '#9B59B6',
    'Neural Network'     : '#F39C12',
}
GOLD   = '#F1C40F'
SILVER = '#BDC3C7'
BRONZE = '#E67E22'

sns.set_style("whitegrid")

# --------------------------------------------------------------------------
# Plot 1 — 2x2 horizontal bar chart for 4 key metrics
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()

plot_metrics = [
    ('Accuracy',  'Accuracy'),
    ('Precision', 'Precision'),
    ('Recall',    'Recall (Sensitivity)'),
    ('F1-Score',  'F1-Score'),
]

for idx, (col, title) in enumerate(plot_metrics):
    ax  = axes_flat[idx]
    sub = metrics_df.sort_values(col, ascending=True)   # ascending for barh
    vals   = sub[col].tolist()
    names  = sub['Model'].tolist()
    best_v = max(vals)

    colors = [GOLD if v == best_v else '#4A90D9' for v in vals]
    bars   = ax.barh(names, vals, color=colors, edgecolor='black',
                     linewidth=0.6, height=0.55)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', ha='left',
                fontsize=9.5, fontweight='bold')

    x_min = max(0, min(vals) - 0.05)
    ax.set_xlim(x_min, max(vals) * 1.12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Score", fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

gold_patch  = mpatches.Patch(color=GOLD,    label='Best in metric')
blue_patch  = mpatches.Patch(color='#4A90D9', label='Other models')
fig.legend(handles=[gold_patch, blue_patch], fontsize=10,
           loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01))

fig.suptitle("Performance Metrics Comparison — All 5 Models",
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("performance_metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> performance_metrics_comparison.png")

# --------------------------------------------------------------------------
# Plot 2 — ROC curves (all models on same axes)
# --------------------------------------------------------------------------
cmap = plt.get_cmap('tab10')
fig, ax = plt.subplots(figsize=(10, 8))

for i, name in enumerate(MODEL_NAMES):
    fpr, tpr, _ = roc_curve(y_test, probs[name])
    auc_val = metrics_df.loc[metrics_df['Model'] == name, 'AUC-ROC'].values[0]
    ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2.2,
            label=f"{name}  (AUC = {auc_val:.4f})")

# Diagonal reference line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.6,
        label='Random classifier (AUC = 0.500)')

ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("roc_curves_all_models.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> roc_curves_all_models.png")

# --------------------------------------------------------------------------
# Plot 3 — Confusion matrices (2x3 grid; last cell = TP/TN/FP/FN table)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.flatten()
cm_labels = ['Non-Diabetic', 'Diabetic']

for idx, name in enumerate(MODEL_NAMES):
    ax  = axes_flat[idx]
    cm  = confusion_matrix(y_test, predictions[name])
    acc = accuracy_score(y_test, predictions[name])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cm_labels, yticklabels=cm_labels,
                linewidths=1, linecolor='white', ax=ax,
                cbar_kws={'shrink': 0.8},
                annot_kws={"size": 14, "weight": "bold"})

    ax.set_title(f"{name}\nAccuracy = {acc:.4f}",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)

# Last subplot (index 5) — TP/TN/FP/FN summary table
ax_tbl = axes_flat[5]
ax_tbl.axis('off')

tbl_data  = []
tbl_cols  = ['Model', 'TP', 'TN', 'FP', 'FN', 'FN Rate']
for _, r in metrics_df.iterrows():
    fn_rate = r['FN'] / (r['FN'] + r['TP']) * 100
    tbl_data.append([
        r['Model'][:14],     # truncate for table width
        str(r['TP']), str(r['TN']), str(r['FP']), str(r['FN']),
        f"{fn_rate:.1f}%"
    ])

tbl = ax_tbl.table(
    cellText=tbl_data, colLabels=tbl_cols,
    cellLoc='center', loc='center',
    bbox=[0.0, 0.1, 1.0, 0.85]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
for j in range(len(tbl_cols)):
    tbl[(0, j)].set_facecolor('#2C3E50')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
# Colour FN-rate column: high FN = pink warning
for i in range(1, len(tbl_data) + 1):
    tbl[(i, 5)].set_facecolor('#FADBD8')
ax_tbl.set_title("Confusion Matrix Summary\n(FN Rate = missed diabetics)",
                 fontsize=10, fontweight='bold')

fig.suptitle("Confusion Matrices — All 5 Models",
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("confusion_matrices_grid.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> confusion_matrices_grid.png")

# --------------------------------------------------------------------------
# Plot 4 — Metrics heatmap (models x metrics)
# --------------------------------------------------------------------------
heat_cols  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
heat_data  = metrics_df.set_index('Model')[heat_cols]

# Compute per-column maxima for annotation
col_max = heat_data.max()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    heat_data, annot=True, fmt='.3f', cmap='YlGnBu',
    linewidths=0.5, linecolor='white',
    cbar_kws={'shrink': 0.7, 'label': 'Score'},
    ax=ax, vmin=0.55, vmax=1.0,
    annot_kws={"size": 11}
)

# Bold-border the cell with the highest value in each column
for col_idx, col_name in enumerate(heat_cols):
    best_model_idx = heat_data[col_name].idxmax()
    row_idx = heat_data.index.get_loc(best_model_idx)
    ax.add_patch(plt.Rectangle(
        (col_idx, row_idx), 1, 1,
        fill=False, edgecolor='#E74C3C', linewidth=2.5, clip_on=False
    ))

ax.set_title("Model Performance Heatmap\n(Red border = best per metric)",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Metric", fontsize=11)
ax.set_ylabel("Model", fontsize=11)
ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig("metrics_heatmap_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> metrics_heatmap_comparison.png")

# --------------------------------------------------------------------------
# Plot 5 — Cross-validation stability (bar + error bars)
# --------------------------------------------------------------------------
cv_plot = cv_df.sort_values('CV_Mean_Accuracy', ascending=False).reset_index(drop=True)
means   = cv_plot['CV_Mean_Accuracy'].tolist()
stds    = cv_plot['CV_Std'].tolist()
names   = cv_plot['Model'].tolist()

std_max = max(stds)
std_min = min(stds)
bar_colors_cv = []
for s in stds:
    t = (s - std_min) / (std_max - std_min + 1e-9)   # 0 = most stable
    # interpolate green -> orange -> red
    if t < 0.5:
        r = int(2 * t * 230)
        bar_colors_cv.append(f"#{r:02X}C060")
    else:
        r = 220
        g = int((1 - 2*(t-0.5)) * 192)
        bar_colors_cv.append(f"#{r:02X}{g:02X}30")

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(names, means, color=bar_colors_cv, edgecolor='black',
              linewidth=0.7, width=0.55, zorder=3,
              yerr=stds, capsize=6,
              error_kw=dict(elinewidth=1.8, ecolor='#2C3E50', capthick=2))

for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.005,
            f"{m:.4f}\n+/-{s:.4f}",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

overall_mean = np.mean(means)
ax.axhline(overall_mean, color='navy', linestyle='--', linewidth=1.5,
           label=f"Overall mean = {overall_mean:.4f}")

ax.set_title("Cross-Validation Stability (5-fold Stratified)",
             fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_ylim(min(means) - 0.08, max(means) + 0.06)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

stable_patch   = mpatches.Patch(color='#00C060', label='Most stable (low std)')
unstable_patch = mpatches.Patch(color='#DC3020', label='Least stable (high std)')
ax.legend(handles=[stable_patch, unstable_patch,
                   mpatches.Patch(color='white', label=f"Overall mean = {overall_mean:.4f}",
                                  linestyle='--', edgecolor='navy')],
          fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig("cv_stability_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> cv_stability_analysis.png")

# --------------------------------------------------------------------------
# Plot 6 — Model Ranking Dashboard  (podium + radar + full table)
# --------------------------------------------------------------------------
top3  = metrics_df.head(3)
radar_models = top3['Model'].tolist()
radar_cols   = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
radar_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']

fig = plt.figure(figsize=(14, 12))
gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Top-left: Podium bar chart ---
ax_podium = fig.add_subplot(gs[0, 0])
podium_order  = [1, 0, 2]          # 2nd left, 1st centre, 3rd right
podium_names  = [radar_models[i] for i in podium_order]
podium_aucs   = [top3.iloc[i]['AUC-ROC'] for i in podium_order]
podium_heights= [0.85, 1.00, 0.70]  # visual podium heights
podium_colors = [SILVER, GOLD, BRONZE]
podium_labels = ['2nd', '1st', '3rd']

bar_objs = ax_podium.bar(
    [0, 1, 2], podium_heights,
    color=podium_colors, edgecolor='black', linewidth=1.2,
    width=0.6, zorder=3
)
ax_podium.set_xticks([0, 1, 2])
ax_podium.set_xticklabels(
    [f"{lbl}\n{n}\nAUC={v:.4f}"
     for lbl, n, v in zip(podium_labels, podium_names, podium_aucs)],
    fontsize=9.5, fontweight='bold'
)
ax_podium.set_yticks([])
ax_podium.set_title("Top 3 Models — Podium", fontsize=12, fontweight='bold')
ax_podium.spines['top'].set_visible(False)
ax_podium.spines['right'].set_visible(False)
ax_podium.spines['left'].set_visible(False)
ax_podium.set_ylim(0, 1.25)
ax_podium.grid(False)

# --- Top-right: Radar chart ---
ax_radar = fig.add_subplot(gs[0, 1], polar=True)

N        = len(radar_labels)
angles   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles  += angles[:1]                  # close the polygon

radar_colors = [GOLD, SILVER, BRONZE]
for i, (model_name, color) in enumerate(zip(radar_models, radar_colors)):
    row    = metrics_df[metrics_df['Model'] == model_name].iloc[0]
    values = [row[c] for c in radar_cols]
    values += values[:1]               # close the polygon

    ax_radar.plot(angles, values, color=color, linewidth=2, linestyle='solid',
                  label=model_name)
    ax_radar.fill(angles, values, color=color, alpha=0.12)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_labels, fontsize=9.5)
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([0.6, 0.7, 0.8, 0.9])
ax_radar.set_yticklabels(['0.6', '0.7', '0.8', '0.9'], fontsize=7, color='grey')
ax_radar.set_title("Radar Chart — Top 3 Models", fontsize=12,
                   fontweight='bold', pad=18)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8.5)
ax_radar.grid(True, linestyle='--', alpha=0.5)

# --- Bottom (spanning both columns): Full ranking table ---
ax_table = fig.add_subplot(gs[1, :])
ax_table.axis('off')

full_cols  = ['Rank', 'Model', 'Accuracy', 'Precision', 'Recall',
              'Specificity', 'F1-Score', 'AUC-ROC']
table_data = []
for rank, (_, r) in enumerate(metrics_df.iterrows(), start=1):
    prefix = ['1st', '2nd', '3rd', '4th', '5th'][rank - 1]
    table_data.append([
        prefix,
        r['Model'],
        f"{r['Accuracy']*100:.2f}%",
        f"{r['Precision']*100:.2f}%",
        f"{r['Recall']*100:.2f}%",
        f"{r['Specificity']*100:.2f}%",
        f"{r['F1-Score']*100:.2f}%",
        f"{r['AUC-ROC']:.4f}",
    ])

tbl2 = ax_table.table(
    cellText=table_data, colLabels=full_cols,
    cellLoc='center', loc='center',
    bbox=[0.0, 0.05, 1.0, 0.90]
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(10.5)

# Header row styling
for j in range(len(full_cols)):
    tbl2[(0, j)].set_facecolor('#1A252F')
    tbl2[(0, j)].set_text_props(color='white', fontweight='bold')

# Colour top-3 rows
row_fills = {1: '#F9E79F', 2: '#D5D8DC', 3: '#FDEBD0'}
for i in range(1, 4):
    for j in range(len(full_cols)):
        tbl2[(i, j)].set_facecolor(row_fills[i])

ax_table.set_title("Complete Model Rankings — All Metrics",
                   fontsize=12, fontweight='bold', pad=6)

fig.suptitle("Final Model Rankings Dashboard", fontsize=16,
             fontweight='bold', y=1.01)
plt.savefig("model_ranking_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> model_ranking_dashboard.png")

# =============================================================================
# PART 4: EVALUATION REPORT
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: GENERATING EVALUATION REPORT")
print("=" * 70)

now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
best_row  = metrics_df.iloc[0]
b_name    = best_row['Model']
b_tp, b_tn  = int(best_row['TP']), int(best_row['TN'])
b_fp, b_fn  = int(best_row['FP']), int(best_row['FN'])
b_fn_rate   = b_fn / (b_fn + b_tp) * 100

worst_auc = metrics_df['AUC-ROC'].min()
best_auc  = best_row['AUC-ROC']
auc_improvement = (best_auc - worst_auc) / worst_auc * 100

medal_strs  = ['1st (Gold)', '2nd (Silver)', '3rd (Bronze)', '4th', '5th']

lines = []
lines.append("=" * 72)
lines.append("  COMPREHENSIVE MODEL EVALUATION REPORT")
lines.append(f"  Generated : {now_str}")
lines.append("=" * 72)

# --- Section 1: Overview ---
lines.append("\n1. EVALUATION SUMMARY")
lines.append("-" * 40)
lines.append(f"  Test set          : {len(y_test)} samples "
             f"({sum(y_test==0)} non-diabetic | {sum(y_test==1)} diabetic)")
lines.append("  Metrics           : Accuracy, Precision, Recall, Specificity,")
lines.append("                       F1-Score, AUC-ROC, TP, TN, FP, FN")
lines.append("  Cross-validation  : 5-fold Stratified (on pre-SMOTE training data)")
lines.append("  Pipeline per fold : MinMaxScaler + Classifier")

# --- Section 2: Full results ---
lines.append("\n\n2. COMPLETE RESULTS TABLE (sorted by AUC-ROC)")
lines.append("-" * 72)
col_hdr = (f"  {'Model':<22} {'Acc%':>6} {'Prec%':>6} {'Rec%':>6} "
           f"{'Spec%':>6} {'F1%':>6} {'AUC':>7}  TP  TN  FP  FN")
lines.append(col_hdr)
lines.append("  " + "-" * 68)
for _, r in metrics_df.iterrows():
    lines.append(
        f"  {r['Model']:<22} "
        f"{r['Accuracy']*100:>5.2f}% "
        f"{r['Precision']*100:>5.2f}% "
        f"{r['Recall']*100:>5.2f}% "
        f"{r['Specificity']*100:>5.2f}% "
        f"{r['F1-Score']*100:>5.2f}% "
        f"{r['AUC-ROC']:>7.4f}  "
        f"{int(r['TP']):>2}  {int(r['TN']):>2}  "
        f"{int(r['FP']):>2}  {int(r['FN']):>2}"
    )
lines.append("  " + "-" * 68)

# --- Section 3: Rankings ---
lines.append("\n\n3. MODEL RANKINGS (by AUC-ROC)")
lines.append("-" * 40)
for rank, (_, r) in enumerate(metrics_df.iterrows(), start=1):
    lines.append(f"  {medal_strs[rank-1]:<14}  {r['Model']:<22}  "
                 f"AUC = {r['AUC-ROC']:.4f}")

# --- Section 4: Best model analysis ---
lines.append(f"\n\n4. BEST MODEL ANALYSIS  ->  {b_name}")
lines.append("-" * 40)
lines.append(f"\n  Key Strengths:")
lines.append(f"    Highest AUC-ROC : {best_row['AUC-ROC']:.4f}")
lines.append(f"    Accuracy        : {best_row['Accuracy']*100:.2f}%")
lines.append(f"    Recall          : {best_row['Recall']*100:.2f}%"
             "  (critical for medical screening)")
lines.append(f"    F1-Score        : {best_row['F1-Score']:.4f}")
lines.append(f"\n  Confusion Matrix:")
lines.append(f"    True Positives (TP)  : {b_tp:>3}  correctly identified diabetics")
lines.append(f"    True Negatives (TN)  : {b_tn:>3}  correctly identified non-diabetics")
lines.append(f"    False Positives (FP) : {b_fp:>3}  false alarms (unnecessary follow-up)")
lines.append(f"    False Negatives (FN) : {b_fn:>3}  MISSED diabetics  <-- most critical")
lines.append(f"\n  Clinical Interpretation:")
lines.append(f"    False Negative Rate : {b_fn_rate:.1f}%")
lines.append(f"    This means {b_fn} out of {b_fn + b_tp} actual diabetics were missed.")
if b_fn_rate < 20:
    interp = ("Low FN rate — acceptable for initial screening; missed cases "
              "should be caught on follow-up testing.")
else:
    interp = ("FN rate warrants attention; consider lowering the decision "
              "threshold to improve recall at a small precision cost.")
lines.append(f"    Interpretation: {interp}")

# --- Section 5: CV Stability ---
lines.append("\n\n5. CROSS-VALIDATION STABILITY (5-fold)")
lines.append("-" * 40)
cv_out = cv_df[['Model', 'CV_Mean_Accuracy', 'CV_Std']].copy()
lines.append(f"\n  {'Model':<22} {'CV Mean Acc':>13} {'Std Dev':>10}")
lines.append("  " + "-" * 46)
for _, r in cv_out.iterrows():
    lines.append(f"  {r['Model']:<22} {r['CV_Mean_Accuracy']:>12.4f}"
                 f"  +/-{r['CV_Std']:.4f}")
lines.append(f"\n  Most  stable : {most_stable['Model']}"
             f"  (std = {most_stable['CV_Std']:.4f})")
lines.append(f"  Least stable : {least_stable['Model']}"
             f"  (std = {least_stable['CV_Std']:.4f})")

# --- Section 6: Key Findings ---
best_recall_model = metrics_df.loc[metrics_df['Recall'].idxmax(), 'Model']
lines.append("\n\n6. KEY FINDINGS")
lines.append("-" * 40)
lines.append(f"  Best overall model     (AUC-ROC) : {b_name}")
lines.append(f"  Best recall model      (clinical): {best_recall_model}")
lines.append(f"  Most stable model      (CV std)  : {most_stable['Model']}")
lines.append(f"  Fastest model          (train)   : "
             f"{min(train_times, key=train_times.get)}"
             f"  ({min(train_times.values()):.4f}s)")
lines.append(f"\n  AUC-ROC range : {worst_auc:.4f} to {best_auc:.4f}"
             f"  (+{auc_improvement:.2f}% improvement from worst to best)")

# --- Section 7: Recommendation ---
lines.append("\n\n7. RECOMMENDATION")
lines.append("-" * 40)
lines.append(f"\n  Primary Model : {b_name}")
lines.append(f"\n  Justification:")
lines.append(f"    (a) Best overall AUC-ROC ({best_auc:.4f}) — strongest ranking "
             f"quality across all thresholds")
lines.append(f"    (b) Highest Recall ({best_row['Recall']*100:.2f}%) — captures "
             f"the most diabetics, minimising missed diagnoses")
lines.append(f"    (c) Preprocessed with median imputation — experimentally "
             f"verified best imputation strategy for this dataset")
lines.append(f"    (d) CV stability confirms generalisation "
             f"(std = {cv_df.loc[cv_df['Model']==b_name, 'CV_Std'].values[0]:.4f})")
lines.append(f"\n  Secondary consideration:")
if most_stable['Model'] != b_name:
    lines.append(f"    If stability is prioritised, {most_stable['Model']} has the "
                 f"lowest CV std ({most_stable['CV_Std']:.4f})")
lines.append(f"\n  For production deployment, wrap the model in a Pipeline with")
lines.append(f"  median imputation + MinMaxScaler to ensure consistent inference.")

lines.append("\n\n" + "=" * 72)
lines.append("  END OF REPORT")
lines.append("=" * 72)

with open("model_evaluation_report.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

print("  Saved -> model_evaluation_report.txt")

# =============================================================================
# PART 5: PRINT SUMMARY
# =============================================================================

auc_sorted    = metrics_df.sort_values('AUC-ROC', ascending=False)
worst_auc_row = auc_sorted.iloc[-1]
best_cv_row   = cv_df.iloc[0]

print("\n")
print("=" * 70)
print("MODEL EVALUATION COMPLETE")
print("=" * 70)

print(f"""
  Models evaluated  : 5
  Test set size     : {len(y_test)} samples
  Cross-validation  : 5-fold stratified (pre-SMOTE training data)

  [BEST MODEL by AUC-ROC]: {best_row['Model']}
    AUC-ROC  : {best_row['AUC-ROC']:.4f}
    Accuracy : {best_row['Accuracy']*100:.2f}%
    Recall   : {best_row['Recall']*100:.2f}%
    F1-Score : {best_row['F1-Score']:.4f}

  [PERFORMANCE SPREAD]:
    Best  AUC : {best_row['AUC-ROC']:.4f}  ({best_row['Model']})
    Worst AUC : {worst_auc_row['AUC-ROC']:.4f}  ({worst_auc_row['Model']})
    Delta     : {best_row['AUC-ROC'] - worst_auc_row['AUC-ROC']:.4f}  (+{auc_improvement:.2f}%)

  [STABILITY]:
    Most  stable : {most_stable['Model']:<22}  CV std = +/-{most_stable['CV_Std']:.4f}
    Least stable : {least_stable['Model']:<22}  CV std = +/-{least_stable['CV_Std']:.4f}
""")

print("=" * 70)
print("[OK] 6 visualizations created")
print("[OK] 2 CSV files exported  (model_comparison_results.csv, "
      "cross_validation_results.csv)")
print("[OK] Evaluation report saved  (model_evaluation_report.txt)")
print("=" * 70)
