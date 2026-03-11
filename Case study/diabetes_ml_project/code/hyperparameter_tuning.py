# =============================================================================
# PIMA DIABETES — HYPERPARAMETER TUNING: RANDOM FOREST
# Grid Search over n_estimators, max_depth, min_samples_leaf, max_features
# Continues from: model_training.py / model_evaluation.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import time
import warnings
from datetime import datetime

from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing   import MinMaxScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from imblearn.over_sampling  import RandomOverSampler
from preprocessing           import build_pipeline, ENG_FEATURES

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# FULL PREPROCESSING REBUILD  (mirrors model_training.py exactly)
# =============================================================================

print("=" * 70)
print("HYPERPARAMETER TUNING — RANDOM FOREST")
print("Preprocessing: Median imputation (experiment winner)")
print("=" * 70)

COLUMN_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
       "master/pima-indians-diabetes.data.csv")

data        = build_pipeline()
X_train_sc  = data.X_train_sc
X_test_sc   = data.X_test_sc
y_train_bal = data.y_train_res
y_test      = data.y_test

print(f"\n  Train (balanced): {len(X_train_sc)} | "
      f"Test: {len(X_test_sc)} | Features: {len(ENG_FEATURES)}")

# =============================================================================
# BASELINE — DEFAULT RANDOM FOREST
# =============================================================================

print("\n  Training default Random Forest (baseline)...")
DEFAULT_PARAMS = {
    'n_estimators'   : 100,
    'max_depth'      : 10,
    'min_samples_leaf': 5,
    'max_features'   : 'sqrt',
}

t0 = time.perf_counter()
rf_default = RandomForestClassifier(**DEFAULT_PARAMS, random_state=42, n_jobs=-1)
rf_default.fit(X_train_sc, y_train_bal)
default_train_time = time.perf_counter() - t0

y_pred_def  = rf_default.predict(X_test_sc)
y_prob_def  = rf_default.predict_proba(X_test_sc)[:, 1]
default_metrics = {
    'Accuracy' : accuracy_score(y_test, y_pred_def),
    'Precision': precision_score(y_test, y_pred_def, zero_division=0),
    'Recall'   : recall_score(y_test, y_pred_def, zero_division=0),
    'F1'       : f1_score(y_test, y_pred_def, zero_division=0),
    'AUC-ROC'  : roc_auc_score(y_test, y_prob_def),
}
print(f"  Baseline AUC-ROC: {default_metrics['AUC-ROC']:.4f}  "
      f"(train time: {default_train_time:.4f}s)")

# =============================================================================
# PART 1: GRID SEARCH SETUP
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: GRID SEARCH")
print("=" * 70)

# Why each parameter matters:
#   n_estimators   : More trees reduce variance; returns diminish past ~150.
#   max_depth      : Controls bias-variance tradeoff; too deep = overfit.
#   min_samples_leaf: Regularisation — forces each leaf to represent enough data.
#   max_features   : Random feature sub-sampling at each split; reduces
#                    tree correlation ('sqrt' = √p, 'log2' = log2(p)).

PARAM_GRID = {
    'n_estimators'    : [50, 100, 150, 200],
    'max_depth'       : [5, 10, 15, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features'    : ['sqrt', 'log2'],
}

n_combos = (len(PARAM_GRID['n_estimators'])
            * len(PARAM_GRID['max_depth'])
            * len(PARAM_GRID['min_samples_leaf'])
            * len(PARAM_GRID['max_features']))

print(f"\n  Starting Hyperparameter Tuning on Random Forest...")
print(f"  Testing {n_combos} parameter combinations")
print(f"  Scoring metric : roc_auc  |  CV folds: 3  |  n_jobs: -1")

gs_t0 = time.perf_counter()

grid_search = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid = PARAM_GRID,
    cv         = 3,
    scoring    = 'roc_auc',
    n_jobs     = -1,
    verbose    = 1,
    return_train_score=False,
)
grid_search.fit(X_train_sc, y_train_bal)

gs_elapsed = time.perf_counter() - gs_t0

print(f"\n  Grid Search Complete!")
print(f"  Time taken      : {gs_elapsed:.2f} seconds")
print(f"  Best CV AUC-ROC : {grid_search.best_score_:.4f}")
print(f"  Best parameters : {grid_search.best_params_}")

best_params   = grid_search.best_params_
best_cv_score = grid_search.best_score_

# =============================================================================
# PART 2: TRAIN TUNED MODEL & COMPARE
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: TUNED MODEL vs DEFAULT")
print("=" * 70)

t0 = time.perf_counter()
rf_tuned = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
rf_tuned.fit(X_train_sc, y_train_bal)
tuned_train_time = time.perf_counter() - t0

y_pred_tuned = rf_tuned.predict(X_test_sc)
y_prob_tuned = rf_tuned.predict_proba(X_test_sc)[:, 1]
tuned_metrics = {
    'Accuracy' : accuracy_score(y_test, y_pred_tuned),
    'Precision': precision_score(y_test, y_pred_tuned, zero_division=0),
    'Recall'   : recall_score(y_test, y_pred_tuned, zero_division=0),
    'F1'       : f1_score(y_test, y_pred_tuned, zero_division=0),
    'AUC-ROC'  : roc_auc_score(y_test, y_prob_tuned),
}

METRIC_KEYS = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']

print(f"\n  {'Metric':<12} {'Default':>10} {'Tuned':>10} {'Improvement':>13}")
print("  " + "-" * 48)
for mk in METRIC_KEYS:
    d_val  = default_metrics[mk]
    t_val  = tuned_metrics[mk]
    delta  = t_val - d_val
    pct    = delta / d_val * 100 if d_val > 0 else 0.0
    sign   = "+" if delta >= 0 else ""
    print(f"  {mk:<12} {d_val:>10.4f} {t_val:>10.4f} "
          f"{sign}{delta:>7.4f} ({sign}{pct:.2f}%)")

# =============================================================================
# PART 3: PARSE GRID SEARCH RESULTS
# =============================================================================

# Build clean results DataFrame from cv_results_
cv_res = pd.DataFrame(grid_search.cv_results_)

# Explode the params dict into individual columns
param_names = list(PARAM_GRID.keys())
for pn in param_names:
    cv_res[pn] = cv_res['params'].apply(lambda p: p[pn])

cv_res_clean = cv_res[[
    'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features',
    'mean_test_score', 'std_test_score', 'rank_test_score'
]].copy()
cv_res_clean.columns = [
    'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features',
    'mean_AUC', 'std_AUC', 'rank'
]
cv_res_clean = cv_res_clean.sort_values('rank').reset_index(drop=True)
cv_res_clean.to_csv("grid_search_results.csv", index=False)
print(f"\n  Saved -> grid_search_results.csv  ({len(cv_res_clean)} rows)")

# --- Compute marginal effects per parameter ---
def marginal(col):
    """Mean (and std) CV AUC across all other parameter combinations."""
    grp = cv_res_clean.groupby(col)['mean_AUC']
    return grp.mean(), grp.std().fillna(0)

ne_mean, ne_std = marginal('n_estimators')
md_mean, md_std = marginal('max_depth')
ml_mean, ml_std = marginal('min_samples_leaf')
mf_mean, mf_std = marginal('max_features')

# --- Parameter importance = range of marginal means ---
param_range = {
    'n_estimators'    : ne_mean.max() - ne_mean.min(),
    'max_depth'       : md_mean.max() - md_mean.min(),
    'min_samples_leaf': ml_mean.max() - ml_mean.min(),
    'max_features'    : mf_mean.max() - mf_mean.min(),
}
most_sensitive  = max(param_range, key=param_range.get)
least_sensitive = min(param_range, key=param_range.get)

# =============================================================================
# PART 4: VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: CREATING VISUALIZATIONS")
print("=" * 70)

sns.set_style("whitegrid")
GOLD = '#F1C40F'

# --------------------------------------------------------------------------
# Plot 1 — n_estimators x max_depth heatmap of mean AUC
# --------------------------------------------------------------------------
pivot = cv_res_clean.groupby(
    ['n_estimators', 'max_depth']
)['mean_AUC'].mean().reset_index()
pivot_mat = pivot.pivot(index='max_depth', columns='n_estimators', values='mean_AUC')
# Sort rows so shallow depth is at top (most interpretable orientation)
pivot_mat = pivot_mat.sort_index(ascending=False)

best_ne = best_params['n_estimators']
best_md = best_params['max_depth']

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(
    pivot_mat, annot=True, fmt='.3f', cmap='viridis',
    linewidths=0.5, linecolor='white',
    cbar_kws={'shrink': 0.8, 'label': 'Mean AUC-ROC (averaged over other params)'},
    annot_kws={'size': 11, 'weight': 'bold'}, ax=ax,
    vmin=pivot_mat.values.min() - 0.002,
    vmax=pivot_mat.values.max() + 0.002,
)

# Mark optimal cell with a white star
col_idx = list(pivot_mat.columns).index(best_ne)
row_idx = list(pivot_mat.index).index(best_md)
ax.text(col_idx + 0.5, row_idx + 0.5, '*', ha='center', va='center',
        fontsize=22, color='white', fontweight='bold')

ax.set_title(f"Hyperparameter Sensitivity: n_estimators vs max_depth\n"
             f"(* = optimal combination: n_est={best_ne}, depth={best_md})",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("n_estimators", fontsize=11)
ax.set_ylabel("max_depth", fontsize=11)
plt.tight_layout()
plt.savefig("parameter_sensitivity_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> parameter_sensitivity_heatmap.png")

# --------------------------------------------------------------------------
# Plot 2 — 2x2 marginal effect plots per parameter
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()

# --- n_estimators (line + fill) ---
ax = axes_flat[0]
x_ne = ne_mean.index.tolist()
ax.plot(x_ne, ne_mean.values, 'o-', color='#3498DB', linewidth=2.2,
        markersize=8, label='Mean AUC')
ax.fill_between(x_ne,
                ne_mean.values - ne_std.values,
                ne_mean.values + ne_std.values,
                alpha=0.2, color='#3498DB', label='±1 std')
opt_val = ne_mean[best_params['n_estimators']]
ax.axvline(best_params['n_estimators'], color=GOLD, linestyle='--',
           linewidth=2, label=f"Optimal: {best_params['n_estimators']}")
ax.scatter([best_params['n_estimators']], [opt_val], color=GOLD,
           s=120, zorder=5)
ax.set_title("n_estimators", fontsize=12, fontweight='bold')
ax.set_xlabel("Number of Trees", fontsize=10)
ax.set_ylabel("Mean CV AUC-ROC", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# --- max_depth (line + fill) ---
ax = axes_flat[1]
x_md = md_mean.index.tolist()
ax.plot(x_md, md_mean.values, 's-', color='#2ECC71', linewidth=2.2,
        markersize=8, label='Mean AUC')
ax.fill_between(x_md,
                md_mean.values - md_std.values,
                md_mean.values + md_std.values,
                alpha=0.2, color='#2ECC71', label='±1 std')
opt_val = md_mean[best_params['max_depth']]
ax.axvline(best_params['max_depth'], color=GOLD, linestyle='--',
           linewidth=2, label=f"Optimal: {best_params['max_depth']}")
ax.scatter([best_params['max_depth']], [opt_val], color=GOLD,
           s=120, zorder=5)
ax.set_title("max_depth", fontsize=12, fontweight='bold')
ax.set_xlabel("Maximum Tree Depth", fontsize=10)
ax.set_ylabel("Mean CV AUC-ROC", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# --- min_samples_leaf (line + fill) ---
ax = axes_flat[2]
x_ml = ml_mean.index.tolist()
ax.plot(x_ml, ml_mean.values, '^-', color='#E67E22', linewidth=2.2,
        markersize=8, label='Mean AUC')
ax.fill_between(x_ml,
                ml_mean.values - ml_std.values,
                ml_mean.values + ml_std.values,
                alpha=0.2, color='#E67E22', label='±1 std')
opt_val = ml_mean[best_params['min_samples_leaf']]
ax.axvline(best_params['min_samples_leaf'], color=GOLD, linestyle='--',
           linewidth=2, label=f"Optimal: {best_params['min_samples_leaf']}")
ax.scatter([best_params['min_samples_leaf']], [opt_val], color=GOLD,
           s=120, zorder=5)
ax.set_title("min_samples_leaf", fontsize=12, fontweight='bold')
ax.set_xlabel("Minimum Samples per Leaf", fontsize=10)
ax.set_ylabel("Mean CV AUC-ROC", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# --- max_features (bar chart) ---
ax = axes_flat[3]
x_mf  = mf_mean.index.tolist()
colors_mf = [GOLD if v == best_params['max_features'] else '#9B59B6'
             for v in x_mf]
bars = ax.bar(x_mf, mf_mean.values, color=colors_mf,
              edgecolor='black', linewidth=0.7, width=0.4,
              yerr=mf_std.values, capsize=8,
              error_kw=dict(elinewidth=1.8, ecolor='#2C3E50', capthick=2))
for bar, val in zip(bars, mf_mean.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + mf_std.values.max() + 0.0005,
            f"{val:.4f}", ha='center', va='bottom', fontsize=10,
            fontweight='bold')
ax.set_title("max_features", fontsize=12, fontweight='bold')
ax.set_xlabel("Feature Subset Strategy", fontsize=10)
ax.set_ylabel("Mean CV AUC-ROC", fontsize=10)
ax.set_ylim(mf_mean.min() - 0.003,
            mf_mean.max() + mf_std.values.max() + 0.006)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
gold_patch = mpatches.Patch(color=GOLD, label=f"Optimal: {best_params['max_features']}")
ax.legend(handles=[gold_patch], fontsize=9)

fig.suptitle("Parameter Effect on CV AUC-ROC (Marginal Averages)\n"
             "Gold dashed line / gold bar = optimal value",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("parameter_effect_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> parameter_effect_plots.png")

# --------------------------------------------------------------------------
# Plot 3 — Default vs Tuned grouped bar chart
# --------------------------------------------------------------------------
bar_metric_colors = ['#2980B9', '#27AE60', '#E74C3C', '#8E44AD', '#F39C12']
bar_width = 0.32
x_pos     = np.arange(len(METRIC_KEYS))

fig, ax = plt.subplots(figsize=(10, 6))

bars_def = ax.bar(x_pos - bar_width / 2,
                  [default_metrics[m] for m in METRIC_KEYS],
                  width=bar_width, label='Default RF', color='#85C1E9',
                  edgecolor='black', linewidth=0.6, alpha=0.85)
bars_tun = ax.bar(x_pos + bar_width / 2,
                  [tuned_metrics[m] for m in METRIC_KEYS],
                  width=bar_width, label='Tuned RF', color='#F9E79F',
                  edgecolor='black', linewidth=0.6, alpha=0.85)

# Value labels on default bars
for bar, mk in zip(bars_def, METRIC_KEYS):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{default_metrics[mk]:.4f}", ha='center', va='bottom',
            fontsize=8, color='#1A5276')

# Value labels + improvement on tuned bars
for bar, mk in zip(bars_tun, METRIC_KEYS):
    d_val  = default_metrics[mk]
    t_val  = tuned_metrics[mk]
    delta  = (t_val - d_val) / d_val * 100 if d_val > 0 else 0.0
    sign   = "+" if delta >= 0 else ""
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{t_val:.4f}", ha='center', va='bottom',
            fontsize=8, color='#784212')
    # Improvement annotation above tuned bar
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.022,
            f"{sign}{delta:.1f}%",
            ha='center', va='bottom', fontsize=7.5,
            color='#27AE60' if delta >= 0 else '#E74C3C',
            fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(METRIC_KEYS, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, max(tuned_metrics.values()) * 1.18)
ax.set_title("Random Forest: Default vs Tuned Performance",
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("tuning_improvement_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> tuning_improvement_comparison.png")

# --------------------------------------------------------------------------
# Plot 4 — Parameter importance (range-based variance)
# --------------------------------------------------------------------------
param_range_sorted = dict(sorted(param_range.items(),
                                 key=lambda x: x[1], reverse=True))
pnames = list(param_range_sorted.keys())
pranges = list(param_range_sorted.values())
max_r   = max(pranges)

# Colour gradient: high variance = deeper colour (more important)
bar_colors_imp = [plt.cm.YlOrRd(0.35 + 0.6 * v / max_r) for v in pranges]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(pnames, pranges, color=bar_colors_imp,
              edgecolor='black', linewidth=0.7, width=0.5, zorder=3)

for bar, rng, pn in zip(bars, pranges, pnames):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_r * 0.01,
            f"{rng:.5f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold')

ax.set_ylabel("AUC-ROC Range  (max marginal mean - min marginal mean)",
              fontsize=10)
ax.set_title("Parameter Sensitivity Analysis\n"
             "Which parameters matter most for tuning?",
             fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0, max_r * 1.25)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Importance labels below each bar
importance_labels = ['Most important', '2nd', '3rd', 'Least important']
for bar, lbl in zip(bars, importance_labels[:len(bars)]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            -max_r * 0.07, lbl, ha='center', va='top',
            fontsize=8.5, color='#555', style='italic')

plt.tight_layout()
plt.savefig("parameter_importance_variance.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> parameter_importance_variance.png")

# =============================================================================
# PART 5: GENERATE TUNING REPORT
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: GENERATING TUNING REPORT")
print("=" * 70)

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Is tuning "worth it"?
auc_improvement_abs = tuned_metrics['AUC-ROC'] - default_metrics['AUC-ROC']
auc_improvement_pct = auc_improvement_abs / default_metrics['AUC-ROC'] * 100
tuning_worth_it     = auc_improvement_abs > 0.005 or \
                      tuned_metrics['Recall'] > default_metrics['Recall']

lines = []
lines.append("=" * 72)
lines.append("  HYPERPARAMETER TUNING REPORT — RANDOM FOREST")
lines.append(f"  Generated : {now_str}")
lines.append("=" * 72)

# --- Section 1: Configuration ---
lines.append("\n1. TUNING CONFIGURATION")
lines.append("-" * 40)
lines.append(f"  Model            : Random Forest (sklearn)")
lines.append(f"  Parameter grid   :")
for pn, vals in PARAM_GRID.items():
    lines.append(f"    {pn:<20} : {vals}")
lines.append(f"  Total combinations: {n_combos}")
lines.append(f"  Cross-validation : 3-fold  (stratified via RF internal)")
lines.append(f"  Scoring metric   : roc_auc")
lines.append(f"  Search time      : {gs_elapsed:.2f} seconds")

# --- Section 2: Optimal parameters ---
lines.append("\n\n2. OPTIMAL PARAMETERS FOUND")
lines.append("-" * 50)
lines.append(f"\n  {'Parameter':<20} {'Default':>10} {'Optimal':>10} {'Change':>12}")
lines.append("  " + "-" * 54)
param_defaults = {
    'n_estimators'    : 100, 'max_depth': 10,
    'min_samples_leaf': 5,   'max_features': 'sqrt',
}
for pn in PARAM_GRID.keys():
    d_val = param_defaults[pn]
    o_val = best_params[pn]
    if isinstance(d_val, int) and isinstance(o_val, int):
        change = f"{'+' if o_val - d_val >= 0 else ''}{o_val - d_val}"
    elif d_val == o_val:
        change = "unchanged"
    else:
        change = f"{d_val} -> {o_val}"
    lines.append(f"  {pn:<20} {str(d_val):>10} {str(o_val):>10} {change:>12}")

lines.append(f"\n  Best CV AUC-ROC (3-fold): {best_cv_score:.4f}")

# --- Section 3: Performance improvement ---
lines.append("\n\n3. PERFORMANCE IMPROVEMENT ON TEST SET")
lines.append("-" * 55)
lines.append(f"\n  {'Metric':<12} {'Default RF':>12} {'Tuned RF':>10} "
             f"{'Abs Gain':>10} {'% Gain':>10}")
lines.append("  " + "-" * 56)
for mk in METRIC_KEYS:
    d_v = default_metrics[mk]
    t_v = tuned_metrics[mk]
    ab  = t_v - d_v
    pct = ab / d_v * 100 if d_v > 0 else 0.0
    sgn = "+" if ab >= 0 else ""
    lines.append(f"  {mk:<12} {d_v:>12.4f} {t_v:>10.4f} "
                 f"{sgn}{ab:>9.4f} {sgn}{pct:>9.2f}%")

# --- Section 4: Parameter impact analysis ---
lines.append("\n\n4. PARAMETER IMPACT ANALYSIS")
lines.append("-" * 40)

param_interp = {
    'n_estimators'    : ("More trees consistently reduce variance up to a "
                         "saturation point. Beyond ~150, returns diminish as "
                         "new trees add little uncorrelated signal."),
    'max_depth'       : ("Controls model complexity. Too shallow = underfitting; "
                         "too deep = overfitting. Dataset complexity determines "
                         "the optimal depth."),
    'min_samples_leaf': ("Acts as a regulariser — forces each terminal node to "
                         "represent at least N training samples, preventing "
                         "the model from memorising noise."),
    'max_features'    : ("Controls correlation between trees. 'sqrt' and 'log2' "
                         "both reduce correlation; difference is small for 10 "
                         "features, explaining low sensitivity."),
}

for rank_i, (pn, rng) in enumerate(param_range_sorted.items(), 1):
    mn = cv_res_clean.groupby(pn)['mean_AUC'].mean()
    lines.append(f"\n  Rank {rank_i}: {pn}")
    lines.append(f"    Score range  : {mn.min():.4f} to {mn.max():.4f}")
    lines.append(f"    Range (delta): {rng:.5f}")
    lines.append(f"    Interpretation: {param_interp[pn]}")

# --- Section 5: Key insights ---
lines.append("\n\n5. KEY INSIGHTS")
lines.append("-" * 40)

# Auto-generate insights from data
ne_vals = ne_mean.sort_index()
if ne_vals.diff().iloc[1:].mean() > 0:
    insight_ne = (f"n_estimators: Increasing trees from "
                  f"{ne_vals.index[0]} to {ne_vals.index[-1]} "
                  f"improved AUC from {ne_vals.iloc[0]:.4f} to "
                  f"{ne_vals.iloc[-1]:.4f} (+{ne_vals.diff().sum():.4f}).")
else:
    peak_ne = ne_vals.idxmax()
    insight_ne = (f"n_estimators: AUC peaked at {peak_ne} trees "
                  f"({ne_vals[peak_ne]:.4f}) and plateaued — diminishing "
                  f"returns beyond that point.")

md_vals = md_mean.sort_index()
peak_md = md_vals.idxmax()
insight_md = (f"max_depth: Optimal depth is {peak_md}. Shallow trees "
              f"(depth=5: {md_vals.get(5, 0):.4f}) underfit; very deep "
              f"trees add noise without meaningful gain.")

ml_vals = ml_mean.sort_index()
best_ml = ml_vals.idxmax()
insight_ml = (f"min_samples_leaf: Best value = {best_ml}. Small values "
              f"(e.g. 1) risk overfitting individual noisy samples; "
              f"larger values add too much regularisation.")

mf_delta = mf_mean.max() - mf_mean.min()
insight_mf = (f"max_features ('{best_params['max_features']}'): Low sensitivity "
              f"(range {mf_delta:.5f}) — for 10 features, 'sqrt' (~3) and "
              f"'log2' (~3) are nearly identical, so the choice matters little.")

for ins in [insight_ne, insight_md, insight_ml, insight_mf]:
    lines.append(f"  - {ins}")

# --- Section 6: Computational cost ---
lines.append("\n\n6. COMPUTATIONAL COST ANALYSIS")
lines.append("-" * 40)
lines.append(f"  Default RF training time   : {default_train_time:.4f}s")
lines.append(f"  Grid search total time     : {gs_elapsed:.2f}s "
             f"({n_combos} combinations x 3-fold)")
lines.append(f"  Tuned RF training time     : {tuned_train_time:.4f}s")
lines.append(f"  Total tuning investment    : {gs_elapsed + tuned_train_time:.2f}s")
lines.append(f"  AUC improvement            : "
             f"{'+' if auc_improvement_abs >= 0 else ''}"
             f"{auc_improvement_abs:.4f} "
             f"({'+' if auc_improvement_pct >= 0 else ''}{auc_improvement_pct:.2f}%)")
if tuning_worth_it:
    lines.append(f"  Is tuning worth it?        : YES — meaningful gain achieved.")
else:
    lines.append(f"  Is tuning worth it?        : MARGINAL — gain is small; default "
                 f"params are near-optimal for this dataset size.")

# --- Section 7: Recommendation ---
lines.append("\n\n7. RECOMMENDATION")
lines.append("-" * 40)
if tuning_worth_it:
    use_tuned = "YES"
    just = (f"The tuned model achieves AUC {tuned_metrics['AUC-ROC']:.4f} "
            f"vs {default_metrics['AUC-ROC']:.4f} default "
            f"(+{auc_improvement_abs:.4f} absolute, +{auc_improvement_pct:.2f}%). "
            f"Recall also {'improved' if tuned_metrics['Recall'] > default_metrics['Recall'] else 'unchanged'}, "
            f"which is critical for medical screening.")
else:
    use_tuned = "OPTIONAL"
    just = (f"Marginal AUC gain ({auc_improvement_abs:+.4f}). "
            f"Default parameters are already well-tuned for this dataset. "
            f"Use tuned model only if infrastructure cost is low.")

lines.append(f"\n  Use Tuned Random Forest: {use_tuned}")
lines.append(f"  Justification: {just}")
lines.append(f"\n  For production deployment:")
lines.append(f"    Recommended parameters:")
for pn, pv in best_params.items():
    lines.append(f"      {pn:<20}: {pv}")
lines.append(f"    Expected AUC-ROC         : {tuned_metrics['AUC-ROC']:.4f}")
lines.append(f"    Retuning frequency       : Every 6-12 months with new patient data")
lines.append(f"    Retune if               : Dataset grows >50% or AUC drops >2%")

lines.append("\n\n" + "=" * 72)
lines.append("  END OF TUNING REPORT")
lines.append("=" * 72)

with open("hyperparameter_tuning_report.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("  Saved -> hyperparameter_tuning_report.txt")

# =============================================================================
# PART 6: PRINT SUMMARY
# =============================================================================

auc_def   = default_metrics['AUC-ROC']
auc_tuned = tuned_metrics['AUC-ROC']
auc_delta = auc_tuned - auc_def
auc_pct   = auc_delta / auc_def * 100

print("\n")
print("=" * 70)
print("HYPERPARAMETER TUNING COMPLETE")
print("=" * 70)

print(f"""
  Model                : Random Forest
  Combinations tested  : {n_combos}
  Search time          : {gs_elapsed:.2f} seconds

  OPTIMAL PARAMETERS:
    n_estimators    : {best_params['n_estimators']}
    max_depth       : {best_params['max_depth']}
    min_samples_leaf: {best_params['min_samples_leaf']}
    max_features    : {best_params['max_features']}

  PERFORMANCE IMPROVEMENT:
    Default RF  ->  AUC-ROC: {auc_def:.4f}
    Tuned RF    ->  AUC-ROC: {auc_tuned:.4f}
    Improvement : {'+' if auc_delta >= 0 else ''}{auc_pct:.2f}%  ({'+' if auc_delta >= 0 else ''}{auc_delta:.4f} absolute)

  MOST  SENSITIVE PARAMETER: {most_sensitive}
    Range: {cv_res_clean.groupby(most_sensitive)['mean_AUC'].mean().min():.4f}
       to: {cv_res_clean.groupby(most_sensitive)['mean_AUC'].mean().max():.4f}
    Variance (range): {param_range[most_sensitive]:.5f}

  LEAST SENSITIVE PARAMETER: {least_sensitive}
    Variance (range): {param_range[least_sensitive]:.5f}

  RECOMMENDATION: {use_tuned} — {just[:80]}
""")

print("=" * 70)
print("[OK] 4 visualizations created")
print("[OK] Grid search results saved  (grid_search_results.csv)")
print("[OK] Tuning report generated    (hyperparameter_tuning_report.txt)")
print("=" * 70)
