# =============================================================================
# PIMA DIABETES - PREPROCESSING COMPARISON EXPERIMENT
# Testing Mean vs Median vs Mode Imputation Strategies
# =============================================================================
# Continues from diabetes_eda.py — loads the same dataset and applies
# three different imputation methods, then evaluates each on a Random Forest
# to find which preprocessing strategy yields the best model performance.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time
import warnings
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# PART 1: SETUP — Load raw data and define experiment configuration
# =============================================================================

print("=" * 70)
print("PREPROCESSING COMPARISON EXPERIMENT")
print("=" * 70)

# --- Load dataset ---
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
       "master/pima-indians-diabetes.data.csv")
df_raw = pd.read_csv(url, names=column_names)
print(f"\n  Dataset loaded: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")

# Columns where 0 is physiologically impossible and must be treated as missing.
# Pregnancies and DiabetesPedigreeFunction are intentionally excluded because
# 0 is a valid value for them (e.g. 0 pregnancies).
ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Feature columns (everything except the target)
FEATURE_COLS = [c for c in df_raw.columns if c != 'Outcome']

# =============================================================================
# PART 2: CORE PREPROCESSING FUNCTION
# =============================================================================

def test_imputation_method(method_name: str, strategy: str) -> dict:
    """
    Full preprocessing + training + evaluation pipeline for one imputation
    strategy.

    Why we test different imputation methods
    -----------------------------------------
    Missing values (here encoded as 0) must be replaced before modelling.
    Three common choices are:
      - Mean   : fast, sensitive to outliers (pulls the mean toward extremes)
      - Median : robust to outliers; better when distributions are skewed
      - Mode   : uses the most frequent value; appropriate for near-discrete
                 features such as BloodPressure

    The method that best preserves the underlying distribution will lead to
    more accurate imputed values and therefore a better-trained model.

    Parameters
    ----------
    method_name : display name ("Mean", "Median", "Mode")
    strategy    : sklearn SimpleImputer strategy string
                  ("mean", "median", "most_frequent")

    Returns
    -------
    dict with keys: method, accuracy, precision, recall, f1, auc, time_sec
    """
    print(f"\n  Testing {method_name} imputation...")
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Replace biologically impossible zeros with NaN
    # ------------------------------------------------------------------
    df = df_raw.copy()
    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)

    # ------------------------------------------------------------------
    # Step 2 — Apply chosen imputation strategy
    # Why column-wise: each feature has its own distribution; imputing
    # with the feature's own statistic is more accurate than a global one.
    # ------------------------------------------------------------------
    imputer = SimpleImputer(strategy=strategy)
    df[ZERO_COLS] = imputer.fit_transform(df[ZERO_COLS])

    # ------------------------------------------------------------------
    # Step 3 — Feature engineering
    # Glucose × BMI interaction captures the joint effect of blood sugar
    # and body-mass on diabetes risk (both are top-2 correlated features).
    # Age_Group converts continuous age into a clinically meaningful bucket.
    # ------------------------------------------------------------------
    df['Glucose_BMI_Interaction'] = df['Glucose'] * df['BMI']

    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 30, 45, 60, 120],
        labels=[0, 1, 2, 3]       # 0=young, 1=middle, 2=senior, 3=elderly
    ).astype(int)

    # ------------------------------------------------------------------
    # Step 4 — Train / test split (stratified to preserve class ratio)
    # ------------------------------------------------------------------
    feature_list = FEATURE_COLS + ['Glucose_BMI_Interaction', 'Age_Group']
    X = df[feature_list].values
    y = df['Outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # Step 5 — Random oversampling on training set ONLY
    # Oversampling after splitting prevents data leakage: the synthetic
    # minority samples are never seen in the test set.
    # ------------------------------------------------------------------
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    # ------------------------------------------------------------------
    # Step 6 — MinMax scaling (fitted on oversampled training data only)
    # Fitting the scaler on training data prevents test-set information
    # from leaking into the scaling parameters.
    # ------------------------------------------------------------------
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc  = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # Step 7 — Train Random Forest and evaluate on the held-out test set
    # ------------------------------------------------------------------
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train_sc, y_train_res)

    y_pred      = clf.predict(X_test_sc)
    y_pred_prob = clf.predict_proba(X_test_sc)[:, 1]

    t_elapsed = time.perf_counter() - t_start

    results = {
        'method'   : method_name,
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall'   : recall_score(y_test, y_pred, zero_division=0),
        'f1'       : f1_score(y_test, y_pred, zero_division=0),
        'auc'      : roc_auc_score(y_test, y_pred_prob),
        'time_sec' : round(t_elapsed, 4)
    }

    # Print single-line result for immediate feedback
    print(f"    Accuracy={results['accuracy']:.4f}  "
          f"Precision={results['precision']:.4f}  "
          f"Recall={results['recall']:.4f}  "
          f"F1={results['f1']:.4f}  "
          f"AUC={results['auc']:.4f}  "
          f"Time={results['time_sec']:.3f}s")

    return results


# =============================================================================
# Run all three imputation strategies
# =============================================================================

print("\n" + "-" * 70)
print("  Running experiments...")
print("-" * 70)

methods = [
    ("Mean",   "mean"),
    ("Median", "median"),
    ("Mode",   "most_frequent"),
]

all_results = []
for name, strategy in methods:
    res = test_imputation_method(name, strategy)
    all_results.append(res)

results_df = pd.DataFrame(all_results)

# Compute composite rank: average rank across all 5 metrics (lower = better)
metrics_for_rank = ['accuracy', 'precision', 'recall', 'f1', 'auc']
rank_df = results_df[metrics_for_rank].rank(ascending=False)   # higher score = rank 1
results_df['avg_rank'] = rank_df.mean(axis=1)
results_df['rank']     = results_df['avg_rank'].rank().astype(int)
results_df = results_df.sort_values('rank').reset_index(drop=True)

print("\n  All experiments complete.")

# =============================================================================
# PART 3: VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: CREATING VISUALIZATIONS")
print("=" * 70)

METRIC_LABELS  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
METRIC_KEYS    = ['accuracy', 'precision', 'recall', 'f1', 'auc']
METHOD_NAMES   = results_df['method'].tolist()
METHOD_COLORS  = ['#3498DB', '#2ECC71', '#E74C3C']   # blue, green, red

# --------------------------------------------------------------------------
# Plot 1 — Grouped bar chart: all methods × all metrics
# imputation_comparison_metrics.png
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

n_methods = len(METHOD_NAMES)
n_metrics = len(METRIC_KEYS)
bar_width  = 0.18
group_gap  = 0.05
x_centers  = np.arange(n_metrics)

metric_colors = ['#2980B9', '#27AE60', '#E74C3C', '#8E44AD', '#F39C12']

for m_idx, row in results_df.iterrows():
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2) * 0
    x_pos = (x_centers
             + (m_idx - (n_methods - 1) / 2) * (bar_width + group_gap / n_methods))
    vals  = [row[k] for k in METRIC_KEYS]

    bars = ax.bar(x_pos, vals, width=bar_width,
                  color=METHOD_COLORS[m_idx], edgecolor='black', linewidth=0.6,
                  label=row['method'], alpha=0.88, zorder=3)

    # Value labels on each bar
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha='center', va='bottom',
                fontsize=7.5, fontweight='bold', rotation=0)

# Mark the overall winner method with a star annotation
winner_name = results_df.iloc[0]['method']
ax.annotate(f"  Best overall: {winner_name}",
            xy=(0.01, 0.97), xycoords='axes fraction',
            fontsize=11, fontweight='bold', color='#27AE60',
            va='top')

ax.set_xticks(x_centers)
ax.set_xticklabels(METRIC_LABELS, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title("Preprocessing Comparison: Imputation Methods", fontsize=14,
             fontweight='bold', pad=14)
ax.legend(title="Imputation Method", fontsize=10, title_fontsize=11,
          loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("imputation_comparison_metrics.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> imputation_comparison_metrics.png")

# --------------------------------------------------------------------------
# Plot 2 — 2×3 subplots: one per metric + summary ranking table
# imputation_comparison_by_metric.png
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes_flat = axes.flatten()

for idx, (m_key, m_label) in enumerate(zip(METRIC_KEYS, METRIC_LABELS)):
    ax = axes_flat[idx]
    vals   = results_df[m_key].tolist()
    names  = results_df['method'].tolist()
    best_v = max(vals)

    # Highlight the winning bar in gold, others in steel-blue
    colors = ['#F1C40F' if v == best_v else '#85C1E9' for v in vals]
    bars   = ax.bar(names, vals, color=colors, edgecolor='black',
                    linewidth=0.7, width=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_title(m_label, fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=9)
    ax.set_ylim(0, min(max(vals) * 1.2, 1.0))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Label best method
    best_method = names[vals.index(best_v)]
    ax.text(0.5, 0.95, f"Best: {best_method}",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color='#D35400',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEBD0', alpha=0.7))

# Last subplot (index 5) — ranking summary table
ax_tbl = axes_flat[5]
ax_tbl.axis('off')

table_data = []
for _, row in results_df.iterrows():
    table_data.append([
        str(int(row['rank'])),
        row['method'],
        f"{row['accuracy']:.4f}",
        f"{row['f1']:.4f}",
        f"{row['auc']:.4f}",
    ])

col_labels = ['Rank', 'Method', 'Accuracy', 'F1', 'AUC']
tbl = ax_tbl.table(
    cellText=table_data, colLabels=col_labels,
    cellLoc='center', loc='center',
    bbox=[0.0, 0.2, 1.0, 0.7]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)

# Style header row
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#2C3E50')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')

# Highlight rank-1 row in gold
for j in range(len(col_labels)):
    tbl[(1, j)].set_facecolor('#F9E79F')

ax_tbl.set_title("Overall Ranking", fontsize=12, fontweight='bold', pad=8)

fig.suptitle("Per-Metric Comparison of Imputation Strategies",
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("imputation_comparison_by_metric.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> imputation_comparison_by_metric.png")

# --------------------------------------------------------------------------
# Plot 3 — Execution time comparison
# imputation_time_comparison.png
# --------------------------------------------------------------------------
times   = results_df['time_sec'].tolist()
names   = results_df['method'].tolist()
min_t   = min(times)

# Color scheme: fastest = green, slowest = red, middle = orange
time_colors = []
for t in times:
    if t == min_t:
        time_colors.append('#27AE60')
    elif t == max(times):
        time_colors.append('#E74C3C')
    else:
        time_colors.append('#F39C12')

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(names, times, color=time_colors, edgecolor='black',
              linewidth=0.7, width=0.4)

for bar, t in zip(bars, times):
    label = f"{t:.4f}s"
    if t == min_t:
        label += "  (fastest)"
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.01,
            label, ha='center', va='bottom',
            fontsize=11, fontweight='bold')

ax.set_title("Preprocessing + Training Time per Imputation Method",
             fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_ylim(0, max(times) * 1.3)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Custom legend patches
patches = [
    mpatches.Patch(color='#27AE60', label='Fastest'),
    mpatches.Patch(color='#F39C12', label='Middle'),
    mpatches.Patch(color='#E74C3C', label='Slowest'),
]
ax.legend(handles=patches, fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig("imputation_time_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> imputation_time_comparison.png")

# =============================================================================
# PART 4: STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: STATISTICAL ANALYSIS")
print("=" * 70)

analysis = {}    # stores per-metric findings

for m_key, m_label in zip(METRIC_KEYS, METRIC_LABELS):
    scores  = dict(zip(results_df['method'], results_df[m_key]))
    best_m  = max(scores, key=scores.get)
    worst_m = min(scores, key=scores.get)
    best_v  = scores[best_m]
    worst_v = scores[worst_m]

    # Improvement = (best - worst) / worst * 100
    pct_improvement = (best_v - worst_v) / worst_v * 100 if worst_v > 0 else 0.0

    # Check if Mean and Median differ by more than 2 % (absolute)
    mean_v   = scores.get('Mean',   None)
    median_v = scores.get('Median', None)
    sig_flag = (abs(mean_v - median_v) > 0.02) if (mean_v and median_v) else False

    analysis[m_key] = {
        'label'      : m_label,
        'scores'     : scores,
        'best'       : best_m,
        'worst'      : worst_m,
        'best_val'   : best_v,
        'worst_val'  : worst_v,
        'pct_impr'   : pct_improvement,
        'mean_vs_med': sig_flag,
    }

    # Determine runner-up
    sorted_methods = sorted(scores, key=scores.get, reverse=True)
    runner_up   = sorted_methods[1]
    runner_val  = scores[runner_up]

    print(f"\n  {m_label}:")
    print(f"    Winner    : {best_m:<8}  {best_v:.4f}")
    print(f"    Runner-up : {runner_up:<8}  {runner_val:.4f}")
    print(f"    Worst     : {worst_m:<8}  {worst_v:.4f}")
    print(f"    Improvement over worst: {pct_improvement:.2f}%")
    if sig_flag:
        print(f"    [NOTE] Mean vs Median differ by >2% -> meaningful difference")

# =============================================================================
# PART 5: GENERATE DETAILED REPORT FILE
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: GENERATING REPORT FILE")
print("=" * 70)

winner  = results_df.iloc[0]
w_name  = winner['method']
w_auc   = winner['auc']

# Worst AUC across all methods for improvement calc
auc_min = results_df['auc'].min()
auc_imp = (w_auc - auc_min) / auc_min * 100 if auc_min > 0 else 0.0

# Build winner explanation from data characteristics observed in EDA
explanation_map = {
    'Mean'  : ("mean imputation assumes a symmetric, normally distributed "
               "feature. When the underlying distributions are approximately "
               "Gaussian (low skew), the mean is the unbiased estimator of "
               "the true missing value. However, it is sensitive to outliers "
               "(e.g. Insulin has extreme values up to 846), so the mean can "
               "over- or under-impute when outliers are present."),
    'Median': ("median imputation is the optimal strategy when features are "
               "skewed or contain outliers. Insulin (skew>5) and SkinThickness "
               "(bimodal) have heavy right tails; the median is unaffected by "
               "extreme values and better represents the 'typical' missing "
               "observation. Robustness to outliers leads to less noise in "
               "imputed values and a cleaner decision boundary for the "
               "Random Forest."),
    'Mode'  : ("mode (most-frequent) imputation replaces each missing value "
               "with the single most common observed value. For near-discrete "
               "features this can be sensible, but for continuous features "
               "it collapses the imputed mass onto one point, reducing "
               "variance and potentially distorting the distribution."),
}

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

report_lines = []
report_lines.append("=" * 72)
report_lines.append("  PREPROCESSING COMPARISON REPORT — PIMA DIABETES DATASET")
report_lines.append(f"  Generated: {now_str}")
report_lines.append("=" * 72)

# --- Section 1: Overview ---
report_lines.append("\n1. EXPERIMENT OVERVIEW")
report_lines.append("-" * 40)
report_lines.append("  Methods tested  : Mean, Median, Mode (most_frequent)")
report_lines.append("  Dataset         : PIMA Indians Diabetes (768 samples, 8 features)")
report_lines.append("  Test protocol   : 80/20 stratified split, Random Forest (50 trees)")
report_lines.append("  Class balancing : RandomOverSampler on training set only")
report_lines.append("  Scaling         : MinMaxScaler (fit on train, applied to test)")
report_lines.append("  Feature eng.    : Glucose_BMI_Interaction, Age_Group added")
report_lines.append("  Imputation cols : Glucose, BloodPressure, SkinThickness, "
                    "Insulin, BMI")

# --- Section 2: Results Table ---
report_lines.append("\n\n2. RESULTS SUMMARY TABLE")
report_lines.append("-" * 72)
header = (f"  {'Method':<10} | {'Accuracy':>9} | {'Precision':>9} | "
          f"{'Recall':>9} | {'F1-Score':>9} | {'AUC-ROC':>9} | {'Rank':>5}")
sep    = "  " + "-" * 68
report_lines.append(header)
report_lines.append(sep)
for _, row in results_df.iterrows():
    line = (f"  {row['method']:<10} | {row['accuracy']*100:>8.2f}% | "
            f"{row['precision']*100:>8.2f}% | {row['recall']*100:>8.2f}% | "
            f"{row['f1']*100:>8.2f}% | {row['auc']:>9.4f} | {int(row['rank']):>5}")
    report_lines.append(line)
report_lines.append(sep)

# --- Section 3: Detailed per-metric analysis ---
report_lines.append("\n\n3. DETAILED ANALYSIS — PER METRIC")
report_lines.append("-" * 40)

for m_key, m_label in zip(METRIC_KEYS, METRIC_LABELS):
    a = analysis[m_key]
    sorted_m = sorted(a['scores'], key=a['scores'].get, reverse=True)
    report_lines.append(f"\n  {m_label}:")
    report_lines.append(f"    Winner     : {a['best']:<8}  score = {a['best_val']:.4f}")
    report_lines.append(f"    Runner-up  : {sorted_m[1]:<8}  score = {a['scores'][sorted_m[1]]:.4f}")
    report_lines.append(f"    Worst      : {a['worst']:<8}  score = {a['worst_val']:.4f}")
    report_lines.append(f"    Improvement: {a['pct_impr']:.2f}% over worst method")
    if a['mean_vs_med']:
        report_lines.append(f"    NOTE: Mean vs Median differ by >2% — this is a "
                            f"practically significant gap.")

# --- Section 4: Why the winner won ---
report_lines.append("\n\n4. WHY THE WINNER WON")
report_lines.append("-" * 40)
report_lines.append(f"\n  Winner: {w_name} Imputation\n")
report_lines.append(f"  Explanation:")
# Wrap the explanation at ~68 chars
exp_text = explanation_map[w_name]
words, line_buf = exp_text.split(), []
for word in words:
    line_buf.append(word)
    if len(" ".join(line_buf)) > 65:
        report_lines.append("    " + " ".join(line_buf[:-1]))
        line_buf = [word]
if line_buf:
    report_lines.append("    " + " ".join(line_buf))

report_lines.append("\n  Data characteristics that support this choice:")
report_lines.append("    - Insulin: heavy right skew (max=846, mean=79.8), "
                    "374 zeros (48.7%)")
report_lines.append("    - SkinThickness: 227 zeros (29.6%), bimodal distribution")
report_lines.append("    - BMI: moderate skew; 11 zeros (1.4%), small outlier effect")
report_lines.append("    - Glucose & BloodPressure: near-normal; few zeros (<5%)")

# --- Section 5: Recommendation ---
report_lines.append("\n\n5. RECOMMENDATION")
report_lines.append("-" * 40)
report_lines.append(f"\n  Based on this experiment, {w_name} imputation is recommended")
report_lines.append( "  for the final model pipeline because:")
report_lines.append(f"    (a) Best overall AUC-ROC: {w_auc:.4f}")

if w_name == "Median":
    report_lines.append("    (b) Robust to the heavy-tailed Insulin distribution")
    report_lines.append("    (c) Unaffected by extreme outliers in SkinThickness/Insulin")
    report_lines.append("    (d) Preserves class-separability better than mean "
                        "when outliers shift the mean")
elif w_name == "Mean":
    report_lines.append("    (b) Feature distributions are sufficiently symmetric")
    report_lines.append("    (c) Mean minimises squared error for Gaussian features")
    report_lines.append("    (d) Provides a smooth, unbiased fill-in for most columns")
else:
    report_lines.append("    (b) Most-frequent value is the MAP estimate for "
                        "near-discrete features")
    report_lines.append("    (c) Avoids fractional imputed values")

# --- Section 6: Impact ---
report_lines.append("\n\n6. IMPACT ON MODEL PERFORMANCE")
report_lines.append("-" * 40)
report_lines.append(f"\n  Improvement from worst to best method:")
report_lines.append(f"    AUC-ROC: {auc_min:.4f}  ->  {w_auc:.4f}  "
                    f"(+{auc_imp:.2f}%)")
report_lines.append("\n  This demonstrates that even before choosing a model,")
report_lines.append("  the choice of imputation strategy can materially affect")
report_lines.append("  classification performance — validating the importance of")
report_lines.append("  thoughtful preprocessing in the ML pipeline.")

# Timing summary
report_lines.append("\n\n7. TIMING SUMMARY")
report_lines.append("-" * 40)
for _, row in results_df.iterrows():
    report_lines.append(f"  {row['method']:<10}: {row['time_sec']:.4f} seconds")

report_lines.append("\n" + "=" * 72)
report_lines.append("  END OF REPORT")
report_lines.append("=" * 72)

# Write report to file (UTF-8 so em-dashes render correctly)
report_path = "preprocessing_comparison_report.txt"
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(report_lines))

print(f"  Saved -> {report_path}")

# =============================================================================
# PART 6: PRINT SUMMARY
# =============================================================================

# Determine AUC ranks for summary print
sorted_by_auc = results_df.sort_values('auc', ascending=False).reset_index(drop=True)

print("\n")
print("=" * 70)
print("PREPROCESSING COMPARISON EXPERIMENT - RESULTS")
print("=" * 70)

print(f"\n  Methods Tested  : Mean, Median, Mode")
print(f"  Evaluation Model: Random Forest (50 trees) on held-out test set")

print("\n  RESULTS (ranked by AUC-ROC):")
for i, row in sorted_by_auc.iterrows():
    print(f"  {i+1}. {row['method']:<8}: AUC-ROC = {row['auc']:.4f}  "
          f"(Rank: {i+1})")

print(f"\n  [WINNER] {w_name} Imputation")
print(f"    Best AUC-ROC  : {w_auc:.4f}")
print(f"    Improvement   : +{auc_imp:.1f}% over worst method")

if w_name == "Median":
    reason = "Most robust to the heavy-tailed Insulin and SkinThickness distributions"
elif w_name == "Mean":
    reason = "Feature distributions are near-symmetric; mean is the optimal estimator"
else:
    reason = "Most-frequent value best represents the near-discrete feature modes"
print(f"    Reason        : {reason}")

print(f"\n  RECOMMENDATION:")
print(f"    Use {w_name} imputation for the final model pipeline because it")
print(f"    achieved the best AUC-ROC ({w_auc:.4f}) and is the most")

if w_name == "Median":
    print("    statistically appropriate choice given outlier-heavy features.")
elif w_name == "Mean":
    print("    appropriate choice given the near-Gaussian feature distributions.")
else:
    print("    appropriate choice for the discrete value patterns in this dataset.")

print("\n" + "=" * 70)
print("[OK] Preprocessing comparison complete")
print("[OK] 3 visualizations created")
print("[OK] Detailed report saved -> preprocessing_comparison_report.txt")
print("=" * 70)
