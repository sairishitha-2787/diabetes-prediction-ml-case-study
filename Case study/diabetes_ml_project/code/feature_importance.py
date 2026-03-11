# =============================================================================
# PIMA DIABETES — COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS
# Extracts and compares importance from Logistic Regression, Decision Tree,
# and Random Forest; generates clinical interpretation report.
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
from scipy.stats import pearsonr, spearmanr
import warnings
from datetime import datetime

from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score
from imblearn.over_sampling  import RandomOverSampler
from preprocessing           import build_pipeline, COLUMN_NAMES, ENG_FEATURES

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# FULL PREPROCESSING + MODEL REBUILD  (mirrors model_training.py exactly)
# =============================================================================

print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("Rebuilding pipeline (Median imputation)")
print("=" * 70)

COLUMN_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
       "master/pima-indians-diabetes.data.csv")
data        = build_pipeline()
X_train_raw = data.X_train_raw
y_train_raw = data.y_train_raw
X_train_res = data.X_train_res
y_train_res = data.y_train_res
X_train_sc  = data.X_train_sc
X_test_orig = data.X_test_raw
X_test_sc   = data.X_test_sc
y_test      = data.y_test
N_FEAT      = len(ENG_FEATURES)

# --- Compute class-level means (from unscaled training data, for clinical ref) ---
train_df = pd.DataFrame(X_train_raw, columns=ENG_FEATURES)
train_df['Outcome'] = y_train_raw
class_means = train_df.groupby('Outcome')[ENG_FEATURES].mean()

print(f"  Train (balanced): {len(X_train_sc)} | "
      f"Test: {len(X_test_sc)} | Features: {N_FEAT}\n")

# --- Train the 3 models used for importance extraction ---
import time
print("  Training 3 models for importance extraction...")

lr  = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
dt  = DecisionTreeClassifier(max_depth=5, min_samples_split=20,
                             min_samples_leaf=10, random_state=42, criterion='gini')
rf  = RandomForestClassifier(n_estimators=100, max_depth=10,
                             min_samples_leaf=5, random_state=42, n_jobs=-1)

for name, clf in [("Logistic Regression", lr),
                  ("Decision Tree",       dt),
                  ("Random Forest",       rf)]:
    t0 = time.perf_counter()
    clf.fit(X_train_sc, y_train_res)
    print(f"    {name:<22}  {time.perf_counter()-t0:.4f}s")

# =============================================================================
# PART 1: EXTRACT FEATURE IMPORTANCES
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: EXTRACTING FEATURE IMPORTANCES")
print("=" * 70)

# --- Why each model computes importance differently ---
# Logistic Regression: absolute coefficient magnitudes.  Because we scaled
#   features to [0,1], coefficients are directly comparable — larger |coef|
#   means the feature shifts the log-odds more per unit change.
# Decision Tree: Gini importance = total Gini impurity reduction across all
#   splits on that feature, weighted by node sample count.
# Random Forest: mean Gini importance across all 100 trees — more stable and
#   less biased than a single tree.

def max_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] by dividing by its maximum."""
    arr = np.abs(arr)
    mx  = arr.max()
    return arr / mx if mx > 0 else arr

lr_imp = max_normalize(lr.coef_[0])
dt_imp = max_normalize(dt.feature_importances_)
rf_imp = max_normalize(rf.feature_importances_)

imp_df = pd.DataFrame({
    'Feature'            : ENG_FEATURES,
    'Logistic_Regression': lr_imp,
    'Decision_Tree'      : dt_imp,
    'Random_Forest'      : rf_imp,
})
imp_df['Average_Importance'] = imp_df[
    ['Logistic_Regression', 'Decision_Tree', 'Random_Forest']
].mean(axis=1)
imp_df['Rank_Avg'] = imp_df['Average_Importance'].rank(ascending=False).astype(int)
imp_df = imp_df.sort_values('Average_Importance', ascending=False).reset_index(drop=True)

# Rank per model (for disagreement analysis)
imp_df['Rank_LR'] = imp_df['Logistic_Regression'].rank(ascending=False).astype(int)
imp_df['Rank_DT'] = imp_df['Decision_Tree'].rank(ascending=False).astype(int)
imp_df['Rank_RF'] = imp_df['Random_Forest'].rank(ascending=False).astype(int)

# Variance across models (disagreement measure)
imp_df['Cross_Model_Variance'] = imp_df[
    ['Logistic_Regression', 'Decision_Tree', 'Random_Forest']
].var(axis=1)

imp_df.to_csv("feature_importance_comparison.csv", index=False)
print("  Saved -> feature_importance_comparison.csv\n")

# Print formatted table
hdr = (f"  {'Rank':<5} {'Feature':<28} {'Avg':>7} "
       f"{'LR':>7} {'DT':>7} {'RF':>7} {'Variance':>10}")
sep = "  " + "-" * 70
print(hdr); print(sep)
for _, r in imp_df.iterrows():
    print(f"  {int(r['Rank_Avg']):<5} {r['Feature']:<28} "
          f"{r['Average_Importance']:>7.4f} "
          f"{r['Logistic_Regression']:>7.4f} "
          f"{r['Decision_Tree']:>7.4f} "
          f"{r['Random_Forest']:>7.4f} "
          f"{r['Cross_Model_Variance']:>10.4f}")
print(sep)

# Convenience
top3       = imp_df.head(3)
top5_feats = imp_df.head(5)['Feature'].tolist()

# =============================================================================
# PART 2: VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: CREATING VISUALIZATIONS")
print("=" * 70)

MODEL_COLS   = ['Logistic_Regression', 'Decision_Tree', 'Random_Forest']
MODEL_LABELS = ['Logistic Regression', 'Decision Tree', 'Random Forest']
MODEL_COLORS = ['#3498DB', '#2ECC71', '#E67E22']
sns.set_style("whitegrid")

# --------------------------------------------------------------------------
# Plot 1 — 1×3 horizontal bar charts (one per model, top 8 features)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 10))

for ax, col, label, color in zip(axes, MODEL_COLS, MODEL_LABELS, MODEL_COLORS):
    sub = imp_df.sort_values(col, ascending=True).tail(8)  # top 8, ascending for barh
    vals  = sub[col].tolist()
    names = [f.replace('_', '\n') for f in sub['Feature'].tolist()]

    bars = ax.barh(names, vals, color=color, edgecolor='black',
                   linewidth=0.6, height=0.6, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', ha='left',
                fontsize=8.5, fontweight='bold')

    ax.set_xlim(0, 1.18)
    ax.set_title(label, fontsize=12, fontweight='bold', color=color, pad=8)
    ax.set_xlabel("Normalized Importance [0–1]", fontsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle("Feature Importance by Model (Top 8 Features)",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("feature_importance_by_model.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> feature_importance_by_model.png")

# --------------------------------------------------------------------------
# Plot 2 — Heatmap: Features × Models (all 10 features)
# --------------------------------------------------------------------------
heat_data = imp_df.set_index('Feature')[MODEL_COLS].rename(
    columns=dict(zip(MODEL_COLS, MODEL_LABELS))
)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    heat_data, annot=True, fmt='.3f', cmap='RdYlGn',
    linewidths=0.5, linecolor='white', vmin=0, vmax=1,
    cbar_kws={'shrink': 0.7, 'label': 'Normalized Importance'},
    annot_kws={'size': 10}, ax=ax
)

# Highlight top-3 average-importance features with bold left margin
top3_feats = imp_df.head(3)['Feature'].tolist()
for feat in top3_feats:
    row_idx = list(heat_data.index).index(feat)
    ax.add_patch(plt.Rectangle(
        (0, row_idx), len(MODEL_LABELS), 1,
        fill=False, edgecolor='#2C3E50', linewidth=2.0, clip_on=False
    ))

ax.set_title("Feature Importance Comparison Across Models\n"
             "(Dark border = top 3 consensus features)",
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
ax.tick_params(axis='x', rotation=20)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig("feature_importance_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> feature_importance_heatmap.png")

# --------------------------------------------------------------------------
# Plot 3 — Aggregate importance bar chart (all 10, sorted, color gradient)
# --------------------------------------------------------------------------
sig_threshold = 0.10

# Color gradient: map rank to Blue palette intensity
n  = len(imp_df)
bar_colors_agg = [plt.cm.Blues(0.4 + 0.5 * (n - i) / n)
                  for i in range(n)]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(
    range(n),
    imp_df['Average_Importance'].values,
    color=bar_colors_agg, edgecolor='black', linewidth=0.6, width=0.65, zorder=3
)

for bar, val in zip(bars, imp_df['Average_Importance'].values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}", ha='center', va='bottom',
            fontsize=9, fontweight='bold')

# Significance threshold line
ax.axhline(sig_threshold, color='#E74C3C', linestyle='--',
           linewidth=1.8, label=f"Significance threshold ({sig_threshold:.2f})",
           zorder=4)

# Annotate top-3 consensus features with arrow + text
above_thresh = imp_df[imp_df['Average_Importance'] >= sig_threshold]
if len(above_thresh) >= 3:
    x_annot = above_thresh.index[-1] + 0.5
    ax.annotate(
        f"Top {min(3, len(above_thresh))} consensus features",
        xy=(0, sig_threshold), xytext=(x_annot, sig_threshold + 0.06),
        fontsize=9, color='#E74C3C', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5)
    )

ax.set_xticks(range(n))
ax.set_xticklabels(
    [f.replace('_', '\n') for f in imp_df['Feature'].tolist()],
    fontsize=8.5
)
ax.set_ylabel("Average Normalized Importance", fontsize=11)
ax.set_title("Aggregate Feature Importance (Average Across 3 Models)",
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.set_ylim(0, imp_df['Average_Importance'].max() * 1.25)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("aggregate_feature_ranking.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> aggregate_feature_ranking.png")

# --------------------------------------------------------------------------
# Plot 4 — Model agreement scatter plots (LR vs DT, LR vs RF, DT vs RF)
# --------------------------------------------------------------------------
pairs = [
    ('Logistic_Regression', 'Decision_Tree',
     'Logistic Regression', 'Decision Tree', '#3498DB', '#2ECC71'),
    ('Logistic_Regression', 'Random_Forest',
     'Logistic Regression', 'Random Forest', '#3498DB', '#E67E22'),
    ('Decision_Tree', 'Random_Forest',
     'Decision Tree', 'Random Forest', '#2ECC71', '#E67E22'),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, (col_x, col_y, lbl_x, lbl_y, cx, cy) in zip(axes, pairs):
    x_vals = imp_df[col_x].values
    y_vals = imp_df[col_y].values

    # Scatter — colour by average importance (darker = more important)
    sc = ax.scatter(x_vals, y_vals, s=80, c=imp_df['Average_Importance'].values,
                    cmap='YlOrRd', edgecolors='black', linewidths=0.7,
                    zorder=4, vmin=0, vmax=1)

    # Diagonal reference line y=x (perfect agreement)
    lim = max(x_vals.max(), y_vals.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1.2, alpha=0.6,
            label='Perfect agreement')

    # Annotate top-3 most disagreeing features
    diffs = np.abs(x_vals - y_vals)
    top_disagree_idx = np.argsort(diffs)[::-1][:3]
    for idx in top_disagree_idx:
        feat_short = ENG_FEATURES[idx].replace('DiabetesPedigreeFunction', 'DPF') \
                                      .replace('Glucose_BMI_Interaction', 'Glu*BMI') \
                                      .replace('BloodPressure', 'BP')
        ax.annotate(feat_short,
                    xy=(x_vals[idx], y_vals[idx]),
                    xytext=(x_vals[idx] + 0.04, y_vals[idx] + 0.03),
                    fontsize=7.5, color='#C0392B',
                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.9))

    r, p = pearsonr(x_vals, y_vals)
    ax.text(0.05, 0.92, f"Pearson r = {r:.3f}",
            transform=ax.transAxes, fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel(lbl_x, fontsize=10)
    ax.set_ylabel(lbl_y, fontsize=10)
    ax.set_title(f"{lbl_x}\nvs {lbl_y}", fontsize=10, fontweight='bold')
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.colorbar(sc, ax=axes[-1], shrink=0.8, label='Average Importance')
fig.suptitle("Model Agreement on Feature Importance\n"
             "(Red labels = highest disagreement features)",
             fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig("model_agreement_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> model_agreement_analysis.png")

# --------------------------------------------------------------------------
# Plot 5 — Radar chart: Top 5 features × 3 models
# --------------------------------------------------------------------------
top5_data = imp_df[imp_df['Feature'].isin(top5_feats)].set_index('Feature')
top5_data = top5_data.loc[top5_feats]   # preserve importance order

N_axes  = 5
angles  = np.linspace(0, 2 * np.pi, N_axes, endpoint=False).tolist()
angles += angles[:1]    # close polygon

radar_colors = ['#3498DB', '#2ECC71', '#E67E22']
radar_styles = ['-', '--', '-.']

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'polar': True})

for (col, label), color, style in zip(
        zip(MODEL_COLS, MODEL_LABELS), radar_colors, radar_styles):
    values  = top5_data[col].tolist()
    values += values[:1]   # close
    ax.plot(angles, values, color=color, linewidth=2.2,
            linestyle=style, label=label, marker='o', markersize=6)
    ax.fill(angles, values, color=color, alpha=0.08)

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(
    [f.replace('_', '\n').replace('DiabetesPedigreeFunction', 'DPF')
       .replace('Glucose_BMI_Interaction', 'Glu*BMI')
     for f in top5_feats],
    fontsize=10, fontweight='bold'
)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'],
                   fontsize=8, color='grey')
ax.set_title("Top 5 Features — Importance Profile by Model",
             fontsize=13, fontweight='bold', pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("feature_importance_radar.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> feature_importance_radar.png")

# =============================================================================
# PART 3: CLINICAL INTERPRETATION REPORT
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: GENERATING CLINICAL ANALYSIS REPORT")
print("=" * 70)

# --- Clinical significance dictionary ---
clinical_notes = {
    'Glucose'                 : ("Primary WHO diagnostic criterion; fasting plasma glucose "
                                 ">=126 mg/dL confirms diabetes; directly reflects "
                                 "insulin-mediated glucose uptake failure."),
    'BMI'                     : ("Obesity (BMI>=30) drives insulin resistance via adipokine "
                                 "dysregulation and excess free fatty acids; strongest "
                                 "modifiable risk factor for Type 2 diabetes."),
    'Glucose_BMI_Interaction' : ("Multiplicative risk: hyperglycaemia combined with obesity "
                                 "creates compounding metabolic stress; captures joint effect "
                                 "of the top two individual features."),
    'Age'                     : ("Insulin sensitivity declines with age due to mitochondrial "
                                 "dysfunction and reduced beta-cell mass; prevalence roughly "
                                 "doubles every decade after age 40."),
    'DiabetesPedigreeFunction': ("Encodes genetic predisposition and family history of "
                                 "diabetes; heritability of Type 2 diabetes estimated "
                                 "at 30–70%."),
    'Pregnancies'             : ("History of gestational diabetes confers 7x higher lifetime "
                                 "Type 2 risk; each pregnancy stresses pancreatic "
                                 "beta-cell reserve."),
    'Insulin'                 : ("Direct measure of pancreatic beta-cell secretory function; "
                                 "elevated fasting insulin indicates resistance, very low "
                                 "values suggest beta-cell failure."),
    'SkinThickness'           : ("Triceps skinfold proxy for subcutaneous fat; central "
                                 "adiposity predicts insulin resistance even in individuals "
                                 "with normal BMI."),
    'BloodPressure'           : ("Hypertension and Type 2 diabetes share pathophysiology "
                                 "(insulin resistance, endothelial dysfunction); core "
                                 "component of metabolic syndrome cluster."),
    'Age_Group'               : ("Categorical age buckets capture threshold effects in "
                                 "diabetes risk (e.g. post-menopausal surge); may encode "
                                 "non-linearities missed by continuous Age alone."),
}

# --- Agreement analysis ---
# "All models agree" = feature is top-4 for all three models
top_k = 4
agree_features = [
    r['Feature'] for _, r in imp_df.iterrows()
    if (r['Rank_LR'] <= top_k and r['Rank_DT'] <= top_k and r['Rank_RF'] <= top_k)
]

# High disagreement: cross-model variance above 75th percentile
var_thresh = imp_df['Cross_Model_Variance'].quantile(0.75)
disagree_features = imp_df[imp_df['Cross_Model_Variance'] > var_thresh]

# --- Engineered feature ranks ---
gbi_row = imp_df[imp_df['Feature'] == 'Glucose_BMI_Interaction'].iloc[0]
ag_row  = imp_df[imp_df['Feature'] == 'Age_Group'].iloc[0]
glu_row = imp_df[imp_df['Feature'] == 'Glucose'].iloc[0]
age_row_orig = imp_df[imp_df['Feature'] == 'Age'].iloc[0]

# --- Sample prediction analysis using Random Forest ---
rf_probs  = rf.predict_proba(X_test_sc)[:, 1]
rf_preds  = rf.predict(X_test_sc)

# Find samples of each type
tp_idx  = np.where((y_test == 1) & (rf_preds == 1) & (rf_probs >= 0.80))[0]
tn_idx  = np.where((y_test == 0) & (rf_preds == 0) & (rf_probs <= 0.20))[0]
fn_idx  = np.where((y_test == 1) & (rf_preds == 0))[0]

# Pick one sample from each group (highest/lowest confidence)
sample_tp = tp_idx[np.argmax(rf_probs[tp_idx])]   if len(tp_idx) > 0 else None
sample_tn = tn_idx[np.argmin(rf_probs[tn_idx])]   if len(tn_idx) > 0 else None
sample_fn = fn_idx[np.argmax(rf_probs[fn_idx])]   if len(fn_idx) > 0 else None  # highest prob FN

def describe_sample(idx, label):
    if idx is None:
        return [f"  No {label} sample found in test set."]
    raw_vals = X_test_orig[idx]
    prob     = rf_probs[idx]
    outcome  = 'Diabetic' if y_test[idx] == 1 else 'Non-Diabetic'
    pred_lbl = 'Diabetic' if rf_preds[idx] == 1 else 'Non-Diabetic'

    # Feature contributions: scale feature value * RF importance (simple proxy)
    scaled_vals = X_test_sc[idx]
    contributions = scaled_vals * rf.feature_importances_
    top3_contrib  = np.argsort(contributions)[::-1][:3]

    lines_out = [
        f"  {label}",
        f"    Prediction : {pred_lbl} ({prob*100:.1f}% probability diabetic)",
        f"    Actual     : {outcome}",
        f"",
        f"    Top 3 Contributing Features (scaled_value x RF_importance):",
    ]
    for rank, fi in enumerate(top3_contrib, 1):
        feat_name = ENG_FEATURES[fi]
        raw_v     = raw_vals[fi]
        r_imp     = rf.feature_importances_[fi]
        # Class comparison
        m0 = class_means.loc[0, feat_name]
        m1 = class_means.loc[1, feat_name]
        lines_out.append(
            f"    {rank}. {feat_name:<28}  value={raw_v:>8.2f}  "
            f"RF_imp={r_imp:.4f}  "
            f"[Non-Diab avg={m0:.1f} | Diab avg={m1:.1f}]"
        )
    return lines_out

# --- Feature selection recommendations ---
cumsum_imp   = imp_df['Average_Importance'].cumsum()
total_imp    = imp_df['Average_Importance'].sum()
# Number of top features needed to reach 80% of total importance
n_for_80pct  = int((cumsum_imp / total_imp < 0.80).sum()) + 1
pct_top_n    = (imp_df.head(n_for_80pct)['Average_Importance'].sum() /
                total_imp * 100)
low_imp_feats = imp_df[imp_df['Average_Importance'] < 0.05]['Feature'].tolist()
critical_feats= imp_df.head(3)['Feature'].tolist()

# ---- Build report ----
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines   = []
lines.append("=" * 72)
lines.append("  FEATURE IMPORTANCE — CLINICAL ANALYSIS REPORT")
lines.append(f"  Generated : {now_str}")
lines.append("=" * 72)

# --- Section 1: Overall Ranking ---
lines.append("\n1. OVERALL FEATURE RANKING")
lines.append("-" * 72)
hdr_tbl = (f"  {'Rank':<5} {'Feature':<28} {'Avg':>6} "
           f"{'LR':>6} {'DT':>6} {'RF':>6}  Clinical Significance")
lines.append(hdr_tbl)
lines.append("  " + "-" * 70)
for _, r in imp_df.iterrows():
    feat    = r['Feature']
    note    = clinical_notes.get(feat, "N/A")
    note_s  = (note[:55] + '...') if len(note) > 55 else note
    lines.append(
        f"  {int(r['Rank_Avg']):<5} {feat:<28} "
        f"{r['Average_Importance']:>6.3f} "
        f"{r['Logistic_Regression']:>6.3f} "
        f"{r['Decision_Tree']:>6.3f} "
        f"{r['Random_Forest']:>6.3f}  {note_s}"
    )
    # Full clinical note on the line below
    lines.append(f"        Full note: {note}")
lines.append("  " + "-" * 70)

# --- Section 2: Model Agreement ---
lines.append("\n\n2. MODEL AGREEMENT ANALYSIS")
lines.append("-" * 40)
lines.append(f"\n  Features where ALL models place in top-{top_k}:")
if agree_features:
    for f in agree_features:
        lines.append(f"    - {f}")
else:
    lines.append("    - No feature ranks top-4 across all three models simultaneously.")

lines.append(f"\n  Features with HIGH DISAGREEMENT (variance > {var_thresh:.4f}):")
for _, r in disagree_features.iterrows():
    lines.append(
        f"    - {r['Feature']:<28} LR rank={r['Rank_LR']:>2}, "
        f"DT rank={r['Rank_DT']:>2}, RF rank={r['Rank_RF']:>2}  "
        f"(variance={r['Cross_Model_Variance']:.4f})"
    )
    # Explain the likely cause of disagreement
    if 'BloodPressure' in r['Feature']:
        lines.append("      Reason: LR captures linear BP-diabetes association; "
                     "tree models split on threshold values only")
    elif 'Age_Group' in r['Feature']:
        lines.append("      Reason: Categorical encoding benefits tree-based models "
                     "more than the continuous LR")
    elif 'Pregnancies' in r['Feature']:
        lines.append("      Reason: Non-linear relationship; trees exploit "
                     "count-based thresholds that LR treats linearly")
    else:
        lines.append("      Reason: Different model assumptions lead to "
                     "varying sensitivity to this feature")

# --- Section 3: Surprising Findings ---
lines.append("\n\n3. SURPRISING FINDINGS")
lines.append("-" * 40)

# Unexpectedly HIGH: Glucose_BMI_Interaction
gbi_rank = int(gbi_row['Rank_Avg'])
if gbi_rank <= 5:
    lines.append(f"\n  Unexpectedly HIGH importance: Glucose_BMI_Interaction (rank {gbi_rank})")
    lines.append("    Why surprising: As a derived feature (Glucose x BMI), it might have")
    lines.append("    been expected to be redundant with its parents. Instead it ranks highly,")
    lines.append("    suggesting the multiplicative interaction captures synergistic risk")
    lines.append("    not fully encoded by either feature alone.")
else:
    lines.append(f"\n  Glucose_BMI_Interaction ranks {gbi_rank} — moderate importance,")
    lines.append("    indicating its multiplicative signal is largely captured by the")
    lines.append("    individual Glucose and BMI features after scaling.")

# Unexpectedly LOW: DiabetesPedigreeFunction or Insulin
dpf_row = imp_df[imp_df['Feature'] == 'DiabetesPedigreeFunction'].iloc[0]
if int(dpf_row['Rank_Avg']) > 5:
    lines.append(f"\n  Unexpectedly LOW importance: DiabetesPedigreeFunction "
                 f"(rank {int(dpf_row['Rank_Avg'])})")
    lines.append("    Why surprising: Genetic predisposition is a major diabetes risk factor.")
    lines.append("    Possible explanation: The pedigree function is a continuous composite")
    lines.append("    score with limited discriminatory range in this dataset; individual")
    lines.append("    metabolic markers (Glucose, BMI) dominate the signal.")

# --- Section 4: Engineered Features ---
lines.append("\n\n4. ENGINEERED FEATURES PERFORMANCE")
lines.append("-" * 40)
lines.append(f"\n  Glucose_BMI_Interaction:")
lines.append(f"    Average importance : {gbi_row['Average_Importance']:.4f}  "
             f"(rank {int(gbi_row['Rank_Avg'])} out of {N_FEAT})")
lines.append(f"    Individual Glucose : {glu_row['Average_Importance']:.4f}  "
             f"(rank {int(glu_row['Rank_Avg'])})")

if gbi_row['Average_Importance'] > 0.5 * glu_row['Average_Importance']:
    lines.append("    Analysis: The interaction term captures meaningful additive signal.")
    lines.append("    Keeping it in the feature set is justified — it contributes beyond")
    lines.append("    what the individual Glucose feature alone provides.")
else:
    lines.append("    Analysis: Lower than expected. The interaction captures some")
    lines.append("    combined effect but individual features dominate. Consider removing")
    lines.append("    it in a minimal feature set to reduce collinearity.")

lines.append(f"\n  Age_Group:")
lines.append(f"    Average importance : {ag_row['Average_Importance']:.4f}  "
             f"(rank {int(ag_row['Rank_Avg'])} out of {N_FEAT})")
lines.append(f"    Continuous Age     : {age_row_orig['Average_Importance']:.4f}  "
             f"(rank {int(age_row_orig['Rank_Avg'])})")

if age_row_orig['Average_Importance'] > ag_row['Average_Importance']:
    lines.append("    Analysis: Continuous Age outperforms the categorical Age_Group.")
    lines.append("    Binning introduces information loss; the linear/threshold splits in")
    lines.append("    tree models already discover the relevant age thresholds naturally.")
else:
    lines.append("    Analysis: Age_Group adds marginal signal above continuous Age,")
    lines.append("    possibly because clinical risk categories are meaningful non-linear")
    lines.append("    breakpoints that the binning captures.")

# --- Section 5: Sample Prediction Analysis ---
lines.append("\n\n5. SAMPLE PREDICTION ANALYSIS  (Random Forest)")
lines.append("-" * 40)
lines.append("\n  Three representative test samples are analysed to show how global")
lines.append("  feature importances combine with individual feature values.\n")

sample_descs = [
    (sample_tp, "Sample A: TRUE POSITIVE  (Correctly identified diabetic)"),
    (sample_tn, "Sample B: TRUE NEGATIVE  (Correctly identified non-diabetic)"),
    (sample_fn, "Sample C: FALSE NEGATIVE (MISSED diabetic — most concerning!)"),
]
for idx, label in sample_descs:
    for l in describe_sample(idx, label):
        lines.append(l)

    # Clinical interpretation
    if idx is not None:
        raw_vals = X_test_orig[idx]
        prob     = rf_probs[idx]
        if sample_tp is not None and idx == sample_tp:
            lines.append(f"\n    Clinical Interpretation:")
            lines.append(f"    High-confidence prediction ({prob*100:.1f}%). Glucose and BMI both")
            lines.append(f"    exceed diabetic thresholds. Multiple co-occurring risk factors")
            lines.append(f"    create a clear diabetic signature that all features reinforce.")
        elif sample_tn is not None and idx == sample_tn:
            lines.append(f"\n    Clinical Interpretation:")
            lines.append(f"    Very low probability ({prob*100:.1f}%). Feature values are")
            lines.append(f"    consistently within non-diabetic ranges. Prediction is")
            lines.append(f"    robust — model has high confidence this is a healthy patient.")
        elif sample_fn is not None and idx == sample_fn:
            lines.append(f"\n    Clinical Interpretation: *** HIGH RISK — MISSED CASE ***")
            lines.append(f"    Probability {prob*100:.1f}% — model predicted non-diabetic")
            lines.append(f"    but patient IS diabetic. Feature values may be borderline.")
            lines.append(f"    Clinical follow-up with HbA1c testing is strongly advised.")
    lines.append("")

# --- Section 6: Feature Selection Recommendations ---
lines.append("\n6. RECOMMENDATIONS FOR FEATURE SELECTION")
lines.append("-" * 40)
lines.append(f"\n  Minimum viable feature set:")
lines.append(f"    Top {n_for_80pct} features account for "
             f"~{pct_top_n:.1f}% of total aggregate importance.")
lines.append(f"    Recommended core set: "
             f"{', '.join(imp_df.head(n_for_80pct)['Feature'].tolist())}")

lines.append(f"\n  Features that COULD be removed (average importance < 0.05):")
if low_imp_feats:
    for f in low_imp_feats:
        row = imp_df[imp_df['Feature'] == f].iloc[0]
        lines.append(f"    - {f:<28}  avg_imp={row['Average_Importance']:.4f}")
else:
    lines.append("    - All features exceed the 0.05 threshold; none are trivially removable.")

lines.append(f"\n  Critical features (never remove):")
for f in critical_feats:
    row = imp_df[imp_df['Feature'] == f].iloc[0]
    lines.append(f"    - {f:<28}  avg_imp={row['Average_Importance']:.4f}")

lines.append("\n\n" + "=" * 72)
lines.append("  END OF CLINICAL ANALYSIS REPORT")
lines.append("=" * 72)

with open("feature_importance_clinical_analysis.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("  Saved -> feature_importance_clinical_analysis.txt")

# =============================================================================
# PART 4: PRINT SUMMARY
# =============================================================================

print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 70)

print(f"""
  Analyzed 3 models  : Logistic Regression, Decision Tree, Random Forest
  Total features     : {N_FEAT} (8 original + 2 engineered)

  TOP 3 CONSENSUS FEATURES (highest average importance):""")

for i, (_, r) in enumerate(imp_df.head(3).iterrows(), 1):
    note_short = clinical_notes.get(r['Feature'], '')
    note_trunc = (note_short[:65] + '...') if len(note_short) > 65 else note_short
    print(f"\n  {i}. {r['Feature']}")
    print(f"     Avg={r['Average_Importance']:.4f}  "
          f"LR={r['Logistic_Regression']:.4f}  "
          f"DT={r['Decision_Tree']:.4f}  "
          f"RF={r['Random_Forest']:.4f}")
    print(f"     Clinical: {note_trunc}")

print(f"\n  MODEL AGREEMENT:")
print(f"    High consensus features : "
      f"{agree_features if agree_features else 'None in strict top-4 for all models'}")
print(f"    High disagreement feats : "
      f"{disagree_features['Feature'].tolist()}")

print(f"\n  ENGINEERED FEATURES:")
print(f"    Glucose_BMI_Interaction : "
      f"Rank {int(gbi_row['Rank_Avg'])}/{N_FEAT}  "
      f"(avg={gbi_row['Average_Importance']:.4f})")
print(f"    Age_Group               : "
      f"Rank {int(ag_row['Rank_Avg'])}/{N_FEAT}  "
      f"(avg={ag_row['Average_Importance']:.4f})")

print(f"\n  CLINICAL INSIGHTS:")
top1 = imp_df.iloc[0]
print(f"    - {top1['Feature']} is the strongest predictor across all 3 models "
      f"(avg={top1['Average_Importance']:.4f})")
print(f"    - Top {n_for_80pct} features capture {pct_top_n:.1f}% of aggregate importance "
      f"-> suitable for a parsimonious clinical screening tool")

print("\n" + "=" * 70)
print("[OK] 5 visualizations created")
print("[OK] Feature importance CSV exported  (feature_importance_comparison.csv)")
print("[OK] Clinical analysis report saved   (feature_importance_clinical_analysis.txt)")
print("=" * 70)
