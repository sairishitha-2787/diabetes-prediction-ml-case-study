# =============================================================================
# PIMA DIABETES DATASET - ADVANCED EXPLORATORY DATA ANALYSIS
# =============================================================================
# Libraries: pandas, numpy, matplotlib, seaborn, scipy.stats
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# PART 1: DATA LOADING
# =============================================================================

print("=" * 70)
print("PART 1: DATA LOADING")
print("=" * 70)

# Define column names as per PIMA dataset specification
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Load dataset directly from GitHub repository
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(url, names=column_names)

# --- Shape ---
print(f"\nDataset Shape: {df.shape}")
print(f"  Rows   : {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")

# --- Info ---
print("\n--- df.info() ---")
df.info()

# --- Descriptive Statistics ---
print("\n--- df.describe() ---")
print(df.describe().round(3).to_string())

# --- First 5 Rows ---
print("\n--- First 5 Rows ---")
print(df.head().to_string())

# --- Class Distribution ---
print("\n--- Class Distribution ---")
class_counts = df['Outcome'].value_counts().sort_index()
total = len(df)
for label, count in class_counts.items():
    name = "Non-Diabetic" if label == 0 else "Diabetic"
    pct = count / total * 100
    print(f"  Outcome={label} ({name:>12s}): {count:4d} samples  ({pct:.2f}%)")

# =============================================================================
# PART 2: MISSING VALUE ANALYSIS (Zero-encoded missing values)
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: MISSING VALUE ANALYSIS (Zero = physiologically impossible)")
print("=" * 70)

# Features where 0 is biologically impossible → treated as missing
zero_check_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

zero_counts = {}
zero_pcts   = {}

print(f"\n{'Feature':<30} {'Zero Count':>12} {'Percentage':>12}")
print("-" * 56)
for feat in zero_check_features:
    cnt = (df[feat] == 0).sum()
    pct = cnt / total * 100
    zero_counts[feat] = cnt
    zero_pcts[feat]   = pct
    print(f"  {feat:<28} {cnt:>12d} {pct:>11.2f}%")

# --- Bar Chart: missing_values_analysis.png ---
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#E74C3C' if zero_counts[f] > 0 else '#2ECC71' for f in zero_check_features]
bars = ax.bar(zero_check_features, [zero_counts[f] for f in zero_check_features],
              color=colors, edgecolor='black', linewidth=0.8, width=0.6)

# Annotate each bar with count and percentage
for bar, feat in zip(bars, zero_check_features):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
            f"{zero_counts[feat]}\n({zero_pcts[feat]:.1f}%)",
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title("Zero-Value (Missing) Count per Feature", fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("Feature", fontsize=12)
ax.set_ylabel("Number of Zero Values", fontsize=12)
ax.set_ylim(0, max(zero_counts.values()) * 1.25)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("missing_values_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("\n  Saved -> missing_values_analysis.png")

# =============================================================================
# PART 3: STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: STATISTICAL ANALYSIS")
print("=" * 70)

feature_cols = [c for c in df.columns if c != 'Outcome']
diabetic     = df[df['Outcome'] == 1]
non_diabetic = df[df['Outcome'] == 0]

# --- Pearson Correlation & Independent T-tests ---
stat_rows = []
for feat in feature_cols:
    r, p_r = stats.pearsonr(df[feat], df['Outcome'])
    t, p_t = stats.ttest_ind(diabetic[feat], non_diabetic[feat])
    stat_rows.append({
        'Feature'    : feat,
        'Pearson_r'  : r,
        'r_p_value'  : p_r,
        't_statistic': t,
        't_p_value'  : p_t
    })

stat_df = pd.DataFrame(stat_rows)
stat_df['abs_r'] = stat_df['Pearson_r'].abs()
stat_df = stat_df.sort_values('abs_r', ascending=False).reset_index(drop=True)

# --- Print Correlation Table ---
print("\n--- Pearson Correlation with Outcome (sorted by |r|) ---")
print(f"\n  {'#':<4} {'Feature':<30} {'Pearson r':>10} {'p-value':>12}")
print("  " + "-" * 58)
for i, row in stat_df.iterrows():
    sig = "**" if row['r_p_value'] < 0.05 else "  "
    print(f"  {i+1:<4} {row['Feature']:<30} {row['Pearson_r']:>10.4f} {row['r_p_value']:>12.6f} {sig}")

# --- Print T-test Results ---
print("\n--- Independent T-test: Diabetic vs Non-Diabetic ---")
print(f"\n  {'Feature':<30} {'t-statistic':>13} {'p-value':>14} {'Significant':>13}")
print("  " + "-" * 72)
for _, row in stat_df.iterrows():
    sig_str = "YES (p<0.05)" if row['t_p_value'] < 0.05 else "No"
    print(f"  {row['Feature']:<30} {row['t_statistic']:>13.4f} {row['t_p_value']:>14.6f} {sig_str:>13}")

# --- Significant Features ---
sig_features = stat_df[stat_df['t_p_value'] < 0.05]['Feature'].tolist()
print(f"\n  Features with significant difference between classes: {sig_features}")

# =============================================================================
# PART 4: VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: CREATING VISUALIZATIONS")
print("=" * 70)

# Shared color palette
COLOR_0   = '#3498DB'   # blue  → Non-Diabetic
COLOR_1   = '#E74C3C'   # red   → Diabetic
ALPHA_VAL = 0.6

# --------------------------------------------------------------------------
# 1. Feature Distributions  (feature_distributions.png)
# --------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, feat in enumerate(feature_cols):
    ax = axes[idx]
    ax.hist(non_diabetic[feat], bins=30, alpha=ALPHA_VAL, color=COLOR_0,
            label='Non-Diabetic (0)', edgecolor='white', linewidth=0.4)
    ax.hist(diabetic[feat],     bins=30, alpha=ALPHA_VAL, color=COLOR_1,
            label='Diabetic (1)',     edgecolor='white', linewidth=0.4)
    ax.set_title(feat, fontsize=13, fontweight='bold')
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

# Hide the unused 9th subplot
axes[8].set_visible(False)
fig.suptitle("Feature Distributions by Diabetes Outcome", fontsize=16,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> feature_distributions.png")

# --------------------------------------------------------------------------
# 2. Correlation Heatmap  (correlation_heatmap.png)
# --------------------------------------------------------------------------
sns.set_style("white")
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(corr_matrix, dtype=bool)  # no masking – show full matrix
hm = sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
    square=True, linewidths=0.5, linecolor='white',
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    ax=ax
)
ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> correlation_heatmap.png")

# --------------------------------------------------------------------------
# 3. Box Plots – Outlier Detection  (box_plots_outliers.png)
# --------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, feat in enumerate(feature_cols):
    ax = axes[idx]
    bp = ax.boxplot(
        df[feat], vert=True, patch_artist=True,
        flierprops=dict(marker='o', markerfacecolor='#E74C3C',
                        markersize=4, alpha=0.6, linestyle='none'),
        medianprops=dict(color='green', linewidth=2.5),
        boxprops=dict(facecolor='#AED6F1', alpha=0.8),
        whiskerprops=dict(linestyle='--', color='gray'),
        capprops=dict(color='black', linewidth=1.5)
    )
    ax.set_title(feat, fontsize=13, fontweight='bold')
    ax.set_ylabel("Value", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[8].set_visible(False)
fig.suptitle("Box Plots – Distribution & Outliers per Feature",
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("box_plots_outliers.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> box_plots_outliers.png")

# --------------------------------------------------------------------------
# 4. Violin Plots  (violin_plots.png)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()

violin_palette = [COLOR_0, COLOR_1]   # index 0 -> Outcome=0, index 1 -> Outcome=1
for idx, feat in enumerate(feature_cols):
    ax = axes[idx]
    sns.violinplot(
        data=df, x='Outcome', y=feat, palette=violin_palette,
        inner='box', linewidth=1.2, ax=ax
    )
    ax.set_title(feat, fontsize=12, fontweight='bold')
    ax.set_xlabel("Outcome (0=No, 1=Yes)", fontsize=9)
    ax.set_ylabel(feat, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

fig.suptitle("Violin Plots – Feature Distributions by Outcome",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("violin_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> violin_plots.png")

# --------------------------------------------------------------------------
# 5. Pair Plot – Top 4 Correlated Features  (pair_plot_key_features.png)
# --------------------------------------------------------------------------
top4_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Outcome']
pair_df = df[top4_features].copy()

pair_palette = {0: COLOR_0, 1: COLOR_1}
g = sns.pairplot(
    pair_df, hue='Outcome', palette=pair_palette,
    diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20},
    diag_kws={'fill': True, 'alpha': 0.5}
)
g.fig.suptitle("Pair Plot – Top 4 Correlated Features", y=1.02,
               fontsize=15, fontweight='bold')

# Rename legend labels
handles = g._legend_data
new_labels = ['Non-Diabetic (0)', 'Diabetic (1)']
g._legend.set_title("Outcome")
for text, label in zip(g._legend.get_texts(), new_labels):
    text.set_text(label)

g.fig.set_size_inches(12, 12)
plt.savefig("pair_plot_key_features.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> pair_plot_key_features.png")

# --------------------------------------------------------------------------
# 6. Class Distribution  (class_distribution.png)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
bar_colors = ['#87CEEB', '#FA8072']   # skyblue, salmon
bars = ax.bar(['Non-Diabetic (0)', 'Diabetic (1)'],
              [class_counts[0], class_counts[1]],
              color=bar_colors, edgecolor='black', linewidth=0.8, width=0.5)

for bar, (label, count) in zip(bars, class_counts.items()):
    pct = count / total * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{count}\n({pct:.1f}%)", ha='center', va='bottom',
            fontsize=13, fontweight='bold')

# Horizontal line at 50 %
ax.axhline(total * 0.5, color='gray', linestyle='--',
           linewidth=1.5, label='50% mark')
ax.set_title("Class Distribution: Diabetic vs Non-Diabetic",
             fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_ylim(0, class_counts.max() * 1.2)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> class_distribution.png")

# --------------------------------------------------------------------------
# 7. Glucose vs BMI Scatter  (glucose_bmi_scatter.png)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

for outcome, color, label in [(0, COLOR_0, 'Non-Diabetic (0)'),
                               (1, COLOR_1, 'Diabetic (1)')]:
    subset = df[df['Outcome'] == outcome]
    ax.scatter(subset['Glucose'], subset['BMI'],
               c=color, alpha=0.5, s=40, label=label, edgecolors='none')

    # Regression line
    m, b, r, p, _ = stats.linregress(subset['Glucose'], subset['BMI'])
    x_line = np.linspace(subset['Glucose'].min(), subset['Glucose'].max(), 200)
    ax.plot(x_line, m * x_line + b, color=color, linestyle='--',
            linewidth=2, label=f"  r={r:.3f} (Outcome={outcome})")

ax.set_xlabel("Glucose", fontsize=12)
ax.set_ylabel("BMI", fontsize=12)
ax.set_title("Glucose vs BMI – Colored by Diabetes Outcome", fontsize=14,
             fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("glucose_bmi_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> glucose_bmi_scatter.png")

# --------------------------------------------------------------------------
# 8. Age Distribution by Outcome  (age_distribution_by_outcome.png)
# --------------------------------------------------------------------------
mean_age_0 = non_diabetic['Age'].mean()
mean_age_1 = diabetic['Age'].mean()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(non_diabetic['Age'], bins=30, alpha=ALPHA_VAL, color=COLOR_0,
        label='Non-Diabetic (0)', edgecolor='white')
ax.hist(diabetic['Age'],     bins=30, alpha=ALPHA_VAL, color=COLOR_1,
        label='Diabetic (1)',     edgecolor='white')

# Vertical lines for means
ax.axvline(mean_age_0, color=COLOR_0, linestyle='--', linewidth=2,
           label=f'Mean Non-Diabetic = {mean_age_0:.1f}')
ax.axvline(mean_age_1, color=COLOR_1, linestyle='--', linewidth=2,
           label=f'Mean Diabetic     = {mean_age_1:.1f}')

# Text annotations
ymax = ax.get_ylim()[1]
ax.text(mean_age_0 + 0.8, ymax * 0.88,
        f"μ = {mean_age_0:.1f}", color=COLOR_0, fontsize=11, fontweight='bold')
ax.text(mean_age_1 + 0.8, ymax * 0.80,
        f"μ = {mean_age_1:.1f}", color=COLOR_1, fontsize=11, fontweight='bold')

ax.set_title("Age Distribution by Diabetes Outcome", fontsize=14, fontweight='bold')
ax.set_xlabel("Age (years)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("age_distribution_by_outcome.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> age_distribution_by_outcome.png")

# --------------------------------------------------------------------------
# 9. Feature Correlation with Outcome  (feature_correlation_with_outcome.png)
# --------------------------------------------------------------------------
corr_outcome = stat_df.set_index('Feature')['Pearson_r'].sort_values(key=abs, ascending=True)
bar_colors_corr = ['#27AE60' if v >= 0 else '#E74C3C' for v in corr_outcome.values]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(corr_outcome.index, corr_outcome.values,
               color=bar_colors_corr, edgecolor='black', linewidth=0.6, height=0.6)

# Value labels
for bar, val in zip(bars, corr_outcome.values):
    x_pos = val + (0.005 if val >= 0 else -0.005)
    ha    = 'left' if val >= 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va='center', ha=ha, fontsize=10, fontweight='bold')

# Vertical line at 0
ax.axvline(0, color='black', linewidth=1.2, linestyle='-')
ax.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=12)
ax.set_title("Feature Correlation with Diabetes Outcome",
             fontsize=14, fontweight='bold', pad=12)
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend patches
pos_patch = mpatches.Patch(color='#27AE60', label='Positive correlation')
neg_patch = mpatches.Patch(color='#E74C3C', label='Negative correlation')
ax.legend(handles=[pos_patch, neg_patch], fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig("feature_correlation_with_outcome.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved -> feature_correlation_with_outcome.png")

# =============================================================================
# PART 5: KEY INSIGHTS SUMMARY
# =============================================================================

# Precompute values used in summary
pct_non_diab = class_counts[0] / total * 100
pct_diab     = class_counts[1] / total * 100

top3 = stat_df.head(3)

mean_gluc_0 = non_diabetic['Glucose'].mean()
mean_gluc_1 = diabetic['Glucose'].mean()
mean_bmi_0  = non_diabetic['BMI'].mean()
mean_bmi_1  = diabetic['BMI'].mean()

print("\n")
print("=" * 70)
print("EXPLORATORY DATA ANALYSIS - KEY INSIGHTS")
print("=" * 70)

print(f"""
1. Dataset Overview:
   - Total samples : {total}
   - Features      : {len(feature_cols)}
   - Target distribution: {pct_non_diab:.1f}% non-diabetic, {pct_diab:.1f}% diabetic
""")

print("2. Missing Values (Zero-encoded):")
for feat in zero_check_features:
    print(f"   - {feat:<25}: {zero_counts[feat]:>3d} ({zero_pcts[feat]:.1f}%)")

print(f"""
3. Top 3 Most Correlated Features with Outcome:""")
for rank, (_, row) in enumerate(top3.iterrows(), start=1):
    print(f"   {rank}. {row['Feature']:<28}: r = {row['Pearson_r']:.4f},  "
          f"p = {row['r_p_value']:.2e}")

print(f"""
4. Statistically Significant Features (p < 0.05):""")
for feat in sig_features:
    print(f"   - {feat}")

print(f"""
5. Key Observations:
   - Mean Glucose : Non-diabetic = {mean_gluc_0:.2f},  Diabetic = {mean_gluc_1:.2f},  Difference = {mean_gluc_1 - mean_gluc_0:.2f}
   - Mean BMI     : Non-diabetic = {mean_bmi_0:.2f},  Diabetic = {mean_bmi_1:.2f},  Difference = {mean_bmi_1 - mean_bmi_0:.2f}
   - Mean Age     : Non-diabetic = {mean_age_0:.2f},  Diabetic = {mean_age_1:.2f},  Difference = {mean_age_1 - mean_age_0:.2f}
""")

print("=" * 70)
print("[OK] EDA Complete: 9 visualizations created")
print("=" * 70)
