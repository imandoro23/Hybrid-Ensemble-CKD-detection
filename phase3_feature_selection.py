"""
PHASE 3 — FEATURE SELECTION PIPELINE
PhD Research: Hybrid ML for Early CKD Detection
Dataset: UCI CKD (UCI_CKD_Processed.csv)

Methods:
  Stage 1 — Statistical filters  (MI, Spearman, ANOVA)
  Stage 2 — Boruta algorithm      (shadow feature comparison)
  Stage 3 — RFECV + LASSO         (wrapper + embedded)
  Stage 4 — SHAP consensus        (TreeExplainer — real shap library)
  Stage 5 — Final consensus table + plots saved as PNG

Output files saved to same folder as this script:
  UCI_Feature_Rankings.csv
  UCI_Consensus_Final.csv
  plots/shap_beeswarm.png
  plots/shap_bar.png
  plots/shap_dependence_creatinine.png
  plots/feature_importance_comparison.png
  plots/boruta_hitrates.png
  plots/rfecv_curve.png

Usage:
  python phase3_feature_selection.py
  python phase3_feature_selection.py my_data.csv   (custom CSV path)
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — saves files instead of showing windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, 'UCI_CKD_Processed.csv')
PLOT_DIR    = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#F4F6FA',
    'axes.edgecolor': '#CCCCCC', 'axes.labelcolor': '#404040',
    'xtick.color': '#404040', 'ytick.color': '#404040',
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'figure.dpi': 150
})
NAVY   = '#1F3864'
BLUE   = '#2E75B6'
TEAL   = '#028090'
AMBER  = '#C65911'
GREEN  = '#375623'
PURPLE = '#4A235A'
RED    = '#C0392B'

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PHASE 3 — FEATURE SELECTION PIPELINE")
print("="*65)
print(f"\nLoading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Drop helper columns — keep only modelling features
drop_cols = [c for c in ['CKD_eGFR','CKD_ACR','KDIGO_Stage','Anaemia_Flag','Comorbidity_Score'] if c in df.columns]
df = df.drop(columns=drop_cols)

X = df.drop(columns=['CKD'])
y = df['CKD'].astype(int)
feature_names = list(X.columns)
n_feat = len(feature_names)

print(f"Shape: {X.shape[0]} rows x {X.shape[1]} features")
print(f"CKD: {y.sum()} ({y.mean()*100:.1f}%)  |  Non-CKD: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")
print(f"Features: {feature_names}\n")

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, f_oneway, binomtest

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — STATISTICAL FILTERS
# ══════════════════════════════════════════════════════════════════════════════
print("─"*65)
print("STAGE 1 — STATISTICAL FILTERS")
print("─"*65)

mi    = mutual_info_classif(X, y, random_state=42)
stat  = {}
for i, f in enumerate(feature_names):
    rho, _ = spearmanr(X[f], y)
    F, p   = f_oneway(X[f][y==0], X[f][y==1])
    stat[f] = {
        'MI':        round(float(mi[i]), 4),
        'Spearman':  round(float(abs(rho)), 4),
        'ANOVA_F':   round(float(F), 2),
        'ANOVA_p':   round(float(p), 6),
        'Stat_sig':  int(p < 0.05)
    }

stat_df = pd.DataFrame(stat).T
stat_df['stat_rank'] = (
    stat_df['MI'].rank(ascending=False) +
    stat_df['Spearman'].rank(ascending=False) +
    stat_df['ANOVA_F'].rank(ascending=False)) / 3
stat_df = stat_df.sort_values('stat_rank')

print(f"\n{'Feature':<28} {'MI':>8} {'Spearman':>10} {'ANOVA-F':>10} {'sig':>6}")
print("─"*65)
for f, row in stat_df.iterrows():
    sig = '***' if row['ANOVA_p']<0.001 else ('**' if row['ANOVA_p']<0.01 else ('*' if row['ANOVA_p']<0.05 else 'ns'))
    print(f"  {f:<26} {row['MI']:>8.4f} {row['Spearman']:>10.4f} {row['ANOVA_F']:>10.2f} {sig:>6}")

stat_selected = stat_df[stat_df['ANOVA_p'] < 0.05].index.tolist()
print(f"\nStatistically significant (p<0.05): {len(stat_selected)}/{n_feat}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — BORUTA
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STAGE 2 — BORUTA ALGORITHM (100 iterations)")
print("─"*65)

Xarr = X.values
rng  = np.random.RandomState(42)
hits = np.zeros(n_feat, dtype=int)

for it in range(100):
    X_shadow = np.column_stack([rng.permutation(Xarr[:, j]) for j in range(n_feat)])
    X_aug    = np.hstack([Xarr, X_shadow])
    rf = RandomForestClassifier(
        n_estimators=100, max_features='sqrt',
        random_state=int(rng.randint(0, 9999)),
        n_jobs=-1, class_weight='balanced')
    rf.fit(X_aug, y)
    imps       = rf.feature_importances_
    shadow_max = imps[n_feat:].max()
    hits      += (imps[:n_feat] > shadow_max).astype(int)
    if (it+1) % 25 == 0:
        print(f"  Iteration {it+1}/100 — "
              f"accepted: {sum(hits/(it+1)>0.5)}, "
              f"rejected: {sum(hits/(it+1)<0.1)}")

# Binomial test
boruta_status = {}
hit_rates     = {}
for i, f in enumerate(feature_names):
    h = int(hits[i])
    hit_rates[f] = round(h / 100, 2)
    p_acc = binomtest(h, 100, 0.5, alternative='greater').pvalue
    p_rej = binomtest(h, 100, 0.5, alternative='less').pvalue
    if p_acc < 0.05:
        boruta_status[f] = 'Accepted'
    elif p_rej < 0.05:
        boruta_status[f] = 'Rejected'
    else:
        boruta_status[f] = 'Tentative'

accepted  = [f for f in feature_names if boruta_status[f] == 'Accepted']
tentative = [f for f in feature_names if boruta_status[f] == 'Tentative']
rejected  = [f for f in feature_names if boruta_status[f] == 'Rejected']

print(f"\n  Accepted  ({len(accepted)}):  {accepted}")
print(f"  Tentative ({len(tentative)}): {tentative}")
print(f"  Rejected  ({len(rejected)}):  {rejected}")

print(f"\n{'Feature':<28} {'Hit Rate':>9}  {'Status'}")
print("─"*55)
for f, hr in sorted(hit_rates.items(), key=lambda x: -x[1]):
    status = boruta_status[f]
    bar    = '█' * int(hr * 25)
    print(f"  {f:<26} {hr:>8.2f}  {bar}  [{status}]")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3A — RFECV
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STAGE 3A — RFECV (Random Forest, 5-fold CV)")
print("─"*65)

rf_est = RandomForestClassifier(
    n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
rfecv = RFECV(
    estimator=rf_est, step=1, cv=cv5,
    scoring='roc_auc', min_features_to_select=5, n_jobs=-1)
rfecv.fit(X, y)

rfecv_selected = [feature_names[i] for i in range(n_feat) if rfecv.support_[i]]
rfecv_ranking  = {feature_names[i]: int(rfecv.ranking_[i]) for i in range(n_feat)}
best_cv_auc    = max(rfecv.cv_results_['mean_test_score'])

print(f"\n  Optimal features: {rfecv.n_features_}")
print(f"  Best CV AUC:      {best_cv_auc:.4f}")
print(f"  Selected: {rfecv_selected}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3B — LASSO
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STAGE 3B — LASSO Regularisation")
print("─"*65)

scaler     = StandardScaler()
X_scaled   = scaler.fit_transform(X)
lasso      = LassoCV(cv=5, random_state=42, max_iter=5000, n_alphas=100)
lasso.fit(X_scaled, y)
lasso_coefs    = {feature_names[i]: abs(float(lasso.coef_[i])) for i in range(n_feat)}
lasso_selected = [f for f, c in lasso_coefs.items() if c > 0]

print(f"\n  Optimal alpha:   {lasso.alpha_:.6f}")
print(f"  Non-zero coefs:  {len(lasso_selected)}")
print(f"\n{'Feature':<28} {'|coef|':>10}  Bar")
print("─"*55)
for f, c in sorted(lasso_coefs.items(), key=lambda x: -x[1]):
    if c > 0:
        bar = '█' * min(int(c * 60), 30)
        print(f"  {f:<26} {c:>10.4f}  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — SHAP (Real TreeExplainer)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STAGE 4 — SHAP (TreeExplainer — real shap library)")
print("─"*65)

try:
    import shap
    print(f"  shap version: {shap.__version__}")

    rf_shap = RandomForestClassifier(
        n_estimators=300, max_features='sqrt',
        min_samples_leaf=2, class_weight='balanced',
        random_state=42, n_jobs=-1)
    rf_shap.fit(X_train, y_train)
    auc_shap = roc_auc_score(y_test, rf_shap.predict_proba(X_test)[:, 1])
    print(f"  RF AUC (for SHAP): {auc_shap:.4f}")

    explainer   = shap.TreeExplainer(rf_shap)
    shap_values = explainer.shap_values(X)

    # Handle all shap output formats (list, 2D, or 3D array)
    if isinstance(shap_values, list):
        sv = shap_values[1]                    # old format: list [class0, class1]
    elif hasattr(shap_values, 'shape') and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]              # new format: (samples, features, classes)
    else:
        sv = shap_values                       # already 2D

    # Ensure 2D before computing mean
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs_shap = np.abs(sv).mean(axis=0)    # shape: (n_features,)
    shap_df = pd.DataFrame({
        'Feature':   feature_names,
        'SHAP_Mean': mean_abs_shap
    }).sort_values('SHAP_Mean', ascending=False)
    shap_df['SHAP_Rank'] = range(1, len(shap_df) + 1)

    print(f"\n{'Rank':<5} {'Feature':<28} {'Mean |SHAP|':>12}  Bar")
    print("─"*60)
    for _, row in shap_df.iterrows():
        bar = '█' * min(int(row['SHAP_Mean'] * 150), 30)
        print(f"  {int(row['SHAP_Rank']):<4} {row['Feature']:<26} {row['SHAP_Mean']:>12.4f}  {bar}")

    # ── SHAP PLOTS ────────────────────────────────────────────────────────────
    print("\n  Saving SHAP plots...")

    # 1. Beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X, plot_type='dot',
                      show=False, max_display=15, color_bar=True)
    plt.title('SHAP Beeswarm — Feature Impact on CKD Prediction', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: plots/shap_beeswarm.png")

    # 2. Bar chart
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(sv, X, plot_type='bar',
                      show=False, max_display=15, color=NAVY)
    plt.title('SHAP Bar Chart — Mean Absolute Feature Importance', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: plots/shap_bar.png")

    # 3. Dependence plot — Serum Creatinine
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'Serum_Creatinine' in feature_names:
        shap.dependence_plot('Serum_Creatinine', sv, X,
                             interaction_index='eGFR_MDRD' if 'eGFR_MDRD' in feature_names else None,
                             ax=ax, show=False)
        ax.set_title('SHAP Dependence — Serum Creatinine (coloured by eGFR)', pad=10)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'shap_dependence_creatinine.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: plots/shap_dependence_creatinine.png")

    SHAP_AVAILABLE = True

except ImportError:
    print("\n  shap not installed — run: pip install shap")
    print("  Falling back to RF Gini importance for Stage 4")
    rf_fb = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_fb.fit(X, y)
    shap_df = pd.DataFrame({'Feature': feature_names,
                             'SHAP_Mean': rf_fb.feature_importances_}).sort_values('SHAP_Mean', ascending=False)
    shap_df['SHAP_Rank'] = range(1, len(shap_df)+1)
    sv = None
    SHAP_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — CONSENSUS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STAGE 5 — CONSENSUS ANALYSIS")
print("─"*65)

consensus = pd.DataFrame(index=feature_names)
consensus['MI_rank']      = stat_df['MI'].rank(ascending=False).astype(int)
consensus['Spearman_rank']= stat_df['Spearman'].rank(ascending=False).astype(int)
consensus['ANOVA_sig']    = stat_df['ANOVA_p'].apply(lambda p: 1 if p<0.05 else 0)
consensus['Boruta']       = [boruta_status[f] for f in feature_names]
consensus['Boruta_HR']    = [hit_rates[f] for f in feature_names]
consensus['RFECV']        = [1 if f in rfecv_selected else 0 for f in feature_names]
consensus['RFECV_rank']   = [rfecv_ranking[f] for f in feature_names]
consensus['LASSO']        = [1 if f in lasso_selected else 0 for f in feature_names]
consensus['LASSO_coef']   = [lasso_coefs[f] for f in feature_names]
consensus['SHAP_rank']    = [int(shap_df[shap_df['Feature']==f]['SHAP_Rank'].values[0]) for f in feature_names]
consensus['SHAP_mean']    = [float(shap_df[shap_df['Feature']==f]['SHAP_Mean'].values[0]) for f in feature_names]

# N methods score
consensus['N_methods'] = (
    (consensus['Boruta'] == 'Accepted').astype(int) +
    consensus['RFECV'] +
    consensus['LASSO'] +
    (consensus['SHAP_rank'] <= 10).astype(int)
)

# Tier assignment
def assign_tier(row):
    if row['N_methods'] == 4:
        return 'TIER 1 — Core'
    elif row['N_methods'] >= 3 or (row['Boruta']=='Accepted' and row['ANOVA_sig']==1):
        return 'TIER 2 — Strong'
    elif row['N_methods'] >= 2 or row['Boruta']=='Tentative':
        return 'TIER 3 — Moderate'
    else:
        return 'TIER 4 — Weak/Reject'

consensus['Tier'] = consensus.apply(assign_tier, axis=1)
consensus = consensus.sort_values(['N_methods','SHAP_rank'], ascending=[False, True])

print(f"\n{'Feature':<28} {'Stat':>5} {'Boruta':>9} {'RFECV':>6} {'LASSO':>6} {'SHAPrk':>7} {'N':>4}  Tier")
print("─"*82)
for f, row in consensus.iterrows():
    bs = row['Boruta'][:3].upper()
    print(f"  {f:<26} {row['ANOVA_sig']:>5} {bs:>9} {row['RFECV']:>6} {row['LASSO']:>6} "
          f"{row['SHAP_rank']:>7} {row['N_methods']:>4}  {row['Tier']}")

# Summary by tier
print(f"\n{'─'*65}")
tier_order = ['TIER 1 — Core','TIER 2 — Strong','TIER 3 — Moderate','TIER 4 — Weak/Reject']
for t in tier_order:
    feats = consensus[consensus['Tier'] == t].index.tolist()
    if feats:
        print(f"\n{t} ({len(feats)}):")
        for f in feats:
            print(f"  ✓ {f}")

final_features = consensus[consensus['Tier'].isin(['TIER 1 — Core','TIER 2 — Strong'])].index.tolist()
print(f"\nFINAL RECOMMENDED FEATURE SET ({len(final_features)} features):")
for f in final_features:
    print(f"  → {f}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("SAVING ADDITIONAL PLOTS")
print("─"*65)

# ── Plot 1: Boruta hit rates ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sorted_feats = sorted(hit_rates.items(), key=lambda x: -x[1])
names  = [x[0] for x in sorted_feats]
values = [x[1] for x in sorted_feats]
colors = [GREEN if boruta_status[n]=='Accepted' else
          AMBER if boruta_status[n]=='Tentative' else RED for n in names]
bars = ax.barh(names, values, color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(0.5, color=NAVY, linestyle='--', linewidth=1.2, label='Decision threshold (0.5)')
ax.set_xlabel('Hit Rate (proportion of iterations beating shadow max)')
ax.set_title('Boruta Algorithm — Feature Hit Rates')
ax.set_xlim(0, 1.05)
patches = [mpatches.Patch(color=GREEN, label='Accepted'),
           mpatches.Patch(color=AMBER, label='Tentative'),
           mpatches.Patch(color=RED,   label='Rejected')]
ax.legend(handles=patches, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'boruta_hitrates.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/boruta_hitrates.png")

# ── Plot 2: RFECV curve ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
n_features_range = range(1, len(rfecv.cv_results_['mean_test_score']) + 1)
mean_scores = rfecv.cv_results_['mean_test_score']
std_scores  = rfecv.cv_results_['std_test_score']
ax.plot(n_features_range, mean_scores, color=BLUE, linewidth=2, marker='o', markersize=4)
ax.fill_between(n_features_range,
                mean_scores - std_scores,
                mean_scores + std_scores,
                alpha=0.15, color=BLUE)
ax.axvline(rfecv.n_features_, color=TEAL, linestyle='--', linewidth=1.5,
           label=f'Optimal: {rfecv.n_features_} features (AUC={best_cv_auc:.4f})')
ax.set_xlabel('Number of Features')
ax.set_ylabel('CV AUC-ROC')
ax.set_title('RFECV — Cross-Validated AUC vs Number of Features')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rfecv_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/rfecv_curve.png")

# ── Plot 3: Feature importance comparison across all methods ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.suptitle('Feature Importance — Multi-Method Comparison', fontsize=14, fontweight='bold', y=1.01)

# MI
mi_sorted = stat_df.sort_values('MI', ascending=True).tail(15)
axes[0].barh(mi_sorted.index, mi_sorted['MI'], color=NAVY)
axes[0].set_title('Mutual Information')
axes[0].set_xlabel('MI Score')

# Boruta
bor_sorted = pd.Series(hit_rates).sort_values().tail(15)
bar_colors = [GREEN if boruta_status[f]=='Accepted' else
              AMBER if boruta_status[f]=='Tentative' else RED for f in bor_sorted.index]
axes[1].barh(bor_sorted.index, bor_sorted.values, color=bar_colors)
axes[1].set_title('Boruta Hit Rates')
axes[1].set_xlabel('Hit Rate')
axes[1].axvline(0.5, color='black', linestyle='--', linewidth=1)

# SHAP
shap_sorted = shap_df.sort_values('SHAP_Mean', ascending=True).tail(15)
axes[2].barh(shap_sorted['Feature'], shap_sorted['SHAP_Mean'], color=TEAL)
axes[2].set_title('SHAP Mean |Value|')
axes[2].set_xlabel('Mean |SHAP|')

for ax in axes:
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'feature_importance_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/feature_importance_comparison.png")

# ── Plot 4: Consensus tier chart ──────────────────────────────────────────────
tier_colors = {
    'TIER 1 — Core':        NAVY,
    'TIER 2 — Strong':      BLUE,
    'TIER 3 — Moderate':    TEAL,
    'TIER 4 — Weak/Reject': '#AAAAAA'
}
fig, ax = plt.subplots(figsize=(10, 7))
cons_plot = consensus.sort_values('N_methods', ascending=True)
bar_colors = [tier_colors[t] for t in cons_plot['Tier']]
ax.barh(cons_plot.index, cons_plot['N_methods'],
        color=bar_colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Number of Methods Selecting Feature (out of 4)')
ax.set_title('Feature Selection Consensus — Methods Agreeing per Feature')
ax.set_xlim(0, 4.5)
for i, (feat, row) in enumerate(cons_plot.iterrows()):
    ax.text(row['N_methods'] + 0.05, i, row['Tier'].split('—')[0].strip(),
            va='center', fontsize=8, color='#444444')
patches = [mpatches.Patch(color=c, label=t) for t, c in tier_colors.items()]
ax.legend(handles=patches, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'consensus_tiers.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/consensus_tiers.png")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CSV RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("SAVING CSV RESULTS")
print("─"*65)

# Full rankings CSV
rankings = pd.DataFrame({
    'Feature':       feature_names,
    'MI_Score':      [stat[f]['MI'] for f in feature_names],
    'Spearman_rho':  [stat[f]['Spearman'] for f in feature_names],
    'ANOVA_F':       [stat[f]['ANOVA_F'] for f in feature_names],
    'ANOVA_p':       [stat[f]['ANOVA_p'] for f in feature_names],
    'Boruta_Status': [boruta_status[f] for f in feature_names],
    'Boruta_HitRate':[hit_rates[f] for f in feature_names],
    'RFECV_Selected':[1 if f in rfecv_selected else 0 for f in feature_names],
    'RFECV_Rank':    [rfecv_ranking[f] for f in feature_names],
    'LASSO_coef':    [lasso_coefs[f] for f in feature_names],
    'SHAP_rank':     [int(shap_df[shap_df['Feature']==f]['SHAP_Rank'].values[0]) for f in feature_names],
    'SHAP_mean':     [float(shap_df[shap_df['Feature']==f]['SHAP_Mean'].values[0]) for f in feature_names],
}).sort_values('SHAP_rank')

out1 = os.path.join(SCRIPT_DIR, 'UCI_Feature_Rankings.csv')
out2 = os.path.join(SCRIPT_DIR, 'UCI_Consensus_Final.csv')
rankings.to_csv(out1, index=False)
consensus.to_csv(out2)
print(f"\n  Saved: UCI_Feature_Rankings.csv")
print(f"  Saved: UCI_Consensus_Final.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PHASE 3 COMPLETE — SUMMARY")
print("="*65)
print(f"\n  Total features:       {n_feat}")
print(f"  Statistically sig:    {len(stat_selected)}")
print(f"  Boruta accepted:      {len(accepted)}")
print(f"  RFECV optimal:        {rfecv.n_features_}  (AUC {best_cv_auc:.4f})")
print(f"  LASSO non-zero:       {len(lasso_selected)}")
print(f"  Tier 1 core:          {len(consensus[consensus['Tier']=='TIER 1 — Core'])}")
print(f"  Tier 2 strong:        {len(consensus[consensus['Tier']=='TIER 2 — Strong'])}")
print(f"  Final model features: {len(final_features)}")
print(f"\n  Output files:")
print(f"    UCI_Feature_Rankings.csv")
print(f"    UCI_Consensus_Final.csv")
print(f"    plots/shap_beeswarm.png")
print(f"    plots/shap_bar.png")
print(f"    plots/shap_dependence_creatinine.png")
print(f"    plots/boruta_hitrates.png")
print(f"    plots/rfecv_curve.png")
print(f"    plots/feature_importance_comparison.png")
print(f"    plots/consensus_tiers.png")
print(f"\n{'='*65}")
