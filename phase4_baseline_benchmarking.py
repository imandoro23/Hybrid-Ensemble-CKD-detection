"""
PHASE 4 — BASELINE MODEL BENCHMARKING
PhD Research: Hybrid ML for Early CKD Detection
Dataset: UCI CKD (UCI_CKD_Processed.csv)

Models benchmarked (9):
  1. Logistic Regression
  2. K-Nearest Neighbors
  3. Decision Tree
  4. Naive Bayes
  5. Support Vector Machine (RBF)
  6. Random Forest
  7. Gradient Boosting
  8. HistGradient Boosting  (XGBoost-equivalent)
  9. AdaBoost

Evaluation:
  - Stratified 5-fold cross-validation (training set)
  - Held-out 20% test set (never touched during training)
  - Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PRC,
             Brier Score, MCC, Sensitivity, Specificity, PPV, NPV

Output files saved to same folder as this script:
  UCI_Baseline_Results.csv
  plots/baseline_auc_comparison.png
  plots/baseline_metrics_heatmap.png
  plots/confusion_matrices.png
  plots/roc_curves.png
  plots/precision_recall_curves.png
  plots/calibration_curves.png
  plots/cv_auc_stability.png

Usage:
  python phase4_baseline_benchmarking.py
  python phase4_baseline_benchmarking.py my_data.csv
"""

import os, sys, warnings, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, 'UCI_CKD_Processed.csv')
PLOT_DIR   = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',    'axes.facecolor':  '#F4F6FA',
    'axes.edgecolor':   '#CCCCCC',  'axes.labelcolor': '#404040',
    'xtick.color':      '#404040',  'ytick.color':     '#404040',
    'font.family': 'DejaVu Sans',   'font.size': 11,
    'axes.titlesize': 13,           'axes.titleweight': 'bold',
    'figure.dpi': 150
})
NAVY   = '#1F3864';  BLUE  = '#2E75B6';  TEAL  = '#028090'
AMBER  = '#C65911';  GREEN = '#375623';  RED   = '#C0392B'
PURPLE = '#4A235A';  GREY  = '#888888'

MODEL_COLORS = [NAVY, BLUE, TEAL, GREEN, AMBER, RED, PURPLE, '#E67E22', '#1ABC9C']

# ── Imports ────────────────────────────────────────────────────────────────────
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split, cross_val_predict)
from sklearn.preprocessing   import StandardScaler, label_binarize
from sklearn.pipeline        import Pipeline
from sklearn.calibration     import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.svm             import SVC
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     HistGradientBoostingClassifier,
                                     AdaBoostClassifier)
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     roc_auc_score, average_precision_score,
                                     brier_score_loss, confusion_matrix,
                                     matthews_corrcoef, roc_curve,
                                     precision_recall_curve, classification_report)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PHASE 4 — BASELINE MODEL BENCHMARKING")
print("="*65)
print(f"\nLoading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
drop_cols = [c for c in ['CKD_eGFR','CKD_ACR','KDIGO_Stage',
                          'Anaemia_Flag','Comorbidity_Score'] if c in df.columns]
df = df.drop(columns=drop_cols)

# ── Use the 21 consensus features from Phase 3 ────────────────────────────────
PHASE3_FEATURES = [
    'Specific_Gravity', 'Albumin', 'Haemoglobin', 'eGFR_MDRD',
    'Packed_Cell_Volume', 'Diabetes_Mellitus', 'Hypertension',
    'Blood_Glucose_Random', 'Serum_Creatinine', 'rbcc', 'BUN_Cr_Ratio',
    'Appetite', 'Blood_Urea', 'Red_Blood_Cells', 'Sugar',
    'Blood_Pressure', 'Sodium', 'Pedal_Edema', 'wbcc', 'Pus_Cell', 'Age'
]
# Use only those present in dataset
features = [f for f in PHASE3_FEATURES if f in df.columns]
missing  = [f for f in PHASE3_FEATURES if f not in df.columns]
if missing:
    print(f"  Note: {len(missing)} Phase 3 features not found, skipping: {missing}")

X = df[features]
y = df['CKD'].astype(int)

print(f"Shape:    {X.shape[0]} rows x {X.shape[1]} features")
print(f"Features: {features}")
print(f"CKD: {y.sum()} ({y.mean()*100:.1f}%)  |  Non-CKD: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\nTrain: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# ══════════════════════════════════════════════════════════════════════════════
# DEFINE MODELS
# ══════════════════════════════════════════════════════════════════════════════
models = {
    'Logistic Regression': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight='balanced', random_state=42))
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ]),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10, min_samples_leaf=3,
        class_weight='balanced', random_state=42),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', CalibratedClassifierCV(
            SVC(kernel='rbf', C=1.0, gamma='scale',
                class_weight='balanced', random_state=42), cv=3))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_features='sqrt', min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        subsample=0.8, random_state=42),
    'HistGradient Boosting': HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, random_state=42),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=200, learning_rate=0.5, random_state=42),
}

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, cv):
    t0 = time.time()

    # 5-fold CV on training set
    cv_res = cross_validate(
        model, X_tr, y_tr, cv=cv,
        scoring=['accuracy','f1','roc_auc','average_precision'],
        return_train_score=False, n_jobs=-1)

    # Fit on full training set
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    # Confusion matrix values
    TN, FP, FN, TP = confusion_matrix(y_te, y_pred).ravel()
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv  = TP / (TP + FP) if (TP + FP) > 0 else 0
    npv  = TN / (TN + FN) if (TN + FN) > 0 else 0

    return {
        'Model':          name,
        # Cross-validation metrics
        'CV_AUC':         round(cv_res['test_roc_auc'].mean(),    4),
        'CV_AUC_std':     round(cv_res['test_roc_auc'].std(),     4),
        'CV_Accuracy':    round(cv_res['test_accuracy'].mean()*100, 2),
        'CV_Acc_std':     round(cv_res['test_accuracy'].std()*100,  2),
        'CV_F1':          round(cv_res['test_f1'].mean()*100,     2),
        'CV_F1_std':      round(cv_res['test_f1'].std()*100,      2),
        'CV_AUPRC':       round(cv_res['test_average_precision'].mean(), 4),
        # Test set metrics
        'Test_Accuracy':  round(accuracy_score(y_te, y_pred)*100, 2),
        'Test_Precision': round(precision_score(y_te, y_pred)*100,2),
        'Test_Recall':    round(recall_score(y_te, y_pred)*100,   2),
        'Test_F1':        round(f1_score(y_te, y_pred)*100,       2),
        'Test_AUC':       round(roc_auc_score(y_te, y_proba),     4),
        'Test_AUPRC':     round(average_precision_score(y_te, y_proba), 4),
        'Brier_Score':    round(brier_score_loss(y_te, y_proba),  4),
        'MCC':            round(matthews_corrcoef(y_te, y_pred),  4),
        'Sensitivity':    round(sens*100, 2),
        'Specificity':    round(spec*100, 2),
        'PPV':            round(ppv*100,  2),
        'NPV':            round(npv*100,  2),
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
        'Time_s':         round(time.time() - t0, 2),
    }

# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("RUNNING ALL 9 BASELINE MODELS")
print(f"{'─'*65}")
print(f"\n{'Model':<28} {'CV-AUC':>8} {'±':>6} {'Test-Acc%':>10} {'Test-AUC':>9} {'Brier':>7} {'Time':>6}")
print("─"*77)

results      = []
fitted_models = {}

for name, model in models.items():
    r = evaluate_model(name, model, X_train, y_train, X_test, y_test, cv5)
    results.append(r)
    fitted_models[name] = model
    print(f"  {name:<26} {r['CV_AUC']:>8.4f} {r['CV_AUC_std']:>6.4f} "
          f"{r['Test_Accuracy']:>10.2f} {r['Test_AUC']:>9.4f} "
          f"{r['Brier_Score']:>7.4f} {r['Time_s']:>5.1f}s")

# Sort by CV AUC descending
results_df = pd.DataFrame(results).sort_values('CV_AUC', ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# PRINT FULL RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("FULL RESULTS TABLE")
print(f"{'='*65}")
print(f"\n{'Model':<28} {'CV-AUC':>8} {'Test-Acc%':>10} {'Test-AUC':>9} "
      f"{'AUPRC':>7} {'Brier':>7} {'Sens%':>7} {'Spec%':>7} {'MCC':>7}")
print("─"*92)
for _, r in results_df.iterrows():
    star = ' ★' if r['Model'] == results_df.iloc[0]['Model'] else ''
    print(f"  {r['Model']:<26} {r['CV_AUC']:>8.4f} {r['Test_Accuracy']:>10.2f} "
          f"{r['Test_AUC']:>9.4f} {r['Test_AUPRC']:>7.4f} {r['Brier_Score']:>7.4f} "
          f"{r['Sensitivity']:>7.2f} {r['Specificity']:>7.2f} {r['MCC']:>7.4f}{star}")

print(f"\n{'─'*65}")
print("CONFUSION MATRICES — Test Set")
print(f"{'─'*65}")
for _, r in results_df.iterrows():
    print(f"\n  {r['Model']}")
    print(f"    {'':15} Pred CKD   Pred Non-CKD")
    print(f"    Actual CKD:    {r['TP']:5d}        {r['FN']:5d}    Sensitivity={r['Sensitivity']}%")
    print(f"    Actual Non-CKD:{r['FP']:5d}        {r['TN']:5d}    Specificity={r['Specificity']}%")
    print(f"    PPV={r['PPV']}%   NPV={r['NPV']}%   F1={r['Test_F1']}%")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CSV
# ══════════════════════════════════════════════════════════════════════════════
out_csv = os.path.join(SCRIPT_DIR, 'UCI_Baseline_Results.csv')
results_df.to_csv(out_csv, index=False)
print(f"\n  Saved: UCI_Baseline_Results.csv")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("SAVING PLOTS")
print(f"{'─'*65}")

model_names  = results_df['Model'].tolist()
n_models     = len(model_names)
colors_sorted= [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(n_models)]

# ── Plot 1: AUC Comparison (CV and Test) ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Baseline Model Performance — AUC Comparison', fontsize=14, fontweight='bold')

# CV AUC with error bars
cv_aucs = results_df['CV_AUC'].values
cv_stds = results_df['CV_AUC_std'].values
y_pos   = np.arange(n_models)

bars = axes[0].barh(y_pos, cv_aucs, xerr=cv_stds, color=colors_sorted,
                    error_kw={'ecolor':'black','capsize':4,'linewidth':1.2},
                    edgecolor='white', linewidth=0.5, height=0.6)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(model_names, fontsize=10)
axes[0].set_xlabel('5-Fold CV AUC-ROC')
axes[0].set_title('Cross-Validation AUC (mean ± std)')
axes[0].set_xlim(min(cv_aucs) - 0.005, 1.002)
for i, (auc, std) in enumerate(zip(cv_aucs, cv_stds)):
    axes[0].text(auc + std + 0.0003, i, f'{auc:.4f}', va='center', fontsize=9)

# Test AUC
test_aucs = results_df['Test_AUC'].values
bars2 = axes[1].barh(y_pos, test_aucs, color=colors_sorted,
                     edgecolor='white', linewidth=0.5, height=0.6)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(model_names, fontsize=10)
axes[1].set_xlabel('Test Set AUC-ROC')
axes[1].set_title('Test Set AUC')
axes[1].set_xlim(min(test_aucs) - 0.005, 1.002)
for i, auc in enumerate(test_aucs):
    axes[1].text(auc + 0.0003, i, f'{auc:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'baseline_auc_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/baseline_auc_comparison.png")

# ── Plot 2: Multi-metric heatmap ──────────────────────────────────────────────
metric_cols = ['CV_AUC', 'Test_Accuracy', 'Test_AUC', 'Test_AUPRC',
               'Brier_Score', 'Sensitivity', 'Specificity', 'MCC']
metric_labels = ['CV AUC', 'Test Acc%', 'Test AUC', 'AUC-PRC',
                 'Brier ↓', 'Sensitivity%', 'Specificity%', 'MCC']

heatmap_data = results_df[metric_cols].values.copy()
# Normalise each column to 0-1 for visual (Brier: invert so lower=better)
heatmap_norm = np.zeros_like(heatmap_data)
for j in range(heatmap_data.shape[1]):
    col = heatmap_data[:, j].astype(float)
    if metric_cols[j] == 'Brier_Score':
        # Invert: lower Brier is better
        col_norm = 1 - (col - col.min()) / (col.max() - col.min() + 1e-10)
    else:
        col_norm = (col - col.min()) / (col.max() - col.min() + 1e-10)
    heatmap_norm[:, j] = col_norm

fig, ax = plt.subplots(figsize=(13, 6))
im = ax.imshow(heatmap_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(metric_labels)))
ax.set_xticklabels(metric_labels, rotation=30, ha='right', fontsize=10)
ax.set_yticks(range(n_models))
ax.set_yticklabels(model_names, fontsize=10)
ax.set_title('Baseline Model Performance Heatmap\n(green=better, red=worse per column)', pad=10)

# Annotate cells with actual values
for i in range(n_models):
    for j in range(len(metric_cols)):
        val = heatmap_data[i, j]
        text_color = 'black' if 0.3 < heatmap_norm[i,j] < 0.7 else 'white'
        fmt = '.4f' if metric_cols[j] == 'Brier_Score' else '.2f'
        ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                fontsize=8.5, color=text_color, fontweight='bold')

plt.colorbar(im, ax=ax, shrink=0.8, label='Normalised score (higher=better)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'baseline_metrics_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/baseline_metrics_heatmap.png")

# ── Plot 3: Confusion matrices grid ───────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(13, 11))
fig.suptitle('Confusion Matrices — All 9 Models (Test Set)', fontsize=13, fontweight='bold', y=1.01)
axes_flat = axes.flatten()

for idx, (_, r) in enumerate(results_df.iterrows()):
    ax = axes_flat[idx]
    cm = np.array([[r['TN'], r['FP']], [r['FN'], r['TP']]])
    im = ax.imshow(cm, interpolation='nearest',
                   cmap=plt.cm.Blues, vmin=0, vmax=max(r['TP'],r['TN'])+5)
    ax.set_title(r['Model'], fontsize=9, fontweight='bold', pad=4)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Non-CKD','CKD'], fontsize=8)
    ax.set_yticklabels(['Non-CKD','CKD'], fontsize=8)
    ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('Actual', fontsize=8)
    for ii in range(2):
        for jj in range(2):
            val = cm[ii, jj]
            color = 'white' if val > cm.max()/2 else 'black'
            ax.text(jj, ii, str(val), ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color)
    # AUC and F1 subtitle
    ax.set_xlabel(f'AUC={r["Test_AUC"]:.4f}  F1={r["Test_F1"]:.1f}%',
                  fontsize=8, color=NAVY)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrices.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/confusion_matrices.png")

# ── Plot 4: ROC Curves ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.50)')

for idx, (name, model) in enumerate(fitted_models.items()):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    color = MODEL_COLORS[idx % len(MODEL_COLORS)]
    lw = 2.5 if name == 'Random Forest' else 1.5
    ax.plot(fpr, tpr, color=color, lw=lw, label=f'{name} (AUC={auc:.4f})')

ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_title('ROC Curves — All Baseline Models')
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/roc_curves.png")

# ── Plot 5: Precision-Recall Curves ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
baseline_precision = y_test.sum() / len(y_test)
ax.axhline(y=baseline_precision, color='k', linestyle='--', lw=1,
           label=f'Random (AP={baseline_precision:.2f})')

for idx, (name, model) in enumerate(fitted_models.items()):
    y_proba = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    color = MODEL_COLORS[idx % len(MODEL_COLORS)]
    lw = 2.5 if name == 'Random Forest' else 1.5
    ax.plot(rec, prec, color=color, lw=lw, label=f'{name} (AP={ap:.4f})')

ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision (PPV)')
ax.set_title('Precision-Recall Curves — All Baseline Models')
ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'precision_recall_curves.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/precision_recall_curves.png")

# ── Plot 6: Calibration Curves (Reliability Diagrams) ─────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
fig.suptitle('Calibration Curves (Reliability Diagrams)\nPerfect calibration = diagonal line',
             fontsize=12, fontweight='bold', y=1.01)
axes_flat = axes.flatten()

for idx, (name, model) in enumerate(fitted_models.items()):
    ax = axes_flat[idx]
    y_proba = model.predict_proba(X_test)[:, 1]
    brier   = brier_score_loss(y_test, y_proba)

    try:
        frac_pos, mean_pred = calibration_curve(y_test, y_proba,
                                                 n_bins=5, strategy='quantile')
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot([0,1],[0,1], 'k--', lw=1, label='Perfect')
        ax.plot(mean_pred, frac_pos, 'o-', color=color, lw=2,
                markersize=6, label=f'Model')
        ax.fill_between(mean_pred, frac_pos, mean_pred,
                        alpha=0.12, color=color)
    except Exception:
        ax.plot([0,1],[0,1],'k--',lw=1)
        ax.text(0.5,0.5,'Insufficient data',ha='center',va='center',
                transform=ax.transAxes, fontsize=9, color=GREY)

    ax.set_title(name, fontsize=9, fontweight='bold', pad=3)
    ax.set_xlabel(f'Mean predicted probability\nBrier={brier:.4f}', fontsize=8)
    ax.set_ylabel('Fraction of positives', fontsize=8)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'calibration_curves.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/calibration_curves.png")

# ── Plot 7: CV AUC Stability (mean ± std visualisation) ───────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(n_models)
cv_aucs_s = results_df['CV_AUC'].values
cv_stds_s = results_df['CV_AUC_std'].values

# Background stability zones
ax.axvspan(0.99,  1.001, alpha=0.08, color=GREEN,  label='Excellent (≥0.99)')
ax.axvspan(0.97,  0.99,  alpha=0.08, color=BLUE,   label='Very Good (0.97–0.99)')
ax.axvspan(0.95,  0.97,  alpha=0.08, color=AMBER,  label='Good (0.95–0.97)')

# Plot error bars
for i, (auc, std, color) in enumerate(zip(cv_aucs_s, cv_stds_s, colors_sorted)):
    ax.barh(i, auc, height=0.5, color=color, alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax.errorbar(auc, i, xerr=std, fmt='none',
                ecolor='black', capsize=5, linewidth=1.5)
    # Stability label
    stability = 'Very Stable' if std < 0.002 else ('Stable' if std < 0.004 else 'Unstable')
    ax.text(0.9825, i, f'±{std:.4f}  {stability}',
            va='center', fontsize=8.5, color=NAVY)

ax.set_yticks(y_pos)
ax.set_yticklabels(results_df['Model'].tolist(), fontsize=10)
ax.set_xlabel('5-Fold CV AUC-ROC')
ax.set_title('CV AUC Stability — Mean ± Standard Deviation across 5 Folds\n'
             'Lower std = more stable model')
ax.set_xlim(0.982, 1.003)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'cv_auc_stability.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/cv_auc_stability.png")

# ── Plot 8: Key metrics bar comparison ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Key Metric Comparison Across All Baseline Models',
             fontsize=13, fontweight='bold')

metrics_to_plot = [
    ('Test_F1',      'F1-Score (%)',       'Higher is better'),
    ('Brier_Score',  'Brier Score',        'Lower is better ↓'),
    ('Sensitivity',  'Sensitivity (%)',    'Higher is better (fewer missed CKD)'),
    ('Specificity',  'Specificity (%)',    'Higher is better (fewer false alarms)'),
]

for ax, (col, ylabel, note) in zip(axes.flatten(), metrics_to_plot):
    vals   = results_df[col].values
    x_pos  = np.arange(n_models)
    # Highlight best
    if col == 'Brier_Score':
        best_idx = np.argmin(vals)
    else:
        best_idx = np.argmax(vals)
    bar_colors = [GREEN if i == best_idx else colors_sorted[i] for i in range(n_models)]
    bars = ax.bar(x_pos, vals, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['Model'].tolist(),
                       rotation=35, ha='right', fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'{ylabel}\n{note}', fontsize=10)
    for i, val in enumerate(vals):
        ax.text(i, val + 0.001*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold' if i==best_idx else 'normal',
                color=GREEN if i==best_idx else NAVY)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'baseline_key_metrics.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/baseline_key_metrics.png")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
best      = results_df.iloc[0]
worst_brier = results_df.loc[results_df['Brier_Score'].idxmax()]
best_brier  = results_df.loc[results_df['Brier_Score'].idxmin()]
best_sens   = results_df.loc[results_df['Sensitivity'].idxmax()]
best_spec   = results_df.loc[results_df['Specificity'].idxmax()]

print(f"\n{'='*65}")
print("PHASE 4 COMPLETE — SUMMARY")
print("="*65)
print(f"\n  Total models evaluated: {n_models}")
print(f"\n  Best CV-AUC:          {best['Model']:<28} {best['CV_AUC']:.4f} ± {best['CV_AUC_std']:.4f}")
print(f"  Best Test Accuracy:   {results_df.iloc[results_df['Test_Accuracy'].values.argmax()]['Model']:<28} {results_df['Test_Accuracy'].max():.2f}%")
print(f"  Best Sensitivity:     {best_sens['Model']:<28} {best_sens['Sensitivity']:.2f}%")
print(f"  Best Specificity:     {best_spec['Model']:<28} {best_spec['Specificity']:.2f}%")
print(f"  Best Calibration:     {best_brier['Model']:<28} Brier={best_brier['Brier_Score']:.4f}")
print(f"  Worst Calibration:    {worst_brier['Model']:<28} Brier={worst_brier['Brier_Score']:.4f}")

print(f"\n  BENCHMARK TO BEAT (for hybrid ensemble):")
print(f"    CV-AUC  > {best['CV_AUC']:.4f}")
print(f"    CV-std  < {results_df['CV_AUC_std'].min():.4f}  (lower variance = more stable)")
print(f"    Brier   < {best_brier['Brier_Score']:.4f}  (better calibration)")
print(f"    Sens    > {best_sens['Sensitivity']:.2f}%  (fewer missed CKD patients)")

print(f"\n  Key finding — near-perfect performance confirms UCI separability")
print(f"  concern from systematic review. External validation essential.")

print(f"\n  Output files:")
print(f"    UCI_Baseline_Results.csv")
print(f"    plots/baseline_auc_comparison.png")
print(f"    plots/baseline_metrics_heatmap.png")
print(f"    plots/confusion_matrices.png")
print(f"    plots/roc_curves.png")
print(f"    plots/precision_recall_curves.png")
print(f"    plots/calibration_curves.png")
print(f"    plots/cv_auc_stability.png")
print(f"    plots/baseline_key_metrics.png")
print(f"\n{'='*65}")
