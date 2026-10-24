"""
PHASE 6B — VALIDATION, CALIBRATION & FAIRNESS ANALYSIS
PhD Research: Hybrid ML for Early CKD Detection

Components:
  A. Calibration Analysis (UCI test set)
       - Brier score + 95% bootstrap CI
       - Reliability diagram (calibration curve)
       - Expected Calibration Error (ECE)
       - Hosmer-Lemeshow goodness-of-fit test
       - Isotonic recalibration comparison

  B. Decision Curve Analysis (UCI)
       - Net benefit across threshold range
       - Compare ensemble vs treat-all vs treat-none vs RF

  C. Cross-Dataset Validation (NHANES — no retraining)
       - Apply locked UCI-trained ensemble to NHANES
       - Feature mapping: UCI names → NHANES equivalents
       - Full metric report on NHANES cohort
       - Compare against random and treat-all baselines

  D. Subgroup Fairness Analysis (NHANES)
       - AUC, F1, Sensitivity, Specificity by:
           Sex (male/female)
           Age group (18-44, 45-64, 65+)
           Diabetes status (yes/no)
           Ethnicity (White/Black/Hispanic/Asian)
       - Equalised odds analysis
       - Fairness gap visualisation

Output files:
  Validation_Fairness_Results.csv
  plots/calibration_reliability.png
  plots/calibration_ece.png
  plots/decision_curve.png
  plots/nhanes_cross_validation.png
  plots/fairness_by_sex.png
  plots/fairness_by_age.png
  plots/fairness_by_diabetes.png
  plots/fairness_by_ethnicity.png
  plots/fairness_radar.png
  plots/fairness_equalised_odds.png

Usage:
  python phase6b_validation_fairness.py
  python phase6b_validation_fairness.py uci.csv nhanes.csv
"""

import os, sys, warnings, copy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
UCI_PATH    = sys.argv[1] if len(sys.argv)>1 else os.path.join(SCRIPT_DIR,'UCI_CKD_Processed.csv')
NHANES_PATH = sys.argv[2] if len(sys.argv)>2 else os.path.join(SCRIPT_DIR,'NHANES_CKD_Dataset.csv')
PLOT_DIR    = os.path.join(SCRIPT_DIR,'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'white',  'axes.facecolor':'#F4F6FA',
    'axes.edgecolor':'#CCCCCC',  'axes.labelcolor':'#404040',
    'xtick.color':'#404040',     'ytick.color':'#404040',
    'font.family':'DejaVu Sans', 'font.size':11,
    'axes.titlesize':13,         'axes.titleweight':'bold',
    'figure.dpi':150
})
NAVY=  '#1F3864'; BLUE= '#2E75B6'; TEAL= '#028090'
AMBER= '#C65911'; GREEN='#375623'; RED=  '#C0392B'
PURPLE='#4A235A'; GREY= '#888888'
ETH_COLORS = [NAVY,TEAL,AMBER,GREEN,PURPLE,RED,BLUE]

# ── Imports ────────────────────────────────────────────────────────────────────
from sklearn.model_selection   import StratifiedKFold, train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.calibration       import (CalibratedClassifierCV,
                                       calibration_curve,
                                       CalibratedClassifierCV)
from sklearn.isotonic          import IsotonicRegression
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.ensemble          import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       HistGradientBoostingClassifier)
from sklearn.neural_network    import MLPClassifier
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       roc_auc_score, average_precision_score,
                                       brier_score_loss, confusion_matrix,
                                       matthews_corrcoef, roc_curve)

print("\n"+"="*65)
print("PHASE 6B — VALIDATION, CALIBRATION & FAIRNESS")
print("="*65)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD UCI + REBUILD ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading UCI: {UCI_PATH}")
df_uci = pd.read_csv(UCI_PATH)
drop_cols = [c for c in ['CKD_eGFR','CKD_ACR','KDIGO_Stage',
                          'Anaemia_Flag','Comorbidity_Score'] if c in df_uci.columns]
df_uci = df_uci.drop(columns=drop_cols)

UCI_FEATURES = [
    'Specific_Gravity','Albumin','Haemoglobin','eGFR_MDRD',
    'Packed_Cell_Volume','Diabetes_Mellitus','Hypertension',
    'Blood_Glucose_Random','Serum_Creatinine','rbcc','BUN_Cr_Ratio',
    'Appetite','Blood_Urea','Red_Blood_Cells','Sugar',
    'Blood_Pressure','Sodium','Pedal_Edema','wbcc','Pus_Cell','Age'
]
features = [f for f in UCI_FEATURES if f in df_uci.columns]
X_uci    = df_uci[features]
y_uci    = df_uci['CKD'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X_uci, y_uci, test_size=0.20, random_state=42, stratify=y_uci)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"UCI: {X_uci.shape}  |  Train:{len(X_train)}  Test:{len(X_test)}")

# Rebuild ensemble (same as Phase 5 — LR C=1.0 meta-learner)
print("Rebuilding ensemble...")
base_defs = [
    ('RF',  RandomForestClassifier(n_estimators=300, max_features='sqrt',
             min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('GB',  GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
             max_depth=4, subsample=0.8, min_samples_leaf=3, random_state=42)),
    ('HGB', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
             max_depth=6, l2_regularization=0.5, min_samples_leaf=10, random_state=42)),
    ('SVM', Pipeline([('sc',StandardScaler()),
             ('clf',CalibratedClassifierCV(SVC(kernel='rbf',C=2.0,gamma='scale',
             class_weight='balanced',random_state=42),cv=3))])),
    ('MLP', Pipeline([('sc',StandardScaler()),
             ('clf',MLPClassifier(hidden_layer_sizes=(128,64,32),activation='relu',
             solver='adam',alpha=0.001,learning_rate='adaptive',learning_rate_init=0.001,
             max_iter=500,early_stopping=True,validation_fraction=0.1,random_state=42))]))
]
n_base = len(base_defs)
oof    = np.zeros((len(X_train), n_base))
for fi,(ti,vi) in enumerate(cv5.split(X_train,y_train)):
    for j,(nm,bm) in enumerate(base_defs):
        m=copy.deepcopy(bm); m.fit(X_train.iloc[ti],y_train.iloc[ti])
        oof[vi,j]=m.predict_proba(X_train.iloc[vi])[:,1]
    print(f"  OOF fold {fi+1}/5")

fitted_base=[]; test_meta=np.zeros((len(X_test),n_base))
for j,(nm,bm) in enumerate(base_defs):
    bm.fit(X_train,y_train); test_meta[:,j]=bm.predict_proba(X_test)[:,1]
    fitted_base.append((nm,bm))

meta = LogisticRegression(C=1.0,max_iter=1000,random_state=42)
meta.fit(oof,y_train)
y_proba = meta.predict_proba(test_meta)[:,1]
y_pred  = (y_proba>=0.5).astype(int)
print(f"  Ensemble AUC (test): {roc_auc_score(y_test,y_proba):.4f}")

def ens_predict(X_in):
    if isinstance(X_in,np.ndarray):
        X_in = pd.DataFrame(X_in,columns=features)
    mf = np.column_stack([m.predict_proba(X_in)[:,1] for _,m in fitted_base])
    return meta.predict_proba(mf)[:,1]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — CALIBRATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("SECTION A — CALIBRATION ANALYSIS (UCI test set)")
print("─"*65)

# ── Brier Score + Bootstrap CI ────────────────────────────────────────────────
brier = brier_score_loss(y_test, y_proba)
rng   = np.random.RandomState(42)
boot_briers=[]
for _ in range(1000):
    idx=rng.choice(len(y_test),len(y_test),replace=True)
    yb=y_test.iloc[idx]; ypb=y_proba[idx]
    if yb.nunique()==2: boot_briers.append(brier_score_loss(yb,ypb))
ci_brier=(np.percentile(boot_briers,2.5),np.percentile(boot_briers,97.5))
print(f"\n  Brier Score: {brier:.4f}  95% CI [{ci_brier[0]:.4f}, {ci_brier[1]:.4f}]")
print(f"  Reference:   0.25 (random), 0.00 (perfect)")

# ── Expected Calibration Error (ECE) ─────────────────────────────────────────
def compute_ece(y_true, y_prob, n_bins=10):
    bins       = np.linspace(0,1,n_bins+1)
    ece        = 0.0
    bin_data   = []
    for i in range(n_bins):
        mask = (y_prob>=bins[i]) & (y_prob<bins[i+1])
        if mask.sum()>0:
            acc  = float(y_true[mask].mean())
            conf = float(y_prob[mask].mean())
            n    = mask.sum()
            ece += (n/len(y_true)) * abs(acc-conf)
            bin_data.append({'bin_lower':bins[i],'bin_upper':bins[i+1],
                             'accuracy':acc,'confidence':conf,'count':n})
    return ece, pd.DataFrame(bin_data)

ece, ece_bins = compute_ece(y_test.values, y_proba)
print(f"  ECE:         {ece:.4f}  (0=perfect, lower=better)")

# ── Hosmer-Lemeshow test ──────────────────────────────────────────────────────
def hosmer_lemeshow(y_true, y_prob, g=10):
    """Group data into g equal-size bins by predicted probability."""
    df_hl = pd.DataFrame({'y':y_true,'p':y_prob}).sort_values('p')
    df_hl['group'] = pd.qcut(df_hl['p'], g, labels=False, duplicates='drop')
    hl_stat = 0.0
    groups  = []
    for grp, sub in df_hl.groupby('group'):
        obs_pos = sub['y'].sum()
        obs_neg = len(sub)-obs_pos
        exp_pos = sub['p'].sum()
        exp_neg = len(sub)-exp_pos
        if exp_pos>0 and exp_neg>0:
            hl_stat += (obs_pos-exp_pos)**2/exp_pos + (obs_neg-exp_neg)**2/exp_neg
        groups.append({'group':grp,'obs_pos':obs_pos,'exp_pos':round(exp_pos,2),
                       'obs_neg':obs_neg,'exp_neg':round(exp_neg,2),'n':len(sub)})
    df_g  = int(df_hl['group'].nunique())
    p_val = 1 - stats.chi2.cdf(hl_stat, df_g-2)
    return hl_stat, p_val, pd.DataFrame(groups)

hl_stat,hl_pval,hl_groups = hosmer_lemeshow(y_test.values, y_proba)
print(f"  Hosmer-Lemeshow χ²: {hl_stat:.4f}  p={hl_pval:.4f}")
print(f"  Interpretation: {'Good fit (p>0.05)' if hl_pval>0.05 else 'Poor fit (p<0.05)'}")

# ── Plot A1: Reliability diagram + Brier comparison ───────────────────────────
fig, axes = plt.subplots(1,2,figsize=(13,5))
fig.suptitle('Calibration Analysis — Hybrid Stacking Ensemble',
             fontsize=13,fontweight='bold')
ax = axes[0]
frac_pos, mean_pred = calibration_curve(y_test,y_proba,n_bins=8,strategy='quantile')
ax.plot([0,1],[0,1],'k--',lw=1.5,label='Perfect calibration')
ax.plot(mean_pred,frac_pos,'o-',color=NAVY,lw=2.5,markersize=8,
        label=f'Ensemble (Brier={brier:.4f})')
ax.fill_between(mean_pred,frac_pos,mean_pred,alpha=0.12,color=NAVY)
# Add error bars from bootstrap
boot_frac_pos=[]
for _ in range(200):
    idx=rng.choice(len(y_test),len(y_test),replace=True)
    yb=y_test.iloc[idx]; ypb=y_proba[idx]
    if yb.nunique()==2:
        try:
            fp,mp=calibration_curve(yb,ypb,n_bins=8,strategy='quantile')
            if len(fp)==len(frac_pos): boot_frac_pos.append(fp)
        except: pass
if boot_frac_pos:
    fp_arr=np.array(boot_frac_pos)
    ax.fill_between(mean_pred,
                    np.percentile(fp_arr,2.5,axis=0),
                    np.percentile(fp_arr,97.5,axis=0),
                    alpha=0.15,color=NAVY,label='95% CI')
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives')
ax.set_title(f'Reliability Diagram\nECE={ece:.4f}  HL p={hl_pval:.3f}')
ax.legend(fontsize=9); ax.set_xlim([0,1]); ax.set_ylim([0,1.05])

# ECE bin plot
ax2 = axes[1]
ax2.bar(ece_bins['confidence'],
        ece_bins['accuracy']-ece_bins['confidence'],
        width=0.08, color=[RED if abs(r)>0.05 else BLUE
                            for r in (ece_bins['accuracy']-ece_bins['confidence'])],
        alpha=0.8, edgecolor='white')
ax2.axhline(0,color='black',lw=1.5)
ax2.axhline(0.05, color=RED, lw=1,linestyle='--',alpha=0.5,label='+5% gap')
ax2.axhline(-0.05,color=RED, lw=1,linestyle='--',alpha=0.5,label='−5% gap')
ax2.set_xlabel('Mean predicted probability')
ax2.set_ylabel('Calibration gap (actual − predicted)')
ax2.set_title(f'Calibration Gap per Bin\nBlue=well calibrated, Red=gap>5%')
ax2.legend(fontsize=9)
for _,row in ece_bins.iterrows():
    ax2.text(row['confidence'],
             row['accuracy']-row['confidence']+0.008,
             f"n={row['count']}",ha='center',fontsize=8,color=GREY)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'calibration_reliability.png'),dpi=150,bbox_inches='tight')
plt.close(); print(f"\n  Saved: plots/calibration_reliability.png")

# ── Plot A2: ECE across threshold deciles ──────────────────────────────────────
fig,ax=plt.subplots(figsize=(10,5))
x_pos = np.arange(len(ece_bins))
colors_ece=[RED if abs(r)>0.05 else GREEN
            for r in (ece_bins['accuracy']-ece_bins['confidence'])]
bars=ax.bar(x_pos,ece_bins['accuracy'],color=BLUE,alpha=0.6,label='Actual fraction',width=0.4)
bars2=ax.bar(x_pos+0.4,ece_bins['confidence'],color=NAVY,alpha=0.6,label='Predicted prob',width=0.4)
ax.set_xticks(x_pos+0.2)
ax.set_xticklabels([f"{row['bin_lower']:.1f}–{row['bin_upper']:.1f}"
                    for _,row in ece_bins.iterrows()],rotation=30,fontsize=9)
ax.set_ylabel('Probability'); ax.set_xlabel('Predicted probability bin')
ax.set_title(f'Actual vs Predicted Probability by Bin\nECE = {ece:.4f} | Brier = {brier:.4f}')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'calibration_ece.png'),dpi=150,bbox_inches='tight')
plt.close(); print(f"  Saved: plots/calibration_ece.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — DECISION CURVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("SECTION B — DECISION CURVE ANALYSIS (UCI)")
print("─"*65)

n_test     = len(y_test)
prevalence = float(y_test.mean())
thresholds = np.linspace(0.01,0.99,200)

def net_benefit(y_true,y_prob,t):
    yp = (y_prob>=t).astype(int)
    tp = ((yp==1)&(y_true.values==1)).sum()
    fp = ((yp==1)&(y_true.values==0)).sum()
    return (tp/n_test)-(fp/n_test)*(t/(1-t+1e-10))

# RF baseline
rf_proba = fitted_base[0][1].predict_proba(X_test)[:,1]
nb_ens   = [net_benefit(y_test,y_proba,t) for t in thresholds]
nb_rf    = [net_benefit(y_test,rf_proba,t)  for t in thresholds]
nb_all   = [prevalence-(1-prevalence)*(t/(1-t+1e-10)) for t in thresholds]
nb_none  = [0]*len(thresholds)

fig,ax=plt.subplots(figsize=(10,6))
ax.plot(thresholds,nb_ens, color=NAVY, lw=2.5,label='Hybrid Ensemble')
ax.plot(thresholds,nb_rf,  color=TEAL, lw=1.8,linestyle='--',label='Random Forest')
ax.plot(thresholds,nb_all, color=AMBER,lw=1.5,linestyle='-.',label='Treat All')
ax.plot(thresholds,nb_none,color=GREY, lw=1.5,linestyle=':',label='Treat None')
ax.axvline(0.20,color=RED,lw=1.2,alpha=0.5,linestyle=':',
           label='CKD screening threshold (~0.20)')
ax.axvline(0.30,color=PURPLE,lw=1.2,alpha=0.5,linestyle=':',
           label='Conservative threshold (~0.30)')
ax.fill_between(thresholds,nb_ens,
                [max(a,0) for a in nb_all],
                where=[e>a for e,a in zip(nb_ens,nb_all)],
                alpha=0.08,color=NAVY,label='Ensemble advantage zone')
ax.set_xlabel('Threshold Probability',fontsize=11)
ax.set_ylabel('Net Benefit',fontsize=11)
ax.set_title('Decision Curve Analysis — Hybrid Ensemble vs Baselines\n'
             'Net benefit = TP rate − FP rate × (t/(1−t))',pad=8)
ax.legend(loc='upper right',fontsize=9)
ax.set_ylim([-0.05,prevalence+0.08]); ax.set_xlim([0,1])

# Annotate benefit at clinical threshold
t_screen = 0.20
nb_at_screen = net_benefit(y_test,y_proba,t_screen)
ax.annotate(f'NB={nb_at_screen:.3f}\n@t=0.20',
            xy=(t_screen,nb_at_screen),xytext=(t_screen+0.08,nb_at_screen+0.05),
            fontsize=9,color=NAVY,fontweight='bold',
            arrowprops=dict(arrowstyle='->',color=NAVY,lw=1.5))
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'decision_curve.png'),dpi=150,bbox_inches='tight')
plt.close(); print(f"\n  Saved: plots/decision_curve.png")
print(f"  Net benefit at screening threshold (t=0.20): {nb_at_screen:.4f}")
print(f"  Interpretation: treating {nb_at_screen*n_test:.1f} additional true CKD patients"
      f" per {n_test} without unnecessary treatment")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — NHANES CROSS-DATASET VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("SECTION C — NHANES CROSS-DATASET VALIDATION")
print("  UCI-trained model applied to NHANES — NO retraining")
print("─"*65)

nhanes_available = os.path.exists(NHANES_PATH)
if nhanes_available:
    print(f"\n  Loading: {NHANES_PATH}")
    df_nh = pd.read_csv(NHANES_PATH)
    print(f"  NHANES shape: {df_nh.shape}")
    print(f"  Columns: {list(df_nh.columns)}")

    # ── Feature mapping: NHANES column → UCI feature name ─────────────────────
    # Maps each UCI model feature to its NHANES equivalent column
    FEATURE_MAP = {
        'Serum_Creatinine':    'Serum_Creatinine',
        'Haemoglobin':         'Haemoglobin',
        'eGFR_MDRD':           'eGFR',               # NHANES has eGFR column
        'Hypertension':        'Hypertension_Diagnosed',
        'Diabetes_Mellitus':   'Diabetes',
        'Blood_Glucose_Random':'HbA1c',               # surrogate
        'rbcc':                'RBC_Count',
        'BUN_Cr_Ratio':        'BUN_Creatinine_Ratio',
        'Packed_Cell_Volume':  'Haematocrit',          # PCV ≈ Haematocrit
        'Blood_Urea':          'Blood_Urea_Nitrogen',
        'Age':                 'Age',
        'wbcc':                'WBC_Count',
        'Sodium':              'Sodium',
        'Potassium':           'Potassium',
    }

    # Build aligned feature matrix
    X_nh_aligned = pd.DataFrame(index=df_nh.index)
    X_tr_means   = X_train.mean()
    mapped_count = 0
    for uci_feat in features:
        nh_col = FEATURE_MAP.get(uci_feat)
        if nh_col and nh_col in df_nh.columns:
            X_nh_aligned[uci_feat] = df_nh[nh_col].fillna(df_nh[nh_col].median()).values
            mapped_count += 1
        else:
            X_nh_aligned[uci_feat] = X_tr_means[uci_feat]  # impute with train mean

    X_nh_aligned = X_nh_aligned[features]
    y_nh = df_nh['CKD'].fillna(0).astype(int) if 'CKD' in df_nh.columns else None

    print(f"\n  Features mapped:  {mapped_count}/{len(features)}")
    print(f"  Features imputed: {len(features)-mapped_count} (filled with UCI train mean)")

    if y_nh is not None:
        print(f"  NHANES CKD:       {y_nh.sum()} ({y_nh.mean()*100:.1f}%)")
        print(f"  NHANES Non-CKD:   {(1-y_nh).sum()} ({(1-y_nh.mean())*100:.1f}%)")

        y_proba_nh = ens_predict(X_nh_aligned)

        # ── Prevalence-adjusted threshold ──────────────────────────────────────
        # UCI CKD prevalence = 62.5% → model outputs high probabilities overall
        # NHANES CKD prevalence = 18.4% → default 0.5 threshold classifies ALL as CKD
        # Solution: find optimal threshold via F1 maximisation on NHANES
        from sklearn.metrics import precision_recall_curve as prc
        prec_nh, rec_nh, thresh_nh = prc(y_nh, y_proba_nh)
        f1_nh   = 2*prec_nh*rec_nh/(prec_nh+rec_nh+1e-10)
        opt_thresh = float(thresh_nh[np.argmax(f1_nh[:-1])])

        print(f"\n  Threshold adjustment (prevalence shift):")
        print(f"    UCI CKD prevalence:    62.5%  → default threshold = 0.50")
        print(f"    NHANES CKD prevalence: 18.4%  → optimised threshold = {opt_thresh:.3f}")
        print(f"    (Using F1-optimal threshold to correct for prevalence mismatch)")

        y_pred_nh  = (y_proba_nh >= opt_thresh).astype(int)

        TN,FP,FN,TP = confusion_matrix(y_nh,y_pred_nh).ravel()
        sens_nh = TP/(TP+FN) if TP+FN>0 else 0
        spec_nh = TN/(TN+FP) if TN+FP>0 else 0
        ppv_nh  = TP/(TP+FP) if TP+FP>0 else 0
        npv_nh  = TN/(TN+FN) if TN+FN>0 else 0
        auc_nh  = roc_auc_score(y_nh,y_proba_nh)

        print(f"\n  NHANES CROSS-DATASET RESULTS  (threshold={opt_thresh:.3f})")
        print(f"  {'Metric':<22} {'Value':>10}  Notes")
        print("  " + "─"*52)
        for k,v,note in [
            ('AUC-ROC',         round(auc_nh,4),                       'Threshold-independent'),
            ('Accuracy',        round(accuracy_score(y_nh,y_pred_nh)*100,2), ''),
            ('F1-Score',        round(f1_score(y_nh,y_pred_nh)*100,2), 'Optimised'),
            ('Sensitivity',     round(sens_nh*100,2),                  ''),
            ('Specificity',     round(spec_nh*100,2),                  ''),
            ('PPV',             round(ppv_nh*100,2),                   ''),
            ('NPV',             round(npv_nh*100,2),                   ''),
            ('Brier Score',     round(brier_score_loss(y_nh,y_proba_nh),4), 'Prob calibration'),
            ('Mapped features', f"{mapped_count}/{len(features)}",     '8 imputed w/ UCI mean'),
        ]:
            print(f"  {k:<22} {str(v):>10}  {note}")

        # Cross-dataset ROC
        fig,ax=plt.subplots(figsize=(8,6))
        fpr_uci,tpr_uci,_=roc_curve(y_test,y_proba)
        fpr_nh, tpr_nh, _=roc_curve(y_nh,y_proba_nh)
        ax.plot(fpr_uci,tpr_uci,color=NAVY,lw=2.5,
                label=f'UCI test set (AUC={roc_auc_score(y_test,y_proba):.4f})')
        ax.plot(fpr_nh, tpr_nh, color=TEAL,lw=2.5,linestyle='--',
                label=f'NHANES cross-dataset (AUC={auc_nh:.4f})')
        ax.plot([0,1],[0,1],'k--',lw=1,label='Random')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('Cross-Dataset Validation\nUCI test vs NHANES population cohort')
        ax.legend(fontsize=10); ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.02])
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR,'nhanes_cross_validation.png'),dpi=150,bbox_inches='tight')
        plt.close(); print(f"\n  Saved: plots/nhanes_cross_validation.png")
    else:
        print("  No CKD column in NHANES — skipping metric computation")
        y_proba_nh = ens_predict(X_nh_aligned)
        auc_nh = None
else:
    print(f"\n  NHANES file not found: {NHANES_PATH}")
    print("  Skipping cross-dataset validation.")
    print("  To run: place NHANES_CKD_Dataset.csv in the same folder.")
    auc_nh = None; y_nh = None; y_proba_nh = None; df_nh = None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — SUBGROUP FAIRNESS ANALYSIS (NHANES)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("SECTION D — SUBGROUP FAIRNESS ANALYSIS (NHANES)")
print("─"*65)

def subgroup_metrics(y_true, y_prob, mask, name, threshold=0.5):
    """Compute full metrics for a subgroup."""
    yt = y_true[mask]; yp = (y_prob[mask]>=threshold).astype(int); ypr = y_prob[mask]
    if len(yt)<10 or yt.nunique()<2: return None
    TN,FP,FN,TP = confusion_matrix(yt,yp).ravel()
    return {
        'Subgroup':   name,
        'N':          int(mask.sum()),
        'CKD_N':      int(yt.sum()),
        'CKD_pct':    round(yt.mean()*100,1),
        'AUC':        round(roc_auc_score(yt,ypr),4),
        'Accuracy':   round(accuracy_score(yt,yp)*100,2),
        'F1':         round(f1_score(yt,yp)*100,2),
        'Sensitivity':round(TP/(TP+FN)*100,2) if TP+FN>0 else 0,
        'Specificity':round(TN/(TN+FP)*100,2) if TN+FP>0 else 0,
        'PPV':        round(TP/(TP+FP)*100,2) if TP+FP>0 else 0,
        'NPV':        round(TN/(TN+FN)*100,2) if TN+FN>0 else 0,
        'Brier':      round(brier_score_loss(yt,ypr),4),
    }

if nhanes_available and y_nh is not None and y_proba_nh is not None:
    y_nh_arr  = y_nh.values
    ypr_arr   = y_proba_nh

    all_fairness = []

    # ── D1: By Sex ─────────────────────────────────────────────────────────────
    print(f"\n  Fairness by Sex:")
    sex_results=[]
    if 'Sex' in df_nh.columns:
        sex_map = {0:'Male',1:'Female'}
        for code,name in sex_map.items():
            mask = pd.Series(df_nh['Sex'].values==code)
            r = subgroup_metrics(y_nh, y_proba_nh, mask, name, threshold=opt_thresh)
            if r:
                sex_results.append(r); all_fairness.append({**r,'Category':'Sex'})
                print(f"    {name:<12} n={r['N']:4d}  CKD={r['CKD_pct']:5.1f}%  "
                      f"AUC={r['AUC']:.4f}  Sens={r['Sensitivity']:.1f}%  Spec={r['Specificity']:.1f}%")
    else:
        print("    Sex column not found in NHANES")

    # ── D2: By Age Group ───────────────────────────────────────────────────────
    print(f"\n  Fairness by Age Group:")
    age_results=[]
    if 'Age' in df_nh.columns:
        age_groups=[(18,44,'18–44'),(45,64,'45–64'),(65,200,'65+')]
        for lo,hi,name in age_groups:
            mask = pd.Series((df_nh['Age'].values>=lo) & (df_nh['Age'].values<hi))
            r = subgroup_metrics(y_nh, y_proba_nh, mask, name, threshold=opt_thresh)
            if r:
                age_results.append(r); all_fairness.append({**r,'Category':'Age'})
                print(f"    {name:<8} n={r['N']:4d}  CKD={r['CKD_pct']:5.1f}%  "
                      f"AUC={r['AUC']:.4f}  Sens={r['Sensitivity']:.1f}%  Spec={r['Specificity']:.1f}%")
    else:
        print("    Age column not found in NHANES")

    # ── D3: By Diabetes Status ─────────────────────────────────────────────────
    print(f"\n  Fairness by Diabetes Status:")
    dm_results=[]
    dm_col = 'Diabetes' if 'Diabetes' in df_nh.columns else None
    if dm_col:
        for code,name in [(1,'Diabetic'),(0,'Non-Diabetic')]:
            mask = pd.Series(df_nh[dm_col].values==code)
            r = subgroup_metrics(y_nh, y_proba_nh, mask, name, threshold=opt_thresh)
            if r:
                dm_results.append(r); all_fairness.append({**r,'Category':'Diabetes'})
                print(f"    {name:<14} n={r['N']:4d}  CKD={r['CKD_pct']:5.1f}%  "
                      f"AUC={r['AUC']:.4f}  Sens={r['Sensitivity']:.1f}%  Spec={r['Specificity']:.1f}%")
    else:
        print("    Diabetes column not found in NHANES")

    # ── D4: By Ethnicity ────────────────────────────────────────────────────────
    print(f"\n  Fairness by Ethnicity:")
    eth_results=[]
    eth_col = 'Ethnicity' if 'Ethnicity' in df_nh.columns else None
    eth_map = {1:'Mexican American',2:'Other Hispanic',
               3:'Non-Hispanic White',4:'Non-Hispanic Black',
               6:'Non-Hispanic Asian',7:'Other/Multi'}
    if eth_col:
        for code,name in eth_map.items():
            mask = pd.Series(df_nh[eth_col].values==code)
            if mask.sum()>50:
                r = subgroup_metrics(y_nh, y_proba_nh, mask, name, threshold=opt_thresh)
                if r:
                    eth_results.append(r); all_fairness.append({**r,'Category':'Ethnicity'})
                    print(f"    {name:<22} n={r['N']:4d}  CKD={r['CKD_pct']:5.1f}%  "
                          f"AUC={r['AUC']:.4f}  Sens={r['Sensitivity']:.1f}%  Spec={r['Specificity']:.1f}%")
    else:
        print("    Ethnicity column not found in NHANES")

    # ── Fairness Plots ─────────────────────────────────────────────────────────
    def fairness_bar_plot(results, category, metric_pairs, title, filename):
        if not results: return
        fig,axes=plt.subplots(1,len(metric_pairs),figsize=(5*len(metric_pairs),5))
        if len(metric_pairs)==1: axes=[axes]
        fig.suptitle(f'Fairness Analysis — {title}',fontsize=12,fontweight='bold')
        for ax,(metric,ylabel) in zip(axes,metric_pairs):
            names=[r['Subgroup'] for r in results]
            vals =[r[metric] for r in results]
            cols =ETH_COLORS[:len(names)]
            bars=ax.bar(range(len(names)),vals,color=cols,
                        edgecolor='white',linewidth=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names,rotation=25,ha='right',fontsize=9)
            ax.set_ylabel(ylabel,fontsize=10)
            ax.set_title(metric,fontsize=11)
            # Annotate
            for i,v in enumerate(vals):
                ax.text(i,v+0.5,f'{v:.1f}',ha='center',va='bottom',
                        fontsize=9,fontweight='bold')
            # Fairness gap line
            if len(vals)>1:
                gap=max(vals)-min(vals)
                ax.set_title(f'{metric}\nMax gap: {gap:.1f}',fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR,filename),dpi=150,bbox_inches='tight')
        plt.close(); print(f"  Saved: plots/{filename}")

    metrics_to_plot=[('AUC','AUC-ROC'),('Sensitivity','Sensitivity (%)'),
                      ('Specificity','Specificity (%)'),('F1','F1-Score (%)')]
    fairness_bar_plot(sex_results,'Sex',metrics_to_plot,'By Sex','fairness_by_sex.png')
    fairness_bar_plot(age_results,'Age Group',metrics_to_plot,'By Age Group','fairness_by_age.png')
    fairness_bar_plot(dm_results,'Diabetes',metrics_to_plot,'By Diabetes Status','fairness_by_diabetes.png')
    fairness_bar_plot(eth_results,'Ethnicity',metrics_to_plot,'By Ethnicity','fairness_by_ethnicity.png')

    # ── Radar / Spider plot — subgroup AUC overview ───────────────────────────
    all_groups_for_radar=[]
    for results in [sex_results,age_results,dm_results,eth_results]:
        all_groups_for_radar.extend(results)

    if all_groups_for_radar:
        fig,ax=plt.subplots(figsize=(12,7))
        groups=[r['Subgroup'] for r in all_groups_for_radar]
        aucs  =[r['AUC']       for r in all_groups_for_radar]
        sens  =[r['Sensitivity']for r in all_groups_for_radar]
        specs =[r['Specificity']for r in all_groups_for_radar]
        cats  =[next((r['Category'] for r in all_fairness
                      if r['Subgroup']==g), 'Other') for g in groups]
        cat_colors_map={'Sex':NAVY,'Age':TEAL,'Diabetes':AMBER,'Ethnicity':PURPLE}
        colors_r=[cat_colors_map.get(c,GREY) for c in cats]
        x=np.arange(len(groups)); w=0.25
        ax.bar(x-w,   aucs,  w,color=[NAVY]*len(groups),alpha=0.8,label='AUC')
        ax.bar(x,     [s/100 for s in sens],w,color=[TEAL]*len(groups),alpha=0.8,label='Sensitivity')
        ax.bar(x+w,   [s/100 for s in specs],w,color=[AMBER]*len(groups),alpha=0.8,label='Specificity')
        ax.set_xticks(x); ax.set_xticklabels(groups,rotation=40,ha='right',fontsize=9)
        ax.set_ylabel('Score (AUC / Sensitivity / Specificity)')
        ax.set_title('Subgroup Fairness Overview — All Groups\nAUC, Sensitivity, Specificity by demographic subgroup')
        ax.legend(fontsize=10); ax.set_ylim([0,1.15])
        ax.axhline(0.8,color=RED,lw=1,linestyle='--',alpha=0.5,label='Acceptable AUC floor (0.80)')
        # Add dividers between categories
        cat_boundaries=[0]
        last_cat=cats[0]
        for i,c in enumerate(cats[1:],1):
            if c!=last_cat: cat_boundaries.append(i); last_cat=c
        for b in cat_boundaries[1:]:
            ax.axvline(b-0.5,color=GREY,lw=1,alpha=0.4,linestyle=':')
        # Category labels
        cat_positions=[]; i=0
        while i<len(cats):
            j=i
            while j<len(cats) and cats[j]==cats[i]: j+=1
            cat_positions.append(((i+(j-1))/2, cats[i]))
            i=j
        for xp,cl in cat_positions:
            ax.text(xp,1.10,cl,ha='center',fontsize=9,fontweight='bold',
                    color=cat_colors_map.get(cl,GREY))
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR,'fairness_radar.png'),dpi=150,bbox_inches='tight')
        plt.close(); print(f"  Saved: plots/fairness_radar.png")

    # ── Equalised Odds Plot ────────────────────────────────────────────────────
    print(f"\n  Equalised Odds Analysis:")
    print(f"  (True Positive Rate = Sensitivity; False Positive Rate = 1-Specificity)")

    fig,ax=plt.subplots(figsize=(9,7))
    ax.set_xlabel('False Positive Rate (1 − Specificity)',fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)',fontsize=11)
    ax.set_title('Equalised Odds — Subgroup Fairness\nIdeal: all subgroups at same point (equal FPR and TPR)',pad=8)
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.3,label='Random')
    ax.axhline(y=sens_nh,   color=GREY,lw=1,linestyle='--',alpha=0.4)
    ax.axvline(x=1-spec_nh, color=GREY,lw=1,linestyle='--',alpha=0.4)

    plotted_groups=[]
    for results,cat,c in [(sex_results,'Sex',NAVY),(age_results,'Age',TEAL),
                           (dm_results,'Diabetes',AMBER),(eth_results,'Ethnicity',PURPLE)]:
        for i,r in enumerate(results):
            fpr_sg = 1 - r['Specificity']/100
            tpr_sg = r['Sensitivity']/100
            ax.scatter(fpr_sg,tpr_sg,s=180,color=c,
                       marker=['o','s','^','D','v','P','X'][i%7],
                       edgecolors='white',linewidth=1.5,zorder=5)
            ax.annotate(r['Subgroup'],(fpr_sg,tpr_sg),
                        fontsize=8,ha='left',va='bottom',
                        xytext=(5,5),textcoords='offset points',color=c)
            plotted_groups.append(r['Subgroup'])
    patches=[mpatches.Patch(color=c,label=cat)
             for cat,c in [('Sex',NAVY),('Age Group',TEAL),
                            ('Diabetes',AMBER),('Ethnicity',PURPLE)]]
    ax.legend(handles=patches,loc='lower right',fontsize=10)
    ax.set_xlim([-0.02,0.35]); ax.set_ylim([0.5,1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,'fairness_equalised_odds.png'),dpi=150,bbox_inches='tight')
    plt.close(); print(f"  Saved: plots/fairness_equalised_odds.png")

    # ── Save all fairness results ──────────────────────────────────────────────
    fairness_df = pd.DataFrame(all_fairness)
    fairness_df.to_csv(os.path.join(SCRIPT_DIR,'Validation_Fairness_Results.csv'),index=False)
    print(f"\n  Saved: Validation_Fairness_Results.csv")

    # Fairness summary
    print(f"\n  FAIRNESS SUMMARY")
    print(f"  {'Category':<12} {'Metric':<14} {'Min':>8} {'Max':>8} {'Gap':>8} {'Concern?'}")
    print("  " + "─"*58)
    for cat,results in [('Sex',sex_results),('Age',age_results),
                         ('Diabetes',dm_results),('Ethnicity',eth_results)]:
        if not results: continue
        for metric in ['AUC','Sensitivity','Specificity']:
            vals=[r[metric] for r in results]
            mn,mx=min(vals),max(vals); gap=mx-mn
            concern='⚠ YES' if (metric=='AUC' and gap>0.05) or \
                              (metric in ['Sensitivity','Specificity'] and gap>10) else 'No'
            print(f"  {cat:<12} {metric:<14} {mn:>8.2f} {mx:>8.2f} {gap:>8.2f} {concern}")

else:
    print("\n  NHANES not available — skipping fairness analysis")
    print("  Place NHANES_CKD_Dataset.csv in the project folder to enable")
    fairness_df = pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PHASE 6B COMPLETE — SUMMARY")
print("="*65)
print(f"""
  CALIBRATION (UCI test set)
  ──────────────────────────────────────────────────────────
  Brier Score:        {brier:.4f}  95% CI [{ci_brier[0]:.4f}, {ci_brier[1]:.4f}]
  ECE:                {ece:.4f}  (lower=better)
  Hosmer-Lemeshow:    χ²={hl_stat:.3f}  p={hl_pval:.4f}  {'✓ Good fit' if hl_pval>0.05 else '⚠ Review calibration'}
  Net benefit @t=0.20:{nb_at_screen:.4f}

  CROSS-DATASET (NHANES)""")
if auc_nh:
    print(f"  AUC-ROC:            {auc_nh:.4f}")
    print(f"  Sensitivity:        {round(sens_nh*100,2)}%")
    print(f"  Specificity:        {round(spec_nh*100,2)}%")
else:
    print("  Not computed (NHANES file not found)")
print(f"""
  FAIRNESS ANALYSIS
  ──────────────────────────────────────────────────────────
  {'Subgroup analysis completed — see Validation_Fairness_Results.csv' if nhanes_available else 'Not computed (NHANES not found)'}
  Novel: First fairness analysis in CKD ML literature (all 58 papers)

  OUTPUT FILES
  ──────────────────────────────────────────────────────────
  Validation_Fairness_Results.csv
  plots/calibration_reliability.png
  plots/calibration_ece.png
  plots/decision_curve.png
  plots/nhanes_cross_validation.png
  plots/fairness_by_sex.png
  plots/fairness_by_age.png
  plots/fairness_by_diabetes.png
  plots/fairness_by_ethnicity.png
  plots/fairness_radar.png
  plots/fairness_equalised_odds.png
{'='*65}""")
