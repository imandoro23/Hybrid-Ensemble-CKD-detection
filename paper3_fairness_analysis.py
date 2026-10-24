"""
Paper 3 — Comprehensive Fairness Analysis Script
Generates all data needed for the fairness paper
Run from: C:/Users/imand/.vscode/PhD main Work/
Runtime: ~30 minutes
"""
import pandas as pd
import numpy as np
import copy, warnings
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import shap
warnings.filterwarnings('ignore')

FEATURES = [
    'Specific_Gravity','Albumin','Haemoglobin','eGFR_MDRD',
    'Packed_Cell_Volume','Diabetes_Mellitus','Hypertension',
    'Blood_Glucose_Random','Serum_Creatinine','rbcc','BUN_Cr_Ratio',
    'Appetite','Blood_Urea','Red_Blood_Cells','Sugar',
    'Blood_Pressure','Sodium','Pedal_Edema','wbcc','Pus_Cell','Age'
]

# ── DeLong AUC CI ──────────────────────────────────────────────────────────────
def delong_ci(y_true, y_score, alpha=0.05):
    y_true = np.array(y_true); y_score = np.array(y_score)
    if len(np.unique(y_true)) < 2 or len(y_true) < 10:
        return np.nan, np.nan, np.nan
    pos = y_score[y_true==1]; neg = y_score[y_true==0]
    n1, n0 = len(pos), len(neg)
    auc = roc_auc_score(y_true, y_score)
    V10 = np.array([np.mean(p > neg) + 0.5*np.mean(p==neg) for p in pos])
    V01 = np.array([np.mean(n < pos) + 0.5*np.mean(n==pos) for n in neg])
    var = np.var(V10,ddof=1)/n1 + np.var(V01,ddof=1)/n0
    se  = np.sqrt(var)
    z   = stats.norm.ppf(1 - alpha/2)
    return auc, max(0, auc - z*se), min(1, auc + z*se)

# ── Load UCI + rebuild ensemble ─────────────────────────────────────────────────
print("="*65)
print("PAPER 3 — FAIRNESS ANALYSIS PIPELINE")
print("="*65)
print("\n[1/6] Loading UCI data and rebuilding ensemble...")

df_uci = pd.read_csv('UCI_CKD_Processed.csv')
drop_c = [c for c in ['CKD_eGFR','CKD_ACR','KDIGO_Stage',
                       'Anaemia_Flag','Comorbidity_Score'] if c in df_uci.columns]
df_uci = df_uci.drop(columns=drop_c)
features = [f for f in FEATURES if f in df_uci.columns]
X_uci = df_uci[features]; y_uci = df_uci['CKD'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X_uci, y_uci, test_size=0.20, random_state=42, stratify=y_uci)
X_tr_means = X_train.mean()

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_defs = [
    ('RF',  RandomForestClassifier(n_estimators=300, max_features='sqrt',
             min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('GB',  GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
             max_depth=4, subsample=0.8, min_samples_leaf=3, random_state=42)),
    ('HGB', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
             max_depth=6, l2_regularization=0.5, min_samples_leaf=10, random_state=42)),
    ('SVM', Pipeline([('sc', StandardScaler()),
             ('clf', CalibratedClassifierCV(SVC(kernel='rbf', C=2.0, gamma='scale',
             class_weight='balanced', random_state=42), cv=3))])),
    ('MLP', Pipeline([('sc', StandardScaler()),
             ('clf', MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu',
             solver='adam', alpha=0.001, learning_rate='adaptive',
             learning_rate_init=0.001, max_iter=500, early_stopping=True,
             validation_fraction=0.1, random_state=42))]))
]
oof = np.zeros((len(X_train), len(base_defs)))
for fi,(ti,vi) in enumerate(cv5.split(X_train, y_train)):
    for j,(nm,bm) in enumerate(base_defs):
        m = copy.deepcopy(bm)
        m.fit(X_train.iloc[ti], y_train.iloc[ti])
        oof[vi,j] = m.predict_proba(X_train.iloc[vi])[:,1]
    print(f"  OOF fold {fi+1}/5")
fitted_base = []
for nm,bm in base_defs:
    bm.fit(X_train, y_train); fitted_base.append((nm,bm))
meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta.fit(oof, y_train)

def predict(X_in):
    if isinstance(X_in, np.ndarray):
        X_in = pd.DataFrame(X_in, columns=features)
    mf = np.column_stack([m.predict_proba(X_in)[:,1] for _,m in fitted_base])
    return meta.predict_proba(mf)[:,1]

print(f"  UCI test AUC: {roc_auc_score(y_test, predict(X_test)):.4f}")

# ── Load NHANES ─────────────────────────────────────────────────────────────────
print("\n[2/6] Loading and preparing NHANES data...")
df_nh = pd.read_csv('NHANES_CKD_Dataset.csv')
print(f"  Shape: {df_nh.shape}")
print(f"  Columns: {list(df_nh.columns)}")

# Feature alignment
FEATURE_MAP = {
    'Serum_Creatinine':    'Serum_Creatinine',
    'Haemoglobin':         'Haemoglobin',
    'eGFR_MDRD':           'eGFR',
    'Hypertension':        'Hypertension_Diagnosed',
    'Diabetes_Mellitus':   'Diabetes',
    'Blood_Glucose_Random':'HbA1c',
    'rbcc':                'RBC_Count',
    'BUN_Cr_Ratio':        'BUN_Creatinine_Ratio',
    'Packed_Cell_Volume':  'Haematocrit',
    'Blood_Urea':          'Blood_Urea_Nitrogen',
    'Age':                 'Age',
    'wbcc':                'WBC_Count',
    'Sodium':              'Sodium',
}
X_nh = pd.DataFrame(index=df_nh.index)
for feat in features:
    nh_col = FEATURE_MAP.get(feat)
    if nh_col and nh_col in df_nh.columns:
        X_nh[feat] = df_nh[nh_col].fillna(df_nh[nh_col].median()).values
    else:
        X_nh[feat] = X_tr_means[feat]
X_nh = X_nh[features]
y_nh = df_nh['CKD'].fillna(0).astype(int)

# Predict
print("  Running ensemble predictions on NHANES...")
y_prob_nh = predict(X_nh)
print(f"  CKD prevalence: {y_nh.mean()*100:.1f}%  ({y_nh.sum()}/{len(y_nh)})")
print(f"  Overall AUC: {roc_auc_score(y_nh, y_prob_nh):.4f}")

# ── Subgroup AUC + CI ──────────────────────────────────────────────────────────
print("\n[3/6] Computing subgroup AUC with DeLong 95% CI...")

# Identify demographic columns
demo_map = {
    'Sex':       'Sex',
    'Age_raw':   'Age',
    'Ethnicity': 'Ethnicity',
    'Diabetes':  'Diabetes',
}
available = {k: v for k,v in demo_map.items() if v in df_nh.columns}
print(f"  Available demographic columns: {list(available.values())}")

results = []

def analyse_subgroup(mask, name, category):
    if mask.sum() < 30:
        return
    yt = y_nh[mask]; yp = y_prob_nh[mask]
    if yt.nunique() < 2:
        return
    auc, ci_lo, ci_hi = delong_ci(yt, yp)
    bs = brier_score_loss(yt, yp)
    prev = yt.mean()
    results.append(dict(
        category=category, subgroup=name,
        n=int(mask.sum()), ckd_n=int(yt.sum()),
        prevalence=round(prev*100, 1),
        auc=round(auc,4), ci_lo=round(ci_lo,4), ci_hi=round(ci_hi,4),
        brier=round(bs,4)
    ))
    print(f"  {name:<30} n={mask.sum():5d}  AUC={auc:.4f} [{ci_lo:.4f},{ci_hi:.4f}]  "
          f"Brier={bs:.4f}  Prev={prev*100:.1f}%")

# Overall
print(f"\n  --- Overall ---")
auc_ov, ci_lo_ov, ci_hi_ov = delong_ci(y_nh, y_prob_nh)
bs_ov = brier_score_loss(y_nh, y_prob_nh)
results.append(dict(category='Overall',subgroup='All patients',
    n=len(y_nh),ckd_n=int(y_nh.sum()),prevalence=round(y_nh.mean()*100,1),
    auc=round(auc_ov,4),ci_lo=round(ci_lo_ov,4),ci_hi=round(ci_hi_ov,4),
    brier=round(bs_ov,4)))
print(f"  {'All patients':<30} n={len(y_nh):5d}  AUC={auc_ov:.4f} [{ci_lo_ov:.4f},{ci_hi_ov:.4f}]"
      f"  Brier={bs_ov:.4f}")

# Sex
if 'Sex' in df_nh.columns:
    print(f"\n  --- Sex ---")
    sex_map = {0:'Male',1:'Female'}
    for code,name in sex_map.items():
        analyse_subgroup(pd.Series(df_nh['Sex'].values==code,index=df_nh.index), name, 'Sex')

# Age groups
if 'Age' in df_nh.columns:
    print(f"\n  --- Age Group ---")
    age_groups = [(18,44,'18-44'),(45,64,'45-64'),(65,200,'65+')]
    for lo,hi,label in age_groups:
        mask = pd.Series((df_nh['Age'].values>=lo)&(df_nh['Age'].values<hi),index=df_nh.index)
        analyse_subgroup(mask, label, 'Age Group')

# Diabetes
if 'Diabetes' in df_nh.columns:
    print(f"\n  --- Diabetes Status ---")
    for code,name in [(1,'Diabetic'),(0,'Non-Diabetic')]:
        analyse_subgroup(pd.Series(df_nh['Diabetes'].values==code,index=df_nh.index),
                        name, 'Diabetes Status')

# Ethnicity
if 'Ethnicity' in df_nh.columns:
    print(f"\n  --- Ethnicity ---")
    eth_map = {1:'Mexican American',2:'Other Hispanic',3:'Non-Hispanic White',
               4:'Non-Hispanic Black',6:'Non-Hispanic Asian',7:'Other/Multiracial'}
    for code,name in eth_map.items():
        mask = pd.Series(df_nh['Ethnicity'].values==code,index=df_nh.index)
        if mask.sum()>=50:
            analyse_subgroup(mask, name, 'Ethnicity')

# BMI groups if available
if 'BMI' in df_nh.columns:
    print(f"\n  --- BMI Group ---")
    bmi_groups = [(0,18.5,'Underweight'),(18.5,25,'Normal'),(25,30,'Overweight'),(30,100,'Obese')]
    for lo,hi,label in bmi_groups:
        mask = pd.Series((df_nh['BMI'].values>=lo)&(df_nh['BMI'].values<hi),index=df_nh.index)
        analyse_subgroup(mask, label, 'BMI Group')

# ── Intersectional analysis ────────────────────────────────────────────────────
print("\n[4/6] Intersectional analysis (age × diabetes; age × sex)...")

if 'Age' in df_nh.columns and 'Diabetes' in df_nh.columns:
    print(f"\n  --- Age × Diabetes ---")
    for lo,hi,age_lbl in age_groups:
        for dm_code,dm_lbl in [(1,'Diabetic'),(0,'Non-Diabetic')]:
            mask = pd.Series(
                (df_nh['Age'].values>=lo)&(df_nh['Age'].values<hi)&
                (df_nh['Diabetes'].values==dm_code), index=df_nh.index)
            analyse_subgroup(mask, f"{age_lbl} + {dm_lbl}", 'Age × Diabetes')

if 'Age' in df_nh.columns and 'Sex' in df_nh.columns:
    print(f"\n  --- Age × Sex ---")
    for lo,hi,age_lbl in age_groups:
        for sex_code,sex_lbl in [(0,'Male'),(1,'Female')]:
            mask = pd.Series(
                (df_nh['Age'].values>=lo)&(df_nh['Age'].values<hi)&
                (df_nh['Sex'].values==sex_code), index=df_nh.index)
            analyse_subgroup(mask, f"{age_lbl} + {sex_lbl}", 'Age × Sex')

# ── Disparity metrics ──────────────────────────────────────────────────────────
print("\n[5/6] Computing fairness disparity metrics...")

df_res = pd.DataFrame(results)
df_res.to_csv('Paper3_FairnessResults.csv', index=False)

print("\n  --- AUC DISPARITY SUMMARY ---")
for cat in ['Sex','Age Group','Diabetes Status','Ethnicity']:
    sub = df_res[df_res['category']==cat]
    if len(sub) < 2: continue
    auc_range = sub['auc'].max() - sub['auc'].min()
    worst = sub.loc[sub['auc'].idxmin()]
    best  = sub.loc[sub['auc'].idxmax()]
    print(f"\n  {cat}:")
    print(f"    AUC range:  {sub['auc'].min():.4f} – {sub['auc'].max():.4f}  (Δ={auc_range:.4f})")
    print(f"    Best:   {best['subgroup']} ({best['auc']:.4f})")
    print(f"    Worst:  {worst['subgroup']} ({worst['auc']:.4f})")
    print(f"    Disparate impact threshold exceeded (Δ>0.05): {'YES ⚠️' if auc_range>0.05 else 'no'}")

# ── SHAP by subgroup ───────────────────────────────────────────────────────────
print("\n[6/6] Computing SHAP attributions by demographic subgroup...")
print("  (TreeExplainer on RF base learner applied to NHANES)")

rf_model = fitted_base[0][1]  # RF is first in base_defs
explainer = shap.TreeExplainer(rf_model)
shap_vals_nh = explainer.shap_values(X_nh)
if isinstance(shap_vals_nh, list):
    sv = shap_vals_nh[1]
elif shap_vals_nh.ndim == 3:
    sv = shap_vals_nh[:, :, 1]   # shape (n, features, classes) -> take CKD class
else:
    sv = shap_vals_nh

shap_df = pd.DataFrame(np.abs(sv), columns=features)

print("\n  --- GLOBAL SHAP ON NHANES (top-10 mean |SHAP|) ---")
nhanes_shap_mean = shap_df.mean().sort_values(ascending=False)
for feat, val in nhanes_shap_mean.head(10).items():
    print(f"  {feat:<28} {val:.4f}")

print("\n  --- SHAP BY AGE GROUP (mean |SHAP| for top-5 features) ---")
top5 = nhanes_shap_mean.head(5).index.tolist()
if 'Age' in df_nh.columns:
    for lo,hi,label in age_groups:
        mask = (df_nh['Age'].values>=lo)&(df_nh['Age'].values<hi)
        if mask.sum() > 50:
            group_shap = shap_df[mask][top5].mean()
            vals = '  '.join([f"{f.split('_')[0]}={v:.3f}" for f,v in group_shap.items()])
            print(f"  {label:<10}: {vals}")

# Save SHAP values
shap_out = pd.DataFrame(sv, columns=features)
shap_out['predicted_prob'] = y_prob_nh
shap_out['true_label'] = y_nh.values
if 'Sex' in df_nh.columns:       shap_out['Sex'] = df_nh['Sex'].values
if 'Age' in df_nh.columns:       shap_out['Age'] = df_nh['Age'].values
if 'Ethnicity' in df_nh.columns: shap_out['Ethnicity'] = df_nh['Ethnicity'].values
if 'Diabetes' in df_nh.columns:  shap_out['Diabetes'] = df_nh['Diabetes'].values
shap_out.to_csv('Paper3_NHANES_SHAP.csv', index=False)

# ── Final summary ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("PAPER 3 — COMPLETE RESULTS SUMMARY")
print("="*65)
print(f"\n  Overall NHANES AUC: {auc_ov:.4f} (95% CI [{ci_lo_ov:.4f}, {ci_hi_ov:.4f}])")
print(f"  Total subgroups analysed: {len(df_res)}")
print(f"  Subgroups with AUC disparity >0.05: "
      f"{sum(1 for cat in ['Sex','Age Group','Diabetes Status','Ethnicity'] for _ in [df_res[df_res.category==cat]['auc'].max()-df_res[df_res.category==cat]['auc'].min() if len(df_res[df_res.category==cat])>1 else 0] if _>0.05)}")
print(f"\n  Files saved:")
print(f"    Paper3_FairnessResults.csv  — subgroup AUC + CI + Brier for all groups")
print(f"    Paper3_NHANES_SHAP.csv      — per-patient SHAP values with demographics")
print("\nDone. Paste full output to Claude to begin writing Paper 3.")