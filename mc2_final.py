"""
MC2 Fix v3 — SHAP-LIME agreement across full test set (n=80)
Fix: use labels=[0,1] and fall back to available label if 1 not present
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
from sklearn.metrics import roc_auc_score
import shap
import lime.lime_tabular
warnings.filterwarnings('ignore')

FEATURES = [
    'Specific_Gravity','Albumin','Haemoglobin','eGFR_MDRD',
    'Packed_Cell_Volume','Diabetes_Mellitus','Hypertension',
    'Blood_Glucose_Random','Serum_Creatinine','rbcc','BUN_Cr_Ratio',
    'Appetite','Blood_Urea','Red_Blood_Cells','Sugar',
    'Blood_Pressure','Sodium','Pedal_Edema','wbcc','Pus_Cell','Age'
]

print("Loading UCI data...")
df = pd.read_csv('UCI_CKD_Processed.csv')
drop_c = [c for c in ['CKD_eGFR','CKD_ACR','KDIGO_Stage',
                       'Anaemia_Flag','Comorbidity_Score'] if c in df.columns]
df = df.drop(columns=drop_c)
features = [f for f in FEATURES if f in df.columns]
X = df[features]; y = df['CKD'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train:{len(X_train)}  Test:{len(X_test)}  Features:{len(features)}")

print("\nRebuilding ensemble...")
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

def ens_1d(X_in):
    if isinstance(X_in, np.ndarray):
        X_in = pd.DataFrame(X_in, columns=features)
    mf = np.column_stack([m.predict_proba(X_in)[:,1] for _,m in fitted_base])
    return meta.predict_proba(mf)[:,1]

def ens_2d(X_arr):
    p = ens_1d(pd.DataFrame(X_arr, columns=features))
    return np.column_stack([1-p, p])

y_prob_test = ens_1d(X_test)
print(f"  Test AUC: {roc_auc_score(y_test, y_prob_test):.4f}")

print("\nSetting up explainers...")
np.random.seed(42)
bg = X_train.iloc[np.random.choice(len(X_train), 50, replace=False)].values
k_explainer = shap.KernelExplainer(ens_1d, bg)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=features,
    class_names=['Non-CKD','CKD'], mode='classification',
    discretize_continuous=False, random_state=42)

print(f"\nComputing SHAP+LIME for all {len(X_test)} test patients...")
print("Progress every 5 patients\n")

results = []
X_arr = X_test.values; y_arr = y_test.values

for i in range(len(X_test)):
    if (i+1) % 5 == 0:
        print(f"  Patient {i+1}/{len(X_test)}  prob={y_prob_test[i]:.3f}")
    pt    = X_arr[i:i+1]
    prob  = float(y_prob_test[i])
    label = int(y_arr[i])

    sv_raw = k_explainer.shap_values(pt, nsamples=200, silent=True)
    if isinstance(sv_raw, list):
        sv = sv_raw[1][0]
    else:
        sv = np.array(sv_raw).flatten()[:len(features)]
    abs_shap = np.abs(sv)

    lime_exp = lime_explainer.explain_instance(
        pt[0], ens_2d, num_features=len(features),
        num_samples=500, labels=[0, 1])
    avail     = lime_exp.available_labels()
    use_label = 1 if 1 in avail else avail[0]
    lime_raw  = dict(lime_exp.as_list(label=use_label))
    sign      = 1 if use_label == 1 else -1
    lime_dict = {k: sign*v for k,v in lime_raw.items()}
    abs_lime  = np.array([abs(lime_dict.get(f, 0.0)) for f in features])

    rho, pval = stats.spearmanr(abs_shap, abs_lime)

    top_idx  = int(np.argmax(abs_shap))
    top_feat = features[top_idx]
    top_sv   = float(sv[top_idx])
    top_lc   = lime_dict.get(top_feat, np.nan)
    dir_agree = bool((top_sv>0)==(top_lc>0)) if not np.isnan(top_lc) else None

    results.append(dict(
        patient_idx=i, prob=prob, true_label=label,
        rho=float(rho), pval=float(pval), significant=bool(pval<0.05),
        top_feat=top_feat, top_sv=top_sv,
        top_lc=float(top_lc) if not np.isnan(top_lc) else np.nan,
        dir_agree=dir_agree, lime_label_used=use_label
    ))

df_res = pd.DataFrame(results)
df_res.to_csv('shap_lime_agreement_n80.csv', index=False)

rho_v  = df_res['rho'].values
prob_v = df_res['prob'].values
conf_v = np.abs(prob_v - 0.5)

print("\n" + "="*65)
print("SHAP-LIME AGREEMENT — FULL TEST SET (n=80)")
print("="*65)
print(f"\nMean rho:  {np.mean(rho_v):.4f}  SD={np.std(rho_v):.4f}")
print(f"Median rho:{np.median(rho_v):.4f}")
print(f"Range:     [{np.min(rho_v):.4f}, {np.max(rho_v):.4f}]")
print(f"Significant (p<0.05): {df_res['significant'].sum()}/80")

r_conf, p_conf = stats.spearmanr(conf_v, rho_v)
r_prob, p_prob = stats.spearmanr(prob_v, rho_v)
print(f"\n--- PRIMARY HYPOTHESIS ---")
print(f"rho vs |prob-0.5| confidence: r={r_conf:.4f}  p={p_conf:.4f}")
print(f"rho vs predicted probability: r={r_prob:.4f}  p={p_prob:.4f}")
verdict = ("CONFIRMED" if r_conf>0 and p_conf<0.05 else
           "TREND p<0.10" if r_conf>0 and p_conf<0.10 else "NOT CONFIRMED")
print(f"Result: {verdict}")

print(f"\n--- rho BY CONFIDENCE QUINTILE ---")
df_res['conf'] = conf_v
qlabels = ['Q1 least','Q2','Q3','Q4','Q5 most']
df_res['conf_q'] = pd.qcut(df_res['conf'], q=5, labels=qlabels)
for ql in qlabels:
    s = df_res[df_res['conf_q']==ql]['rho']
    print(f"  {ql}: n={len(s):2d}  mean rho={s.mean():.4f}  SD={s.std():.4f}")

print(f"\n--- rho BY PROBABILITY QUINTILE ---")
df_res['prob_q'] = pd.qcut(df_res['prob'], q=5,
    labels=['0-20pct','20-40pct','40-60pct','60-80pct','80-100pct'])
for pl in ['0-20pct','20-40pct','40-60pct','60-80pct','80-100pct']:
    s = df_res[df_res['prob_q']==pl]['rho']
    print(f"  prob {pl}: n={len(s):2d}  mean rho={s.mean():.4f}")

ckd_r = df_res[df_res['true_label']==1]['rho']
non_r = df_res[df_res['true_label']==0]['rho']
mwu   = stats.mannwhitneyu(ckd_r, non_r, alternative='two-sided')
print(f"\n--- BY TRUE LABEL ---")
print(f"CKD    (n={len(ckd_r)}): mean rho={ckd_r.mean():.4f}  SD={ckd_r.std():.4f}")
print(f"Non-CKD(n={len(non_r)}): mean rho={non_r.mean():.4f}  SD={non_r.std():.4f}")
print(f"Mann-Whitney: U={mwu.statistic:.1f}  p={mwu.pvalue:.4f}")

hi = df_res[conf_v>0.35]; lo = df_res[conf_v<=0.35]
print(f"\n--- HIGH vs LOW CONFIDENCE ---")
print(f"High |p-0.5|>0.35 (n={len(hi)}): mean rho={hi['rho'].mean():.4f}")
print(f"Low  |p-0.5|<=0.35 (n={len(lo)}): mean rho={lo['rho'].mean():.4f}")
if len(hi)>1 and len(lo)>1:
    mwu2 = stats.mannwhitneyu(hi['rho'], lo['rho'], alternative='greater')
    print(f"Mann-Whitney (high>low): p={mwu2.pvalue:.4f}")

dc  = df_res['dir_agree'].dropna()
inv = df_res[df_res['dir_agree']==False]
print(f"\n--- TOP-FEATURE DIRECTION AGREEMENT ---")
print(f"Direction AGREE:  {dc.sum()}/{len(dc)} ({dc.mean()*100:.1f}%)")
print(f"Direction INVERT: {(~dc).sum()}/{len(dc)} ({(~dc).mean()*100:.1f}%)")
if len(inv):
    print("Top inversion features:")
    for feat,cnt in inv['top_feat'].value_counts().head(5).items():
        sub = inv[inv['top_feat']==feat]
        print(f"  {feat}: {cnt} patients  mean prob={sub['prob'].mean():.3f}")

print(f"\nLIME label-1 used: {(df_res['lime_label_used']==1).sum()}/80")
print("Saved: shap_lime_agreement_n80.csv")
print("Done.")
