"""
MIMIC-IV Sensitivity Analysis — Supplementary Table S4
=======================================================
Response to Reviewer Issue III (JMCMS rejection):
  "the model contains creatinine-derived eGFR, so the reported AUC of 0.7029
   cannot establish genuine emergent staging capability without demonstrating
   that the signal is not mechanically inherited."

This script re-runs the MIMIC-IV staging analysis TWICE:
  1. PRIMARY:     All 21 features including eGFR_MDRD (reproduces AUC=0.7029)
  2. SENSITIVITY: eGFR_MDRD replaced by UCI training mean (removes eGFR signal)

If AUC in run 2 remains significantly above 0.50, staging signal is genuinely
emergent from other biomarkers (creatinine, haemoglobin, albumin, etc.).

Run from: C:/Users/imand/.vscode/PhD main Work/
Prerequisites: Same CLIF parquet files used in phase6c_mimic_validation.py

Usage:
    & ".venv/Scripts/python.exe" mimic_sensitivity_s4.py
    & ".venv/Scripts/python.exe" mimic_sensitivity_s4.py "C:/path/to/clif/folder"
"""

import os, sys, copy, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.model_selection   import StratifiedKFold, train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.ensemble          import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       HistGradientBoostingClassifier)
from sklearn.neural_network    import MLPClassifier
from sklearn.metrics           import roc_auc_score, roc_curve
from scipy                     import stats

os.makedirs('plots', exist_ok=True)

plt.rcParams.update({
    'font.family':'Times New Roman','font.size':11,
    'axes.titlesize':12,'figure.dpi':150,
    'savefig.dpi':300,'savefig.bbox':'tight',
    'savefig.facecolor':'white',
})
NAVY='#1F3864'; BLUE='#2E75B6'; RED='#C00000'; GREY='#404040'
TEAL='#028090'; AMBER='#C65911'

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CLIF_FOLDER = sys.argv[1] if len(sys.argv) > 1 else (
    r"C:\Users\imand\Downloads\mimic-iv-ext-clif-mimic-iv-in-the-common-longitudinal-icu-data-format-clif-1.1.0"
    r"\mimic-iv-ext-clif-mimic-iv-in-the-common-longitudinal-icu-data-format-clif-1.1.0"
)
UCI_PATH = os.path.join(SCRIPT_DIR, 'UCI_CKD_Processed.csv')

# ── DeLong CI ─────────────────────────────────────────────────────────────────
def delong_ci(y_true, y_score, alpha=0.05):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    pos = y_score[y_true==1]; neg = y_score[y_true==0]
    n1, n0 = len(pos), len(neg)
    auc = np.mean([np.mean(p>neg)+0.5*np.mean(p==neg) for p in pos])
    v10 = np.array([np.mean(p>neg)+0.5*np.mean(p==neg) for p in pos])
    v01 = np.array([np.mean(q<pos)+0.5*np.mean(q==pos) for q in neg])
    se  = np.sqrt(np.var(v10,ddof=1)/n1 + np.var(v01,ddof=1)/n0)
    z   = stats.norm.ppf(1-alpha/2)
    return auc, max(0,auc-z*se), min(1,auc+z*se)

def delong_z_test(y_true, s1, s2):
    a1,l1,h1 = delong_ci(y_true,s1)
    a2,l2,h2 = delong_ci(y_true,s2)
    se1=(h1-l1)/(2*1.96); se2=(h2-l2)/(2*1.96)
    r = np.corrcoef(s1[y_true==1],s2[y_true==1])[0,1]
    se_d = np.sqrt(se1**2+se2**2-2*r*se1*se2)
    z = (a1-a2)/se_d if se_d>0 else 0
    p = 2*(1-stats.norm.cdf(abs(z)))
    return a1,a2,z,p

print("="*65)
print("SUPPLEMENTARY TABLE S4 — eGFR_MDRD SENSITIVITY ANALYSIS")
print("="*65)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Rebuild UCI ensemble (identical to phase6c)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Rebuilding UCI ensemble...")

df_uci = pd.read_csv(UCI_PATH)
drop_c = [c for c in ['CKD_eGFR','CKD_ACR','KDIGO_Stage',
                       'Anaemia_Flag','Comorbidity_Score'] if c in df_uci.columns]
df_uci = df_uci.drop(columns=drop_c)

UCI_FEATURES = [
    'Specific_Gravity','Albumin','Haemoglobin','eGFR_MDRD',
    'Packed_Cell_Volume','Diabetes_Mellitus','Hypertension',
    'Blood_Glucose_Random','Serum_Creatinine','rbcc','BUN_Cr_Ratio',
    'Appetite','Blood_Urea','Red_Blood_Cells','Sugar',
    'Blood_Pressure','Sodium','Pedal_Edema','wbcc','Pus_Cell','Age'
]
features = [f for f in UCI_FEATURES if f in df_uci.columns]
X_uci = df_uci[features]; y_uci = df_uci['CKD'].astype(int)
X_train,X_test,y_train,y_test = train_test_split(
    X_uci,y_uci,test_size=0.20,random_state=42,stratify=y_uci)
cv5 = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
X_tr_means = X_train.mean()

egfr_idx   = features.index('eGFR_MDRD')
egfr_mean  = X_tr_means['eGFR_MDRD']
print(f"  eGFR_MDRD UCI training mean (sensitivity imputation value): {egfr_mean:.4f}")

base_defs = [
    ('RF',  RandomForestClassifier(n_estimators=300,max_features='sqrt',
             min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1)),
    ('GB',  GradientBoostingClassifier(n_estimators=200,learning_rate=0.08,
             max_depth=4,subsample=0.8,min_samples_leaf=3,random_state=42)),
    ('HGB', HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,
             max_depth=6,l2_regularization=0.5,min_samples_leaf=10,random_state=42)),
    ('SVM', Pipeline([('sc',StandardScaler()),
             ('clf',CalibratedClassifierCV(SVC(kernel='rbf',C=2.0,gamma='scale',
             class_weight='balanced',random_state=42),cv=3))])),
    ('MLP', Pipeline([('sc',StandardScaler()),
             ('clf',MLPClassifier(hidden_layer_sizes=(128,64,32),activation='relu',
             solver='adam',alpha=0.001,learning_rate='adaptive',learning_rate_init=0.001,
             max_iter=500,early_stopping=True,validation_fraction=0.1,random_state=42))])),
]
n_base = len(base_defs)
oof    = np.zeros((len(X_train),n_base))
for fi,(ti,vi) in enumerate(cv5.split(X_train,y_train)):
    for j,(nm,bm) in enumerate(base_defs):
        m=copy.deepcopy(bm); m.fit(X_train.iloc[ti],y_train.iloc[ti])
        oof[vi,j]=m.predict_proba(X_train.iloc[vi])[:,1]
    print(f"  OOF fold {fi+1}/5")
fitted_base=[]
for nm,bm in base_defs:
    bm.fit(X_train,y_train); fitted_base.append((nm,bm))
meta=LogisticRegression(C=1.0,max_iter=1000,random_state=42); meta.fit(oof,y_train)

def ens_predict(X_in):
    if isinstance(X_in,np.ndarray):
        X_in=pd.DataFrame(X_in,columns=features)
    mf=np.column_stack([m.predict_proba(X_in)[:,1] for _,m in fitted_base])
    return meta.predict_proba(mf)[:,1]

uci_auc=roc_auc_score(y_test,ens_predict(X_test))
print(f"  UCI test AUC: {uci_auc:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load CLIF files (same as phase6c)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Loading CLIF parquet files...")
import time

def load_parquet(fname):
    path=os.path.join(CLIF_FOLDER,fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    t0=time.time()
    df=pd.read_parquet(path)
    print(f"  {fname:<45} {df.shape[0]:>8,} rows  ({time.time()-t0:.1f}s)")
    return df

patients = load_parquet('clif_patient.parquet')
hosp     = load_parquet('clif_hospitalization.parquet')
diag     = load_parquet('clif_hospital_diagnosis.parquet')
print("  Loading labs (may take 2-3 minutes)...")
labs     = load_parquet('clif_labs.parquet')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Identify CKD patients + staging (same as phase6c)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Identifying CKD patients and staging...")

CKD_ICD10=('N181','N182','N183','N184','N185','N186','N189','N18')
CKD_ICD9 =('5851','5852','5853','5854','5855','5856','5859','585')
ckd_mask=(
    (diag['diagnosis_code_format'].str.upper().str.contains('ICD10|ICD-10',na=False)) &
    (diag['diagnosis_code'].str.startswith(CKD_ICD10))
)|(
    (diag['diagnosis_code_format'].str.upper().str.contains('ICD9|ICD-9',na=False)) &
    (diag['diagnosis_code'].str.startswith(CKD_ICD9))
)
ckd_diag=diag[ckd_mask].copy()
ckd_hosp_ids=ckd_diag['hospitalization_id'].unique()

def get_stage(code):
    code=str(code)
    for k,v in {'N181':1,'N182':2,'N183':3,'N184':4,'N185':5,'N186':5,
                '5851':1,'5852':2,'5853':3,'5854':4,'5855':5}.items():
        if code.startswith(k): return v
    return None

ckd_diag['CKD_Stage']=ckd_diag['diagnosis_code'].apply(get_stage)
ckd_hosp=hosp[hosp['hospitalization_id'].isin(ckd_hosp_ids)][
    ['patient_id','hospitalization_id','age_at_admission']].copy()
ckd_hosp['patient_id']=ckd_hosp['patient_id'].astype(str)
ckd_hosp=ckd_hosp.sort_values('age_at_admission',ascending=False)
ckd_per_patient=ckd_hosp.drop_duplicates(subset='patient_id',keep='first')
patients['patient_id']=patients['patient_id'].astype(str)
ckd_pts=ckd_per_patient.merge(
    patients[['patient_id','sex_category','race_category']],
    on='patient_id',how='left')
print(f"  CKD patients: {len(ckd_pts):,}")

# Labs
LAB_MAP={'Serum_Creatinine':['creatinine'],'Albumin':['albumin'],
         'Haemoglobin':['hemoglobin'],'Blood_Urea':['bun'],
         'Sodium':['sodium'],'wbcc':['wbc'],
         'Blood_Glucose_Random':['glucose_serum'],'Potassium':['potassium']}
ckd_labs=labs[labs['hospitalization_id'].isin(ckd_hosp_ids)][
    ['hospitalization_id','lab_collect_dttm','lab_category','lab_value_numeric']].copy()
del labs
lab_dfs={}
for uci_feat,targets in LAB_MAP.items():
    sub=ckd_labs[ckd_labs['lab_category'].isin(targets)].copy()
    sub=sub.dropna(subset=['lab_value_numeric'])
    sub=sub.sort_values('lab_collect_dttm',ascending=False)
    sub=sub.drop_duplicates(subset='hospitalization_id',keep='first')
    sub=sub.rename(columns={'lab_value_numeric':uci_feat})
    lab_dfs[uci_feat]=sub[['hospitalization_id',uci_feat]]
del ckd_labs
labs_merged=ckd_hosp[['patient_id','hospitalization_id']].copy()
for feat,df_lab in lab_dfs.items():
    labs_merged=labs_merged.merge(df_lab,on='hospitalization_id',how='left')

vitals_path=os.path.join(CLIF_FOLDER,'clif_vitals.parquet')
if os.path.exists(vitals_path):
    vitals=pd.read_parquet(vitals_path,
        columns=['hospitalization_id','recorded_dttm','vital_category','vital_value'])
    vitals_ckd=vitals[(vitals['hospitalization_id'].isin(ckd_hosp_ids)) &
                       (vitals['vital_category']=='sbp')].copy()
    vitals_ckd['vital_value']=pd.to_numeric(vitals_ckd['vital_value'],errors='coerce')
    vitals_ckd=vitals_ckd[vitals_ckd['vital_value'].between(60,200)]
    vitals_ckd=vitals_ckd.sort_values('recorded_dttm',ascending=False)
    vitals_ckd=vitals_ckd.drop_duplicates(subset='hospitalization_id',keep='first')
    vitals_ckd=vitals_ckd.rename(columns={'vital_value':'Blood_Pressure'})
    labs_merged=labs_merged.merge(
        vitals_ckd[['hospitalization_id','Blood_Pressure']],
        on='hospitalization_id',how='left')
    del vitals,vitals_ckd
else:
    labs_merged['Blood_Pressure']=np.nan

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build feature matrix × 2 (with and without eGFR_MDRD)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Building feature matrices (PRIMARY and SENSITIVITY)...")

mimic_df=ckd_pts.merge(labs_merged,on=['patient_id','hospitalization_id'],how='left')
mimic_df['Age']=pd.to_numeric(mimic_df['age_at_admission'],errors='coerce')

def compute_egfr(row):
    cr=row.get('Serum_Creatinine'); age=row.get('Age')
    sex=str(row.get('sex_category','')).lower()
    if pd.isna(cr) or pd.isna(age) or cr<=0 or age<=0: return np.nan
    return round(175*(cr**-1.154)*(age**-0.203)*(0.742 if 'female' in sex else 1.0),2)

mimic_df['eGFR_MDRD']=mimic_df.apply(compute_egfr,axis=1)
mimic_df['BUN_Cr_Ratio']=np.where(
    mimic_df['Serum_Creatinine'].notna()&(mimic_df['Serum_Creatinine']>0)&
    mimic_df['Blood_Urea'].notna(),
    mimic_df['Blood_Urea']/mimic_df['Serum_Creatinine'],np.nan)

# PRIMARY: all 21 features (eGFR_MDRD from creatinine)
X_full=pd.DataFrame(index=mimic_df.index)
for feat in features:
    if feat in mimic_df.columns and mimic_df[feat].notna().sum()>100:
        X_full[feat]=mimic_df[feat].fillna(X_tr_means[feat])
    else:
        X_full[feat]=X_tr_means[feat]
X_full=X_full[features]

# SENSITIVITY: eGFR_MDRD replaced by UCI training mean
X_noe=X_full.copy()
X_noe['eGFR_MDRD']=egfr_mean
print(f"  eGFR_MDRD set to UCI train mean={egfr_mean:.4f} in sensitivity matrix")

# Remove patients missing all core features
core_feats=['Serum_Creatinine','Haemoglobin','eGFR_MDRD']
valid_mask=mimic_df[core_feats].notna().any(axis=1)
X_full=X_full[valid_mask].reset_index(drop=True)
X_noe =X_noe[valid_mask].reset_index(drop=True)
mimic_df=mimic_df[valid_mask].reset_index(drop=True)
print(f"  Valid patients: {len(X_full):,}")

# Add stage labels
stage_map=ckd_diag.groupby('hospitalization_id')['CKD_Stage'].max().reset_index()
mimic_df=mimic_df.merge(stage_map,on='hospitalization_id',how='left')
mimic_df['CKD_Severe']=(mimic_df['CKD_Stage']>=3).astype(float)
staged_mask=mimic_df['CKD_Severe'].notna()
X_staged_full=X_full[staged_mask]
X_staged_noe =X_noe[staged_mask]
y_staged=mimic_df[staged_mask]['CKD_Severe'].values.astype(int)

print(f"  Staged patients: n={len(y_staged):,}")
print(f"  Stage 1-2 (early):    n={int((y_staged==0).sum()):,}")
print(f"  Stage 3-5 (advanced): n={int((y_staged==1).sum()):,}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Predict, compute AUCs, generate Table S4
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Running predictions and computing Supplementary Table S4...")

prob_full = ens_predict(X_staged_full)
prob_noe  = ens_predict(X_staged_noe)

auc_f,lo_f,hi_f = delong_ci(y_staged,prob_full)
auc_n,lo_n,hi_n = delong_ci(y_staged,prob_noe)
_,_,z_stat,p_val = delong_z_test(y_staged,prob_full,prob_noe)

delta     = auc_f - auc_n
pct_chg   = (delta/auc_f)*100

# Interpretation
if lo_n > 0.50:
    interp = (f"AUC={auc_n:.4f} [95% CI {lo_n:.4f}, {hi_n:.4f}] remains significantly "
              f"above chance after removing eGFR_MDRD (lower CI {lo_n:.4f} > 0.50). "
              f"The staging signal is not solely mechanically inherited; other biomarkers "
              f"(creatinine, haemoglobin, albumin) provide independent discriminative signal.")
elif lo_n > 0.55:
    interp = (f"AUC={auc_n:.4f} remains above chance but is reduced from {auc_f:.4f}, "
              f"suggesting partial mechanical inheritance from eGFR. The claim has been "
              f"reframed to acknowledge this caveat.")
else:
    interp = (f"AUC={auc_n:.4f} approaches chance level, indicating that the staging "
              f"discrimination is predominantly mechanically inherited from eGFR. "
              f"The staged severity claim should be substantially reframed.")

print(f"\n  PRIMARY  (eGFR_MDRD computed): AUC={auc_f:.4f} [{lo_f:.4f}, {hi_f:.4f}]")
print(f"  SENSITIVITY (eGFR_MDRD=mean):  AUC={auc_n:.4f} [{lo_n:.4f}, {hi_n:.4f}]")
print(f"  ΔAUC: {delta:+.4f} ({pct_chg:.1f}% change)")
print(f"  DeLong Z={z_stat:.3f}, p={p_val:.4f}")
print(f"\n  INTERPRETATION: {interp}")

# ── Figure S4 ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14,5))
gs  = GridSpec(1,2,figure=fig,wspace=0.35)

# Table panel
ax_t = fig.add_subplot(gs[0])
ax_t.axis('off')
rows = [
    ['Configuration','n (staged)','AUC-ROC','95% CI','Note'],
    ['Primary\n(all 21 features, eGFR computed)',
     f"{len(y_staged):,}",f"{auc_f:.4f}",f"[{lo_f:.4f}, {hi_f:.4f}]",'Reported result'],
    ['Sensitivity\n(eGFR_MDRD = UCI train mean)',
     f"{len(y_staged):,}",f"{auc_n:.4f}",f"[{lo_n:.4f}, {hi_n:.4f}]",
     f'ΔAUC={delta:+.4f}'],
    ['DeLong Z-test','','','',f'Z={z_stat:.3f}, p={p_val:.4f}'],
]
cols=[0.33,0.12,0.12,0.25,0.18]
bgs=[NAVY,'#D6E4F0','#E2EFDA','#FFF2CC']
tcs=['white',GREY,GREY,GREY]
ys=[0.90,0.65,0.38,0.12]
for ri,(row,y,bg,tc) in enumerate(zip(rows,ys,bgs,tcs)):
    x=0.0
    for ci,(cell,w) in enumerate(zip(row,cols)):
        ax_t.add_patch(plt.Rectangle((x,y-0.18),w,0.24 if ri==0 else 0.28,
            facecolor=bg,edgecolor='white',linewidth=1.5,
            transform=ax_t.transAxes,clip_on=False))
        ax_t.text(x+w/2,y-0.05,cell,ha='center',va='center',
            fontsize=8 if ri==0 else 8.5,fontweight='bold' if ri==0 else 'normal',
            color='white' if ri==0 else GREY,
            transform=ax_t.transAxes,multialignment='center')
        x+=w
ax_t.set_title('Supplementary Table S4\nMIMIC-IV Staging Sensitivity Analysis',
    fontsize=10,color=NAVY,fontweight='bold',pad=8)

# ROC panel
ax_r = fig.add_subplot(gs[1])
fpr_f,tpr_f,_ = roc_curve(y_staged,prob_full)
fpr_n,tpr_n,_ = roc_curve(y_staged,prob_noe)
ax_r.plot(fpr_f,tpr_f,color=BLUE,lw=2,
    label=f'Primary — eGFR computed\nAUC={auc_f:.4f} [{lo_f:.4f}, {hi_f:.4f}]')
ax_r.plot(fpr_n,tpr_n,color=RED,lw=2,ls='--',
    label=f'Sensitivity — eGFR=UCI mean\nAUC={auc_n:.4f} [{lo_n:.4f}, {hi_n:.4f}]')
ax_r.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Chance (0.50)')
ax_r.fill_between(fpr_f,tpr_f,alpha=0.08,color=BLUE)
ax_r.fill_between(fpr_n,tpr_n,alpha=0.08,color=RED)
ax_r.set_xlabel('False Positive Rate',color=GREY)
ax_r.set_ylabel('True Positive Rate',color=GREY)
ax_r.set_title('Supplementary Figure S4\nROC: Stage 3-5 (Severe) vs Stage 1-2 (Early)',
    color=NAVY,fontweight='bold')
ax_r.legend(fontsize=8.5,loc='lower right',framealpha=0.95,edgecolor=GREY)
ax_r.spines['top'].set_visible(False); ax_r.spines['right'].set_visible(False)
ax_r.annotate(f'ΔAUC={delta:+.4f}\n({pct_chg:.1f}% change)\nZ={z_stat:.3f}, p={p_val:.4f}',
    xy=(0.52,0.28),xycoords='axes fraction',fontsize=8.5,color=GREY,
    bbox=dict(boxstyle='round,pad=0.4',facecolor='white',edgecolor=GREY,alpha=0.9))

plt.savefig('plots/supp_fig_s4_roc.png')
plt.close()

print("\n" + "="*65)
print("SUPPLEMENTARY TABLE S4 — COPY THESE VALUES INTO MANUSCRIPT")
print("="*65)
print(f"\n  n staged:              {len(y_staged):,}")
print(f"  Stage 1-2 (early):     {int((y_staged==0).sum()):,}")
print(f"  Stage 3-5 (advanced):  {int((y_staged==1).sum()):,}")
print()
print(f"  PRIMARY (all 21 features, eGFR_MDRD computed from creatinine):")
print(f"    AUC = {auc_f:.4f}  [95% CI {lo_f:.4f}, {hi_f:.4f}]")
print()
print(f"  SENSITIVITY (eGFR_MDRD replaced by UCI training mean = {egfr_mean:.4f}):")
print(f"    AUC = {auc_n:.4f}  [95% CI {lo_n:.4f}, {hi_n:.4f}]")
print()
print(f"  ΔAUC = {delta:+.4f}  ({pct_chg:.1f}% relative change)")
print(f"  DeLong Z = {z_stat:.3f},  p = {p_val:.4f}")
print(f"  Lower CI above 0.50: {'YES ✅' if lo_n>0.50 else 'NO ❌'}")
print()
print(f"  INTERPRETATION:")
print(f"  {interp}")
print()
print("  Files saved:")
print("    plots/supp_fig_s4_roc.png  — dual ROC curve for Supplementary Fig S4")
print("="*65)
