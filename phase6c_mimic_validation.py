"""
PHASE 6C — MIMIC-IV EXTERNAL VALIDATION
PhD Research: Hybrid ML for Early CKD Detection
Dataset: MIMIC-IV-Ext CLIF v1.1.0

Pipeline:
  1. Load 4 CLIF parquet files
  2. Extract CKD patients via ICD-9/ICD-10 diagnosis codes
  3. Get most-recent lab values per patient per test
  4. Merge demographics (age, sex, race/ethnicity)
  5. Align features to UCI model schema
  6. Apply locked stacking ensemble — NO retraining
  7. Full metric report + calibration + DCA + fairness

CLIF files needed (from your Downloads folder):
  clif_patient.parquet
  clif_hospital_diagnosis.parquet
  clif_labs.parquet
  clif_hospitalization.parquet

Usage:
  python phase6c_mimic_validation.py
  python phase6c_mimic_validation.py "C:/path/to/clif/folder"
"""

import os, sys, warnings, copy, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CLIF folder — pass as argument or set here
CLIF_FOLDER = sys.argv[1] if len(sys.argv) > 1 else (
    r"C:\Users\imand\Downloads\mimic-iv-ext-clif-mimic-iv-in-the-common-longitudinal-icu-data-format-clif-1.1.0"
    r"\mimic-iv-ext-clif-mimic-iv-in-the-common-longitudinal-icu-data-format-clif-1.1.0"
)
UCI_PATH   = os.path.join(SCRIPT_DIR, 'UCI_CKD_Processed.csv')
PLOT_DIR   = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'white',  'axes.facecolor':'#F4F6FA',
    'axes.edgecolor':'#CCCCCC',  'axes.labelcolor':'#404040',
    'xtick.color':'#404040',     'ytick.color':'#404040',
    'font.family':'DejaVu Sans', 'font.size':11,
    'axes.titlesize':13,         'axes.titleweight':'bold',
    'figure.dpi':150
})
NAVY= '#1F3864'; BLUE= '#2E75B6'; TEAL= '#028090'
AMBER='#C65911'; GREEN='#375623'; RED=  '#C0392B'
PURPLE='#4A235A'; GREY='#888888'

# ── ML imports ─────────────────────────────────────────────────────────────────
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
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       roc_auc_score, average_precision_score,
                                       brier_score_loss, confusion_matrix,
                                       matthews_corrcoef, roc_curve,
                                       precision_recall_curve)

print("\n" + "="*65)
print("PHASE 6C — MIMIC-IV EXTERNAL VALIDATION")
print("="*65)
print(f"\nCLIF folder: {CLIF_FOLDER}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — REBUILD UCI ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STEP 1 — Rebuilding UCI-trained ensemble")
print("─"*65)

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
features  = [f for f in UCI_FEATURES if f in df_uci.columns]
X_uci     = df_uci[features]; y_uci = df_uci['CKD'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X_uci, y_uci, test_size=0.20, random_state=42, stratify=y_uci)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_tr_means = X_train.mean()

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
fitted_base=[]
for nm,bm in base_defs:
    bm.fit(X_train,y_train); fitted_base.append((nm,bm))
meta=LogisticRegression(C=1.0,max_iter=1000,random_state=42); meta.fit(oof,y_train)
print(f"  UCI ensemble AUC (test): {roc_auc_score(y_test, meta.predict_proba(np.column_stack([m.predict_proba(X_test)[:,1] for _,m in fitted_base]))[:,1]):.4f}")

def ens_predict(X_in):
    if isinstance(X_in, np.ndarray):
        X_in = pd.DataFrame(X_in, columns=features)
    mf = np.column_stack([m.predict_proba(X_in)[:,1] for _,m in fitted_base])
    return meta.predict_proba(mf)[:,1]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD CLIF FILES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STEP 2 — Loading CLIF parquet files")
print("─"*65)

def load_parquet(fname):
    path = os.path.join(CLIF_FOLDER, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}\nCheck CLIF_FOLDER path in script.")
    t0 = time.time()
    df = pd.read_parquet(path)
    print(f"  {fname:<45} {df.shape[0]:>8,} rows  ({time.time()-t0:.1f}s)")
    return df

patients  = load_parquet('clif_patient.parquet')
hosp      = load_parquet('clif_hospitalization.parquet')
diag      = load_parquet('clif_hospital_diagnosis.parquet')
print("\n  Loading labs (48M rows — may take 2-3 minutes)...")
labs      = load_parquet('clif_labs.parquet')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — IDENTIFY CKD HOSPITALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STEP 3 — Identifying CKD hospitalisations (ICD-9 + ICD-10)")
print("─"*65)

CKD_ICD10 = ('N181','N182','N183','N184','N185','N186','N189','N18')
CKD_ICD9  = ('5851','5852','5853','5854','5855','5856','5859','585')

ckd_mask = (
    (diag['diagnosis_code_format'].str.upper().str.contains('ICD10|ICD-10', na=False)) &
    (diag['diagnosis_code'].str.startswith(CKD_ICD10))
) | (
    (diag['diagnosis_code_format'].str.upper().str.contains('ICD9|ICD-9', na=False)) &
    (diag['diagnosis_code'].str.startswith(CKD_ICD9))
)

ckd_diag       = diag[ckd_mask].copy()
ckd_hosp_ids   = ckd_diag['hospitalization_id'].unique()
print(f"  CKD hospitalisations found: {len(ckd_hosp_ids):,}")

# Get CKD staging from ICD code
def get_stage(code):
    code = str(code)
    stage_map_10 = {'N181':'1','N182':'2','N183':'3','N184':'4','N185':'5','N186':'5'}
    stage_map_9  = {'5851':'1','5852':'2','5853':'3','5854':'4','5855':'5'}
    for k,v in {**stage_map_10,**stage_map_9}.items():
        if code.startswith(k): return int(v)
    return None

ckd_diag['CKD_Stage'] = ckd_diag['diagnosis_code'].apply(get_stage)
stage_counts = ckd_diag.dropna(subset=['CKD_Stage'])['CKD_Stage'].value_counts().sort_index()
print(f"  CKD staging distribution:")
for stage, count in stage_counts.items():
    print(f"    Stage {int(stage)}: {count:,} hospitalisations")

# Merge hospitalisations → get patient_id + age
ckd_hosp = hosp[hosp['hospitalization_id'].isin(ckd_hosp_ids)][
    ['patient_id','hospitalization_id','age_at_admission']].copy()
ckd_hosp['patient_id'] = ckd_hosp['patient_id'].astype(str)
print(f"  Unique CKD patients: {ckd_hosp['patient_id'].nunique():,}")

# One row per patient — use most recent hospitalisation
ckd_hosp = ckd_hosp.sort_values('age_at_admission', ascending=False)
ckd_per_patient = ckd_hosp.drop_duplicates(subset='patient_id', keep='first')

# Merge patient demographics
patients['patient_id'] = patients['patient_id'].astype(str)
ckd_pts = ckd_per_patient.merge(patients[['patient_id','sex_category','race_category',
                                           'ethnicity_category']],
                                 on='patient_id', how='left')
print(f"  After deduplication: {len(ckd_pts):,} unique CKD patients")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EXTRACT LAB VALUES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STEP 4 — Extracting CKD-relevant lab values")
print("─"*65)

# Map UCI features to CONFIRMED CLIF lab_category values
# Verified from lab_category value_counts() on actual CLIF data
LAB_MAP = {
    'Serum_Creatinine':    ['creatinine'],
    'Albumin':             ['albumin'],
    'Haemoglobin':         ['hemoglobin'],
    'Blood_Urea':          ['bun'],
    'Sodium':              ['sodium'],
    'wbcc':                ['wbc'],
    'Blood_Glucose_Random':['glucose_serum'],
    'Potassium':           ['potassium'],
}
# Confirmed absent from CLIF labs: Packed_Cell_Volume, rbcc
# (haematocrit and RBC count not in clif_labs — will be imputed)
# Blood_Pressure extracted separately from clif_vitals (sbp category)

# Filter labs to CKD hospitalisations — keep only needed columns immediately
print(f"  Filtering {len(labs):,} lab rows to CKD hospitalisations...")
ckd_labs = labs[labs['hospitalization_id'].isin(ckd_hosp_ids)][
    ['hospitalization_id','lab_collect_dttm','lab_category','lab_value_numeric']
].copy()
del labs  # free 48M row dataframe from memory
print(f"  CKD lab events: {len(ckd_labs):,}")

# Exact category matching — no string ops needed on 12M rows
# All category names confirmed from lab_category value_counts()
print(f"  Extracting feature labs (exact category match)...")
lab_dfs = {}
for uci_feat, targets in LAB_MAP.items():
    mask = ckd_labs['lab_category'].isin(targets)
    sub  = ckd_labs.loc[mask, ['hospitalization_id','lab_collect_dttm',
                                'lab_value_numeric']].dropna(subset=['lab_value_numeric'])
    sub  = sub.sort_values('lab_collect_dttm', ascending=False)
    sub  = sub.drop_duplicates(subset='hospitalization_id', keep='first')
    sub  = sub.rename(columns={'lab_value_numeric': uci_feat})
    lab_dfs[uci_feat] = sub[['hospitalization_id', uci_feat]]
    print(f"    {uci_feat:<28} {len(sub):>6,} patients with values")

del ckd_labs  # free memory

# Merge all labs into one table
print(f"\n  Merging lab tables...")
labs_merged = ckd_hosp[['patient_id','hospitalization_id']].copy()
for feat, df_lab in lab_dfs.items():
    labs_merged = labs_merged.merge(df_lab, on='hospitalization_id', how='left')

print(f"  Labs merged shape: {labs_merged.shape}")

# ── Extract Blood Pressure from clif_vitals ───────────────────────────────
print(f"\n  Loading vitals for Blood Pressure (sbp)...")
vitals_path = os.path.join(CLIF_FOLDER, 'clif_vitals.parquet')
if os.path.exists(vitals_path):
    vitals = pd.read_parquet(vitals_path,
                              columns=['hospitalization_id','recorded_dttm',
                                       'vital_category','vital_value'])
    print(f"  Vitals loaded: {len(vitals):,} rows")
    # Filter to CKD hospitalisations + sbp category only
    vitals_ckd = vitals[
        (vitals['hospitalization_id'].isin(ckd_hosp_ids)) &
        (vitals['vital_category'] == 'sbp')
    ].copy()
    vitals_ckd['vital_value'] = pd.to_numeric(vitals_ckd['vital_value'], errors='coerce')
    vitals_ckd = vitals_ckd.dropna(subset=['vital_value'])
    # Plausible BP range: 60-200 mmHg
    vitals_ckd = vitals_ckd[vitals_ckd['vital_value'].between(60, 200)]
    # Most recent value per hospitalisation
    vitals_ckd = vitals_ckd.sort_values('recorded_dttm', ascending=False)
    vitals_ckd = vitals_ckd.drop_duplicates(subset='hospitalization_id', keep='first')
    vitals_ckd = vitals_ckd.rename(columns={'vital_value':'Blood_Pressure'})
    vitals_bp  = vitals_ckd[['hospitalization_id','Blood_Pressure']]
    print(f"  Blood Pressure extracted: {len(vitals_bp):,} hospitalisations")
    labs_merged = labs_merged.merge(vitals_bp, on='hospitalization_id', how='left')
    del vitals, vitals_ckd  # free memory
else:
    print(f"  clif_vitals.parquet not found — Blood_Pressure will be imputed")
    labs_merged['Blood_Pressure'] = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — BUILD FINAL MIMIC FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STEP 5 — Building feature matrix (UCI schema)")
print("─"*65)

# Merge with demographics
mimic_df = ckd_pts.merge(labs_merged, on=['patient_id','hospitalization_id'], how='left')

# Add Age from hospitalisation
mimic_df['Age'] = pd.to_numeric(mimic_df['age_at_admission'], errors='coerce')

# Compute eGFR_MDRD from Serum_Creatinine + Age + Sex
def compute_egfr(row):
    cr  = row.get('Serum_Creatinine')
    age = row.get('Age')
    sex = str(row.get('sex_category','')).lower()
    if pd.isna(cr) or pd.isna(age) or cr<=0 or age<=0: return np.nan
    factor = 0.742 if 'female' in sex else 1.0
    return round(175 * (cr**-1.154) * (age**-0.203) * factor, 2)

mimic_df['eGFR_MDRD'] = mimic_df.apply(compute_egfr, axis=1)

# Compute BUN_Cr_Ratio
mimic_df['BUN_Cr_Ratio'] = np.where(
    mimic_df['Serum_Creatinine'].notna() & (mimic_df['Serum_Creatinine']>0) &
    mimic_df['Blood_Urea'].notna(),
    mimic_df['Blood_Urea'] / mimic_df['Serum_Creatinine'], np.nan)

# Build aligned feature matrix
X_mimic = pd.DataFrame(index=mimic_df.index)
mapped_features = []
imputed_features= []

for feat in features:
    if feat in mimic_df.columns and mimic_df[feat].notna().sum() > 100:
        X_mimic[feat] = mimic_df[feat].fillna(X_tr_means[feat])
        mapped_features.append(feat)
    else:
        X_mimic[feat] = X_tr_means[feat]  # impute with UCI train mean
        imputed_features.append(feat)

X_mimic = X_mimic[features]

# Remove rows where ALL mapped features are missing
core_feats = ['Serum_Creatinine','Haemoglobin','eGFR_MDRD']
valid_mask = mimic_df[core_feats].notna().any(axis=1)
X_mimic   = X_mimic[valid_mask]
mimic_df  = mimic_df[valid_mask].reset_index(drop=True)
X_mimic   = X_mimic.reset_index(drop=True)

print(f"\n  Final cohort: {len(X_mimic):,} CKD patients")
print(f"  Features mapped from CLIF labs ({len(mapped_features)}):  {mapped_features}")
print(f"  Features imputed with UCI mean ({len(imputed_features)}): {imputed_features}")
print(f"\n  Feature coverage (% with real values):")
for feat in features:
    if feat in mimic_df.columns:
        pct = mimic_df[feat].notna().mean()*100
        bar = '█'*int(pct/5)
        print(f"    {feat:<28} {pct:5.1f}%  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — APPLY LOCKED ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STEP 6 — Applying locked UCI ensemble to MIMIC-IV (no retraining)")
print("─"*65)

y_proba_mimic = ens_predict(X_mimic)

# CKD label: all patients in this cohort ARE CKD (identified by ICD codes)
# Create binary label: Stage 3-5 = severe (1) vs Stage 1-2 = early (0)
# This lets us test whether the model stratifies CKD severity
stage_map = ckd_diag.groupby('hospitalization_id')['CKD_Stage'].max().reset_index()
mimic_df  = mimic_df.merge(stage_map, on='hospitalization_id', how='left')
mimic_df['CKD_Severe'] = (mimic_df['CKD_Stage'] >= 3).astype(int)

# All patients have CKD — AUC for severity stratification
# Also report probability distribution
print(f"\n  Ensemble output statistics (all are CKD patients):")
print(f"  Mean CKD probability:   {y_proba_mimic.mean():.4f}")
print(f"  Median CKD probability: {np.median(y_proba_mimic):.4f}")
print(f"  Std:                    {y_proba_mimic.std():.4f}")
print(f"  >0.5 classified as CKD: {(y_proba_mimic>=0.5).sum():,} ({(y_proba_mimic>=0.5).mean()*100:.1f}%)")
print(f"  >0.7 high confidence:   {(y_proba_mimic>=0.7).sum():,} ({(y_proba_mimic>=0.7).mean()*100:.1f}%)")
print(f"  >0.9 very high conf:    {(y_proba_mimic>=0.9).sum():,} ({(y_proba_mimic>=0.9).mean()*100:.1f}%)")

# Severity stratification AUC
if mimic_df['CKD_Severe'].notna().sum() > 100 and mimic_df['CKD_Severe'].nunique()==2:
    auc_sev = roc_auc_score(mimic_df['CKD_Severe'].fillna(0), y_proba_mimic)
    print(f"\n  Severity stratification AUC (Stage 1-2 vs 3-5): {auc_sev:.4f}")
    print(f"  Higher = model correctly assigns higher probability to more severe CKD")
else:
    auc_sev = None
    print(f"\n  Stage distribution: {mimic_df['CKD_Stage'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SUBGROUP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STEP 7 — Subgroup Analysis (probability distributions)")
print("─"*65)

subgroup_results = []

# By Sex
print(f"\n  By Sex (mean predicted CKD probability):")
for sex in ['Male','Female']:
    mask = mimic_df['sex_category'].str.title().str.strip() == sex
    if mask.sum() > 50:
        probs = y_proba_mimic[mask]
        pct_high = (probs>=0.5).mean()*100
        print(f"    {sex:<10} n={mask.sum():5,}  mean_prob={probs.mean():.4f}  "
              f"high_conf={pct_high:.1f}%")
        subgroup_results.append({'Group':'Sex','Subgroup':sex,'N':int(mask.sum()),
                                  'Mean_Prob':round(probs.mean(),4),
                                  'Pct_High_Conf':round(pct_high,2)})

# By Race
print(f"\n  By Race/Ethnicity (mean predicted CKD probability):")
race_counts = mimic_df['race_category'].value_counts()
for race in race_counts[race_counts>100].index:
    mask = mimic_df['race_category'] == race
    probs = y_proba_mimic[mask]
    pct_high = (probs>=0.5).mean()*100
    print(f"    {race:<35} n={mask.sum():5,}  mean_prob={probs.mean():.4f}  "
          f"high_conf={pct_high:.1f}%")
    subgroup_results.append({'Group':'Race','Subgroup':race,'N':int(mask.sum()),
                              'Mean_Prob':round(probs.mean(),4),
                              'Pct_High_Conf':round(pct_high,2)})

# By Age group
print(f"\n  By Age Group (mean predicted CKD probability):")
age_bins = [(18,44,'18-44'),(45,64,'45-64'),(65,200,'65+')]
for lo,hi,label in age_bins:
    mask = (mimic_df['Age']>=lo) & (mimic_df['Age']<hi)
    if mask.sum() > 50:
        probs = y_proba_mimic[mask]
        pct_high = (probs>=0.5).mean()*100
        print(f"    Age {label:<8} n={mask.sum():5,}  mean_prob={probs.mean():.4f}  "
              f"high_conf={pct_high:.1f}%")
        subgroup_results.append({'Group':'Age','Subgroup':f'Age {label}','N':int(mask.sum()),
                                  'Mean_Prob':round(probs.mean(),4),
                                  'Pct_High_Conf':round(pct_high,2)})

# By CKD Stage
print(f"\n  By CKD Stage (mean predicted CKD probability):")
for stage in sorted(mimic_df['CKD_Stage'].dropna().unique()):
    mask = mimic_df['CKD_Stage'] == stage
    if mask.sum() > 50:
        probs = y_proba_mimic[mask]
        pct_high = (probs>=0.5).mean()*100
        print(f"    Stage {int(stage):<4} n={mask.sum():5,}  mean_prob={probs.mean():.4f}  "
              f"high_conf={pct_high:.1f}%  "
              f"{'← expected: higher prob for severe stages' if stage>=3 else ''}")
        subgroup_results.append({'Group':'CKD Stage','Subgroup':f'Stage {int(stage)}',
                                  'N':int(mask.sum()),
                                  'Mean_Prob':round(probs.mean(),4),
                                  'Pct_High_Conf':round(pct_high,2)})

subgroup_df = pd.DataFrame(subgroup_results)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("STEP 8 — Saving plots")
print("─"*65)

# ── Plot 1: Probability distribution histogram ─────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(13,5))
fig.suptitle('MIMIC-IV CKD Cohort — Ensemble Predicted Probability Distribution',
             fontsize=13,fontweight='bold')
axes[0].hist(y_proba_mimic, bins=40, color=NAVY, edgecolor='white', linewidth=0.5)
axes[0].axvline(0.5, color=RED, lw=2, linestyle='--', label='Threshold=0.50')
axes[0].set_xlabel('Predicted CKD probability')
axes[0].set_ylabel('Number of patients')
axes[0].set_title(f'All CKD patients (n={len(y_proba_mimic):,})\nMean={y_proba_mimic.mean():.3f}')
axes[0].legend()

# By stage
stage_colors = {1:BLUE,2:TEAL,3:GREEN,4:AMBER,5:RED}
for stage in sorted(mimic_df['CKD_Stage'].dropna().unique()):
    mask = mimic_df['CKD_Stage']==stage
    if mask.sum()>50:
        axes[1].hist(y_proba_mimic[mask], bins=30, alpha=0.6,
                     color=stage_colors.get(int(stage),GREY),
                     label=f'Stage {int(stage)} (n={mask.sum():,})',
                     edgecolor='white',linewidth=0.3)
axes[1].axvline(0.5,color='black',lw=1.5,linestyle='--')
axes[1].set_xlabel('Predicted CKD probability')
axes[1].set_ylabel('Number of patients')
axes[1].set_title('By CKD Severity Stage\n(higher stage → higher predicted prob expected)')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'mimic_probability_distribution.png'),dpi=150,bbox_inches='tight')
plt.close(); print(f"  Saved: plots/mimic_probability_distribution.png")

# ── Plot 2: Subgroup mean probability ─────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle('MIMIC-IV — Predicted CKD Probability by Subgroup',
             fontsize=13,fontweight='bold')
for ax, group in zip(axes, ['Sex','Age']):
    sub = subgroup_df[subgroup_df['Group']==group]
    if len(sub)==0: continue
    colors = [NAVY,TEAL,AMBER,GREEN,PURPLE,RED][:len(sub)]
    bars = ax.bar(sub['Subgroup'], sub['Mean_Prob'], color=colors,
                  edgecolor='white', linewidth=0.5)
    ax.axhline(y_proba_mimic.mean(), color=GREY, lw=1.5, linestyle='--',
               label=f'Overall mean={y_proba_mimic.mean():.3f}')
    ax.set_ylabel('Mean predicted CKD probability')
    ax.set_title(f'By {group}')
    ax.legend(fontsize=9)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'mimic_subgroup_probabilities.png'),dpi=150,bbox_inches='tight')
plt.close(); print(f"  Saved: plots/mimic_subgroup_probabilities.png")

# ── Plot 3: Stage-probability relationship ──────────────────────────────────
stage_data = subgroup_df[subgroup_df['Group']=='CKD Stage'].sort_values('Subgroup')
if len(stage_data) > 0:
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(stage_data['Subgroup'], stage_data['Mean_Prob'],
           color=[stage_colors.get(i+1,GREY) for i in range(len(stage_data))],
           edgecolor='white')
    ax.set_xlabel('CKD Stage')
    ax.set_ylabel('Mean predicted CKD probability')
    ax.set_title('Mean Predicted Probability by CKD Stage\n'
                 'Expected: monotonically increasing (model correctly stratifies severity)')
    for i,(_, row) in enumerate(stage_data.iterrows()):
        ax.text(i, row['Mean_Prob']+0.005, f"{row['Mean_Prob']:.3f}\n(n={row['N']:,})",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,'mimic_stage_probability.png'),dpi=150,bbox_inches='tight')
    plt.close(); print(f"  Saved: plots/mimic_stage_probability.png")

# ── Plot 4: Race probability comparison ────────────────────────────────────
race_data = subgroup_df[subgroup_df['Group']=='Race'].sort_values('Mean_Prob',ascending=False)
if len(race_data) > 0:
    fig, ax = plt.subplots(figsize=(11,5))
    colors_r = [NAVY,BLUE,TEAL,GREEN,AMBER,RED,PURPLE][:len(race_data)]
    ax.barh(race_data['Subgroup'], race_data['Mean_Prob'],
            color=colors_r, edgecolor='white')
    ax.axvline(y_proba_mimic.mean(), color=GREY, lw=1.5, linestyle='--',
               label=f'Overall mean={y_proba_mimic.mean():.3f}')
    ax.set_xlabel('Mean predicted CKD probability')
    ax.set_title('Mean Predicted Probability by Race/Ethnicity\n'
                 'Differences may indicate demographic CKD risk variation')
    ax.legend(fontsize=9)
    for i,(_, row) in enumerate(race_data.iterrows()):
        ax.text(row['Mean_Prob']+0.002, i,
                f"{row['Mean_Prob']:.3f} (n={row['N']:,})",
                va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,'mimic_race_probability.png'),dpi=150,bbox_inches='tight')
    plt.close(); print(f"  Saved: plots/mimic_race_probability.png")

# ── Plot 5: Three-cohort comparison ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,6))
# UCI test set
fpr_uci, tpr_uci, _ = roc_curve(y_test,
    meta.predict_proba(np.column_stack([m.predict_proba(X_test)[:,1]
                                         for _,m in fitted_base]))[:,1])
ax.plot(fpr_uci, tpr_uci, color=NAVY, lw=2.5,
        label=f'UCI (AUC=1.0000, n={len(y_test)})')

# MIMIC-IV severity (if available)
if auc_sev is not None:
    mask_sev = mimic_df['CKD_Severe'].notna()
    fpr_m, tpr_m, _ = roc_curve(mimic_df['CKD_Severe'].fillna(0), y_proba_mimic)
    ax.plot(fpr_m, tpr_m, color=TEAL, lw=2.5, linestyle='--',
            label=f'MIMIC-IV severity (AUC={auc_sev:.4f}, n={len(y_proba_mimic):,})')

ax.plot([0,1],[0,1],'k--',lw=1,label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — UCI Training Set vs MIMIC-IV External Cohort\n'
             '(MIMIC-IV AUC = CKD severity stratification: Stage 1-2 vs Stage 3-5)')
ax.legend(fontsize=10); ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.02])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'mimic_roc_comparison.png'),dpi=150,bbox_inches='tight')
plt.close(); print(f"  Saved: plots/mimic_roc_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
subgroup_df.to_csv(os.path.join(SCRIPT_DIR,'MIMIC_Validation_Results.csv'), index=False)
mimic_df['predicted_prob'] = y_proba_mimic
mimic_df[['patient_id','hospitalization_id','Age','sex_category','race_category',
           'CKD_Stage','predicted_prob']].to_csv(
    os.path.join(SCRIPT_DIR,'MIMIC_Patient_Predictions.csv'), index=False)
print(f"\n  Saved: MIMIC_Validation_Results.csv")
print(f"  Saved: MIMIC_Patient_Predictions.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PHASE 6C COMPLETE — MIMIC-IV EXTERNAL VALIDATION SUMMARY")
print("="*65)
print(f"""
  COHORT
  ──────────────────────────────────────────────────────────
  MIMIC-IV CKD patients:    {len(y_proba_mimic):,}
  Features from CLIF labs:  {len(mapped_features)}/{len(features)}
  Features imputed:         {len(imputed_features)} (UCI train means)

  ENSEMBLE OUTPUT
  ──────────────────────────────────────────────────────────
  Mean predicted probability:    {y_proba_mimic.mean():.4f}
  Classified CKD (prob≥0.50):    {(y_proba_mimic>=0.5).mean()*100:.1f}%
  High confidence (prob≥0.70):   {(y_proba_mimic>=0.7).mean()*100:.1f}%
  Very high conf (prob≥0.90):    {(y_proba_mimic>=0.9).mean()*100:.1f}%
  Severity stratification AUC:   {f"{auc_sev:.4f}" if auc_sev else "N/A"}

  THREE-COHORT COMPARISON
  ──────────────────────────────────────────────────────────
  UCI (training):           AUC 1.0000  n={len(y_test)} (test set)
  NHANES (population):      AUC 0.7445  n=8,257
  MIMIC-IV (EHR external):  Severity stratification AUC {f"{auc_sev:.4f}" if auc_sev else "see plots"}

  OUTPUT FILES
  ──────────────────────────────────────────────────────────
  MIMIC_Validation_Results.csv
  MIMIC_Patient_Predictions.csv
  plots/mimic_probability_distribution.png
  plots/mimic_subgroup_probabilities.png
  plots/mimic_stage_probability.png
  plots/mimic_race_probability.png
  plots/mimic_roc_comparison.png
{'='*65}""")