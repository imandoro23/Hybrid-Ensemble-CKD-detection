"""
Recompute NHANES metrics at analytically derived threshold t=0.1192
Run from: C:/Users/imand/.vscode/PhD main Work/
"""
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              recall_score, precision_score, confusion_matrix,
                              brier_score_loss)

# ── Load UCI + rebuild ensemble ────────────────────────────────────────────────
print("Loading UCI data and rebuilding ensemble...")
df_uci = pd.read_csv('UCI_CKD_Processed.csv')
drop_c = [c for c in ['CKD_eGFR','CKD_ACR','KDIGO_Stage',
                       'Anaemia_Flag','Comorbidity_Score'] if c in df_uci.columns]
df_uci = df_uci.drop(columns=drop_c)

FEATURES = [
    'Specific_Gravity','Albumin','Haemoglobin','eGFR_MDRD',
    'Packed_Cell_Volume','Diabetes_Mellitus','Hypertension',
    'Blood_Glucose_Random','Serum_Creatinine','rbcc','BUN_Cr_Ratio',
    'Appetite','Blood_Urea','Red_Blood_Cells','Sugar',
    'Blood_Pressure','Sodium','Pedal_Edema','wbcc','Pus_Cell','Age'
]
features = [f for f in FEATURES if f in df_uci.columns]
X_uci = df_uci[features]
y_uci = df_uci['CKD'].astype(int)
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
    ('SVM', Pipeline([('sc', StandardScaler()),
             ('clf', CalibratedClassifierCV(SVC(kernel='rbf', C=2.0, gamma='scale',
             class_weight='balanced', random_state=42), cv=3))])),
    ('MLP', Pipeline([('sc', StandardScaler()),
             ('clf', MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu',
             solver='adam', alpha=0.001, learning_rate='adaptive',
             learning_rate_init=0.001, max_iter=500, early_stopping=True,
             validation_fraction=0.1, random_state=42))]))
]
n_base = len(base_defs)
oof = np.zeros((len(X_train), n_base))
for fi,(ti,vi) in enumerate(cv5.split(X_train, y_train)):
    for j,(nm,bm) in enumerate(base_defs):
        m = copy.deepcopy(bm)
        m.fit(X_train.iloc[ti], y_train.iloc[ti])
        oof[vi,j] = m.predict_proba(X_train.iloc[vi])[:,1]
    print(f"  OOF fold {fi+1}/5")

fitted_base = []
for nm,bm in base_defs:
    bm.fit(X_train, y_train)
    fitted_base.append((nm, bm))
meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta.fit(oof, y_train)

def ens_predict(X_in):
    if isinstance(X_in, np.ndarray):
        X_in = pd.DataFrame(X_in, columns=features)
    mf = np.column_stack([m.predict_proba(X_in)[:,1] for _,m in fitted_base])
    return meta.predict_proba(mf)[:,1]

# ── Load NHANES ────────────────────────────────────────────────────────────────
print("\nLoading NHANES data...")
df_nh = pd.read_csv('NHANES_CKD_Dataset.csv')
print(f"NHANES shape: {df_nh.shape}")

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

# ── Predict ────────────────────────────────────────────────────────────────────
print("Running ensemble predictions on NHANES...")
y_proba_nh = ens_predict(X_nh)

# ── Analytical threshold ───────────────────────────────────────────────────────
T_ANALYTICAL = 0.1192
print(f"\nAnalytically derived threshold: {T_ANALYTICAL}")

y_pred_nh = (y_proba_nh >= T_ANALYTICAL).astype(int)
TN,FP,FN,TP = confusion_matrix(y_nh, y_pred_nh).ravel()
sens = TP/(TP+FN) if TP+FN>0 else 0
spec = TN/(TN+FP) if TN+FP>0 else 0
ppv  = TP/(TP+FP) if TP+FP>0 else 0
npv  = TN/(TN+FN) if TN+FN>0 else 0

print("\n" + "="*55)
print("NHANES RESULTS AT ANALYTICAL THRESHOLD (t=0.1192)")
print("="*55)
print(f"AUC-ROC:     {roc_auc_score(y_nh, y_proba_nh):.4f}  (unchanged — threshold-independent)")
print(f"Accuracy:    {accuracy_score(y_nh, y_pred_nh)*100:.2f}%")
print(f"Sensitivity: {sens*100:.2f}%")
print(f"Specificity: {spec*100:.2f}%")
print(f"PPV:         {ppv*100:.2f}%")
print(f"NPV:         {npv*100:.2f}%")
print(f"F1-Score:    {f1_score(y_nh, y_pred_nh)*100:.2f}%")
print(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")

# ── Subgroup fairness at new threshold ────────────────────────────────────────
print("\n" + "="*55)
print("SUBGROUP FAIRNESS AT t=0.1192")
print("="*55)

def subgroup_metrics(mask, name):
    yt  = y_nh[mask]; ypr = y_proba_nh[mask]
    yp  = (ypr >= T_ANALYTICAL).astype(int)
    if len(yt)<10 or yt.nunique()<2: return
    TN2,FP2,FN2,TP2 = confusion_matrix(yt,yp).ravel()
    s2 = TP2/(TP2+FN2) if TP2+FN2>0 else 0
    sp2= TN2/(TN2+FP2) if TN2+FP2>0 else 0
    au2= roc_auc_score(yt,ypr)
    print(f"  {name:<25} n={mask.sum():5d}  AUC={au2:.4f}  "
          f"Sens={s2*100:.1f}%  Spec={sp2*100:.1f}%")

# By Sex
print("\nBy Sex:")
for code,name in [(0,'Male'),(1,'Female')]:
    if 'Sex' in df_nh.columns:
        subgroup_metrics(pd.Series(df_nh['Sex'].values==code), name)

# By Age
print("\nBy Age Group:")
if 'Age' in df_nh.columns:
    for lo,hi,label in [(18,44,'18-44'),(45,64,'45-64'),(65,200,'65+')]:
        mask = pd.Series((df_nh['Age'].values>=lo)&(df_nh['Age'].values<hi))
        subgroup_metrics(mask, label)

# By Diabetes
print("\nBy Diabetes:")
dm_col = 'Diabetes' if 'Diabetes' in df_nh.columns else None
if dm_col:
    for code,name in [(1,'Diabetic'),(0,'Non-Diabetic')]:
        subgroup_metrics(pd.Series(df_nh[dm_col].values==code), name)

# By Ethnicity
print("\nBy Ethnicity:")
eth_map = {1:'Mexican American',2:'Other Hispanic',3:'Non-Hispanic White',
           4:'Non-Hispanic Black',6:'Non-Hispanic Asian',7:'Other/Multi'}
if 'Ethnicity' in df_nh.columns:
    for code,name in eth_map.items():
        mask = pd.Series(df_nh['Ethnicity'].values==code)
        if mask.sum()>50:
            subgroup_metrics(mask, name)

print("\nDone. Update Table 10 and Table 13 with these values.")