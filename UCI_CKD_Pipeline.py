"""
UCI CKD Dataset — Complete Preprocessing & EDA Pipeline
PhD Research: Hybrid ML for Early CKD Detection
Run this script once the ARFF or CSV file is available.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD (supports both ARFF and CSV)
# ─────────────────────────────────────────────────────────────────────────────
def load_uci_ckd(filepath):
    """Load UCI CKD dataset from ARFF or CSV format."""
    if filepath.endswith('.arff'):
        # Manual parser — handles dirty whitespace values in ARFF
        cols, data_lines, in_data = [], [], False
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip('\r\n')
                if line.strip().upper().startswith('@ATTRIBUTE'):
                    parts = line.strip().split()
                    cols.append(parts[1].strip("'\""))
                elif line.strip().upper() == '@DATA':
                    in_data = True
                elif in_data and line.strip() and not line.startswith('%'):
                    vals = [v.strip() for v in line.strip().split(',')]
                    if len(vals) == len(cols):
                        data_lines.append(vals)
        df = pd.DataFrame(data_lines, columns=cols)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
    else:
        raise ValueError("Unsupported format. Use .arff or .csv")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — COLUMN STANDARDISATION
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_RENAME = {
    'age':   'Age',          'bp':    'Blood_Pressure',
    'sg':    'Specific_Gravity', 'al': 'Albumin',
    'su':    'Sugar',         'rbc':   'Red_Blood_Cells',
    'pc':    'Pus_Cell',      'pcc':   'Pus_Cell_Clumps',
    'ba':    'Bacteria',      'bgr':   'Blood_Glucose_Random',
    'bu':    'Blood_Urea',    'sc':    'Serum_Creatinine',
    'sod':   'Sodium',        'pot':   'Potassium',
    'hemo':  'Haemoglobin',   'pcv':   'Packed_Cell_Volume',
    'wc':    'WBC_Count',     'rc':    'RBC_Count',
    'htn':   'Hypertension',  'dm':    'Diabetes_Mellitus',
    'cad':   'Coronary_Artery_Disease',
    'appet': 'Appetite',      'pe':    'Pedal_Edema',
    'ane':   'Anemia',        'class': 'CKD'
}

def standardise_columns(df):
    df = df.rename(columns={k:v for k,v in COLUMN_RENAME.items() if k in df.columns})
    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')
    # Standardise column names that may vary by source
    rename_lower = {v.lower(): v for v in COLUMN_RENAME.values()}
    df = df.rename(columns=rename_lower)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CLEAN & ENCODE
# ─────────────────────────────────────────────────────────────────────────────
NOMINAL_COLS = ['Red_Blood_Cells','Pus_Cell','Pus_Cell_Clumps','Bacteria',
                'Hypertension','Diabetes_Mellitus','Coronary_Artery_Disease',
                'Appetite','Pedal_Edema','Anemia']

NUMERIC_COLS = ['Age','Blood_Pressure','Specific_Gravity','Albumin','Sugar',
                'Blood_Glucose_Random','Blood_Urea','Serum_Creatinine',
                'Sodium','Potassium','Haemoglobin','Packed_Cell_Volume',
                'WBC_Count','RBC_Count']

def clean_and_encode(df):
    # Replace common missing value indicators
    df = df.replace(['?', '\t?', ' ?', '??', '', 'nan', 'NaN'], np.nan)

    # Force numeric columns to numeric
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Standardise nominal values
    nominal_maps = {
        'Red_Blood_Cells':           {'normal':1, 'abnormal':0},
        'Pus_Cell':                  {'normal':1, 'abnormal':0},
        'Pus_Cell_Clumps':           {'notpresent':0, 'present':1},
        'Bacteria':                  {'notpresent':0, 'present':1},
        'Hypertension':              {'no':0, 'yes':1},
        'Diabetes_Mellitus':         {'no':0, 'yes':1, '\tno':0, '\tyes':1, ' yes':1, ' no':0},
        'Coronary_Artery_Disease':   {'no':0, 'yes':1, '\tno':0},
        'Appetite':                  {'good':1, 'poor':0},
        'Pedal_Edema':               {'no':0, 'yes':1},
        'Anemia':                    {'no':0, 'yes':1},
    }
    for col, mapping in nominal_maps.items():
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower().map(mapping)

    # Encode target
    if 'CKD' in df.columns:
        df['CKD'] = df['CKD'].str.strip().str.lower()
        df['CKD'] = df['CKD'].map({'ckd':1, 'notckd':0, 'not_ckd':0})
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df):
    # eGFR using CKD-EPI 2021 (assumes sex=unknown → use male formula as default)
    # For UCI dataset sex is not available — use simplified MDRD as fallback
    # MDRD: eGFR = 175 × Cr^(-1.154) × Age^(-0.203)
    if 'Serum_Creatinine' in df.columns and 'Age' in df.columns:
        df['eGFR_MDRD'] = (175 *
                           df['Serum_Creatinine'].pow(-1.154) *
                           df['Age'].pow(-0.203)).round(2)

    # BUN / Creatinine ratio
    if 'Blood_Urea' in df.columns and 'Serum_Creatinine' in df.columns:
        df['BUN_Cr_Ratio'] = (df['Blood_Urea'] / df['Serum_Creatinine']).round(2)

    # Haemoglobin categorisation (anaemia flag)
    if 'Haemoglobin' in df.columns:
        df['Anaemia_Flag'] = (df['Haemoglobin'] < 12).astype(float)

    # KDIGO staging from eGFR
    if 'eGFR_MDRD' in df.columns:
        def kdigo(e):
            if pd.isna(e): return np.nan
            if e >= 90: return 1
            if e >= 60: return 2
            if e >= 45: return 3
            if e >= 30: return 3  # 3a/3b
            if e >= 15: return 4
            return 5
        df['KDIGO_Stage'] = df['eGFR_MDRD'].apply(kdigo)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MISSING VALUE IMPUTATION (MICE)
# ─────────────────────────────────────────────────────────────────────────────
def impute_missing(df, target_col='CKD'):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor

    feature_cols = [c for c in df.columns if c != target_col]
    target = df[target_col].copy()
    features = df[feature_cols].copy()

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=42),
        max_iter=10,
        random_state=42,
        verbose=0
    )
    imputed = imputer.fit_transform(features)
    df_imputed = pd.DataFrame(imputed, columns=feature_cols)

    # Clip binary columns to [0,1] and round
    binary_cols = ['Red_Blood_Cells','Pus_Cell','Pus_Cell_Clumps','Bacteria',
                   'Hypertension','Diabetes_Mellitus','Coronary_Artery_Disease',
                   'Appetite','Pedal_Edema','Anemia','Anaemia_Flag']
    for col in binary_cols:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].clip(0,1).round()

    df_imputed[target_col] = target.values
    return df_imputed

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — EDA REPORT
# ─────────────────────────────────────────────────────────────────────────────
def eda_report(df, label='CKD'):
    print("="*60)
    print("UCI CKD DATASET — EDA REPORT")
    print("="*60)
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    if label in df.columns:
        vc = df[label].value_counts()
        print(f"\nClass distribution:")
        print(f"  CKD     (1): {vc.get(1,0):3d} ({vc.get(1,0)/len(df)*100:.1f}%)")
        print(f"  Non-CKD (0): {vc.get(0,0):3d} ({vc.get(0,0)/len(df)*100:.1f}%)")

    print(f"\nMissing values:")
    miss = df.isnull().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    for col, n in miss.items():
        print(f"  {col:30s}: {n:3d} ({n/len(df)*100:.1f}%)")

    print(f"\nNumeric feature statistics (CKD vs Non-CKD):")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != label]
    for col in num_cols[:8]:
        ckd_mean    = df[df[label]==1][col].mean()
        nonckd_mean = df[df[label]==0][col].mean()
        print(f"  {col:25s}: CKD={ckd_mean:7.2f}  NonCKD={nonckd_mean:7.2f}")
    print("="*60)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run when file is available
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'chronic_kidney_disease_full.arff'
    print(f"Loading: {filepath}")
    df = load_uci_ckd(filepath)
    df = standardise_columns(df)
    print(f"Raw shape: {df.shape}")
    eda_report(df)
    df = clean_and_encode(df)
    df = engineer_features(df)
    df_clean = impute_missing(df)
    eda_report(df_clean)
    df_clean.to_csv('UCI_CKD_Processed.csv', index=False)
    print("\nSaved: UCI_CKD_Processed.csv")
