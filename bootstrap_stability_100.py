"""
Bootstrap SHAP stability — increase to 100 resamples + compute CI
Run from: C:/Users/imand/.vscode/PhD main Work/
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations

np.random.seed(42)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading UCI data...")
df = pd.read_csv('UCI_CKD_Processed.csv')
drop_c = [c for c in ['CKD_eGFR','CKD_ACR','KDIGO_Stage',
                       'Anaemia_Flag','Comorbidity_Score'] if c in df.columns]
df = df.drop(columns=drop_c)

FEATURES = [
    'Specific_Gravity','Albumin','Haemoglobin','eGFR_MDRD',
    'Packed_Cell_Volume','Diabetes_Mellitus','Hypertension',
    'Blood_Glucose_Random','Serum_Creatinine','rbcc','BUN_Cr_Ratio',
    'Appetite','Blood_Urea','Red_Blood_Cells','Sugar',
    'Blood_Pressure','Sodium','Pedal_Edema','wbcc','Pus_Cell','Age'
]
features = [f for f in FEATURES if f in df.columns]
X = df[features]; y = df['CKD'].astype(int)
X_train,_,y_train,_ = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

n_feat = len(features)
print(f"Training set: {len(X_train)} patients, {n_feat} features")

# ── 100 bootstrap resamples ────────────────────────────────────────────────────
print("\nRunning 100 bootstrap resamples (may take ~5 mins)...")
B = 100
boot_imps = np.zeros((B, n_feat))
rng = np.random.RandomState(42)

for b in range(B):
    idx = rng.choice(len(X_train), len(X_train), replace=True)
    Xb, yb = X_train.iloc[idx], y_train.iloc[idx]
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                 min_samples_leaf=2, class_weight='balanced',
                                 random_state=b, n_jobs=-1)
    rf.fit(Xb, yb)
    boot_imps[b] = rf.feature_importances_
    if (b+1) % 10 == 0:
        print(f"  Resample {b+1}/100")

# ── Top-10 sets per resample ───────────────────────────────────────────────────
top10_sets = [set(np.argsort(boot_imps[b])[-10:]) for b in range(B)]

# ── Jaccard@top-10 — all pairs ─────────────────────────────────────────────────
all_pairs = list(combinations(range(B), 2))
all_jaccards = []
for i,j in all_pairs:
    inter = len(top10_sets[i] & top10_sets[j])
    union = len(top10_sets[i] | top10_sets[j])
    all_jaccards.append(inter/union if union>0 else 0)

mean_j = np.mean(all_jaccards)

# ── Jackknife CI on mean Jaccard ───────────────────────────────────────────────
# Leave-one-resample-out jackknife
jack_means = []
for leave_out in range(B):
    remaining = [b for b in range(B) if b != leave_out]
    pairs_sub = [(i,j) for i,j in combinations(remaining,2)]
    jacs = []
    for i,j in pairs_sub:
        inter = len(top10_sets[i] & top10_sets[j])
        union = len(top10_sets[i] | top10_sets[j])
        jacs.append(inter/union if union>0 else 0)
    jack_means.append(np.mean(jacs))

jack_arr = np.array(jack_means)
jack_bias = (B-1)*(np.mean(jack_arr) - mean_j)
jack_se   = np.sqrt((B-1)/B * np.sum((jack_arr - np.mean(jack_arr))**2))
ci_lo = mean_j - 1.96*jack_se
ci_hi = mean_j + 1.96*jack_se

# ── Top-10 frequency per feature ──────────────────────────────────────────────
top10_freq = [sum(1 for s in top10_sets if j in s) for j in range(n_feat)]

# ── Print results ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"BOOTSTRAP SHAP STABILITY — 100 RESAMPLES")
print(f"{'='*60}")
print(f"Mean Jaccard@top-10:  {mean_j:.4f}")
print(f"Jackknife 95% CI:     [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Jackknife SE:         {jack_se:.5f}")
print(f"Total pairs:          {len(all_pairs)}")
print(f"\n{'─'*60}")
print(f"{'Feature':<28} {'Freq':>6}  {'Stability'}")
print(f"{'─'*60}")
stab_df = pd.DataFrame({'Feature':features,'Freq':top10_freq,
                         'MeanImp':boot_imps.mean(axis=0)})
stab_df = stab_df.sort_values('Freq',ascending=False)
for _,row in stab_df.iterrows():
    f    = int(row['Freq'])
    stab = ('VERY STABLE' if f>=90 else 'STABLE' if f>=80
            else 'MODERATE' if f>=50 else 'UNSTABLE')
    bar  = '█'*int(f/5)
    print(f"  {row['Feature']:<26} {f:>5}/100  {bar}  {stab}")

print(f"\nInterpretation: Jaccard {mean_j:.4f} (95% CI [{ci_lo:.4f},{ci_hi:.4f}])")
print(f"= {mean_j*100:.1f}% of top-10 feature list consistent across resamples")
print("\nDone.")