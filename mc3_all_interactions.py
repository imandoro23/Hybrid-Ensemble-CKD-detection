"""
MC3 Fix: All 210 pairwise interaction effects + bootstrap 95% CI
Runtime: ~15-25 minutes
C:/Users/imand/.vscode/PhD main Work/mc3_all_interactions.py
"""
import pandas as pd
import numpy as np
import copy, warnings
from itertools import combinations
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')

FEATURES = [
    'Specific_Gravity','Albumin','Haemoglobin','eGFR_MDRD',
    'Packed_Cell_Volume','Diabetes_Mellitus','Hypertension',
    'Blood_Glucose_Random','Serum_Creatinine','rbcc','BUN_Cr_Ratio',
    'Appetite','Blood_Urea','Red_Blood_Cells','Sugar',
    'Blood_Pressure','Sodium','Pedal_Edema','wbcc','Pus_Cell','Age'
]

# ── Load and split ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('UCI_CKD_Processed.csv')
drop_c = [c for c in ['CKD_eGFR','CKD_ACR','KDIGO_Stage',
                       'Anaemia_Flag','Comorbidity_Score'] if c in df.columns]
df = df.drop(columns=drop_c)
features = [f for f in FEATURES if f in df.columns]
X = df[features]; y = df['CKD'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
X_tr_means = X_train.mean()
print(f"Train:{len(X_train)}  Test:{len(X_test)}  Features:{len(features)}")
n_feat = len(features)

# ── Rebuild ensemble ───────────────────────────────────────────────────────────
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

def predict_batch(X_arr):
    """Predict for a 2D numpy array, returns 1D probability array."""
    Xdf = pd.DataFrame(X_arr, columns=features)
    mf  = np.column_stack([m.predict_proba(Xdf)[:,1] for _,m in fitted_base])
    return meta.predict_proba(mf)[:,1]

# ── Sample 60 test instances ───────────────────────────────────────────────────
np.random.seed(42)
n_inst = 60
idx60  = np.random.choice(len(X_test), n_inst, replace=False)
X60    = X_test.values[idx60]          # shape (60, 21)
means  = X_tr_means.values             # shape (21,)

print(f"\nPrecomputing predictions for all 210 pairs + 21 marginals...")
print(f"This requires {1 + n_feat + int(n_feat*(n_feat-1)/2)} batch calls of {n_inst} instances each")

# ── Precompute: original predictions ──────────────────────────────────────────
f_orig = predict_batch(X60)            # shape (60,)

# ── Precompute: single-feature marginals  f(x̄ᵢ, everything else original) ────
print("  Computing single-feature marginals (21 calls)...")
f_single = np.zeros((n_feat, n_inst))  # f_single[i,k] = f with feature i → mean
for i in range(n_feat):
    X_mod = X60.copy()
    X_mod[:, i] = means[i]
    f_single[i] = predict_batch(X_mod)
    if (i+1) % 5 == 0: print(f"    Feature {i+1}/{n_feat}")

# ── Precompute: double-feature marginals ──────────────────────────────────────
print("  Computing double-feature marginals (210 calls)...")
pairs = list(combinations(range(n_feat), 2))   # 210 pairs
f_double = {}                                   # {(i,j): array(60,)}
for k,(i,j) in enumerate(pairs):
    X_mod = X60.copy()
    X_mod[:, i] = means[i]
    X_mod[:, j] = means[j]
    f_double[(i,j)] = predict_batch(X_mod)
    if (k+1) % 42 == 0: print(f"    Pair {k+1}/{len(pairs)}")

print("  All predictions cached.\n")

# ── Compute φᵢⱼ for all 210 pairs ─────────────────────────────────────────────
def compute_phi(i, j, instance_mask=None):
    """Compute mean finite-difference interaction for pair (i,j).
    instance_mask: boolean array of length n_inst (for bootstrap)."""
    if instance_mask is None:
        fo  = f_orig
        fsi = f_single[i]
        fsj = f_single[j]
        fd  = f_double[(i,j)] if (i,j) in f_double else f_double[(j,i)]
    else:
        fo  = f_orig[instance_mask]
        fsi = f_single[i][instance_mask]
        fsj = f_single[j][instance_mask]
        fd  = (f_double[(i,j)] if (i,j) in f_double else f_double[(j,i)])[instance_mask]
    return np.mean(fo - fsi - fsj + fd)

print("Computing φᵢⱼ for all 210 pairs...")
phi_vals = {(i,j): compute_phi(i,j) for i,j in pairs}

# ── Bootstrap 95% CI (1000 resamples, no extra model calls needed) ────────────
print("Computing bootstrap CIs (1000 resamples, no extra model calls)...")
B = 1000
rng = np.random.RandomState(42)
boot_phis = {pair: np.zeros(B) for pair in pairs}

for b in range(B):
    idx_b = rng.choice(n_inst, n_inst, replace=True)
    mask  = np.zeros(n_inst, dtype=bool)
    # Use boolean indexing via unique indices
    X_b   = np.zeros(n_inst, dtype=bool)
    idx_u, counts = np.unique(idx_b, return_counts=True)
    for pair in pairs:
        i,j  = pair
        vals = []
        for inst_idx, cnt in zip(idx_u, counts):
            fo_k  = f_orig[inst_idx]
            fsi_k = f_single[i][inst_idx]
            fsj_k = f_single[j][inst_idx]
            fd_k  = (f_double[pair] if pair in f_double else f_double[(j,i)])[inst_idx]
            term  = fo_k - fsi_k - fsj_k + fd_k
            vals.extend([term]*cnt)
        boot_phis[pair][b] = np.mean(vals)
    if (b+1) % 200 == 0: print(f"  Bootstrap {b+1}/{B}")

print("\nComputing CIs and p-values...")
results = []
for pair in pairs:
    i,j   = pair
    phi   = phi_vals[pair]
    bs    = boot_phis[pair]
    ci_lo = np.percentile(bs, 2.5)
    ci_hi = np.percentile(bs, 97.5)
    # Empirical p-value: proportion of bootstraps with phi ≤ 0
    # (one-sided: is phi significantly > 0?)
    p_val = np.mean(bs <= 0)
    results.append(dict(
        feat_i=features[i], feat_j=features[j],
        phi=phi, ci_lo=ci_lo, ci_hi=ci_hi,
        p_val=p_val,
        sig_05=bool(ci_lo > 0),   # 95% CI excludes zero
        sig_01=bool(p_val < 0.01),
    ))

df_res = pd.DataFrame(results).sort_values('phi', ascending=False)
df_res.to_csv('interaction_all210.csv', index=False)

# ── Print Results ──────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ALL 210 PAIRWISE INTERACTION EFFECTS — SORTED BY φᵢⱼ")
print("="*80)
print(f"{'Feature i':<22} {'Feature j':<22} {'φᵢⱼ':>8} {'95% CI':>22} {'p(>0)':>8} {'Sig?':>6}")
print("-"*80)
for _,r in df_res.iterrows():
    sig = "***" if r['p_val']<0.001 else "**" if r['p_val']<0.01 else "*" if r['p_val']<0.05 else ""
    print(f"  {r['feat_i']:<20} {r['feat_j']:<20} {r['phi']:>8.4f}  "
          f"[{r['ci_lo']:>7.4f}, {r['ci_hi']:>7.4f}]  {r['p_val']:>8.4f}  {sig:>4}")

sig_pairs = df_res[df_res['sig_05']]
print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Total pairs tested:              {len(df_res)}")
print(f"Significant (95% CI > 0):        {df_res['sig_05'].sum()}")
print(f"Significant (p < 0.01):          {df_res['sig_01'].sum()}")
print(f"Top φᵢⱼ in original 10 pairs:    {df_res[df_res['feat_i'].isin(['Haemoglobin','Albumin','eGFR_MDRD']) | df_res['feat_j'].isin(['Haemoglobin','Albumin','eGFR_MDRD'])].nlargest(5,'phi')[['feat_i','feat_j','phi']].to_string()}")

# Check if original 3 pairs remain top
print(f"\nOriginal 3 'meaningful' pairs — confirmed in full 210 analysis?")
orig3 = [('Haemoglobin','Serum_Creatinine'),
         ('Albumin','Serum_Creatinine'),
         ('eGFR_MDRD','Albumin')]
for fi,fj in orig3:
    row = df_res[(df_res['feat_i']==fi)&(df_res['feat_j']==fj)]
    if len(row)==0:
        row = df_res[(df_res['feat_i']==fj)&(df_res['feat_j']==fi)]
    if len(row):
        r = row.iloc[0]
        rank = df_res.index.get_loc(row.index[0]) + 1
        print(f"  {fi} × {fj}: φ={r['phi']:.4f} CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] rank={rank}/210")

print(f"\nSaved: interaction_all210.csv")
print("Done.")