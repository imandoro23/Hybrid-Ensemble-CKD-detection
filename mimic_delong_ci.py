"""
Compute DeLong 95% CI for MIMIC-IV Stage 1-2 vs Stage 3-5 AUC
Run from: C:/Users/imand/.vscode/PhD main Work/
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# ── DeLong variance estimator ──────────────────────────────────────────────────
def delong_ci(y_true, y_score, alpha=0.05):
    from scipy import stats
    y_true = np.array(y_true); y_score = np.array(y_score)
    pos = y_score[y_true==1]; neg = y_score[y_true==0]
    n1, n0 = len(pos), len(neg)
    auc = roc_auc_score(y_true, y_score)
    # Placement values
    V10 = np.array([np.mean(p > neg) + 0.5*np.mean(p == neg) for p in pos])
    V01 = np.array([np.mean(n < pos) + 0.5*np.mean(n == pos) for n in neg])
    S10 = np.var(V10, ddof=1)
    S01 = np.var(V01, ddof=1)
    var = S10/n1 + S01/n0
    se  = np.sqrt(var)
    z   = stats.norm.ppf(1 - alpha/2)
    return auc, auc - z*se, auc + z*se, se, n1, n0

# ── Load MIMIC predictions ──────────────────────────────────────────────────────
print("Loading MIMIC predictions...")
df = pd.read_csv('MIMIC_Patient_Predictions.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"CKD_Stage distribution:\n{df['CKD_Stage'].value_counts().sort_index()}")

# ── Binary label: early (1-2) vs advanced (3-5) ───────────────────────────────
df = df.dropna(subset=['CKD_Stage','predicted_prob'])
df['severe'] = (df['CKD_Stage'] >= 3).astype(int)
# Filter to only staged patients (stages 1-5)
df = df[df['CKD_Stage'].between(1,5)]
print(f"\nStageable patients: {len(df)}")
print(f"Stage 1-2 (early):    {(df['severe']==0).sum()}")
print(f"Stage 3-5 (advanced): {(df['severe']==1).sum()}")

# ── Overall severity AUC + DeLong CI ─────────────────────────────────────────
auc, ci_lo, ci_hi, se, n1, n0 = delong_ci(df['severe'], df['predicted_prob'])
print(f"\n{'='*55}")
print(f"SEVERITY STRATIFICATION AUC (Stage 1-2 vs 3-5)")
print(f"{'='*55}")
print(f"AUC:     {auc:.4f}")
print(f"95% CI:  [{ci_lo:.4f}, {ci_hi:.4f}]  (DeLong method)")
print(f"SE:      {se:.5f}")
print(f"n(advanced): {n1}   n(early): {n0}")
print(f"Significantly above 0.50: {'YES' if ci_lo > 0.50 else 'NO — CI includes 0.50'}")

# ── Per-stage mean probability ─────────────────────────────────────────────────
print(f"\n{'='*55}")
print("PER-STAGE MEAN PROBABILITY")
print(f"{'='*55}")
for stage in sorted(df['CKD_Stage'].unique()):
    sub = df[df['CKD_Stage']==stage]
    print(f"  Stage {int(stage)}: n={len(sub):5d}  "
          f"mean={sub['predicted_prob'].mean():.4f}  "
          f"SD={sub['predicted_prob'].std():.4f}")

# ── Bootstrap CI for the AUC as cross-check ────────────────────────────────────
print(f"\n{'='*55}")
print("BOOTSTRAP CI CROSS-CHECK (1000 resamples)")
print(f"{'='*55}")
rng = np.random.RandomState(42)
boot_aucs = []
for _ in range(1000):
    idx = rng.choice(len(df), len(df), replace=True)
    yb  = df['severe'].values[idx]
    ypb = df['predicted_prob'].values[idx]
    if yb.nunique()==2:
        boot_aucs.append(roc_auc_score(yb, ypb))
print(f"Bootstrap AUC:  {np.mean(boot_aucs):.4f}")
print(f"Bootstrap 95%:  [{np.percentile(boot_aucs,2.5):.4f}, "
      f"{np.percentile(boot_aucs,97.5):.4f}]")
print(f"Significantly above 0.50: "
      f"{'YES' if np.percentile(boot_aucs,2.5) > 0.50 else 'NO — CI includes 0.50'}")
print("\nDone.")