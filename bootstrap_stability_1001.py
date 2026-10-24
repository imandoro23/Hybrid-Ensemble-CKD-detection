import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paste the top10_freq and features from your 100-resample output
features_ordered = [
    'Specific_Gravity','Haemoglobin','eGFR_MDRD','Packed_Cell_Volume',
    'Serum_Creatinine','Albumin','rbcc','Hypertension',
    'Blood_Glucose_Random','Diabetes_Mellitus','Blood_Urea','BUN_Cr_Ratio',
    'Sodium','Red_Blood_Cells','Sugar','Blood_Pressure',
    'Appetite','Pedal_Edema','wbcc','Pus_Cell','Age'
]
freqs = [100,100,100,100,100,100,96,94,87,67,21,21,6,2,2,2,1,1,0,0,0]
# Quick add-on to bootstrap_stability_100.py — run after loading boot_imps
for k in [3, 5, 10]:
    top_k = [set(np.argsort(boot_imps[b])[-k:]) for b in range(100)]
    pairs_k = list(combinations(range(100), 2))
    j_k = np.mean([len(top_k[i]&top_k[j])/len(top_k[i]|top_k[j])
                   for i,j in pairs_k])
    print(f"Jaccard@top-{k}: {j_k:.4f}")
colors = ['#375623' if f>=90 else '#2E75B6' if f>=80
          else '#028090' if f>=50 else '#888888' for f in freqs]

fig, ax = plt.subplots(figsize=(11,8))
ax.barh(features_ordered[::-1], freqs[::-1],
        color=colors[::-1], edgecolor='white', linewidth=0.5)
ax.axvline(90, color='#375623', lw=1.5, linestyle='--', label='Very Stable (≥90/100)')
ax.axvline(80, color='#2E75B6', lw=1.5, linestyle='--', label='Stable (≥80/100)')
ax.axvline(50, color='#028090', lw=1.5, linestyle='--', label='Moderate (≥50/100)')
ax.set_xlabel('Top-10 frequency across 100 bootstrap resamples', fontsize=11)
ax.set_title(
    f'Bootstrap SHAP Stability — 100 Resamples (4,950 pairs)\n'
    f'Jaccard@top-10 = 0.8404  (95% CI [0.8163, 0.8644], jackknife)',
    fontsize=12, fontweight='bold'
)
ax.set_xlim(0, 107)
for i, v in enumerate(freqs[::-1]):
    ax.text(v+1, i, f'{v}/100', va='center', fontsize=9)
patches = [mpatches.Patch(color='#375623', label='Very Stable (≥90/100)'),
           mpatches.Patch(color='#2E75B6', label='Stable (80-89/100)'),
           mpatches.Patch(color='#028090', label='Moderate (50-79/100)'),
           mpatches.Patch(color='#888888', label='Unstable (<50/100)')]
ax.legend(handles=patches, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('plots/bootstrap_stability_100.png', dpi=150, bbox_inches='tight')
print("Saved: plots/bootstrap_stability_100.png")