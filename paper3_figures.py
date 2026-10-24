"""
Paper 3 — Figure Generation Script
Generates all 6 figures from Paper3_FairnessResults.csv + Paper3_NHANES_SHAP.csv
Run from: C:/Users/imand/.vscode/PhD main Work/
Runtime: ~2 minutes
Output: plots/paper3_fig1.png through plots/paper3_fig6.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os, warnings
warnings.filterwarnings('ignore')

os.makedirs('plots', exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY   = '#1F3864'
BLUE   = '#2E75B6'
TEAL   = '#028090'
RED    = '#C00000'
AMBER  = '#C65911'
GREEN  = '#375623'
LTGREY = '#F4F6FA'
GREY   = '#404040'

plt.rcParams.update({
    'font.family':     'Times New Roman',
    'font.size':       11,
    'axes.titlesize':  13,
    'axes.labelsize':  11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'savefig.facecolor': 'white',
})

fair = pd.read_csv('Paper3_FairnessResults.csv')
shap = pd.read_csv('Paper3_NHANES_SHAP.csv')

# ── Figure 1 — Study Pipeline Diagram ─────────────────────────────────────────
print("Generating Figure 1 — Study pipeline...")
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis('off')

boxes = [
    (0.3, 1.2, 2.0, 1.6, 'UCI CKD Dataset\nn=397\n(62.5% CKD)', NAVY),
    (2.7, 1.2, 2.0, 1.6, 'Hybrid Stacking\nEnsemble\n(RF+GB+HGB+SVM+MLP)', BLUE),
    (5.1, 1.2, 2.0, 1.6, 'NHANES 2017-2020\nn=8,257\n(18.4% CKD)', TEAL),
    (7.5, 1.2, 2.0, 1.6, 'Subgroup\nFairness Analysis\n(4 demographic axes)', AMBER),
    (9.9, 1.2, 1.8, 1.6, 'Intersectional\nAnalysis\n(Age×DM; Age×Sex)', RED),
]
for x, y, w, h, label, color in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='white', linewidth=2, alpha=0.92)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha='center', va='center',
        color='white', fontsize=9.5, fontweight='bold', linespacing=1.5)

# Arrows
for x_start in [2.3, 4.7, 7.1, 9.5]:
    ax.annotate('', xy=(x_start+0.4, 2.0), xytext=(x_start, 2.0),
        arrowprops=dict(arrowstyle='->', color=GREY, lw=2))

# Labels above arrows
ax.text(2.4, 2.55, 'Train\n(locked)', ha='center', fontsize=8.5, color=GREY)
ax.text(4.8, 2.55, 'Apply\n(no retraining)', ha='center', fontsize=8.5, color=GREY)

ax.text(6.0, 0.35, 'AUC-ROC + DeLong 95% CI per subgroup\n'
    'Disparity threshold: Δ AUC > 0.05 or non-overlapping CIs',
    ha='center', fontsize=9, color=GREY, style='italic')

ax.set_title('Figure 1. Study pipeline: locked UCI-trained ensemble applied to NHANES '
    'for subgroup fairness evaluation', pad=12, color=NAVY, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/paper3_fig1.png')
plt.close()
print("  Saved: plots/paper3_fig1.png")

# ── Figure 2 — AUC by Age Group ────────────────────────────────────────────────
print("Generating Figure 2 — AUC by age group...")
age_data = fair[fair['category']=='Age Group'].sort_values('auc')
fig, ax = plt.subplots(figsize=(7, 5))
colors = [RED if a < 0.65 else AMBER if a < 0.70 else BLUE for a in age_data['auc']]
bars = ax.barh(age_data['subgroup'], age_data['auc'],
    xerr=[age_data['auc']-age_data['ci_lo'], age_data['ci_hi']-age_data['auc']],
    color=colors, edgecolor='white', linewidth=1.2, height=0.5,
    error_kw=dict(ecolor=GREY, lw=1.5, capsize=6, capthick=1.5))

# Add AUC values
for i, row in age_data.iterrows():
    ax.text(row['ci_hi']+0.005, list(age_data.index).index(i),
        f"{row['auc']:.4f}\n[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]",
        va='center', fontsize=8.5, color=GREY)

ax.axvline(0.5, color=RED, lw=1.2, ls='--', alpha=0.6, label='Chance (0.50)')
ax.axvline(fair[fair['subgroup']=='All patients']['auc'].values[0],
    color=NAVY, lw=1.5, ls=':', alpha=0.8, label=f"Overall AUC=0.7445")
ax.set_xlim(0.45, 0.85)
ax.set_xlabel('AUC-ROC (95% DeLong CI)', color=GREY)
ax.set_title('Figure 2. AUC-ROC by Age Group with 95% DeLong CI\n'
    'Δ = 0.1182 (18-44 vs 65+); non-overlapping CIs confirm significant disparity',
    color=NAVY, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')

# Annotate disparity
ax.annotate('', xy=(0.6010, 0), xytext=(0.7192, 2),
    arrowprops=dict(arrowstyle='<->', color=NAVY, lw=1.5))
ax.text(0.660, 1.0, 'Δ=0.1182', ha='center', fontsize=9,
    color=NAVY, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=NAVY, alpha=0.8))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=GREY)
plt.tight_layout()
plt.savefig('plots/paper3_fig2.png')
plt.close()
print("  Saved: plots/paper3_fig2.png")

# ── Figure 3 — AUC by Ethnicity ────────────────────────────────────────────────
print("Generating Figure 3 — AUC by ethnicity...")
eth_data = fair[fair['category']=='Ethnicity'].sort_values('auc')
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = [RED if a < 0.70 else BLUE if a >= 0.74 else TEAL for a in eth_data['auc']]
ax.barh(eth_data['subgroup'], eth_data['auc'],
    xerr=[eth_data['auc']-eth_data['ci_lo'], eth_data['ci_hi']-eth_data['auc']],
    color=colors, edgecolor='white', linewidth=1.2, height=0.5,
    error_kw=dict(ecolor=GREY, lw=1.5, capsize=6, capthick=1.5))

for i, row in eth_data.iterrows():
    ax.text(row['ci_hi']+0.004, list(eth_data.index).index(i),
        f"n={row['n']:,}  AUC={row['auc']:.4f}", va='center', fontsize=8.5, color=GREY)

ax.axvline(fair[fair['subgroup']=='All patients']['auc'].values[0],
    color=NAVY, lw=1.5, ls=':', alpha=0.8, label='Overall AUC=0.7445')
ax.set_xlim(0.58, 0.88)
ax.set_xlabel('AUC-ROC (95% DeLong CI)', color=GREY)
ax.set_title('Figure 3. AUC-ROC by Ethnicity with 95% DeLong CI\n'
    'Δ = 0.0757 (Non-Hispanic White vs Mexican American)', color=NAVY, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(colors=GREY)
plt.tight_layout()
plt.savefig('plots/paper3_fig3.png')
plt.close()
print("  Saved: plots/paper3_fig3.png")

# ── Figure 4 — Heatmap Age × Diabetes ─────────────────────────────────────────
print("Generating Figure 4 — Intersectional heatmap Age × Diabetes...")
age_groups = ['18-44','45-64','65+']
dm_groups  = ['Non-Diabetic','Diabetic']
auc_matrix = np.zeros((2, 3))
ci_lo_mat  = np.zeros((2, 3))
ci_hi_mat  = np.zeros((2, 3))
n_mat      = np.zeros((2, 3), dtype=int)

int_data = fair[fair['category']=='Age × Diabetes']
for j, ag in enumerate(age_groups):
    for i, dm in enumerate(dm_groups):
        label = f"{ag} + {dm}"
        row = int_data[int_data['subgroup']==label]
        if len(row):
            auc_matrix[i,j]  = row['auc'].values[0]
            ci_lo_mat[i,j]   = row['ci_lo'].values[0]
            ci_hi_mat[i,j]   = row['ci_hi'].values[0]
            n_mat[i,j]       = row['n'].values[0]

cmap = LinearSegmentedColormap.from_list('rg', [RED, AMBER, GREEN], N=256)
fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(auc_matrix, cmap=cmap, vmin=0.40, vmax=0.75, aspect='auto')

ax.set_xticks(range(3)); ax.set_xticklabels(age_groups, fontsize=11, color=GREY)
ax.set_yticks(range(2)); ax.set_yticklabels(dm_groups, fontsize=11, color=GREY)
ax.set_xlabel('Age Group', color=GREY, fontsize=11)
ax.set_ylabel('Diabetes Status', color=GREY, fontsize=11)

for i in range(2):
    for j in range(3):
        auc = auc_matrix[i,j]
        txt_color = 'white' if auc < 0.56 else GREY
        flag = '\n⚠️ CI includes 0.50' if ci_lo_mat[i,j] < 0.50 < ci_hi_mat[i,j] else (
               '\n⚠️' if auc < 0.60 else '')
        ax.text(j, i,
            f"AUC={auc:.4f}{flag}\n[{ci_lo_mat[i,j]:.4f}, {ci_hi_mat[i,j]:.4f}]\nn={n_mat[i,j]}",
            ha='center', va='center', color=txt_color, fontsize=8.5,
            fontweight='bold' if flag else 'normal', linespacing=1.4)

plt.colorbar(im, ax=ax, label='AUC-ROC', shrink=0.85)
ax.set_title('Figure 4. Intersectional Heatmap — Age × Diabetes Status\n'
    '18-44 + Diabetic: AUC=0.4708, CI includes 0.50 (chance performance)',
    color=NAVY, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/paper3_fig4.png')
plt.close()
print("  Saved: plots/paper3_fig4.png")

# ── Figure 5 — Heatmap Age × Sex ──────────────────────────────────────────────
print("Generating Figure 5 — Intersectional heatmap Age × Sex...")
sex_groups  = ['Male','Female']
auc_matrix2 = np.zeros((2, 3))
ci_lo_mat2  = np.zeros((2, 3))
ci_hi_mat2  = np.zeros((2, 3))
n_mat2      = np.zeros((2, 3), dtype=int)

int_data2 = fair[fair['category']=='Age × Sex']
for j, ag in enumerate(age_groups):
    for i, sx in enumerate(sex_groups):
        label = f"{ag} + {sx}"
        row = int_data2[int_data2['subgroup']==label]
        if len(row):
            auc_matrix2[i,j] = row['auc'].values[0]
            ci_lo_mat2[i,j]  = row['ci_lo'].values[0]
            ci_hi_mat2[i,j]  = row['ci_hi'].values[0]
            n_mat2[i,j]      = row['n'].values[0]

fig, ax = plt.subplots(figsize=(8, 4.5))
im2 = ax.imshow(auc_matrix2, cmap=cmap, vmin=0.50, vmax=0.77, aspect='auto')
ax.set_xticks(range(3)); ax.set_xticklabels(age_groups, fontsize=11, color=GREY)
ax.set_yticks(range(2)); ax.set_yticklabels(sex_groups, fontsize=11, color=GREY)
ax.set_xlabel('Age Group', color=GREY, fontsize=11)
ax.set_ylabel('Sex', color=GREY, fontsize=11)

for i in range(2):
    for j in range(3):
        auc = auc_matrix2[i,j]
        txt_color = 'white' if auc < 0.57 else GREY
        delta = abs(auc_matrix2[0,j]-auc_matrix2[1,j])
        flag = f'\nΔ={delta:.3f}' if i==0 else ''
        ax.text(j, i,
            f"AUC={auc:.4f}{flag}\n[{ci_lo_mat2[i,j]:.4f}, {ci_hi_mat2[i,j]:.4f}]\nn={n_mat2[i,j]}",
            ha='center', va='center', color=txt_color, fontsize=8.5, linespacing=1.4)

plt.colorbar(im2, ax=ax, label='AUC-ROC', shrink=0.85)
ax.set_title('Figure 5. Intersectional Heatmap — Age × Sex\n'
    'Sex gap largest in 18-44 (Δ=0.044); converges at 65+ (Δ=0.001)',
    color=NAVY, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/paper3_fig5.png')
plt.close()
print("  Saved: plots/paper3_fig5.png")

# ── Figure 6 — SHAP by Age Group ──────────────────────────────────────────────
print("Generating Figure 6 — SHAP by age group...")
FEATURES = ['Specific_Gravity','Albumin','Haemoglobin','eGFR_MDRD',
            'Packed_Cell_Volume','Hypertension','Serum_Creatinine','Diabetes_Mellitus']
feat_cols = [f for f in FEATURES if f in shap.columns]
short_names = {'Specific_Gravity':'Specific\nGravity','Albumin':'Albumin',
    'Haemoglobin':'Haemo-\nglobin','eGFR_MDRD':'eGFR',
    'Packed_Cell_Volume':'PCV','Hypertension':'Hyper-\ntension',
    'Serum_Creatinine':'Serum\nCreat.','Diabetes_Mellitus':'Diabetes\nMellitus'}

age_groups_plot = [('18-44',(shap['Age']<45)), ('45-64',(shap['Age']>=45)&(shap['Age']<65)),
                   ('65+',(shap['Age']>=65))]
colors_age = [AMBER, BLUE, TEAL]

x = np.arange(len(feat_cols))
width = 0.25
fig, ax = plt.subplots(figsize=(11, 5.5))

for k, (label, mask) in enumerate(age_groups_plot):
    vals = shap[mask][feat_cols].abs().mean().values
    bars = ax.bar(x + k*width - width, vals, width,
        label=label, color=colors_age[k], edgecolor='white', linewidth=0.8, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels([short_names.get(f,f) for f in feat_cols],
    fontsize=9.5, color=GREY)
ax.set_ylabel('Mean |SHAP| Value', color=GREY)
ax.set_title('Figure 6. SHAP Feature Attribution by Age Group\n'
    'Hypertension: 2.17× higher in 65+ vs 18-44; Diabetes Mellitus: 2.57× higher in 65+',
    color=NAVY, fontweight='bold')
ax.legend(title='Age Group', fontsize=10, title_fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(colors=GREY)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# Annotate key ratios
for fi, feat in enumerate(feat_cols):
    if feat in ['Hypertension','Diabetes_Mellitus']:
        vals_by_age = [shap[mask][feat].abs().mean()
                       for _,mask in age_groups_plot]
        ratio = vals_by_age[2] / vals_by_age[0] if vals_by_age[0]>0 else 1
        ax.text(fi, max(vals_by_age)*1.08, f'{ratio:.1f}×',
            ha='center', fontsize=8.5, color=RED, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/paper3_fig6.png')
plt.close()
print("  Saved: plots/paper3_fig6.png")

print("\n" + "="*55)
print("ALL FIGURES COMPLETE")
print("="*55)
print("\nFiles saved:")
for i in range(1, 7):
    path = f"plots/paper3_fig{i}.png"
    size = os.path.getsize(path)/1024 if os.path.exists(path) else 0
    print(f"  {path}  ({size:.0f} KB)")
print("\nInsert into Paper 3 in Word:")
print("  Fig 1 → after Introduction Section 1.3")
print("  Fig 2 → after Section 4.2 (Age results)")
print("  Fig 3 → after Section 4.4 (Ethnicity results)")
print("  Fig 4 → after Section 4.5 (Age × Diabetes)")
print("  Fig 5 → after Section 4.6 (Age × Sex)")
print("  Fig 6 → after Section 4.7 (SHAP by subgroup)")
print("\nDone.")