# CKD Hybrid Ensemble — Multi-Cohort ML Pipeline

> Machine learning pipeline for early chronic kidney disease (CKD) detection: 5-learner stacking ensemble, 3-cohort external validation (UCI · NHANES · MIMIC-IV), bootstrap SHAP stability, 210-pair interaction analysis, demographic fairness analysis, and clinical decision support dashboard.

**Supporting code for:**
> Abdulmumini Imam Ibrahim & Manish Madhava Tripathi. "A Hybrid Stacking Ensemble with Deep Neural Network Base Learner for Early Detection of Chronic Kidney Disease: Multi-Cohort Validation, Explainability, and Fairness Analysis." *IEEE Journal of Biomedical and Health Informatics* (under review, 2026).

---

## Overview

This repository contains the complete seven-phase research pipeline developed for a PhD study on machine learning-based CKD detection at Integral University, Lucknow. The pipeline covers systematic review–guided gap analysis through model development, explainability, external validation, demographic fairness analysis, and a clinical decision support dashboard prototype.

### Key Contributions

| # | Contribution | Script |
|---|---|---|
| 1 | 4-stage feature selection consensus (statistical filters → Boruta → RFECV/LASSO → SHAP) | `phase3_feature_selection.py` |
| 2 | 5-learner stacking ensemble with OOF meta-learning and DNN base learner | `phase5_stacking_ensemble.py` |
| 3 | Bootstrap SHAP stability (100 resamples, 4,950 Jaccard pairs, Jaccard@top-10=0.8404) | `bootstrap_stability_100.py` |
| 4 | Full C(21,2)=210 pairwise SHAP interaction analysis | `mc3_all_interactions.py` |
| 5 | SHAP-LIME population agreement study (n=80; 88% directional inversions in CKD patients) | `mc2_final.py` |
| 6 | First demographic fairness analysis in CKD ML (sex, age, ethnicity, diabetes; DeLong CIs) | `paper3_fairness_analysis.py` |
| 7 | Intersectional fairness (age × diabetes; age × sex) | `paper3_fairness_analysis.py` |
| 8 | NHANES imputation sensitivity analysis (Supp. Table S5) | `nhanes_sensitivity_s5.py` |
| 9 | MIMIC-IV CLIF v1.1.0 external validation (n=27,911) + eGFR sensitivity (Supp. Table S4) | `phase6c_mimic_validation.py` · `mimic_sensitivity_s4.py` |
| 10 | Clinical decision support dashboard (Flask; SHAP + LIME + risk categories) | `phase7_clinical_dashboard.py` |

---

## Results at a Glance

| Cohort | n | AUC | Notes |
|---|---|---|---|
| UCI (internal test) | 80 | 1.0000 | Brier=0.0018 [0.0003, 0.0048] |
| NHANES 2017–2020 | 8,257 | 0.7445 [0.7304, 0.7586] | 18.4% CKD prevalence |
| MIMIC-IV CLIF v1.1.0 | 27,911 | 0.7029 [0.6884, 0.7175] | Stage 1–2 vs 3–5 |

**Fairness (NHANES):**

| Axis | Best subgroup | Worst subgroup | Δ AUC |
|---|---|---|---|
| Sex | Male 0.7856 | Female 0.7121 | 0.0735 ⚠ |
| Age | 65+ 0.7192 | 18–44 0.6010 | 0.1182 ⚠ |
| Ethnicity | NHW 0.7698 | Mexican Am. 0.6941 | 0.0757 ⚠ |
| Intersectional | 18–44 + Diabetic: AUC=0.4708 [0.3523, 0.5893] — CI includes 0.50 | | |

---

## Repository Structure

```
ckd-hybrid-ensemble/
│
├── data/
│   ├── chronic_kidney_disease_full.arff   # Raw UCI CKD dataset
│   └── UCI_CKD_Processed.csv             # Preprocessed UCI (397 × 25)
│   # NHANES and MIMIC-IV data not included — see Data Access below
│
├── phase3_feature_selection.py   # 4-stage consensus feature selection
├── phase4_baseline_benchmarking.py  # 9-model baseline benchmarks
├── phase5_stacking_ensemble.py   # 5-learner OOF stacking ensemble
├── phase6_xai_shap_lime.py       # Global SHAP, LIME, dependence plots
├── phase6b_validation_fairness.py # NHANES cross-dataset validation + fairness
├── phase6c_mimic_validation.py   # MIMIC-IV CLIF external validation
├── phase7_clinical_dashboard.py  # Flask clinical decision support dashboard
│
├── bootstrap_stability_100.py    # 100-resample bootstrap SHAP stability
├── mc2_final.py                  # SHAP-LIME agreement, n=80 population
├── mc3_all_interactions.py       # 210-pair pairwise SHAP interactions
├── mn9_jaccard_sensitivity.py    # Jaccard@top-k sensitivity (k=3,5,10)
│
├── paper3_fairness_analysis.py   # Full fairness pipeline (Paper 3)
├── paper3_figures.py             # Figure generation for Paper 3
│
├── mimic_sensitivity_s4.py       # MIMIC eGFR sensitivity (Supp. Table S4)
├── nhanes_sensitivity_s5.py      # NHANES imputation sensitivity (Supp. Table S5)
│
├── recompute_nhanes_threshold.py # Prevalence-matched threshold computation
│
├── plots/                        # Generated figures (PDF + PNG)
├── outputs/                      # CSV result files
│   ├── interaction_all210.csv
│   ├── shap_lime_agreement_n80.csv
│   ├── Paper3_FairnessResults.csv
│   └── Paper3_NHANES_SHAP.csv
│
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/[YOUR-USERNAME]/ckd-hybrid-ensemble.git
cd ckd-hybrid-ensemble
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**requirements.txt**
```
scikit-learn==1.5.1
pandas==2.2.2
numpy==1.26.4
scipy==1.13.1
shap==0.51.0
lime
matplotlib
flask
openpyxl
boruta
```

---

## Data Access

| Dataset | Access | Notes |
|---|---|---|
| **UCI CKD** | Included (`data/`) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) |
| **NHANES 2017–2020** | Free download | [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020) — see `phase6b_validation_fairness.py` for variable mapping |
| **MIMIC-IV CLIF v1.1.0** | Credentialed access via PhysioNet | [PhysioNet](https://physionet.org/content/mimic-iv-ext-clif/1.1.0/) — requires CITI training + data use agreement. Pass parquet file paths as arguments to `phase6c_mimic_validation.py` and `mimic_sensitivity_s4.py` |

---

## Reproducing the Main Results

### 1 — Feature selection and UCI baseline

```bash
python phase3_feature_selection.py
python phase4_baseline_benchmarking.py
```

### 2 — Hybrid stacking ensemble

```bash
python phase5_stacking_ensemble.py
```
Outputs: UCI test AUC, Brier score, calibration metrics, confusion matrix.

### 3 — XAI: bootstrap stability + interactions

```bash
python bootstrap_stability_100.py      # Jaccard@top-10 across 100 resamples
python mc3_all_interactions.py         # All 210 pairwise SHAP interactions
python mn9_jaccard_sensitivity.py      # Jaccard@top-k for k=3,5,10
python mc2_final.py                    # SHAP-LIME agreement, full n=80 test set
```

### 4 — NHANES validation and fairness

```bash
python phase6b_validation_fairness.py
python paper3_fairness_analysis.py
python paper3_figures.py               # Generates plots/paper3_fig1–6.png
```

### 5 — MIMIC-IV external validation

```bash
python phase6c_mimic_validation.py \
  --clif_patient PATH/clif_patient.parquet \
  --clif_labs    PATH/clif_labs.parquet \
  --clif_vitals  PATH/clif_vitals.parquet

python mimic_sensitivity_s4.py \
  --clif_patient PATH/clif_patient.parquet \
  --clif_labs    PATH/clif_labs.parquet \
  --clif_vitals  PATH/clif_vitals.parquet
```

### 6 — Clinical dashboard

```bash
python phase7_clinical_dashboard.py
# Open http://localhost:5000 in a browser
```

---

## Key Numbers (for verification)

```
UCI test AUC:          1.0000
UCI Brier score:       0.0018  [0.0003, 0.0048]
UCI ECE:               0.0021
NHANES AUC:            0.7445  [0.7304, 0.7586]  n=8,257
MIMIC-IV staging AUC:  0.7029  [0.6884, 0.7175]  n=27,911
Jaccard@top-10:        0.8404  [0.8163, 0.8644]  100 resamples, 4,950 pairs
Top SHAP interaction:  Hb × PCV  φ=+0.0882  rank 1/210
SHAP-LIME CKD ρ mean:  0.383   (88% directional inversions, Cohen's d=2.52)
Age AUC Δ (fairness):  0.1182  (18–44: 0.6010 vs 65+: 0.7192)
```

---

## Citation

If you use this code or build on these results, please cite:

```bibtex
@article{ibrahim2026ckd,
  title   = {A Hybrid Stacking Ensemble with Deep Neural Network Base Learner
             for Early Detection of Chronic Kidney Disease: Multi-Cohort
             Validation, Explainability, and Fairness Analysis},
  author  = {Ibrahim, Abdulmumini Imam and Tripathi, Manish Madhava},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

---

## Ethical and Regulatory Notice

- The UCI dataset is publicly available under the Creative Commons Attribution 4.0 licence.
- NHANES data are de-identified public-use files released by the US Centers for Disease Control and Prevention.
- MIMIC-IV data access requires completion of CITI Programme human subjects training and execution of a PhysioNet data use agreement. MIMIC-IV data are **not** included in this repository.
- The clinical dashboard prototype is a **research tool only**. It has not undergone prospective clinical validation, regulatory review, or approval for clinical use. It must not be used for patient diagnosis or treatment decisions.

---

## Licence

MIT Licence — see `LICENSE` for details.

---

## Contact

**Abdulmumini Imam Ibrahim**
PhD Scholar, Department of Computer Science and Engineering
Integral University, Lucknow, India
imandoro23@gmail.com

Supervisor: Dr. Manish Madhava Tripathi — mmt@iul.ac.in
