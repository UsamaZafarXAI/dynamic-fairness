# A Multi-Model Framework for Explainable Sepsis Mortality Prediction with Dynamic Fairness and Drift Analysis

This repository reproduces the analysis for the manuscript submitted to IEEE Journal of Biomedical and Health Informatics (JBHI-00470-2026).

**No patient-level data is included.**

## Data Access

Credentialed access via PhysioNet is required:

- [MIMIC-IV](https://physionet.org/content/mimiciv/)
- [eICU Collaborative Research Database](https://physionet.org/content/eicu-crd/)

Expected MIMIC-IV directory structure:

```
<mimic_base_path>/
    hosp/
        patients.csv
        admissions.csv
        diagnoses_icd.csv
        labevents.csv
    icu/
        icustays.csv
        chartevents.csv
        inputevents.csv
        procedureevents.csv
```

## Installation

Python 3.12 recommended.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config config.yaml
```

Edit `config.yaml` to set `mimic_base_path` and `output_path` before running.

## Pipeline

The script executes the following steps in order:

1. **Preprocessing** — Extracts sepsis cohort from MIMIC-IV, computes 17 predictors (5 temporal vital signs + 7 laboratory values + vasopressor use + mechanical ventilation + SOFA score + age + comorbidity index), and assigns intersectional demographic groups.
2. **LSTM training** — Trains a hybrid temporal-static LSTM on 80% of the cohort with 1,000-iteration bootstrap confidence intervals on AUC.
3. **Baseline comparison** — Evaluates Logistic Regression, Random Forest, XGBoost, LightGBM, and a SOFA-only clinical baseline on the held-out 20%.
4. **Static fairness audit** — Computes TPR gap, σTPR, FPR gap, and FOR gap at t=0 for both baseline and static reweighing models.
5. **Longitudinal simulation** — Runs 30 independent 15-cycle retraining simulations with performative feedback and the Dynamic Adaptive Fairness (DAF) controller. Reports Friedman and Wilcoxon statistics on TPR and FPR gaps.
6. **Robustness experiment** — Introduces an exogenous covariate shift at t=8 (lactate +30%, SOFA +2) to test DAF generalization to unseen distribution shifts.
7. **SHAP analysis** — Computes Spearman rank correlation of feature importance between t=0 and t=14.

## Outputs

All figures and tables are saved to `output_path`
