# Dynamic Fairness and Drift Analysis for Sepsis Mortality Prediction

Code and aggregate results for:

> U. Zafar and F. Wu, "A Multi-Model Framework for Explainable Sepsis Mortality Prediction with Dynamic Fairness and Drift Analysis," *IEEE Journal of Biomedical and Health Informatics*, manuscript JBHI-00470-2026.

## Data access

The analyses use two PhysioNet databases, both available to credentialed users under their data use agreements:

- MIMIC-IV (development cohort, 5,106 ICU stays)
- eICU Collaborative Research Database v2.0 (cross-institutional cohort, 208 hospitals)

This repository contains no patient-level data. Every file in `results/` is an aggregate table.

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/01_jbhi_pipeline.ipynb` | Main analysis: MIMIC-IV cohort construction, hybrid LSTM and tabular baselines, static fairness audit, longitudinal simulation (30 runs of 15 retraining cycles), Dynamic Adaptive Fairness controller, exogenous-shift experiment, SHAP stability analysis, and the eICU cross-institutional audit |
| `notebooks/02_missingness_smallsample.ipynb` | Missing-data and small-sample reporting: pre-imputation missingness by predictor, intersectional subgroup, and hospital; subgroup event counts with Wilson intervals; minimum-cell, bootstrap, and empirical-Bayes gap estimates |
| `src/smallsample_audit.py` | Functions for Wilson score intervals, minimum-cell gaps, within-subgroup bootstrap of max-min gaps, and beta-binomial empirical-Bayes shrinkage |
| `results/missing_data/` | Outputs of notebook 02 (manuscript Table II; Supplementary Tables S1 to S7) |
| `results/subgroup_audit/` | Subgroup mortality with Wilson intervals (Table III) and test-set subgroup counts for the t = 0 audit (Supplementary Table S8) |
| `requirements.txt` | Package versions |

## Reproducing the results

1. Obtain MIMIC-IV and eICU-CRD from PhysioNet.
2. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set the data paths at the top of each notebook (`ROOT`, `MIMIC_HOSP`, `MIMIC_ICU`, `EICU`).
4. Run `01_jbhi_pipeline.ipynb` for the results reported in the manuscript, then `02_missingness_smallsample.ipynb` for the missing-data and small-sample tables.

Both notebooks were executed on Kaggle (CPU only). scikit-learn is pinned to version 1.6.1 because Random Forest outputs vary across library versions. Random seeds are fixed: `random_state = 42` for the train-test split and models, and seed 42 + 100r for simulation run r.

Notebook 02 prints two checks before writing its tables: the MIMIC-IV cohort size and mortality (5,106 stays; 0.2793) and the eICU AUC of the common-feature Random Forest (0.7431; 95% CI 0.7375 to 0.7485).

## Citation

If you use this code, please cite the article listed above.
