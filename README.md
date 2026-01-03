# IJMI CXAI Reproducibility Package

This repository reproduces the analysis for the manuscript:
“Generating Prescriptive Counterfactual Explanations for Clinical Decline”.

It performs:
- MIMIC-IV + eICU preprocessing (first 24h, hourly aggregation, imputation)
- LSTM mortality prediction (“oracle”)
- Gradient-based counterfactual time-series generation
- Exports manuscript artefacts (Figures 1–5, Tables 2–4)

No patient-level data is included. Do not upload MIMIC/eICU data or derived patient-level files to GitHub.

## Data access (required)
You must obtain credentialed access via PhysioNet and comply with each dataset’s DUA:
- MIMIC-IV (e.g., v3.1)
- eICU Collaborative Research Database (e.g., v2.0)

Expected paths:
- MIMIC: <mimic_base_path>/icu/ and <mimic_base_path>/hosp/
- eICU: <eicu_base_path>/

## Install
Python 3.10 recommended.

```bash
pip install -r requirements.txt
