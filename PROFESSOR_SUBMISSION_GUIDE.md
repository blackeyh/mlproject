# Professor Submission Guide

This repository contains the full hospital readmission project, including the raw data, EDA, final model pipeline, report materials, and experiment history.

## Final Model File

Run this file for the final model:

```bash
python FINAL_MODEL_PIPELINE.py
```

Fast smoke check without training:

```bash
python FINAL_MODEL_PIPELINE.py --dry-run
```

`FINAL_MODEL_PIPELINE.py` is the canonical one-file final pipeline. It:

1. Loads `archive/diabetic_data.csv`.
2. Creates `readmitted_30`.
3. Removes hospice/expired discharge cases.
4. Builds the patient-safe train/validation/test split.
5. Applies the final preprocessing and feature engineering.
6. Trains the final CatBoost model.
7. Selects the operating threshold on validation.
8. Evaluates on the held-out test set.
9. Writes outputs to `final_model_outputs/`.

## Required Project Materials

- Raw data: `archive/diabetic_data.csv`
- Project description: `project_description.pdf`
- EDA notebook: `hospital_readmission_eda.ipynb`
- Modeling notebook: `hospital_readmission_modeling.ipynb`
- Final one-file model pipeline: `FINAL_MODEL_PIPELINE.py`
- Package requirements: `requirements.txt`
- Exact local package snapshot: `requirements-pinned.txt`
- Final report package: `final_report_package/`

Use `requirements.txt` for a normal install. Use `requirements-pinned.txt` if an exact package-version snapshot is needed for reproduction.

## Experiment History

The professor asked to include everything tried. The broad experiment history is included in:

- `experiment_results/`: saved CSV/JSON outputs from experiments.
- `modeling_experiments.py`: baseline model comparisons.
- `targeted_modeling_search.py`: targeted preprocessing/model variants.
- `native_catboost_search.py`: native CatBoost tests.
- `ensemble_search.py`: ensemble experiments.
- `neural_network_search.py`: neural-network experiments.
- `imbalance_experiments.py`: imbalance-handling experiments.
- `feature_engineering_search.py`: feature-engineering experiments.
- `catboost_tuning_search.py`: CatBoost tuning.
- `all_encounters_group_split_search.py`: all-encounter patient-safe split experiments.
- `paper_reproduction_search.py`: closest paper reproduction/sensitivity.
- `plateau_diagnostic_search.py`: plateau diagnosis.
- `patient_history_feature_search.py`: prior patient-history features.
- `history_negative_ratio_refinement.py`: final negative-ratio refinement search.
- `history_catboost_optuna_search.py`: Optuna search.

## Best Final Result

Validation-selected final model:

```text
Model: NegRefineCat_d6_lr002_neg7.5_seed37
PR-AUC: 0.2414
ROC-AUC: 0.6827
Recall: 0.4226
Precision: 0.2223
F1: 0.2913
Accuracy: 0.7733
```

The model is best interpreted as a risk-ranking model, not a perfect yes/no clinical decision system.

Important reproducibility note: the older `all_encounters_group_split_search.py` experiment produced the previous 0.2290 PR-AUC result. The current final result above comes from `FINAL_MODEL_PIPELINE.py`.
