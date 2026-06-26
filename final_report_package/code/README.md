# Code Included For Final Report

This folder contains the main notebooks and scripts needed to support the final report. It is a curated copy of the important code from the project root.

## Main Notebooks

- `hospital_readmission_eda.ipynb`
  EDA notebook: dataset understanding, missing values, class imbalance, and visual exploration.

- `hospital_readmission_modeling.ipynb`
  Original modeling notebook with the course-project baseline workflow.

## Final Result Scripts

- `../../FINAL_MODEL_PIPELINE.py`
  Canonical self-contained final model pipeline. This is the clearest file for the professor:
  it loads the raw data, applies final preprocessing/feature engineering, trains the
  validation-selected CatBoost model, and writes final metrics/lift outputs.

- `history_negative_ratio_refinement.py`
  Experiment-history script. Reproduces the CatBoost patient-history negative-ratio
  refinement results:
  - validation-selected final model: PR-AUC `0.2414`
  - best observed exploratory variant: PR-AUC `0.2415`

- `history_catboost_optuna_search.py`
  Optuna hyperparameter search. Confirms Optuna did not improve beyond the known best CatBoost setup.

## Supporting Scripts

These are imported by the copied experiment-history scripts, not by the final root pipeline:

- `modeling_experiments.py`
- `feature_engineering_search.py`
- `imbalance_experiments.py`
- `all_encounters_group_split_search.py`
- `plateau_diagnostic_search.py`
- `patient_history_feature_search.py`

## Additional Experiment Scripts

These document extra attempts used to understand the plateau:

- `history_balanced_bagging_search.py`
- `history_heterogeneous_search.py`
- `history_catboost_seed_sweep.py`
- `history_catboost_order_sensitivity.py`
- `history_catboost_bootstrap_search.py`
- `paper_reproduction_search.py`

## How To Run

From the project root, install requirements and run the canonical root script:

```bash
/opt/anaconda3/bin/python -m pip install -r requirements.txt
/opt/anaconda3/bin/python FINAL_MODEL_PIPELINE.py
```

For the optional broader experiment environment:

```bash
/opt/anaconda3/bin/python -m pip install -r requirements-experiments.txt
```

For a fast wiring check without training:

```bash
/opt/anaconda3/bin/python FINAL_MODEL_PIPELINE.py --dry-run
```

The scripts expect the raw dataset at:

```text
archive/diabetic_data.csv
```

The code in this folder is copied for report convenience. The canonical final pipeline is still in the project root, and runnable experiment scripts are stored under the root-level `experiments/` folder.

If someone wants to run the copied scripts from inside `final_report_package/code/`, copy or symlink the project `archive/` folder beside them first, because the scripts expect `archive/diabetic_data.csv` relative to their working project directory.
