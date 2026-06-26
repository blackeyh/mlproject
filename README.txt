# Hospital Readmission Prediction

Course project for predicting 30-day hospital readmission among diabetic patient encounters using the UCI Diabetes 130-US Hospitals dataset.

The final model is a patient-safe CatBoost risk-ranking model. It is meant to prioritize patients for follow-up review, not to replace clinical judgment.

## Final Result

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

Baseline positive rate / natural PR-AUC baseline: `0.1103`.

Interpretation: the final model more than doubles the natural PR-AUC baseline and works best as a risk-ranking tool for identifying higher-risk discharges.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

For the exact local package snapshot used in the final experiments:

```bash
pip install -r requirements-pinned.txt
```

Run the canonical final model pipeline:

```bash
python FINAL_MODEL_PIPELINE.py
```

Fast wiring check without training:

```bash
python FINAL_MODEL_PIPELINE.py --dry-run
```

Quick smoke-test training run:

```bash
python FINAL_MODEL_PIPELINE.py --quick
```

All code expects the raw UCI file at:

```text
archive/diabetic_data.csv
```

## Interactive Patient Prediction Demo

After a normal terminal run, the final pipeline asks whether you want to enter a patient manually.

You can also force the prompt with:

```bash
python FINAL_MODEL_PIPELINE.py --interactive-predict
```

The script asks for encounter information one field at a time, including age group, diagnoses, admission/discharge details, prior utilization, lab results, and medication status. It then prints:

- estimated 30-day readmission probability
- the validation-selected classification threshold
- whether the patient is above or below that threshold
- where the patient ranks compared with held-out test encounters

This demo uses the same preprocessing and feature engineering path as the final model. It is for project demonstration only and should not be treated as medical advice.

## Main Files

- `FINAL_MODEL_PIPELINE.py`: single-file final model pipeline. It loads the raw data, builds final features, trains CatBoost, evaluates the held-out test set, writes outputs, and optionally runs the interactive prediction prompt.
- `PROFESSOR_SUBMISSION_GUIDE.md`: concise guide to what the professor should inspect.
- `final_report_package/`: final report materials, tables, figures, LaTeX source, PDF, and presentation script.
- `hospital_readmission_eda.ipynb`: EDA notebook.
- `hospital_readmission_modeling.ipynb`: modeling notebook with baseline models and saved experiment summaries.
- `preprocessing_decision_log.md`: preprocessing and modeling decision log.
- `modeling_experiment_report.md`: broad experiment search summary.
- `plateau_analysis_report.md`: explanation of the performance plateau and why results are limited.
- `presentation_results_summary.md`: slide-ready results and interpretation.
- `START_HERE_FOR_PRESENTATION.md`: handoff guide for presentation work.
- `experiment_results/`: CSV audit trail from model searches, feature-engineering loops, imbalance experiments, neural-network tests, paper reproduction, CatBoost tuning, and plateau diagnostics.
- `requirements.txt`: flexible dependency list.
- `requirements-pinned.txt`: exact local package versions used for the final experiments.

## Final Pipeline Outputs

Running `python FINAL_MODEL_PIPELINE.py` writes:

- `final_model_outputs/final_pipeline_summary.json`
- `final_model_outputs/final_split_summary.csv`
- `final_model_outputs/final_model_metrics.csv`
- `final_model_outputs/final_model_lift_table.csv`
- `final_model_outputs/final_validation_scores.npy`
- `final_model_outputs/final_test_scores.npy`

`final_model_outputs/` is intentionally ignored by Git because these files are generated outputs.

## Experiment History

The repository keeps the broader experiment history because the professor asked to include what was tried. Important scripts include:

- `modeling_experiments.py`
- `targeted_modeling_search.py`
- `native_catboost_search.py`
- `ensemble_search.py`
- `neural_network_search.py`
- `imbalance_experiments.py`
- `feature_engineering_search.py`
- `catboost_tuning_search.py`
- `all_encounters_group_split_search.py`
- `paper_reproduction_search.py`
- `plateau_diagnostic_search.py`
- `patient_history_feature_search.py`
- `history_negative_ratio_refinement.py`
- `history_catboost_optuna_search.py`

The older `all_encounters_group_split_search.py` result is part of the experiment history. The current final headline result comes from `FINAL_MODEL_PIPELINE.py`.

## Caveat

The performance plateau is expected for this dataset because the target is imbalanced and the dataset lacks richer clinical information such as exact dates, vitals, continuous lab values, medication doses, discharge plans, hospital/provider identifiers, and social determinants of health.
