Hospital Readmission Prediction (EDA)

Project topic:
Predict whether a diabetes patient encounter leads to hospital readmission within 30 days.

Main files:
- archive/diabetic_data.csv: raw UCI Diabetes 130-US Hospitals dataset.
- project_description.pdf: AI1215 course project description.
- prediction-on-hospital-readmission.ipynb: old starting notebook.
- hospital_readmission_eda.ipynb: new EDA-focused notebook.
- hospital_readmission_modeling.ipynb: modeling notebook with baseline models and extended experiment summary.
- preprocessing_decision_log.md: source-of-truth preprocessing and modeling decision log.
- modeling_experiment_report.md: summary of the extended model search.
- experiment_results/: CSV audit trail for broad, targeted, native CatBoost, ensemble, neural-network, imbalance-handling, feature-engineering, CatBoost-tuning, balanced-test, alternate row-scope, and paper-reproduction experiments.
- modeling_experiments.py, targeted_modeling_search.py, native_catboost_search.py, ensemble_search.py, neural_network_search.py, imbalance_experiments.py, imbalance_refinement_search.py, imbalance_refined_ensemble.py, feature_engineering_search.py, feature_engineering_ensemble.py, catboost_tuning_search.py, balanced_test_evaluation.py, all_encounters_group_split_search.py, paper_reproduction_search.py: experiment scripts used to produce the saved comparison tables.
- requirements.txt: pinned Python packages and versions used for the notebooks and experiment scripts.

How to reproduce the headline result:
1. Create and activate a Python environment, then install the pinned package versions:

   pip install -r requirements.txt

2. Regenerate the final reported 0.2290 PR-AUC result with:

   python all_encounters_group_split_search.py

   This script uses the fixed RANDOM_STATE=42 from modeling_experiments.py, the raw data at
   archive/diabetic_data.csv, all eligible encounters, and a patient-grouped train/validation/test
   split. It rewrites:
   - experiment_results/all_encounters_group_split_summary.csv
   - experiment_results/all_encounters_group_split_validation_results.csv
   - experiment_results/all_encounters_group_split_selected_for_test.csv
   - experiment_results/all_encounters_group_split_test_results.csv
   - experiment_results/all_encounters_group_split_lift_tables.csv

   The expected best selected test row is:
   - PR-AUC: 0.2290
   - ROC-AUC: 0.6797
   - Recall: 0.3817
   - Precision: 0.2214
   - F1: 0.2802
   - Accuracy: 0.7838

3. Open hospital_readmission_modeling.ipynb and run all cells to reproduce the notebook
   baseline and display the saved extended-experiment tables, including the all-encounter
   patient-group result above.

Important reproducibility note:
The first part of hospital_readmission_modeling.ipynb trains the conservative first-encounter
per-patient models. Those notebook-trained models are expected to have lower PR-AUC than the
headline 0.2290 result. The headline 0.2290 result comes from
all_encounters_group_split_search.py and is summarized in the notebook's Extended Experiment
Search section from the saved CSV audit trail.

Other useful runs:
- python paper_reproduction_search.py
  Recreates the local paper-style comparison saved in experiment_results/paper_reproduction_results.csv.
- Open hospital_readmission_eda.ipynb and run all cells for the EDA-only notebook.

Current scope:
This stage covers data loading, target definition, missing-value review, class balance,
patient/encounter granularity, diagnosis grouping, medication summaries, utilization EDA,
preprocessing pipelines, model training, model comparison, threshold tuning, lift tables,
and extended model-search documentation, including neural-network, imbalance-handling,
feature-engineering, balanced-test sensitivity, all-encounter patient-group split,
and paper-reproduction comparisons.
