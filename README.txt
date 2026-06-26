Hospital Readmission Prediction (EDA)

Project topic:
Predict whether a diabetes patient encounter leads to hospital readmission within 30 days.

Main files:
- FINAL_MODEL_PIPELINE.py: canonical one-file final model pipeline. This is the file to run if the professor wants the final preprocessing + final CatBoost training + final evaluation in one obvious place.
- archive/diabetic_data.csv: raw UCI Diabetes 130-US Hospitals dataset.
- project_description.pdf: AI1215 course project description.
- prediction-on-hospital-readmission.ipynb: old starting notebook.
- hospital_readmission_eda.ipynb: new EDA-focused notebook.
- hospital_readmission_modeling.ipynb: modeling notebook with baseline models and extended experiment summary.
- preprocessing_decision_log.md: source-of-truth preprocessing and modeling decision log.
- modeling_experiment_report.md: summary of the extended model search.
- plateau_analysis_report.md: final follow-up explaining the performance plateau and the best advanced patient-history result.
- presentation_results_summary.md: concise presentation-ready result framing.
- START_HERE_FOR_PRESENTATION.md: handoff guide for presentation work.
- experiment_results/: CSV audit trail for broad, targeted, native CatBoost, ensemble, neural-network, imbalance-handling, feature-engineering, CatBoost-tuning, balanced-test, alternate row-scope, paper-reproduction, plateau-diagnostic, patient-history, advanced ensemble, seed/order, bootstrap, and Optuna experiments.
- modeling_experiments.py, targeted_modeling_search.py, native_catboost_search.py, ensemble_search.py, neural_network_search.py, imbalance_experiments.py, imbalance_refinement_search.py, imbalance_refined_ensemble.py, feature_engineering_search.py, feature_engineering_ensemble.py, catboost_tuning_search.py, balanced_test_evaluation.py, all_encounters_group_split_search.py, paper_reproduction_search.py, plateau_diagnostic_search.py, plateau_ensemble_search.py, patient_history_feature_search.py, patient_history_tuning_search.py, history_balanced_bagging_search.py, history_heterogeneous_search.py, history_catboost_seed_sweep.py, history_negative_ratio_refinement.py, history_catboost_order_sensitivity.py, history_catboost_bootstrap_search.py, history_catboost_optuna_search.py: experiment scripts used to produce the saved comparison tables.
- requirements.txt: flexible Python package list for running the notebooks and scripts.
- requirements-pinned.txt: exact package versions from the local environment used for the final experiments.

How to run:
1. Install the packages in requirements.txt.

   pip install -r requirements.txt

   For a closer reproduction of this local environment, use:

   pip install -r requirements-pinned.txt

2. For the final model, run:

   python FINAL_MODEL_PIPELINE.py

   For a fast wiring check without training, run:

   python FINAL_MODEL_PIPELINE.py --dry-run

3. For notebooks, open hospital_readmission_eda.ipynb or hospital_readmission_modeling.ipynb from this folder and run all cells.

All code expects the raw data at archive/diabetic_data.csv.

Final result reproducibility note:
The final headline result is produced by FINAL_MODEL_PIPELINE.py, not by the older
all_encounters_group_split_search.py experiment script. FINAL_MODEL_PIPELINE.py writes:
- final_model_outputs/final_pipeline_summary.json
- final_model_outputs/final_split_summary.csv
- final_model_outputs/final_model_metrics.csv
- final_model_outputs/final_model_lift_table.csv
- final_model_outputs/final_validation_scores.npy
- final_model_outputs/final_test_scores.npy

Expected validation-selected final test metrics:
- PR-AUC: 0.2414
- ROC-AUC: 0.6827
- Recall: 0.4226
- Precision: 0.2223
- F1: 0.2913
- Accuracy: 0.7733

Current scope:
This stage covers data loading, target definition, missing-value review, class balance,
patient/encounter granularity, diagnosis grouping, medication summaries, utilization EDA,
preprocessing pipelines, model training, model comparison, threshold tuning, lift tables,
and extended model-search documentation, including neural-network, imbalance-handling,
feature-engineering, balanced-test sensitivity, all-encounter patient-group split,
paper-reproduction comparisons, plateau diagnostics, prior patient-history features,
advanced CatBoost seed/order sensitivity, negative-ratio refinement, bootstrap checks,
and Optuna hyperparameter search.
