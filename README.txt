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
- plateau_analysis_report.md: final follow-up explaining the performance plateau and the best patient-history result.
- presentation_results_summary.md: concise presentation-ready result framing.
- START_HERE_FOR_PRESENTATION.md: handoff guide for presentation work.
- experiment_results/: CSV audit trail for broad, targeted, native CatBoost, ensemble, neural-network, imbalance-handling, feature-engineering, CatBoost-tuning, balanced-test, alternate row-scope, paper-reproduction, plateau-diagnostic, and patient-history experiments.
- modeling_experiments.py, targeted_modeling_search.py, native_catboost_search.py, ensemble_search.py, neural_network_search.py, imbalance_experiments.py, imbalance_refinement_search.py, imbalance_refined_ensemble.py, feature_engineering_search.py, feature_engineering_ensemble.py, catboost_tuning_search.py, balanced_test_evaluation.py, all_encounters_group_split_search.py, paper_reproduction_search.py, plateau_diagnostic_search.py, plateau_ensemble_search.py, patient_history_feature_search.py, patient_history_tuning_search.py: experiment scripts used to produce the saved comparison tables.
- requirements.txt: minimal Python packages for running the notebook.

How to run:
1. Install the packages in requirements.txt.
2. Open hospital_readmission_eda.ipynb or hospital_readmission_modeling.ipynb from this folder.
3. Run all cells. The notebooks expect the raw data at archive/diabetic_data.csv.

Current scope:
This stage covers data loading, target definition, missing-value review, class balance,
patient/encounter granularity, diagnosis grouping, medication summaries, utilization EDA,
preprocessing pipelines, model training, model comparison, threshold tuning, lift tables,
and extended model-search documentation, including neural-network, imbalance-handling,
feature-engineering, balanced-test sensitivity, all-encounter patient-group split,
paper-reproduction comparisons, plateau diagnostics, and prior patient-history features.
