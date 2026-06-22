# Start Here For Presentation

This is the quick guide for building the presentation.

## Project Goal

Predict whether a diabetic patient will be readmitted to the hospital within 30 days using the UCI Diabetes 130-US Hospitals dataset.

Target:

`readmitted_30 = 1` if `readmitted == "<30"`, otherwise `0`.

## Main Story

This is a hard imbalanced clinical prediction problem. Only about 9% to 11% of encounters are positive depending on the split, so accuracy alone is misleading. A model can get high accuracy by predicting "not readmitted" for almost everyone.

The best way to present the project is as a risk-ranking model:

The model is not perfect at classifying every readmission, but it ranks patients by readmission risk better than the baseline and better than our local reproduction of the closest paper setup.

## Best Result To Use

Best overall model:

All-encounter CatBoost with patient-safe train/validation/test split.

Metrics:

- PR-AUC: 0.2290
- ROC-AUC: 0.6797
- Recall: 0.3817
- Precision: 0.2214
- F1: 0.2802
- Accuracy: 0.7838

Baseline positive rate / majority PR-AUC baseline:

- 0.1103

Simple interpretation:

The model more than doubles the PR-AUC baseline: 0.2290 vs 0.1103.

## Paper Comparison

Closest paper:

Bhuvan et al. 2016, "Identifying Diabetic Patients with High Risk of Readmission".

Paper reported:

- Same dataset
- Same `<30` vs `>30/NO` 30-day readmission target
- Best reported Random Forest PR-AUC: 0.242

Our local reproduction of the visible paper setup:

- PR-AUC: 0.2083
- ROC-AUC: 0.6564
- Recall: 0.4203
- Precision: 0.2079
- F1: 0.2782
- Accuracy: 0.7539

Careful wording:

Our model beats the locally reproduced paper setup on PR-AUC, ROC-AUC, precision, F1, and accuracy, but not recall at the selected threshold. The paper itself reports PR-AUC only for the 30-day target, not all these extra metrics.

## Files To Use

Start with:

- `presentation_results_summary.md`  
  Best slide source. Use this first.

Then use:

- `hospital_readmission_eda.ipynb`  
  EDA plots, missing values, class imbalance, dataset understanding.

- `hospital_readmission_modeling.ipynb`  
  Main modeling notebook and final pipeline.

- `modeling_experiment_report.md`  
  Full record of models tried and results.

- `preprocessing_decision_log.md`  
  Why preprocessing choices were made.

- `experiment_results/all_encounters_group_split_test_results.csv`  
  Final model metrics.

- `experiment_results/all_encounters_group_split_lift_tables.csv`  
  Top-risk group performance. Useful for explaining practical value.

- `experiment_results/paper_reproduction_results.csv`  
  Paper reproduction comparison.

Optional:

- `project_description.pdf`  
  Assignment description and topic wording.

- `README.txt`  
  How to reproduce the work.

## Suggested Slide Order

1. Problem and motivation
2. Dataset and target
3. Class imbalance and why accuracy is misleading
4. Preprocessing and feature engineering
5. Models tried
6. Final model results
7. Comparison to paper and baseline
8. Error analysis and limitations
9. Conclusion and next steps

## Do Not Say

Do not say the model has "high accuracy" as the main result.

Do not compare our 30-day result to online notebooks that predict any readmission or use SMOTE before splitting.

Do not say we beat the paper on every threshold.

## Best One-Sentence Conclusion

Our final CatBoost model achieved PR-AUC 0.2290 on the 30-day readmission task, more than doubling the natural baseline of 0.1103 and outperforming our local reproduction of the closest paper setup, while using a patient-safe evaluation split.
