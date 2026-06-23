# Final Report Package

This folder contains the small set of materials needed to write the final AI1215 report. Use this folder instead of searching through all experiment files.

## Start Here

1. Read `report_outline.md`.
2. Use `tables/model_type_validation_comparison.csv` for the required model comparison table.
3. Use `tables/final_model_metrics_best_f1.csv` for the final model metrics.
4. Use `tables/final_model_lift_table.csv` and `figures/final_model_lift_curve.png` for practical interpretation.
5. Use `source_notes/assignment_requirements_summary.md` to check that the report matches the project description.

## Best Numbers To Report

Clean validation-selected final model:

```text
Model: NegRefineCat_d6_lr002_neg7.5_seed37
PR-AUC: 0.2414
ROC-AUC: 0.6827
Recall: 0.4226
Precision: 0.2223
F1: 0.2913
Accuracy: 0.7733
```

Best observed exploratory variant:

```text
Model: NegRefineCat_d6_lr002_neg8_seed202
PR-AUC: 0.2415
ROC-AUC: 0.6817
Recall: 0.4446
Precision: 0.2160
F1: 0.2907
Accuracy: 0.7608
```

Recommended wording:

```text
The validation-selected CatBoost model achieved PR-AUC 0.2414 on a patient-safe held-out test split, about 2.19x the 0.1103 baseline positive rate. A seed/order-sensitive exploratory variant reached PR-AUC 0.2415, but the disciplined number to lead with is 0.2414.
```

## Figures

- `figures/class_balance_readmitted_30.png`: use in Problem & Data.
- `figures/missing_values_top_columns.png`: use in Problem & Data or Preprocessing.
- `figures/validation_pr_auc_by_modeling_stage.png`: use in Model Selection & Results.
- `figures/final_model_lift_curve.png`: use in Results or Conclusions.

## Tables

- `tables/dataset_key_statistics.csv`: raw rows, eligible rows, first-encounter rows, positive rates.
- `tables/top_missing_values.csv`: missing-value summary.
- `tables/patient_safe_split_summary.csv`: train/validation/test split.
- `tables/model_type_validation_comparison.csv`: required model comparison table.
- `tables/report_model_comparison_test.csv`: short test comparison across modeling stages.
- `tables/final_model_metrics_best_f1.csv`: final model metrics.
- `tables/final_model_lift_table.csv`: top-risk group lift.
- `tables/paper_and_final_model_comparison.csv`: paper comparison.
- `tables/references_to_cite.csv`: references and URLs.

## Main Caveat

The model is useful as a risk-ranking model, not a perfect yes/no clinical decision system. The dataset is imbalanced and lacks richer clinical information such as vitals, continuous labs, medication doses, discharge plans, social determinants, exact dates, and hospital/provider identifiers.
