# Final Report Outline

Course format: up to 7 pages, excluding cover page and references.

## Cover Page

Include:

- Team name
- Student names
- Project title: `Hospital Readmission Prediction Using Machine Learning`
- Chosen topic: `Hospital Readmission Prediction`
- Course: `AI1215 - Introduction to Machine Learning`

## Section 1: Problem & Data

Goal:

Predict whether a diabetic patient encounter will lead to readmission within 30 days.

Who benefits:

Hospitals, clinicians, and care-management teams can prioritize higher-risk patients for discharge planning, follow-up calls, medication review, or case-management support.

Data source:

UCI Diabetes 130-US Hospitals dataset, covering 101,766 encounters from 130 US hospitals/integrated delivery networks from 1999-2008.

Use these assets:

- `tables/dataset_key_statistics.csv`
- `tables/class_balance_all_eligible.csv`
- `tables/top_missing_values.csv`
- `figures/class_balance_readmitted_30.png`
- `figures/missing_values_top_columns.png`

Key points to write:

- Target is `readmitted_30 = 1` if `readmitted == "<30"`, otherwise `0`.
- The class is imbalanced; the held-out test positive rate is about `0.1103`.
- Accuracy alone is misleading because a majority-class model can look accurate while missing nearly all readmissions.
- Missingness is concentrated in `weight`, `medical_specialty`, `payer_code`, and a few diagnosis/race fields.

## Section 2: Preprocessing & Feature Engineering

Use these assets:

- `tables/patient_safe_split_summary.csv`
- `../preprocessing_decision_log.md`
- `../plateau_analysis_report.md`

Key points to write:

- Removed hospice/expired discharge rows because the target is not meaningful for patients who expired or went to hospice.
- Used a patient-safe train/validation/test split so the same patient does not appear in multiple splits.
- Created the binary target `readmitted_30`.
- Grouped ICD-9 diagnosis codes into clinically meaningful categories.
- Grouped admission/discharge/source IDs and age categories.
- Kept high-missing categorical fields with explicit missing/rare handling where useful.
- Created medication summary features and utilization features.
- Added prior patient-history features for the all-encounter setting using only earlier encounters for the same patient.
- One-hot/scaled features were used for scikit-learn models; CatBoost used native categorical handling where appropriate.

Important caveat:

Prior patient-history features are valid only in the all-encounter framing. They should not be described as first-encounter-only features.

## Section 3: Model Selection & Results

Use these assets:

- `tables/model_type_validation_comparison.csv`
- `tables/report_model_comparison_test.csv`
- `tables/final_model_metrics_best_f1.csv`
- `tables/final_model_lift_table.csv`
- `tables/paper_and_final_model_comparison.csv`
- `figures/validation_pr_auc_by_modeling_stage.png`
- `figures/final_model_lift_curve.png`

Required model comparison:

Use `tables/model_type_validation_comparison.csv`. It includes baseline, neural network, tree ensembles, XGBoost, LightGBM, and CatBoost.

Final model:

```text
Validation-selected model: NegRefineCat_d6_lr002_neg7.5_seed37
Test PR-AUC: 0.2414
ROC-AUC: 0.6827
Recall: 0.4226
Precision: 0.2223
F1: 0.2913
Accuracy: 0.7733
```

Best observed exploratory variant:

```text
NegRefineCat_d6_lr002_neg8_seed202
Test PR-AUC: 0.2415
```

Why CatBoost was selected:

- Best validation PR-AUC among model families.
- Handles categorical variables well.
- Performed better than XGBoost, LightGBM, Random Forest, Extra Trees, neural networks, and heterogeneous ensembles.
- Optuna hyperparameter search did not find a better configuration, supporting the final choice.

Practical interpretation:

The model is best as a risk-ranking tool. In the best observed variant, the top 1% highest-risk encounters had about `55.0%` readmission precision, compared with `11.0%` overall. The top 10% risk group had about `28.0%` readmission precision and captured about `25.4%` of positives.

Error analysis:

- The model has modest precision because readmission is rare and driven by factors not present in the dataset.
- Many false positives are patients with administrative/utilization signals that look high-risk but are not readmitted within 30 days.
- Many false negatives likely depend on missing post-discharge factors: follow-up care, social support, disease severity, medication adherence, and external events.
- Validation/test differences and seed/order sensitivity show that the remaining gains are small and unstable.

## Section 4: Conclusions

Main conclusion:

The final CatBoost model meaningfully ranks patients by readmission risk, more than doubling the PR-AUC baseline and essentially matching the closest paper's reported PR-AUC for the `<30` target under a patient-safe evaluation.

Use this wording:

```text
The model should not be treated as a standalone clinical decision system. Its value is risk stratification: it helps identify a smaller group of encounters with substantially higher 30-day readmission risk.
```

Limitations:

- Public dataset is old: 1999-2008.
- Limited clinical depth: no vitals, continuous labs, notes, medication doses, discharge plans, exact dates, or social determinants.
- Target is noisy because readmission depends on events after discharge.
- Performance is capped around PR-AUC 0.24-0.242 on the honest patient-safe setup.

Future improvements:

- Add richer longitudinal data and exact dates.
- Include hospital/provider identifiers if ethically and legally available.
- Add continuous labs, vitals, medication doses, notes, follow-up appointment data, and social determinants.
- Validate on a newer external hospital dataset.

## References

Use `tables/references_to_cite.csv`.

Recommended references:

- UCI Diabetes 130-US Hospitals dataset.
- Bhuvan et al. 2016, `Identifying Diabetic Patients with High Risk of Readmission`.
- scikit-learn, CatBoost, XGBoost, LightGBM documentation if methods need citation.

## AI Assistance Disclosure

Include a short disclosure because the project description requires explaining AI use.

Suggested text:

```text
AI assistance was used to help organize code, run experiment loops, summarize results, and draft report materials. All modeling decisions, code outputs, metrics, and final claims were checked against the saved notebooks/scripts and result CSV files.
```
