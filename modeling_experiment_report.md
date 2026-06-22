# Hospital Readmission Modeling Experiment Report

This report documents the extended modeling search run after the first modeling notebook baseline.

Detailed run-level audit files are in `experiment_results/`. The most important files are:

- `validation_experiment_results.csv`: broad preprocessing/model search.
- `targeted_validation_results.csv`: targeted search around the best broad-search feature family.
- `native_catboost_validation_results.csv`: native CatBoost categorical search.
- `ensemble_validation_results.csv`: averaged score ensembles.
- `nn_validation_results.csv`: neural-network search with PyTorch MLP and TabNet models.
- Matching `*_test_results*.csv` files: test results for candidates selected from validation results.

## Baseline from first modeling notebook

The first executed notebook used the accepted preprocessing decisions and standard scikit-learn models.

Best selected test result from that first version:

```text
HistGradientBoosting: PR-AUC 0.1457, ROC-AUC 0.6231, recall 0.5637, precision 0.1301, F1 0.2115, accuracy 0.6227
Majority baseline: PR-AUC 0.0897, ROC-AUC 0.5000, recall 0.0000, F1 0.0000, accuracy 0.9103
```

Accuracy is high for the majority baseline only because the positive class is rare. PR-AUC, recall, precision, and F1 are more informative.

## Broad Search

Script:

```text
modeling_experiments.py
```

What was tried:

- Accepted preprocessing with rare thresholds 100, 200, 500.
- Detailed administrative grouping.
- Raw administrative IDs as categorical features.
- Raw age vs paper age grouping.
- Gender kept vs dropped.
- Weight dropped, indicator, and category variants.
- Diagnosis groups only, raw diagnosis only, and diagnosis groups plus raw diagnosis codes.
- Medication rare-column dropping, all medication columns, and medication summaries only.
- Log utilization and bucketed utilization variants.
- Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, GaussianNB, LightGBM, XGBoost, and CatBoost.
- Threshold strategies: default 0.5, best F1, best F2, max recall with precision >= 0.12, 0.15, and 0.20.

Best broad-search validation PR-AUC:

```text
raw_admin_raw_age_weight_category + XGBoost_depth4_aucpr
Validation PR-AUC 0.1854
```

Best broad-search selected test PR-AUC:

```text
raw_admin_raw_age_weight_category + LightGBM_balanced_leaves31
Test PR-AUC 0.1834, ROC-AUC 0.6441, recall 0.2431, precision 0.2250, F1 0.2337, accuracy 0.8569
```

Broad-search conclusion:

The biggest lift came from using raw administrative IDs as categorical variables, retaining raw age/gender, and using weight as a category. Raw diagnosis-code detail and keeping all medication columns did not help enough to justify their added complexity.

## Targeted Search

Scripts:

```text
targeted_modeling_search.py
evaluate_targeted_selected.py
```

What was tried:

- Focused feature variants around raw administrative IDs.
- Rare thresholds 50, 100, 200, and 500.
- Weight category, weight indicator, and weight dropped.
- Raw age and paper age grouping.
- Gender kept and dropped.
- Log utilization, no-log utilization, and utilization buckets.
- Medication summaries only.
- Tuned LightGBM, XGBoost, CatBoost, Extra Trees, Random Forest, and Logistic Regression variants.

Best targeted validation PR-AUC:

```text
target_raw_admin_age_paper_weight_category_rare100 + CatBoost_d5_lr0.03_SqrtBalanced
Validation PR-AUC 0.1956
```

Best targeted validation F1:

```text
target_raw_admin_age_paper_weight_category_rare100 + XGBoost_d5_lr0.015_spw0.75
Validation F1 0.2582
```

Best targeted selected test PR-AUC:

```text
CatBoost_d5_lr0.03_SqrtBalanced
Test PR-AUC 0.1887, ROC-AUC 0.6539, recall 0.3089, precision 0.2103, F1 0.2502, accuracy 0.8338
```

Best targeted selected test F1:

```text
XGBoost_d4_lr0.015_spw1 with weight indicator and precision >= 0.20 threshold
Test PR-AUC 0.1785, ROC-AUC 0.6474, recall 0.3408, precision 0.1988, F1 0.2511, accuracy 0.8176
```

Targeted-search conclusion:

XGBoost improved F1 after threshold tuning. CatBoost improved PR-AUC. Threshold tuning was important: default 0.5 was not generally the best operating point.

## Native CatBoost Search

Script:

```text
native_catboost_search.py
```

What was tried:

- CatBoost trained directly on categorical columns instead of one-hot encoded features
- Raw administrative IDs
- Paper age and raw age variants
- Weight category and weight dropped
- Medication summaries only
- Balanced and SqrtBalanced class weighting
- Depth 4, 5, and 6 variants with different learning rates and L2 regularization.

Best native CatBoost validation PR-AUC:

```text
native_cat_raw_admin_age_paper_summaries_only_rare100 + NativeCatBoost_d6_lr0.025_l27.0_SqrtBalanced
Validation PR-AUC 0.1978, ROC-AUC 0.6648, F1 0.2513
```

Best native CatBoost selected test PR-AUC:

```text
native_cat_raw_admin_age_paper_summaries_only_rare100 + NativeCatBoost_d5_lr0.02_l28.0_SqrtBalanced
Threshold: max recall with precision >= 0.20
Test PR-AUC 0.1983, ROC-AUC 0.6542, recall 0.3100, precision 0.2025, F1 0.2450, accuracy 0.8285
```

Native-CatBoost conclusion:

Native categorical handling produced the strongest held-out PR-AUC among selected candidates. Medication summaries only performed better than keeping all individual medication status columns for this model family.

## Ensemble Search

Script:

```text
ensemble_search.py
```

What was tried:

- Simple average score ensembles of the strongest native CatBoost, one-hot CatBoost, XGBoost, and LightGBM candidates.
- Combination sizes 2 through 5.
- A few hand-weighted blends centered on the best validation candidates.

Best ensemble validation PR-AUC:

```text
avg_2_native_summary_d6_sqrt__xgb_rawage_weight_d5
Validation PR-AUC 0.2005, F1 0.2564
```

Best ensemble selected test PR-AUC:

```text
avg_3_native_summary_d6_sqrt__native_agepaper_weight_d6_sqrt__lgbm_agepaper_weight
Test PR-AUC 0.1973
```

Ensemble conclusion:

The ensemble improved validation PR-AUC, but it did not beat the best native CatBoost single model on the selected test results. Because the test set should not be used for repeated model choice, the ensemble should be treated as an interesting sensitivity result rather than a clearly superior final model.

## Neural Network Search

Script:

```text
neural_network_search.py
```

What was tried:

- PyTorch embedding MLPs for mixed categorical/numeric tabular data.
- Hidden layer layouts from 128-64 up to 512-256-128.
- Dropout values from 0.15 through 0.35.
- Positive-class loss weights at 0.5, 0.75, and 1.0 times the empirical imbalance ratio.
- Three strongest feature families from the tree/boosting search: raw administrative IDs with paper age and medication summaries, raw administrative IDs with raw age and weight category, and raw administrative IDs with paper age plus weight indicator.
- PyTorch TabNet with two width/step configurations.
- A lightweight transformer-style prototype was considered, but the CPU-only run was too slow before producing useful validation output, so the completed neural comparison focuses on MLP and TabNet.

Best neural validation PR-AUC:

```text
nn_raw_admin_age_paper_weight_indicator_rare100 + EmbeddingMLP_512_256_do0.30_pw0.75
Validation PR-AUC 0.1818, ROC-AUC 0.6465, recall 0.3188, precision 0.1938, F1 0.2411, accuracy 0.8200
```

Best neural validation F1:

```text
nn_raw_admin_age_paper_summaries_only_rare100 + EmbeddingMLP_512_256_128_do0.35_pw1.0
Validation PR-AUC 0.1753, ROC-AUC 0.6524, recall 0.3177, precision 0.2030, F1 0.2477, accuracy 0.8270
```

Best neural selected test PR-AUC:

```text
nn_raw_admin_age_paper_summaries_only_rare100 + EmbeddingMLP_256_128_64_do0.20_pw0.5
Threshold: best F1 selected on validation
Test PR-AUC 0.1824, ROC-AUC 0.6483, recall 0.3546, precision 0.1715, F1 0.2311, accuracy 0.7883
```

Best neural selected test F1:

```text
nn_raw_admin_age_paper_weight_indicator_rare100 + EmbeddingMLP_512_256_128_do0.35_pw1.0
Threshold: max recall with precision >= 0.20 selected on validation
Test PR-AUC 0.1775, ROC-AUC 0.6491, recall 0.3206, precision 0.1980, F1 0.2448, accuracy 0.8225
```

Neural-network conclusion:

The MLPs were faster and stronger than TabNet on this CPU-only setup, but neither MLP nor TabNet beat the tuned native CatBoost family. TabNet took roughly 74 to 210 seconds per fit and remained below the best MLP validation PR-AUC. For this dataset and feature setup, boosted trees remain the best-performing and easiest-to-defend model family.

## Imbalance-Handling Search

Scripts:

```text
imbalance_experiments.py
imbalance_refinement_search.py
imbalance_refined_ensemble.py
```

What was tried:

- Wider class-weight grids for XGBoost, LightGBM, Logistic Regression, and native CatBoost.
- Random oversampling and random undersampling on the training split only.
- SMOTENC on the training split only.
- Balanced Random Forest, EasyEnsemble, and RUSBoost from `imbalanced-learn`.
- Sigmoid-calibrated XGBoost and LightGBM.
- Focal-loss PyTorch embedding MLPs.
- Lift tables for the selected imbalance-handling candidates.
- Refined rank-average and score-average ensembles built from the strongest old and new candidates.

Important practical result:

Resampling did not improve the best held-out result. SMOTENC and RUSBoost increased recall in some settings but reduced precision and PR-AUC. The useful improvement came from refined class weighting and CatBoost/XGBoost-style threshold selection.

Best refined validation PR-AUC:

```text
imb_ref_age_paper_summaries_only_rare100
RefinedNativeCat_d6_lr0.015_l210.0_customPW0.25
Validation PR-AUC 0.1982
Validation ROC-AUC 0.6666
Validation recall 0.3518
Validation precision 0.1917
Validation F1 0.2481
Validation accuracy 0.8089
```

Best selected held-out PR-AUC after imbalance refinement:

```text
imb_ref_age_paper_summaries_only_rare100
RefinedNativeCat_d6_lr0.015_l210.0_SqrtBalanced
Threshold: max recall with validation precision >= 0.20
Test PR-AUC 0.1991
Test ROC-AUC 0.6535
Test recall 0.3217
Test precision 0.1979
Test F1 0.2450
Test accuracy 0.8221
```

Best selected held-out F1 after imbalance refinement:

```text
rank-average ensemble:
ref_native_summary_d6_custom025
ref_native_indicator_d6_sqrt
ref_xgb_summary_d5_spw050

Threshold: max recall with validation precision >= 0.20
Test PR-AUC 0.1943
Test ROC-AUC 0.6543
Test recall 0.3238
Test precision 0.2068
Test F1 0.2524
Test accuracy 0.8278
```

Lift-table interpretation for the refined native CatBoost model:

```text
Top 1% highest-risk patients: precision 0.4667, recall 0.0520, lift 5.20x
Top 5% highest-risk patients: precision 0.2857, recall 0.1592, lift 3.18x
Top 10% highest-risk patients: precision 0.2305, recall 0.2569, lift 2.57x
Top 20% highest-risk patients: precision 0.1700, recall 0.3790, lift 1.89x
```

Imbalance-handling conclusion:

The best way to work with the imbalance was not to synthetically rebalance the dataset. It was to keep validation/test naturally imbalanced, use class-weighted boosting, tune thresholds on validation, and report lift/risk-group performance. This produced the strongest PR-AUC observed in the project, with a small improvement over the previous native CatBoost result.

## Feature-Engineering and Final Search Loop

Scripts:

```text
feature_engineering_search.py
feature_engineering_ensemble.py
catboost_tuning_search.py
all_encounters_group_split_search.py
```

What was tried:

- ICD-9 chapter, prefix, and numeric diagnosis detail.
- Elixhauser-like comorbidity flags and comorbidity count.
- Medication class summaries, insulin-change flags, medication up/down counts.
- Utilization flags and interactions such as prior acute visits, frequent inpatient use, long stay, labs per day, medications per day, and age by utilization interactions.
- A1C/glucose measurement and high-result flags.
- Administrative risk flags for discharge/admission/source patterns.
- Native CatBoost on engineered features.
- Score-average and rank-average ensembles mixing engineered and earlier native CatBoost models.
- A focused CatBoost tuning loop with validation early stopping, deeper trees, custom positive-class weights, underbagged CatBoost, and validation-selected ensembles.

Best first-encounter validation PR-AUC in this loop:

```text
catboost_tuning_ensemble
score_average_3_d6_lr0015_l210_sqrt_rs1__d6_lr0015_l210_custom025_rs1__d6_lr0015_l210_custom025_rs1
Validation PR-AUC 0.2057
Validation ROC-AUC 0.6698
Validation recall 0.4346
Validation precision 0.1787
Validation F1 0.2533
Validation accuracy 0.7702
```

However, this did not transfer to the original first-encounter held-out test set:

```text
Best catboost_tuning_search.py selected test PR-AUC: 0.1971
Best catboost_tuning_search.py selected test F1: 0.2509
```

This is treated as validation overfitting from repeated tuning, not as the final model.

Best first-encounter held-out PR-AUC after feature engineering and ensembling:

```text
avg_3_fe_rawage_d6_custom025__old_summary_d6_sqrt__old_indicator_d6_sqrt
Threshold: max recall with validation precision >= 0.20

Test PR-AUC 0.2002
Test ROC-AUC 0.6568
Test recall 0.3195
Test precision 0.2024
Test F1 0.2478
Test accuracy 0.8259
Confusion matrix: TN 8368, FP 1186, FN 641, TP 301
```

Best first-encounter held-out F1 after feature engineering:

```text
fe_weight_indicator_core
FE_NativeCat_d6_lr0.015_l210_SqrtBalanced
Threshold: max recall with validation precision >= 0.20

Test PR-AUC 0.1941
Test ROC-AUC 0.6571
Test recall 0.3142
Test precision 0.2119
Test F1 0.2531
Test accuracy 0.8336
Confusion matrix: TN 8453, FP 1101, FN 646, TP 296
```

Feature-engineering conclusion:

The engineered features and small ensembles raised the best first-encounter PR-AUC from 0.1991 to 0.2002 and the best F1 from 0.2524 to 0.2531. This is a real but very small improvement. The aggressive CatBoost tuning loop improved validation PR-AUC but not test PR-AUC, so it should be documented as an explored path rather than used as the final result.

## Alternate Row-Scope Experiment

The accepted course-project row scope remains first encounter per patient with hospice/expired discharges removed. Because the first-encounter models plateaued around PR-AUC 0.20, an additional experiment tested whether the row-scope decision was limiting signal.

Alternate scope:

```text
Use all eligible encounters.
Remove hospice/expired discharges.
Split by patient_nbr so no patient appears in more than one split.
Keep readmitted_30 as the same binary target.
```

Patient-group split summary:

```text
Train rows 69,444, patients 48,992, positive rate 11.38%
Validation rows 15,071, patients 10,499, positive rate 11.80%
Test rows 14,828, patients 10,499, positive rate 11.03%
```

Best alternate-scope validation result:

```text
all_enc_fe_indicator
AllEncCat_d6_lr0015_l210_custom025
Validation PR-AUC 0.2705
Validation ROC-AUC 0.7001
Validation recall 0.3935
Validation precision 0.2738
Validation F1 0.3229
Validation accuracy 0.8052
```

Best alternate-scope held-out test PR-AUC:

```text
all_enc_fe_summary
AllEncCat_d6_lr0015_l210_custom025
Threshold: best F1 selected on validation

Test PR-AUC 0.2290
Test ROC-AUC 0.6797
Test recall 0.3817
Test precision 0.2214
Test F1 0.2802
Test accuracy 0.7838
Confusion matrix: TN 10998, FP 2195, FN 1011, TP 624
```

Best alternate-scope held-out F1:

```text
all_enc_fe_indicator
AllEncCat_d6_lr0015_l210_custom025
Threshold: best F1 selected on validation

Test PR-AUC 0.2290
Test ROC-AUC 0.6794
Test recall 0.3498
Test precision 0.2357
Test F1 0.2816
Test accuracy 0.8032
Confusion matrix: TN 11338, FP 1855, FN 1063, TP 572
```

Alternate-scope interpretation:

Using all encounters with patient-level splitting gives materially better performance than the first-encounter setup. This suggests later encounters contain useful risk signal, and the first-encounter-only design is conservative. The result is not directly comparable to the accepted first-encounter test set because the modeling population changed, but it is the strongest practical direction if the project is allowed to change row scope.

## Paper Reproduction Check

Script:

```text
paper_reproduction_search.py
```

Reference paper:

```text
Bhuvan et al., "Identifying Diabetic Patients with High Risk of Readmission", arXiv:1602.04257.
```

What the paper did differently from the accepted project setup:

- Used all encounter rows after preprocessing, not first encounter only.
- Dropped `weight`, `payer_code`, and `medical_specialty`.
- Removed rows with missing race or diagnosis values.
- Grouped ICD-9 diagnosis codes.
- Kept only insulin from the individual medication columns.
- Used 22 risk factors.
- Used a random 75/25 train/test split.
- Tuned model parameters by five-fold cross-validation on the training set.
- Reported PR-AUC for `<30` vs `>30/NO`; their Random Forest result was 0.242.

The visible paper setup was reproduced with:

```text
Rows after paper-style filtering: 98,053
Positive rate: 11.29%
Random Forest: 250 trees, max_depth=5
One-hot encoded categorical variables
Random 75/25 train/test split
```

Observed reproduction result:

```text
Paper-style all rows, random 75/25 split
PaperRF_250_depth5_unweighted
Test PR-AUC 0.2083
Test ROC-AUC 0.6564
Best-F1 recall 0.4203
Best-F1 precision 0.2079
Best-F1 F1 0.2782
Patient overlap between train and test: 7,906 patients
```

Patient-safe version of the same reproduction:

```text
Paper-style all rows, patient-group 75/25 split
PaperRF_250_depth5_unweighted
Test PR-AUC 0.2027
Test ROC-AUC 0.6510
Best-F1 recall 0.4889
Best-F1 precision 0.1849
Best-F1 F1 0.2683
Patient overlap between train and test: 0 patients
```

First-encounter version of the paper-style features:

```text
First encounter + hospice/expired removed + paper-style features
PaperRF_250_depth5_unweighted
Test PR-AUC 0.1655
Test ROC-AUC 0.6425
Best-F1 recall 0.3182
Best-F1 precision 0.1884
Best-F1 F1 0.2366
```

Seed and RF-depth sensitivity:

```text
paper_rf_sensitivity_results.csv
20 random 75/25 seeds with PaperRF_250_depth5_unweighted:
Mean PR-AUC 0.2172
Max PR-AUC 0.2242
Min PR-AUC 0.2096

Small RF/ExtraTrees hyperparameter grid:
Best fixed-seed PR-AUC 0.2167
```

Additional preprocessing/metric sensitivity:

```text
paper_preprocessing_metric_sensitivity.csv
Compared:
- stratified vs unstratified random 75/25 split
- scikit average precision vs trapezoidal PR-AUC
- one-hot vs ordinal categorical encoding
- default Random Forest feature sampling vs max_features=5/log2

Best observed PR-AUC: 0.2242
Best observed trapezoidal PR-AUC: 0.2237
```

Paper reproduction conclusion:

The paper's broad preprocessing direction confirms the most important project finding: using all encounters gives better apparent performance than first encounter only. However, the visible paper setup did not reproduce their 0.242 PR-AUC locally. The best paper-style reproduction result was 0.2242 across 20 random seeds, while the best all-encounter CatBoost patient-group split result remained stronger at about 0.2290 PR-AUC. Therefore, copying the paper exactly is not better than the current all-encounter CatBoost approach.

## Balanced-Test Sensitivity Check

Script:

```text
balanced_test_evaluation.py
```

This is not the official final test score. It is a sensitivity check requested after the imbalance experiments. The balanced test set was created only from the original held-out test split:

```text
Positive test cases kept: 942
Negative test cases sampled: 942
Balanced test rows: 1,884
Positive rate: 50%
```

Best result on the balanced test by PR-AUC:

```text
Rank-average ensemble:
ref_native_summary_d6_custom025
ref_native_indicator_d6_sqrt
ref_xgb_summary_d5_spw050
Threshold: default 0.5

Balanced-test PR-AUC 0.6583
Balanced-test ROC-AUC 0.6531
Balanced-test recall 0.6168
Balanced-test precision 0.6129
Balanced-test F1 0.6148
Balanced-test accuracy 0.6136
Confusion matrix: TN 575, FP 367, FN 361, TP 581
```

Best refined native CatBoost result on the balanced test:

```text
RefinedNativeCat_d6_lr0.015_l210.0_SqrtBalanced
Threshold: max recall with validation precision >= 0.15

Balanced-test PR-AUC 0.6583
Balanced-test ROC-AUC 0.6516
Balanced-test recall 0.5011
Balanced-test precision 0.6170
Balanced-test F1 0.5530
Balanced-test accuracy 0.5950
Confusion matrix: TN 649, FP 293, FN 470, TP 472
```

Balanced-test interpretation:

The balanced test makes precision, F1, and accuracy look much higher because half the rows are positives. ROC-AUC stays around 0.65, similar to the naturally imbalanced test, because ranking ability did not fundamentally change. This confirms that the model has real ranking signal, but the official result should still use the naturally imbalanced test set because that matches the real dataset prevalence.

## Recommended Final Results to Report

## Presentation-Ready Framing

The best result to lead with is the alternate all-encounter patient-group split model, because it is the strongest performance found while still preventing patient overlap between train, validation, and test.

```text
Final performance headline:
All-encounter patient-safe CatBoost
Test PR-AUC 0.2290
Test ROC-AUC 0.6797
Test F1 0.2802
Test accuracy 0.7838
```

Why this is a strong result:

```text
Natural test positive rate / majority PR-AUC baseline: 0.1103
Best model PR-AUC: 0.2290
Relative improvement over baseline: about 2.1x
```

Comparison to the reproduced paper-style Random Forest:

```text
Paper-style RF, fixed random split: PR-AUC 0.2083
Paper-style RF, patient-safe split: PR-AUC 0.2027
Paper-style RF, best of 20 random seeds: PR-AUC 0.2242
Our patient-safe CatBoost: PR-AUC 0.2290
```

Risk-ranking interpretation:

```text
Top 1% highest-risk encounters: precision 49.7%, lift 4.5x
Top 5% highest-risk encounters: precision 31.9%, lift 2.9x
Top 10% highest-risk encounters: precision 27.0%, recall 24.5%, lift 2.45x
Top 20% highest-risk encounters: precision 21.8%, recall 39.6%, lift 2.0x
```

Recommended wording:

```text
The model is best viewed as a patient risk-ranking tool. It does not perfectly classify every readmission, but it more than doubles PR-AUC over the natural baseline and concentrates readmission risk strongly in the highest-risk groups. For example, the top 10% risk group has a 27% readmission rate compared with 11% overall.
```

Primary validation-selected model family:

```text
Refined native CatBoost with raw administrative IDs, paper age groups, medication summaries only,
rare/unseen category handling, class weighting, and validation-selected threshold tuning.
```

Most defensible final validation-selected model:

```text
imb_ref_age_paper_summaries_only_rare100
RefinedNativeCat_d6_lr0.015_l210.0_customPW0.25

Validation PR-AUC 0.1982
Validation ROC-AUC 0.6666
Validation recall 0.3518
Validation precision 0.1917
Validation F1 0.2481
Validation accuracy 0.8089

Selected test PR-AUC 0.1978
Selected test ROC-AUC 0.6538
Selected test recall 0.3408
Selected test precision 0.1871
Selected test F1 0.2415
Selected test accuracy 0.8079
```

Best selected held-out PR-AUC observed:

```text
imb_ref_age_paper_summaries_only_rare100
RefinedNativeCat_d6_lr0.015_l210.0_SqrtBalanced
Threshold: max recall with validation precision >= 0.20

Test PR-AUC 0.1991
Test ROC-AUC 0.6535
Test recall 0.3217
Test precision 0.1979
Test F1 0.2450
Test accuracy 0.8221
```

Best selected held-out F1 observed:

```text
Rank-average ensemble:
ref_native_summary_d6_custom025
ref_native_indicator_d6_sqrt
ref_xgb_summary_d5_spw050
Threshold: max recall with validation precision >= 0.20

Test PR-AUC 0.1943
Test ROC-AUC 0.6543
Test recall 0.3238
Test precision 0.2068
Test F1 0.2524
Test accuracy 0.8278
```

## Practical Interpretation

The final models are substantially better than the majority baseline and the first modeling notebook:

```text
Majority baseline PR-AUC: 0.0897
First notebook best selected test PR-AUC: 0.1457
Broad search best selected test PR-AUC: 0.1834
Targeted search best selected test PR-AUC: 0.1887
Native CatBoost best selected test PR-AUC: 0.1983
Best neural selected test PR-AUC: 0.1824
Imbalance-refined best selected test PR-AUC: 0.1991
Feature-engineered first-encounter ensemble best selected test PR-AUC: 0.2002
Feature-engineered first-encounter best selected test F1: 0.2531
Alternate all-encounter patient-group split best selected test PR-AUC: 0.2290
Alternate all-encounter patient-group split best selected test F1: 0.2816
```

The models are still not clinically strong in an absolute sense. Precision remains around 0.20 at useful thresholds, meaning many flagged patients are false positives. However, this is a clear course-project improvement: the model can rank and identify higher-risk patients much better than the baseline while maintaining a transparent validation/test workflow.
