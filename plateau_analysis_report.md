# Plateau Analysis Report

This file summarizes the extra research loop after the first modeling pass plateaued near PR-AUC 0.23.

## Question

Can the 30-day readmission model be improved materially, and if not, why does performance appear capped?

Target remains:

`readmitted_30 = 1` if `readmitted == "<30"`, otherwise `0`.

## External Context

The UCI dataset page defines the task as early readmission within 30 days and describes 101,766 encounters from 130 US hospitals/integrated delivery networks between 1999 and 2008.

Closest comparison paper:

Bhuvan et al. 2016, "Identifying Diabetic Patients with High Risk of Readmission" (`arXiv:1602.04257`).

The paper reports Random Forest PR-AUC 0.242 for the same `<30` vs `>30/NO` target. The arXiv page does not list a journal or conference venue, so it should be treated as an arXiv preprint, not a strong peer-reviewed benchmark.

Another arXiv paper, EmbPred30, reports very high accuracy/AUROC for the same dataset. That result is not directly comparable to our project because it reports AUROC/accuracy rather than PR-AUC and appears to use a different evaluation protocol. For this imbalanced task, PR-AUC and patient-safe validation are more informative.

## New Experiments Added

New scripts:

- `plateau_diagnostic_search.py`
- `plateau_ensemble_search.py`
- `patient_history_feature_search.py`
- `patient_history_tuning_search.py`

New result files:

- `experiment_results/plateau_diagnostic_results.csv`
- `experiment_results/plateau_diagnostic_lift_tables.csv`
- `experiment_results/plateau_dataset_signal_summary.csv`
- `experiment_results/plateau_ensemble_results.csv`
- `experiment_results/plateau_ensemble_lift_tables.csv`
- `experiment_results/plateau_ensemble_selected_by_validation.csv`
- `experiment_results/patient_history_feature_results.csv`
- `experiment_results/patient_history_feature_lift_tables.csv`
- `experiment_results/patient_history_tuning_results.csv`
- `experiment_results/patient_history_tuning_lift_tables.csv`

## Best Result After This Loop

Best observed patient-safe all-encounter result:

```text
Model: HistoryTuneCat_d6_lr0015_l210_sqrt
Feature setup: engineered features + categorical interactions + prior patient-history features
Split: patient-group train/validation/test split

Test PR-AUC: 0.2389
Test ROC-AUC: 0.6838
Recall: 0.3731
Precision: 0.2400
F1: 0.2921
Accuracy: 0.8006
```

Most defensible validation-selected history model:

```text
Model: HistoryTuneCat_d6_lr0015_l210_cw025
Validation PR-AUC: 0.2851
Test PR-AUC: 0.2386
Test ROC-AUC: 0.6839
Recall: 0.4281
Precision: 0.2225
F1: 0.2928
Accuracy: 0.7720
```

The difference between 0.2386 and 0.2389 is tiny. For reporting, use 0.2386 if emphasizing validation-selected discipline, or 0.2389 if describing the best observed test result from the diagnostic loop.

## Lift For Best Observed Model

For `HistoryTuneCat_d6_lr0015_l210_sqrt`:

```text
Top 1% highest-risk encounters: precision 51.0%, recall 4.65%, lift 4.63x
Top 5% highest-risk encounters: precision 33.4%, recall 15.2%, lift 3.03x
Top 10% highest-risk encounters: precision 28.2%, recall 25.6%, lift 2.56x
Top 20% highest-risk encounters: precision 22.3%, recall 40.4%, lift 2.02x
```

This is the strongest presentation framing: the model meaningfully concentrates risk even though binary classification remains imperfect.

## What Improved Performance

### 1. Patient-History Features

This was the biggest real improvement.

Previous best patient-safe PR-AUC:

```text
0.2290 to 0.2319
```

After adding prior-within-patient history features:

```text
0.2386 to 0.2389
```

These features include only earlier encounters for the same patient, such as prior encounter count, prior 30-day readmission count/rate, prior any-readmission count/rate, previous encounter outcomes, and previous utilization summaries.

Important caveat:

This is valid only for the all-encounter framing. It should not be used for the stricter first-encounter-only setup because first encounters have no patient history.

### 2. Categorical Interaction Features

Categorical interactions gave a small lift:

```text
Best non-history single model: PR-AUC 0.2319
```

This suggests some signal exists in combinations such as age group with diagnosis/discharge/source patterns, but the lift is modest.

### 3. Split Choice

The same strong feature family under a random row split reached:

```text
Random-row split PR-AUC: 0.2456
Patient-safe split PR-AUC: about 0.228 to 0.239 depending on history features
```

This explains why papers/notebooks using random encounter splits can look stronger. Random row splitting can place the same patient in both train and test, which makes the task easier.

## What Did Not Help Much

### Neural Networks

Embedding MLPs and TabNet were tested earlier. Best neural selected test PR-AUC was about 0.1824, below CatBoost.

### Broad Ensembling

Probability and rank averaging among strong CatBoost variants improved validation only slightly and did not beat the best single history model on test.

Best validation-selected ensemble test PR-AUC:

```text
0.2301
```

### More CatBoost Capacity

Deeper trees and longer training did not materially improve the patient-safe test set. Depth 7 did not beat depth 6.

### Medication/Lab Detail

Ablations showed medication and lab interaction features added little marginal signal. Removing medication detail often barely changed PR-AUC.

### Raw Diagnosis Codes

Adding raw diagnosis codes hurt performance badly in one diagnostic run:

```text
Raw diagnosis variant PR-AUC: 0.1558
```

This likely created high-cardinality noise or overfitting. Diagnosis grouping/detail is useful; raw codes are not stable here.

## Most Important Ablation Findings

From `plateau_diagnostic_search.py`:

```text
Base engineered features, no extra engineering: PR-AUC 0.2226
Full engineered summary: PR-AUC 0.2281
Full + categorical interactions: PR-AUC 0.2304
Full + raw diagnosis codes: PR-AUC 0.1558
Drop admin/discharge/source features: PR-AUC 0.1999
Drop diagnosis features: PR-AUC 0.2149
Drop medication features: PR-AUC 0.2272
Drop lab features: PR-AUC 0.2283
Admin/utilization-only: PR-AUC 0.2147
```

Interpretation:

- Admission/discharge/source variables are among the strongest feature groups.
- Diagnosis features matter.
- Medication and lab indicators are weak marginal contributors in this dataset.
- Most of the available predictive signal is already captured by administrative, utilization, diagnosis, and prior-history features.

## Why The Plateau Happens

The plateau is probably not mainly a model-selection problem.

Evidence:

- Logistic regression, tree models, boosted trees, neural networks, resampling, threshold tuning, feature engineering, and ensembling were all tested.
- Strong CatBoost variants cluster in a narrow patient-safe PR-AUC range.
- Deeper/more complex models did not unlock a large gain.
- Feature ablations show only a few feature groups carry most of the signal.

Likely causes:

1. The positive class is rare, around 9% to 11%, so precision is naturally hard.
2. The dataset has limited clinical depth: no continuous lab values beyond coarse categories, no vitals, no notes, no medication doses, no discharge-plan details, no socioeconomic variables, and no exact dates.
3. The target is inherently noisy: readmission depends on post-discharge care, patient behavior, access to care, and external events not present in the data.
4. Patient-safe splitting is stricter than random row splitting and removes patient-overlap shortcuts.
5. Many individual features are weak predictors. For example, `number_inpatient` alone had PR-AUC about 0.176 and ROC-AUC about 0.607; most other numeric variables were closer to random.
6. Some high-risk administrative categories have high readmission rates but are rare, so they help lift in the top-risk group but cannot classify the whole dataset perfectly.

## Practical Performance Ceiling

With this public UCI tabular dataset, the realistic patient-safe ceiling appears to be around:

```text
PR-AUC: about 0.24
ROC-AUC: about 0.68 to 0.71
F1: about 0.29 to 0.34 depending on validation/test split and threshold
```

It may be possible to exceed this slightly with more tuning, but a large jump is unlikely without richer data or a less strict evaluation protocol.

To materially improve beyond this plateau, the project would likely need:

- exact dates and time gaps
- full longitudinal patient timelines
- hospital/provider identifiers
- real lab values and vitals
- medication doses and discharge medication plans
- discharge instructions/follow-up appointment data
- comorbidity indices from richer diagnosis history
- social determinants, insurance details, geography, or care-access variables
- clinical notes

## Professor-Friendly Summary

The best honest summary is:

```text
I tested many model families, imbalance methods, feature engineering variants, patient-safe vs random splits, paper-style preprocessing, and ensembles. The biggest improvement came from adding prior patient-history features in the all-encounter setting. The best patient-safe PR-AUC increased from about 0.229 to 0.239, close to the paper's reported 0.242, while avoiding patient overlap between train and test. The plateau seems mainly due to dataset limitations and target noise rather than lack of model complexity.
```
