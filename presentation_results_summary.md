# Presentation-Ready Results Summary

Use this framing when presenting the final model. It is intentionally positive, but it does not inflate or invent results.

## Main Result to Lead With

Our strongest model is an all-encounter, patient-safe CatBoost model:

```text
Model: AllEncCat_d6_lr0015_l210_custom025
Split: patient-group train/validation/test split
No patient appears in more than one split.

Test PR-AUC: 0.2290
Test ROC-AUC: 0.6797
Test recall: 0.3817
Test precision: 0.2214
Test F1: 0.2802
Test accuracy: 0.7838
```

Recommended wording:

```text
The final model is best understood as a readmission risk-ranking model. On a patient safe held-out test split, it more than doubled PR-AUC over the natural positive rate baseline and produced meaningful lift in the highest-risk patient groups.
```

## Why This Looks Good

Baseline prevalence in the all-encounter patient-safe test set:

```text
Positive rate / majority PR-AUC baseline: 0.1103
Best model PR-AUC: 0.2290
Relative improvement: about 2.1x baseline
```

Comparison to the reproduced paper-style Random Forest:

```text
Paper-style RF, fixed random split: PR-AUC 0.2083
Paper-style RF, patient-safe split: PR-AUC 0.2027
Paper-style RF, best of 20 random seeds: PR-AUC 0.2242
Our patient-safe CatBoost: PR-AUC 0.2290
```

Recommended wording:

```text
The published paper reports a higher number, but using the visible method details we could not reproduce that score locally. Our patient-safe CatBoost outperformed the locally reproduced paper-style Random Forest while avoiding patient overlap between train and test.
```

## Best Risk-Ranking Story

The model is most useful for prioritizing patients for follow-up, not for making a perfect yes/no prediction.

```text
Top 1% highest-risk encounters:
Precision 49.7%
Lift 4.5x over baseline

Top 5% highest-risk encounters:
Precision 31.9%
Lift 2.9x over baseline

Top 10% highest-risk encounters:
Precision 27.0%
Recall 24.5%
Lift 2.45x over baseline

Top 20% highest-risk encounters:
Precision 21.8%
Recall 39.6%
Lift 2.0x over baseline
```

Recommended wording:

```text
If the hospital reviews only the top 10% highest-risk encounters the readmission rate in that group is about 27%, compared with 11% overall. That is a 2.45x concentration of risk and captures about one quarter of all 30-day readmissions.
```

## Honest Caveat

Do not present this as clinically deployable.

Recommended wording:

```text
The model is not a standalone clinical decision system. Its precision is still modest because 30-day readmission is difficult and imbalanced. The value is in risk stratification: it meaningfully concentrates readmission risk into a smaller group that can be prioritized for additional review or discharge planning.
```

## One-Slide Version

```text
Final model: Patient-safe CatBoost risk-ranking model
Test PR-AUC: 0.2290, about 2.1x the baseline positive rate
ROC-AUC: 0.6797
Best F1: 0.2816
Top 10% risk group: 27.0% readmission rate, 2.45x lift
Beat locally reproduced paper-style RF: 0.2290 vs 0.2242 best seed
Main takeaway: useful for prioritizing high-risk patients, not perfect binary prediction
```
