# Presentation-Ready Results Summary

Use this framing when presenting the final model. It is intentionally positive, but it does not inflate or invent results.

## Main Result to Lead With

Our strongest model after the plateau follow-up is an all-encounter, patient-safe CatBoost model with prior patient-history features:

```text
Model: HistoryTuneCat_d6_lr0015_l210_sqrt
Split: patient-group train/validation/test split
No patient appears in more than one split.

Test PR-AUC: 0.2389
Test ROC-AUC: 0.6838
Test recall: 0.3731
Test precision: 0.2400
Test F1: 0.2921
Test accuracy: 0.8006
```

Recommended wording:

```text
The final model is best understood as a readmission risk-ranking model. On a patient-safe held-out test split, it more than doubled PR-AUC over the natural positive rate baseline and produced meaningful lift in the highest-risk patient groups. The largest improvement came from adding prior patient-history features for the all-encounter setting.
```

## Why This Looks Good

Baseline prevalence in the all-encounter patient-safe test set:

```text
Positive rate / majority PR-AUC baseline: 0.1103
Best model PR-AUC: 0.2389
Relative improvement: about 2.17x baseline
```

Comparison to the reproduced paper-style Random Forest:

```text
Paper-style RF, fixed random split: PR-AUC 0.2083
Paper-style RF, patient-safe split: PR-AUC 0.2027
Paper-style RF, best of 20 random seeds: PR-AUC 0.2242
Our patient-safe CatBoost before patient-history features: PR-AUC 0.2290
Our patient-safe CatBoost with patient-history features: PR-AUC 0.2389
```

Recommended wording:

```text
The published paper reports a higher number, but using the visible method details we could not reproduce that score locally. Our patient-safe CatBoost outperformed the locally reproduced paper-style Random Forest while avoiding patient overlap between train and test. It is also very close to the paper's reported 0.242 PR-AUC, despite using a stricter patient-safe split.
```

## Best Risk-Ranking Story

The model is most useful for prioritizing patients for follow-up, not for making a perfect yes/no prediction.

```text
Top 1% highest-risk encounters:
Precision 51.0%
Lift 4.63x over baseline

Top 5% highest-risk encounters:
Precision 33.4%
Lift 3.03x over baseline

Top 10% highest-risk encounters:
Precision 28.2%
Recall 25.6%
Lift 2.56x over baseline

Top 20% highest-risk encounters:
Precision 22.3%
Recall 40.4%
Lift 2.02x over baseline
```

Recommended wording:

```text
If the hospital reviews only the top 10% highest-risk encounters, the readmission rate in that group is about 28%, compared with 11% overall. That is a 2.56x concentration of risk and captures about one quarter of all 30-day readmissions.
```

## Honest Caveat

Do not present this as clinically deployable.

Recommended wording:

```text
The model is not a standalone clinical decision system. Its precision is still modest because 30-day readmission is difficult and imbalanced. The value is in risk stratification: it meaningfully concentrates readmission risk into a smaller group that can be prioritized for additional review or discharge planning.
```

## One-Slide Version

```text
Final model: Patient-safe CatBoost risk-ranking model with prior patient-history features
Test PR-AUC: 0.2389, about 2.17x the baseline positive rate
ROC-AUC: 0.6838
Best F1: 0.2921
Top 10% risk group: 28.2% readmission rate, 2.56x lift
Beat locally reproduced paper-style RF: 0.2389 vs 0.2242 best seed
Main takeaway: useful for prioritizing high-risk patients, not perfect binary prediction
```
