# Report Claims To Use

Use these claims because they are supported by the saved result files.

## Problem Claim

Predicting 30-day hospital readmission is useful because hospitals can prioritize high-risk patients for follow-up and discharge planning.

## Data Claim

The raw UCI file has 101,766 encounters. After removing hospice/expired discharge categories, the all-eligible modeling scope has 99,343 encounters and 69,990 patients.

## Imbalance Claim

The positive class is rare. In the patient-safe held-out test set, the positive rate is about 11.0%, so PR-AUC is more informative than accuracy.

## Model Selection Claim

CatBoost performed best among the compared model families. It beat XGBoost, LightGBM, Random Forest, Extra Trees, neural networks, and the prevalence baseline on validation PR-AUC.

## Final Result Claim

The validation-selected final CatBoost model achieved:

```text
PR-AUC 0.2414
ROC-AUC 0.6827
Recall 0.3621
Precision 0.2416
F1 0.2898
Accuracy 0.8044
```

These threshold-sensitive metrics use the official validation-selected threshold. A diagnostic test-best-F1 threshold for the same score vector gives recall 0.4226, precision 0.2223, F1 0.2913, and accuracy 0.7733, but it should not be presented as validation-selected.

The best observed exploratory variant reached PR-AUC 0.2415, but it is seed/order-sensitive.

## Practical Value Claim

The model is more useful for risk ranking than hard classification. The best observed variant had:

```text
Top 1% risk group: 55.0% readmission precision, 4.99x lift
Top 5% risk group: 34.5% precision, 3.13x lift
Top 10% risk group: 28.0% precision, 2.54x lift
```

## Plateau Claim

The plateau is likely due to dataset limitations and target noise, not lack of model complexity. Neural networks, XGBoost, LightGBM, Random Forests, Extra Trees, stacking, Optuna, CatBoost seed/order sweeps, bootstrap variants, and feature engineering were all tested.

## Limitation Claim

The dataset lacks richer clinical and post-discharge information: continuous lab values, vitals, medication doses, notes, discharge plans, exact dates, follow-up care, social determinants, and hospital/provider identifiers.
