# What Worked, What Plateaued, And Main Conclusions

This is the file to read before writing the final report discussion/conclusion section.

## Best Final Result

The best clean model to report is:

```text
Model: NegRefineCat_d6_lr002_neg7.5_seed37
Method: CatBoost with engineered features, categorical interactions, and prior patient-history features
Split: patient-safe train/validation/test split
Test PR-AUC: 0.2414
ROC-AUC: 0.6827
Recall: 0.4226
Precision: 0.2223
F1: 0.2913
Accuracy: 0.7733
```

The best observed exploratory variant reached:

```text
PR-AUC: 0.2415
ROC-AUC: 0.6817
Recall: 0.4446
Precision: 0.2160
F1: 0.2907
Accuracy: 0.7608
```

Use `0.2414` as the disciplined final report number because it was selected from validation. Mention `0.2415` only as the best observed sensitivity result.

## What Worked Best

The biggest improvement came from using the all-encounter framing with patient-safe splitting and adding prior patient-history features.

The most useful feature ideas were:

- Prior patient encounters.
- Prior 30-day readmission count/rate.
- Prior any-readmission count/rate.
- Previous encounter outcome and utilization summaries.
- Administrative/discharge/source features.
- Diagnosis grouping rather than raw diagnosis-code memorization.
- Categorical interactions with CatBoost.

The strongest model family was CatBoost.

Why CatBoost likely worked best:

- The dataset has many categorical variables.
- CatBoost handles categorical patterns better than one-hot tree models in this setup.
- It performed best across validation comparisons.
- XGBoost, LightGBM, Random Forest, Extra Trees, neural networks, and heterogeneous stackers did not beat it.

The best practical framing is risk ranking:

```text
Top 1% highest-risk encounters: 55.0% readmission precision, 4.99x lift
Top 5% highest-risk encounters: 34.5% precision, 3.13x lift
Top 10% highest-risk encounters: 28.0% precision, 2.54x lift
```

This is stronger than saying the model is a yes/no classifier. It is not perfect at binary classification, but it is useful for prioritizing patients.

## What Did Not Help Much

These were tried and did not beat the final CatBoost setup:

- Neural networks: embedding MLP and TabNet.
- XGBoost and LightGBM.
- Random Forest and Extra Trees.
- Heterogeneous score/rank ensembles.
- Logistic stackers.
- Balanced CatBoost bagging.
- Deeper CatBoost trees.
- More CatBoost seeds.
- Bayesian, Bernoulli, MVS, no-bootstrap, and Ordered boosting variants.
- Optuna hyperparameter search.
- Raw diagnosis-code detail.
- Heavy class weighting.

Important detail:

Some ensembles improved validation PR-AUC slightly, but they did not beat the best single CatBoost model on test. This suggests the remaining differences are small and unstable.

## Why Performance Plateaued

The plateau is probably caused by the data and target, not by lack of model complexity.

Main reasons:

- The positive class is rare. The test positive rate is only about `11.0%`.
- PR-AUC is naturally hard when the base rate is low.
- Readmission is noisy and depends on many things not in the dataset.
- The dataset has no continuous lab values, vitals, clinical notes, medication doses, discharge plans, exact dates, or social determinants.
- The dataset is old, from 1999-2008.
- Patient-safe splitting is stricter than random row splitting because the same patient cannot appear in both train and test.
- Many available features are weak predictors by themselves.
- Random seeds and row order change the last few decimals, which means the model is near the data's signal limit.

The practical ceiling for this honest setup appears to be:

```text
PR-AUC: about 0.24 to 0.242
ROC-AUC: about 0.68 to 0.71
F1: about 0.29
```

## Most Important Interpretations

Accuracy should not be the main metric.

Reason:

The majority class is "not readmitted within 30 days", so a model can get high accuracy by predicting negative for almost everyone.

PR-AUC is the most important metric.

Reason:

It focuses on performance for the rare positive class and shows whether the model can find true readmission cases among high-risk predictions.

Recall and precision must be discussed together.

Reason:

Higher recall catches more readmissions but also creates more false positives. This is expected in an imbalanced clinical task.

The model is useful for prioritization.

Reason:

The top-risk groups have much higher readmission rates than the overall population. This supports use as a triage/risk-ranking tool.

## Comparison To The Paper

Closest paper:

`Bhuvan et al. 2016, Identifying Diabetic Patients with High Risk of Readmission`

Paper reported:

```text
Random Forest PR-AUC for <30 readmission: 0.242
```

Our result:

```text
Validation-selected CatBoost PR-AUC: 0.2414
Best observed CatBoost PR-AUC: 0.2415
```

Interpretation:

Our final result essentially matches the paper's reported PR-AUC while using a patient-safe split. This is a strong result for the course project, especially because patient-safe evaluation is stricter than a random encounter split.

## Final Conclusion For Report

The final model should be presented as follows:

```text
The final CatBoost model with prior patient-history features achieved PR-AUC 0.2414 on a patient-safe held-out test split, more than doubling the 0.1103 baseline positive rate. The model is most useful as a risk-ranking tool: the top 10% highest-risk encounters had about a 28% readmission rate compared with 11% overall. Performance plateaued around PR-AUC 0.24-0.242 despite extensive model and feature searches, suggesting the main limitation is the dataset's missing clinical and post-discharge information rather than model choice.
```

## Best Next Steps If This Were A Real Clinical Project

- Add exact dates and time gaps between encounters.
- Add continuous lab values and vital signs.
- Add medication dose and discharge medication data.
- Add discharge-plan and follow-up appointment information.
- Add clinical notes.
- Add social determinants and care-access variables.
- Validate on a newer external hospital dataset.
