# Presentation Script

Use the main script for timing. Use the "optional backup detail" lines if the slide is light, if the professor asks, or if you want to sound more technical during Q&A.

## Speaker 1 (Arthur)

Timing: ~4 Minutes (Problem & Motivation + Data Collection)

Slides: 1 to 7

### [Slide 1 & 2: Title & Outline]

Arthur: Hello everyone. We are the Readmission Risk Team. My name is Arthur, and I am here with my teammates Mohamed and Omar. Today, we are presenting our machine learning project: predicting 30-day hospital readmission for diabetic patients.

Arthur: The presentation has four parts. First, I will explain the problem, motivation, and data source. Then Mohamed will explain our preprocessing, feature engineering, metrics, and model search. Finally, Omar will present the final results, practical interpretation, limitations, and next steps.

Optional backup detail:

- Our task is supervised binary classification, but we interpret the output mainly as a risk score.
- The goal is not to replace doctors. The goal is to help care teams prioritize follow-up resources.

### [Slide 3 & 4: Problem & Motivation]

Arthur: Let us start with the problem. Thirty-day readmission is important because it is stressful for patients and costly for hospitals. For diabetes encounters, early readmission can happen because of unresolved illness, medication instability, insufficient discharge planning, poor follow-up access, or lack of support after discharge.

Arthur: In our eligible encounter data, about 11% of encounters led to readmission within 30 days. So roughly 1 in 9 encounters is a positive case. This is exactly the kind of problem where a hospital may not have enough resources to give every patient intensive follow-up, but it can prioritize the highest-risk patients.

Arthur: Our objective was to predict whether a diabetic hospital encounter would be followed by readmission within 30 days. More practically, we wanted to rank discharged patients by risk so that the highest-risk patients can be considered for low-risk interventions such as follow-up calls, medication reconciliation, appointment reminders, or case-manager review.

Optional backup detail:

- We are predicting the short-term `<30` target, which is harder than predicting any readmission.
- We care about finding positive cases, not just maximizing overall accuracy.
- A false positive is not necessarily dangerous if the action is a follow-up call, but a false negative means a readmission risk may be missed.

### [Slide 5 & 6: Data Collection]

Arthur: For the data, we used the Diabetes 130-US Hospitals dataset from the UCI Machine Learning Repository. This is a public dataset covering hospital encounters from 1999 to 2008 across 130 US hospitals and integrated delivery networks.

Arthur: The raw file contains 101,766 encounters, 71,518 unique patients, and 50 original columns. The columns include demographics, admission and discharge information, diagnosis codes, lab indicators, medication variables, prior utilization counts, and the readmission outcome.

Arthur: Our target variable was binary. We created `readmitted_30`, where the value is 1 if the original `readmitted` column is `<30`, and 0 if it is `>30` or `NO`.

Arthur: Before modeling, we removed hospice and expired discharge cases, because readmission prediction is not clinically meaningful in the same way for patients who died or were discharged to hospice. After this filtering, the eligible all-encounter dataset had 99,343 rows and 69,990 patients.

Optional backup detail:

- The raw positive count was 11,357 out of 101,766 rows.
- After hospice/expired removal, the positive count was 11,314 out of 99,343 rows.
- The final patient-safe test split had 14,828 encounters and 1,635 positives.
- The hospice/expired discharge disposition IDs removed were 11, 13, 14, 19, 20, and 21.

### [Slide 7: Challenges in the Data]

Arthur: The dataset had three major challenges.

Arthur: First, class imbalance. Only around 11% of encounters are positive cases, so a model can look good by predicting "not readmitted" almost all the time.

Arthur: Second, missingness. Some fields are heavily missing. For example, weight is missing in most rows, and payer code and medical specialty are also sparse. We cannot treat every missing value as a simple numeric imputation problem.

Arthur: Third, repeated patients. Many patients appear more than once. If we randomly split rows, the same patient could appear in both training and testing. That would create patient leakage and make the model look better than it really is.

Arthur: I will now pass it to Mohamed, who will explain how we handled these challenges in preprocessing, feature engineering, and model selection.

Optional backup detail:

- This repeated-patient issue is one reason many online results on this dataset look inflated.
- Patient-safe splitting is stricter than a random row split.

## Speaker 2 (Mohamed)

Timing: ~5 Minutes (Methodology - Preprocessing, Metrics & Models)

Slides: 8 to 11

### [Slide 8 & 9: Methodology & Preprocessing]

Mohamed: Thank you, Arthur. To handle these data challenges, we built a preprocessing pipeline focused on leakage control, categorical handling, and feature engineering.

Mohamed: First, we used a patient-safe split. This means all visits from the same patient stayed in only one split: train, validation, or test. This was important because otherwise the model could partially recognize the same patient in both training and testing.

Mohamed: Second, we handled missing and categorical values carefully. We changed raw question marks into missing values, kept missing categories as `Missing`, and mapped very rare categories to `Other`. For lab results like A1C and glucose, we kept `None` as meaningful because it means the test was not performed.

Mohamed: Third, we cleaned up the medical and administrative features. Diagnosis codes were grouped into broader disease groups, such as diabetes, circulatory, and respiratory diseases. Medication columns were kept in a balanced way: we used overall medication-change summaries, but also kept indicators for major diabetes medication groups and insulin.

Mohamed: Finally, we added history and utilization features. We summarized previous outpatient, emergency, and inpatient usage. For patients who had earlier encounters in the dataset, we also added safe history features, such as previous readmissions and previous utilization. We only used earlier visits, never future visits.

Optional backup detail:

- Final split sizes: 69,444 train rows, 15,071 validation rows, and 14,828 test rows.
- Direct leakage columns dropped from predictors: `readmitted`, `readmitted_30`, `encounter_id`, and `patient_nbr`.
- `encounter_id` and `patient_nbr` were used only for ordering/history/splitting, not as model inputs.
- Rare categories were mapped to `Other` using the training set only, with `min_count=100`.
- Administrative code fields were treated as nominal categories, not ordered numeric values.
- Diagnosis grouping reduced hundreds of sparse ICD-9 codes into broader groups such as diabetes, circulatory, respiratory, digestive, injury, musculoskeletal, genitourinary, neoplasms, and other.
- Medication features were not fully collapsed into one variable. We used medication-burden summaries plus class-level indicators for metformin, sulfonylureas, meglitinides, thiazolidinediones, alpha-glucosidase inhibitors, combination drugs, and insulin.
- Utilization features included outpatient, emergency, and inpatient counts, plus log and total summaries.
- About 23% of patients had more than one encounter, and about 30% of eligible rows had a prior encounter available.
- History features included prior encounter count, prior 30-day readmission count/rate, prior any-readmission rate, immediately previous outcome, and previous utilization values.
- We tested first-encounter-only modeling first, but the all-encounter patient-safe setup performed better because valid history features became available.
- CatBoost handled categorical columns natively; scikit-learn baselines used one-hot/scaling where needed.

### [Slide 10: Evaluation Metric]

Mohamed: Next, we had to choose the right metric. Because the positive class is only about 11%, accuracy is misleading.

Mohamed: A majority-class model that predicts "not readmitted" for every patient gets about 89% test accuracy. But it has zero recall, zero precision, and zero F1 for the readmission class. It catches no readmissions.

Mohamed: So our primary metric was PR-AUC, or Precision-Recall Area Under the Curve. PR-AUC is appropriate because it focuses on the rare positive class and measures whether true readmission cases are ranked highly.

Mohamed: The natural test baseline for PR-AUC is the positive rate, which is 0.1103. Our final model needed to clearly beat that. We also reported ROC-AUC, recall, precision, F1, accuracy, and the confusion matrix so the tradeoffs were transparent.

Optional backup detail:

- ROC-AUC can look acceptable even when positive-class precision is weak, so PR-AUC is more informative here.
- Threshold-independent metrics: PR-AUC and ROC-AUC.
- Threshold-dependent metrics: recall, precision, F1, accuracy, and confusion matrix.
- We selected the final decision threshold using the validation best-F1 point, not the test set.

### [Slide 11: Models Explored]

Mohamed: For model selection, we compared many model families.

Mohamed: We started with standard models: majority baseline, Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, and Gaussian Naive Bayes where compatible.

Mohamed: Then we tried stronger tabular models: XGBoost, LightGBM, and CatBoost. We also tried neural networks, including embedding MLP-style models, plus imbalance methods, score and rank ensembles, logistic stackers, seed sweeps, bootstrap variants, negative-ratio refinement, and Optuna tuning.

Mohamed: All model selection decisions were made using the validation set. The held-out test set stayed untouched until final reporting and sensitivity checks.

Mohamed: CatBoost performed best. This made sense because our data has many categorical variables and combinations of administrative context, diagnosis patterns, utilization history, medication features, and patient history. CatBoost can use native categorical handling instead of relying only on sparse one-hot encoding.

Mohamed: Our final selected model was CatBoost with engineered features, categorical interactions, and prior patient-history features.

Mohamed: The exact final configuration was depth 6, learning rate 0.02, 1500 iterations, L2 leaf regularization 10, random strength 1, early stopping with `od_wait=140`, and evaluation metric PRAUC. The final training subset used all positives plus about 7.5 negatives per positive, with seed 37.

Mohamed: Now, Omar will present the results and explain what they mean.

Optional backup detail:

- Final model ID: `NegRefineCat_d6_lr002_neg7.5_seed37`.
- Best observed sensitivity variant used negative ratio 8.0 and seed 202, but we report the validation-selected seed-37 model as the disciplined final result.
- Optuna did not discover a materially better setup; it mostly confirmed the CatBoost plateau.

## Speaker 3 (Omar)

Timing: ~4 Minutes (Results, Insights & Conclusion)

Slides: 12 to 18

### [Slide 12 & 13: Results & Final Model Performance]

Omar: Thank you, Mohamed. Now let us look at the final results on the held-out test set.

Omar: Our validation-selected CatBoost model achieved a test PR-AUC of 0.2414. The baseline PR-AUC is 0.1103, so our model more than doubled the baseline. The best observed sensitivity variant reached 0.2415, but we use 0.2414 as the final number because it was selected based on validation.

Omar: The final model also achieved ROC-AUC 0.6827, recall 0.4226, precision 0.2223, F1 0.2913, and accuracy 0.7733.

Omar: The confusion matrix was: 10,775 true negatives, 2,418 false positives, 944 false negatives, and 691 true positives. So at the selected threshold, the model catches 691 of the 1,635 readmissions in the test set.

Omar: The accuracy is lower than the majority baseline, but that is expected. The majority baseline predicts "not readmitted" for everyone, so it gets high accuracy but finds zero positive cases. Our model accepts more false positives in exchange for finding real readmissions, which is more useful for triage.

Optional backup detail:

- Majority baseline test accuracy: 0.8897, but recall 0 and F1 0.
- Our final model accuracy: 0.7733, but recall 0.4226 and F1 0.2913.
- This is why accuracy should not be the headline metric.

### [Slide 14: Risk Stratification]

Omar: The best way to understand the model is as a risk-ranking tool.

Omar: If a hospital sorts patients by predicted risk and focuses only on the top 10% highest-risk encounters, the readmission precision is about 28%. That is much higher than the overall test readmission rate of about 11%.

Omar: For the validation-selected final model, the top 10% group captures about 25% of all readmissions. The top 5% has about 33.4% precision, and the top 1% has about 52.3% precision.

Omar: This means the model concentrates risk. It is not perfect at saying yes or no for every patient, but it is useful for deciding where a hospital should look first.

Optional backup detail:

- Validation-selected final model, top 1%: 149 encounters, 78 positives, 52.3% precision, 4.75x lift.
- Validation-selected final model, top 5%: 742 encounters, 248 positives, 33.4% precision, 3.03x lift.
- Validation-selected final model, top 10%: 1,483 encounters, 413 positives, 27.8% precision, 2.53x lift.
- Validation-selected final model, top 20%: 2,966 encounters, 662 positives, 22.3% precision, 2.02x lift.
- The best observed sensitivity variant was slightly higher at top 1% and top 5%, but we should present the validation-selected final model first.

### [Slide 15 & 16: What Worked and What Did Not]

Omar: During experimentation, an important improvement came from moving to the all-encounter patient-safe setup and adding prior-history features where they were available. This did not help every row equally, because many patients appear only once, but it gave extra signal for repeat patients without leaking future information.

Omar: CatBoost worked best overall. XGBoost, LightGBM, Random Forest, Extra Trees, logistic stacking, neural networks, imbalance methods, and ensembles were tested, but none gave a robust improvement over the final CatBoost setup.

Omar: Neural networks did not perform better. In our validation comparison, the best neural network was much lower than CatBoost on PR-AUC. Heavy imbalance handling also did not solve the problem. It often increased recall, but it reduced precision too much.

Omar: We also compared our result to the closest paper we found on this dataset. That paper reported about 0.242 PR-AUC for the `<30` readmission target. Our final result of 0.2414 essentially matches that level, while using a strict patient-safe evaluation design.

Optional backup detail:

- CatBoost validation PR-AUC: 0.2879.
- XGBoost validation PR-AUC: 0.2708.
- LightGBM validation PR-AUC: 0.2607.
- Random Forest validation PR-AUC: 0.2545.
- Extra Trees validation PR-AUC: 0.2523.
- Neural network validation PR-AUC: 0.1818.
- Baseline validation PR-AUC: 0.1180.
- The plateau appears around test PR-AUC 0.24 to 0.242.

### [Slide 17 & 18: Conclusion & Future Work]

Omar: In conclusion, we built a patient-safe readmission risk model that more than doubles the PR-AUC baseline and provides useful risk ranking.

Omar: The model is not a standalone clinical decision-maker. It should be used as a screening aid for low-risk interventions and human review.

Omar: The performance plateau is likely caused by data limitations, not by a lack of trying models. Readmission depends on many post-discharge events that are not in the dataset.

Omar: The dataset does not include continuous lab values, vital signs, clinical notes, medication doses, discharge plans, exact dates, follow-up appointment information, social determinants, or hospital/provider identifiers. It is also historical data from 1999 to 2008.

Omar: For future work, the biggest improvements would likely come from richer clinical and post-discharge data. We would also add probability calibration, subgroup fairness analysis, external validation on newer hospital data, and prospective testing before any real deployment.

Omar: Thank you very much for listening. We are now happy to answer your questions.

Optional backup detail:

- Our honest setup is stricter than many inflated online notebooks because we use patient-safe splitting.
- If asked why performance is not higher: the target is noisy, the class is rare, and key post-discharge predictors are missing.
- If asked what the model learned: risk is linked to prior utilization, patient-history patterns, admission/discharge context, diagnosis groups, medication burden, and lab-testing status.
