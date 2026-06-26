# Presentation Script

Use this script with `AI1215_Final_Presentation.pptx`.

Suggested timing: 15 minutes total + Q&A.

## Speaker 1: Arthur

Timing: about 4 minutes

Slides: 1 to 5

### Slide 1: Title

Arthur: Hello everyone. We are the Readmission Risk Team. My name is Arthur, and I am here with Mohamed and Omar. Our project is Hospital Readmission Prediction.

Arthur: The goal is to predict whether a diabetic hospital encounter will be followed by readmission within 30 days. We treat the model mainly as a risk-ranking tool, meaning it helps identify which patients may need more follow-up attention after discharge.

### Slide 2: Presentation Roadmap

Arthur: The presentation follows the project requirements. First, I will explain the problem, motivation, and data. Then Mohamed will cover preprocessing, feature engineering, metrics, and model selection. Omar will present the final results, limitations, conclusion, and demo path.

### Slide 3: Problem & Motivation

Arthur: Thirty-day readmission matters because it is stressful for patients and expensive for hospitals. For diabetes patients, readmission can be related to unresolved illness, medication instability, poor follow-up access, or weak discharge support.

Arthur: Our practical goal is not to replace doctors. The practical goal is to rank patients by risk so hospitals can prioritize low-risk interventions such as follow-up calls, appointment reminders, medication review, or case-manager review.

### Slide 4: Data Source and Target

Arthur: We used the public UCI Diabetes 130-US Hospitals dataset. It contains 101,766 hospital encounters from 71,518 patients, with 50 original columns.

Arthur: We created the target variable `readmitted_30`. It equals 1 when the original `readmitted` value is `<30`, and 0 when it is `>30` or `NO`.

Arthur: We removed hospice and expired discharge cases because readmission prediction is not meaningful in the same way for patients who died or were discharged to hospice. After this removal, the eligible dataset had 99,343 encounters and 69,990 patients.

### Slide 5: EDA: Why This Is Hard

Arthur: The first challenge is class imbalance. Only about 11% of eligible encounters are positive cases. This means accuracy can be misleading because a model can get high accuracy by predicting "not readmitted" most of the time.

Arthur: The second challenge is missingness. Weight is missing in most rows, and payer code and medical specialty are also sparse. So we handled missing values carefully rather than treating all missingness as a simple numeric imputation problem.

Arthur: I will now pass to Mohamed, who will explain the methodology.

## Speaker 2: Mohamed

Timing: about 5 minutes

Slides: 6 to 10

### Slide 6: Preprocessing Pipeline

Mohamed: To avoid leakage, we used a patient-safe split. This means all encounters from the same patient stay in only one split: train, validation, or test.

Mohamed: The pipeline loads the raw CSV, creates the target, removes hospice and expired cases, splits by patient ID, builds the final features, trains CatBoost, and evaluates on the held-out test set.

Mohamed: The important point is that `encounter_id` and `patient_nbr` are used only for ordering and splitting. They are not used as direct model predictors.

### Slide 7: Feature Engineering

Mohamed: We created several groups of features.

Mohamed: For diagnoses, we grouped ICD-9 codes into broader categories and added diagnosis-detail indicators. This reduced sparsity while keeping medical meaning.

Mohamed: For medication, we used medication burden, medication changes, insulin status, and diabetes medication class indicators. For utilization, we used prior inpatient, emergency, and outpatient counts.

Mohamed: The most important extra signal came from prior patient history. For patients with earlier encounters in the dataset, we added features like prior readmission count, prior readmission rate, and previous utilization. These were computed only from earlier visits, never future visits.

### Slide 8: Evaluation Metric

Mohamed: Because the positive class is rare, our primary metric is PR-AUC. The natural PR-AUC baseline is the positive rate, which is 0.1103 on the test split.

Mohamed: A majority-class model has high accuracy, but it has zero recall and zero F1 because it catches no readmissions. So accuracy is reported, but it is not the headline metric.

Mohamed: We also report ROC-AUC, recall, precision, F1, accuracy, and confusion matrix so the tradeoff is clear.

### Slide 9: Models Explored

Mohamed: We compared many model families. We started with baselines like majority class, Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, and Naive Bayes where compatible.

Mohamed: Then we tested stronger tabular models such as XGBoost, LightGBM, and CatBoost. We also tried neural networks, imbalance methods, ensembles, seed sweeps, bootstrap variants, and Optuna tuning.

Mohamed: CatBoost performed best because this dataset contains many categorical and administrative variables, and CatBoost handles categorical structure natively.

### Slide 10: Validation Model Comparison

Mohamed: On validation PR-AUC, CatBoost was the strongest model family. CatBoost reached about 0.288 validation PR-AUC. The next best methods, such as the logistic stacker and XGBoost, were lower. Neural networks were much lower.

Mohamed: This supported choosing CatBoost as the final model. The final setup used engineered features, categorical interactions, patient-history features, negative-ratio refinement, and validation-based threshold selection.

Mohamed: Now Omar will explain the final results.

## Speaker 3: Omar

Timing: about 6 minutes

Slides: 11 to 16

### Slide 11: Final Test Performance

Omar: On the held-out test set, the final validation-selected CatBoost model achieved PR-AUC 0.2414 and ROC-AUC 0.6827.

Omar: At the validation-selected threshold, recall was 36.2%, precision was 24.2%, F1 was 0.2898, and accuracy was 0.8044.

Omar: The model found 592 of the 1,635 true readmissions in the test set. The majority baseline has higher accuracy, but it finds zero readmissions, so it is not useful for this task.

### Slide 12: Risk Stratification

Omar: The strongest practical use is risk stratification.

Omar: If a hospital reviews only the top 10% highest-risk encounters, the readmission rate in that group is about 27.8%, compared with about 11.0% overall. The top 5% has 33.4% precision, and the top 1% has 52.3% precision.

Omar: So the model is not perfect at classifying every patient, but it does concentrate risk into a smaller group that a care team can review first.

### Slide 13: What Worked and What Did Not

Omar: What worked best was patient-safe all-encounter modeling, prior patient-history features, native CatBoost categorical handling, and validation-based threshold selection.

Omar: What did not beat the final CatBoost setup were neural networks, heavy imbalance handling, complex ensembles, and Optuna tuning. Some methods improved recall, but usually hurt precision too much.

### Slide 14: Why Performance Plateaus

Omar: The performance plateau is mainly a data limitation. Readmission depends on many factors that are not in the dataset.

Omar: The dataset does not include exact dates, vitals, continuous lab values, medication doses, clinical notes, discharge plans, follow-up appointments, home support, or social determinants.

Omar: Also, our patient-safe split is stricter than random row splitting. It prevents the model from seeing the same patient in both training and testing, which makes the result more honest.

### Slide 15: Demo and Reproducibility

Omar: The final model can be reproduced by running:

`python FINAL_MODEL_PIPELINE.py`

Omar: For a demo, we can run:

`python FINAL_MODEL_PIPELINE.py --interactive-predict`

Omar: The interactive mode asks for patient encounter information one field at a time and prints the estimated 30-day readmission risk.

### Slide 16: Conclusion

Omar: In conclusion, we built a reproducible end-to-end ML pipeline that more than doubles the PR-AUC baseline and produces useful patient risk ranking.

Omar: Our final PR-AUC of 0.2414 is essentially in the same range as the closest paper's reported PR-AUC for the same `<30` target.

Omar: The model should be used as a screening aid for human review, not as an automatic clinical decision-maker. The biggest future improvements would likely come from richer clinical and post-discharge data.

Omar: Thank you. We are happy to answer your questions.
