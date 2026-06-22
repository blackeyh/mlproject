# Hospital Readmission Project - Preprocessing Decision Log

This document is the running record of preprocessing decisions for the hospital readmission prediction project.

The goal is to decide slowly and explicitly. Each decision should record:

- the question being decided
- the options compared
- evidence from EDA or course requirements
- the chosen option
- why it was chosen
- what implementation will follow from the choice
- any risks or assumptions

## Project Context

Project task: predict whether a diabetes patient encounter leads to hospital readmission within 30 days.

Dataset: UCI Diabetes 130-US Hospitals dataset at `archive/diabetic_data.csv`.

Associated dataset/source paper: `archive/description.pdf`, titled "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records."

Important nuance:
This paper is both the associated source paper that describes how the dataset was assembled and a research article focused on a specific HbA1c/readmission analysis. We treat its dataset construction details, feature definitions, and ICD-9 grouping table as authoritative references for understanding the data. We treat its modeling choices as strong references, not mandatory rules for our course project.

Main EDA notebook: `hospital_readmission_eda.ipynb`.

Target variable planned for modeling:

```text
readmitted_30 = 1 if readmitted == "<30"
readmitted_30 = 0 if readmitted == ">30" or readmitted == "NO"
```

Full raw encounter dataset:

```text
Total y samples: 101,766
Positive class: 11,357
Negative class: 90,409
Positive rate: 11.16%
```

Potential stricter modeling scope from EDA:

```text
First encounter per patient, excluding hospice/expired discharges
Total y samples: 69,973
Positive class: 6,277
Negative class: 63,696
Positive rate: 8.97%
```

## Current Workflow Rule

We will make preprocessing decisions one at a time.

After each decision, this file should be updated before moving to the next preprocessing topic.

From Decision 8 onward, we will focus detailed discussion on controversial or high-impact choices. Standard choices that are clearly required by the dataset documentation or basic ML practice can be documented briefly, while choices that affect leakage, target definition, feature meaning, fairness, model performance, or evaluation should receive full comparison and rationale.

## Decision 0 - Use a Dedicated Decision Log

Status: accepted

Question:
How should preprocessing decisions be documented?

Options compared:

1. Keep decisions only in chat.
2. Keep decisions only inside the notebook.
3. Keep a separate decision log and update it after each decision.

Choice:
Use a separate decision log: `preprocessing_decision_log.md`.

Reason:
The project needs a clear written explanation for the report and presentation. A separate log keeps the reasoning visible, avoids burying decisions inside code cells, and makes it easier to reuse the material later.

Implementation impact:
Every preprocessing choice should be added here before being implemented in the modeling notebook.

Risks / assumptions:
This log must be kept up to date manually as decisions are made.

## Decision 1 - Modeling Row Scope

Status: accepted

Question:
What rows should be used for the first modeling dataset?

Options to compare:

1. Use all encounters.
2. Use all encounters but split train/test by patient ID.
3. Use only the first encounter per patient.
4. Use only the first encounter per patient and remove hospice/expired discharges.

Evidence from EDA:

- The raw data has 101,766 encounters.
- The raw data has 71,518 unique patients.
- Some patients appear more than once, which creates possible train/test leakage.
- The original clinical study used one encounter per patient and removed hospice/death outcomes.

Calculations checked during discussion:

Full raw encounter dataset:

```text
Total encounters / y samples: 101,766
Positive class, readmitted <30: 11,357
Negative class, not readmitted <30: 90,409
Positive rate: 11.16%
Negative rate: 88.84%
```

Repeated-patient counts:

```text
Unique patients: 71,518
Patients with more than one visit: 16,773
All encounter rows belonging to repeated patients: 47,021
Extra visit rows beyond each patient's first visit: 30,248
```

Readmission among repeated-patient rows:

```text
Repeated-patient encounter rows: 47,021
Readmitted <30 among repeated-patient rows: 9,191
Percent readmitted <30 among repeated-patient rows: 19.55%
```

Readmission among only extra visits after the first visit:

```text
Extra visit rows after each patient's first visit: 30,248
Readmitted <30 among extra visit rows: 5,064
Percent readmitted <30 among extra visit rows: 16.74%
```

Patient-level repeated-visit readmission:

```text
Patients with more than one visit: 16,773
Repeated-visit patients with at least one <30-labeled encounter: 6,668
Percent: 39.75%
```

First-encounter-only dataset:

```text
Total y samples: 71,518
Positive class, readmitted <30: 6,293
Negative class, not readmitted <30: 65,225
Positive rate: 8.80%
Negative rate: 91.20%
```

First encounter per patient, excluding hospice/expired discharges:

```text
Total y samples: 69,973
Positive class, readmitted <30: 6,277
Negative class, not readmitted <30: 63,696
Positive rate: 8.97%
Negative rate: 91.03%
```

Thinking during discussion:

- Keeping all encounters gives the largest dataset and the largest number of positive cases.
- However, repeated patients create leakage risk if the same patient appears in both training and testing.
- We found that 16,773 patients appear more than once, so repeated-patient leakage is not a small edge case.
- We also found that repeated-patient rows have a higher readmission rate than the full dataset, so repeated encounters are not neutral duplicates; they represent a different risk profile.
- Using all encounters with a patient-level split could avoid leakage while keeping all rows, but it is more complex to explain and still leaves dependence between rows within the training data.
- Using only the first encounter per patient makes rows more independent and makes the project easier to explain.
- Removing hospice/expired discharge outcomes is clinically reasonable because readmission prediction is not meaningful in the same way for patients who died or were discharged to hospice.
- The stricter dataset still has 69,973 samples, which is enough for the course project.

Decision:
Use only the first encounter per patient and remove hospice/expired discharges.

Why this option was chosen:

This is the cleanest and most defensible modeling setup. It avoids repeated-patient leakage, keeps one row per patient, follows the original study's logic more closely, and removes cases where the readmission target is clinically distorted by death or hospice discharge. The dataset becomes smaller, but it remains large enough for reliable modeling and easier to explain in the written report and oral presentation.

Why we also follow the paper here:

The paper used one encounter per patient because repeated encounters from the same patient violate the independence assumption in their analysis. Even though our ML models do not all require the same statistical assumptions as logistic regression, repeated patients still create a practical leakage risk and can make evaluation too optimistic. The paper also removed hospice/death outcomes because readmission prediction is not meaningful in the same way for those discharges. Those reasons apply directly to our course project, so following the paper here improves both validity and explainability.

Implementation impact:

- Sort by `encounter_id`.
- Keep the first row per `patient_nbr`.
- Remove discharge disposition IDs associated with hospice or expired outcomes.
- Define `y` from the remaining rows.

Risks / assumptions:

- This reduces the dataset from 101,766 rows to 69,973 rows.
- It may remove useful information from later encounters.
- The final model would be framed as predicting readmission after a patient's first observed qualifying encounter in this dataset.
- The positive class rate becomes 8.97%, so the class imbalance becomes slightly stronger than in the full raw encounter dataset.

## Decision 2 - Target Definition

Status: accepted

Question:
How exactly should the prediction target `y` be defined after applying the accepted row scope?

Context:
The original dataset has a column called `readmitted` with three values:

```text
<30 = readmitted in less than 30 days
>30 = readmitted after more than 30 days
NO = no recorded readmission
```

The course topic is specifically:

```text
Predict whether a patient will be readmitted within 30 days of discharge.
```

Options to compare:

1. Binary target:

```text
Positive class = <30
Negative class = >30 or NO
```

2. Three-class target:

```text
Class 1 = <30
Class 2 = >30
Class 3 = NO
```

3. Drop `>30` rows and classify only:

```text
Positive class = <30
Negative class = NO
```

Calculations under the accepted row scope:

After applying Decision 1, the original `readmitted` distribution is:

```text
NO: 41,474
>30: 22,222
<30: 6,277
Total: 69,973
```

If converted to the binary target:

```text
Positive class, readmitted_30 = 1: 6,277
Negative class, readmitted_30 = 0: 63,696
Positive rate: 8.97%
Negative rate: 91.03%
```

Thinking during discussion:

- The course topic specifically asks whether the patient will be readmitted within 30 days.
- Therefore `<30` is the only positive class.
- A `>30` readmission is still a real readmission, but it is not a 30-day readmission.
- Dropping `>30` rows would remove 22,222 usable samples and would change the question into `<30` versus never readmitted, which is not exactly the assigned task.
- A three-class target could be interesting, but it would make the project more complex and would no longer directly match the requested binary classification framing.

Decision:
Use Option 1, the binary target.

Choice:

```text
readmitted_30 = 1 if readmitted == "<30"
readmitted_30 = 0 if readmitted == ">30" or readmitted == "NO"
```

Why this option was chosen:

This target definition directly matches the project question. It also keeps all rows from the accepted modeling scope and avoids turning the project into either a three-class classification task or a narrower `<30` versus `NO` task.

Why we also follow the paper here:

The paper's analysis is also centered on early readmission within 30 days. Using `<30` as the positive class aligns our supervised target with both the course prompt and the associated dataset paper's outcome framing. This makes our results easier to compare with the paper's reported readmission percentages while keeping the task as binary classification.

Implementation impact:

- Create `readmitted_30`.
- Set `readmitted_30 = 1` when `readmitted == "<30"`.
- Set `readmitted_30 = 0` when `readmitted == ">30"` or `readmitted == "NO"`.
- Use `readmitted_30` as `y`.
- Drop the original `readmitted` column from `X` to avoid target leakage.

Risks / assumptions:

- The negative class combines two clinically different groups: late readmission and no readmission.
- This is acceptable because the project question is specifically about early readmission within 30 days.

## Decision 3 - Columns Removed for Target Leakage or Identifier Leakage

Status: accepted

Question:
Which columns should be removed from the feature matrix `X` before modeling because they are either the answer itself or non-medical identifiers?

Options compared:

1. Minimal removal:

```text
Drop only readmitted, readmitted_30, and encounter_id.
Keep patient_nbr.
```

2. Safe ID/leakage removal:

```text
Drop readmitted, readmitted_30, encounter_id, and patient_nbr.
Keep all other columns for later preprocessing decisions.
```

3. Aggressive early removal:

```text
Drop readmitted, readmitted_30, encounter_id, patient_nbr,
weight, payer_code, medical_specialty, raw diagnosis columns,
and raw admission/discharge/source ID columns.
```

Thinking during discussion:

- `readmitted` is the original target column. It directly contains the outcome values `<30`, `>30`, and `NO`, so keeping it in `X` would give the model the answer.
- `readmitted_30` is the binary target created from `readmitted`. It must be kept separately as `y`, not included as an input feature.
- `encounter_id` is a unique hospital encounter identifier. It is not medical information and should not be used to predict future patients.
- `patient_nbr` is a patient identifier. Even after keeping only one encounter per patient, it is still not a valid medical predictor. A future patient will have a new unseen ID, so learning patterns from this number would not generalize.
- We do not want to remove `weight`, `payer_code`, `medical_specialty`, diagnosis columns, or admission/discharge/source IDs yet because those require separate decisions. Some may be useful after missing-value handling, grouping, or encoding.

Decision:
Use Option 2, safe ID/leakage removal.

Choice:

```text
Remove from X:
- readmitted
- readmitted_30
- encounter_id
- patient_nbr
```

Why this option was chosen:

This removes only columns that are clearly invalid as predictors. It avoids target leakage and identifier leakage while preserving potentially useful clinical and administrative variables for later, more careful preprocessing decisions.

Implementation impact:

- Create `y = model_df["readmitted_30"]`.
- Create `X` by dropping `readmitted`, `readmitted_30`, `encounter_id`, and `patient_nbr`.
- Do not drop any other feature columns in this step.

Risks / assumptions:

- Removing `patient_nbr` means the model cannot memorize patient-specific history through an ID, which is intended.
- Removing `encounter_id` also removes any ordering/time artifact that may be hidden in the encounter identifier.
- Any remaining problematic columns will be handled in later decisions rather than removed prematurely.

## Decision Progress Checkpoint

The remaining preprocessing and training-pipeline decisions listed after Decision 3 were finalized in Decisions 4 through 14. The project is now ready to move from planning into the first training notebook.

## Missing-Value Scope Check

Status: accepted as project context

Question:
What is the scope of missing values after applying the accepted modeling row scope?

Calculations under the accepted row scope:

```text
Rows: 69,973

Missing columns:
weight: 67,185 missing, 96.02%
medical_specialty: 33,639 missing, 48.07%
payer_code: 30,415 missing, 43.47%
race: 1,918 missing, 2.74%
diag_3: 1,224 missing, 1.75%
diag_2: 293 missing, 0.42%
diag_1: 10 missing, 0.01%
```

Missingness and target rates:

```text
weight present: 10.87% positive readmission
weight missing: 8.89% positive readmission

medical_specialty present: 8.73% positive readmission
medical_specialty missing: 9.23% positive readmission

payer_code present: 8.26% positive readmission
payer_code missing: 9.89% positive readmission

race present: 9.02% positive readmission
race missing: 7.35% positive readmission

diag_3 present: 9.04% positive readmission
diag_3 missing: 4.98% positive readmission

diag_2 present: 8.99% positive readmission
diag_2 missing: 4.44% positive readmission

diag_1 present: 8.97% positive readmission
diag_1 missing: 20.00% positive readmission
```

Thinking during discussion:

- Missing values should not be handled with one blanket rule.
- `weight` is almost empty and should be considered separately.
- `medical_specialty` and `payer_code` have large missingness, but may still contain useful administrative signal.
- `race`, `diag_1`, `diag_2`, and `diag_3` have small missingness and can likely be retained with explicit missing handling.
- Some missingness rates differ in positive readmission rate, especially `payer_code`, so missingness may contain signal.

Decision:
Handle missing values in separate decisions:

1. `weight`
2. `medical_specialty` and `payer_code`
3. small-missing columns: `race`, `diag_1`, `diag_2`, `diag_3`

Why this approach was chosen:

The columns have very different levels and meanings of missingness. Splitting them into separate decisions avoids dropping useful features too early and avoids treating extreme missingness the same way as minor missingness.

## Decision 4 - Handling `weight`

Status: accepted

Question:
How should the `weight` column be handled?

Evidence:

```text
Rows: 69,973
weight missing: 67,185 rows, 96.02%
weight present: 2,788 rows, 3.98%
weight present positive readmission rate: 10.87%
weight missing positive readmission rate: 8.89%
```

Options to compare:

1. Drop `weight` completely.
2. Keep `weight` as a categorical feature with a `Missing` category.
3. Replace `weight` with a binary indicator: `weight_recorded`.
4. Keep both categorical `weight` and binary `weight_recorded`.

Decision:
Use Option 1: drop `weight` completely for the first modeling version.

Why this option was chosen:

`weight` is missing for 96.02% of rows under the accepted modeling scope. Only 2,788 rows have a recorded value, so using the weight categories directly would create a feature where almost every row is just missing. This would add dimensionality and likely noise without enough observed data to learn a stable relationship.

We also considered a `weight_recorded` indicator because weight-present rows have a slightly higher positive readmission rate than weight-missing rows. However, for the first clean baseline, we prefer to keep preprocessing simple and avoid adding a feature that mostly captures documentation behavior. This can be tested later as an experiment if the baseline models need improvement.

Why we also follow the paper here:

The paper explicitly removed `weight` because it was too sparse to use reliably. Our scoped dataset shows the same problem, with 96.02% missing. Since both the original dataset analysis and our EDA identify the same extreme sparsity, dropping `weight` is a low-risk paper-aligned decision.

Implementation impact:

- Drop `weight` from `X`.
- Do not impute `weight`.
- Do not create `weight_recorded` in the first modeling version.

Risks / assumptions:

- We may lose a small amount of information from the 2,788 rows where weight was recorded.
- We assume the extreme missingness makes `weight` too sparse for the initial model.
- A later model version can compare performance with a `weight_recorded` indicator if needed.

## Decision 5 - Handling `medical_specialty` and `payer_code`

Status: accepted

Question:
How should the high-missing administrative columns `medical_specialty` and `payer_code` be handled?

What these columns represent:

- `medical_specialty` represents the specialty of the admitting physician or service, such as Internal Medicine, Cardiology, Emergency/Trauma, Surgery, Nephrology, Psychiatry, or Oncology.
- `payer_code` represents the payer or insurance/payment category. The dataset description says this corresponds to payment sources such as Medicare, Blue Cross / Blue Shield, self-pay, and other payer groups, but the raw data stores these as short codes.

Why this information was inspected before deciding:

These columns are not direct clinical measurements, but they may capture patient context, care setting, insurance/access patterns, and the type of clinical service responsible for the admission. Because both columns have large missingness, we inspected their category sizes and readmission rates before deciding whether to keep, group, or drop them.

Reference evidence inspected under the accepted modeling scope:

```text
Rows: 69,973

medical_specialty missing: 33,639 rows, 48.07%
medical_specialty unique non-missing categories: 70

payer_code missing: 30,415 rows, 43.47%
payer_code unique non-missing categories: 17
```

Top `medical_specialty` categories by count:

```text
Missing: 33,639 rows, 9.23% positive
InternalMedicine: 10,641 rows, 9.76% positive
Family/GeneralPractice: 4,978 rows, 9.74% positive
Emergency/Trauma: 4,393 rows, 7.83% positive
Cardiology: 4,207 rows, 7.18% positive
Surgery-General: 2,205 rows, 8.34% positive
Orthopedics: 1,128 rows, 9.93% positive
Orthopedics-Reconstructive: 1,041 rows, 6.63% positive
Radiologist: 821 rows, 7.06% positive
Nephrology: 797 rows, 11.04% positive
Psychiatry: 613 rows, 11.09% positive
Oncology: 205 rows, 18.05% positive
ObstetricsandGynecology: 593 rows, 3.71% positive
```

Top `payer_code` categories by count:

```text
Missing: 30,415 rows, 9.89% positive
MC: 19,782 rows, 9.36% positive
HM: 3,984 rows, 6.95% positive
BC: 3,397 rows, 5.98% positive
SP: 3,303 rows, 7.96% positive
MD: 2,165 rows, 7.94% positive
CP: 1,939 rows, 6.45% positive
UN: 1,855 rows, 7.60% positive
CM: 1,295 rows, 8.03% positive
OG: 647 rows, 7.73% positive
PO: 457 rows, 5.69% positive
DM: 372 rows, 9.14% positive
```

Initial interpretation from this reference evidence:

- These columns are not almost empty like `weight`; they have substantial missingness but also many usable observed values.
- Missingness itself may carry signal, especially for `payer_code`, where missing rows have a higher positive rate than many observed payer groups.
- `medical_specialty` has many categories, so using all raw categories naively may add sparsity and noise.
- `payer_code` has fewer categories, but still has rare categories that may need grouping.
- Some category-level readmission rates differ meaningfully, suggesting these columns may contain useful signal.

Options to compare:

1. Drop both columns.
2. Keep both columns and replace missing values with `Missing`.
3. Keep both columns, replace missing values with `Missing`, and group rare categories into `Other`.
4. Keep only one of the two columns.

Decision:
Use Option 3: keep both columns, preserve `Missing` as a category, and group rare categories into `Other`.

Rare-category grouping explanation:

Rare-category grouping means common categories are kept as their own values, while categories with too few examples are combined into one shared category called `Other`.

For example, if `medical_specialty = "Hematology/Oncology"` is too rare during training, it will be transformed to:

```text
medical_specialty_grouped = "Other"
```

If a future input contains a specialty never seen during training, it will also be mapped to:

```text
medical_specialty_grouped = "Other"
```

Missing values are handled separately:

```text
missing medical_specialty -> "Missing"
rare/unseen medical_specialty -> "Other"
```

This distinction matters because `Missing` means no value was recorded, while `Other` means a value was recorded but it was rare or unseen.

Why this option was chosen:

This keeps potentially useful administrative signal while controlling sparsity from rare categories and preserving missingness as information. The inspected reference evidence showed that these columns are not almost empty like `weight`; their observed categories have different readmission rates, and missingness itself may carry signal.

Paper-aligned part of this decision:

The paper kept `medical_specialty` and added a missing/unknown category, so we follow that part because the column represents admitting physician/service context and was used in the paper's analysis. We do not follow the paper exactly for `payer_code`: the paper removed it, while we keep it for the first ML version because our EDA showed payer missingness and payer groups may contain signal. This is documented as our own modeling choice, not a direct paper copy.

Implementation impact:

- Keep `medical_specialty`.
- Keep `payer_code`.
- Fill missing values in both columns with `Missing`.
- Learn the frequent categories from the training data only.
- Map rare categories to `Other`.
- Map unseen categories at inference time to `Other`.
- One-hot encode the grouped categorical values later.

Threshold note:

The exact rare-category threshold is not finalized in this decision. A simple starting choice to discuss later is:

```text
categories with fewer than 200 training rows -> Other
```

Risks / assumptions:

- Grouping rare categories loses detail from uncommon specialties or payer codes.
- Keeping these columns may introduce administrative or access-to-care bias, so model interpretation should be careful.
- The grouping threshold should be chosen using training data only to avoid leaking information from validation/test data.

## Decision 6 - Small-Missing Columns: `race`, `diag_1`, `diag_2`, `diag_3`

Status: accepted

Question:
How should columns with small missingness be handled?

Columns:

```text
race
diag_1
diag_2
diag_3
```

Reference evidence inspected under the accepted modeling scope:

```text
Rows: 69,973

race missing: 1,918 rows, 2.74%
diag_1 missing: 10 rows, 0.01%
diag_2 missing: 293 rows, 0.42%
diag_3 missing: 1,224 rows, 1.75%
```

Additional diagnosis-cardinality evidence:

```text
diag_1 unique non-missing raw codes: 694
diag_2 unique non-missing raw codes: 723
diag_3 unique non-missing raw codes: 756

diag_1 raw codes with fewer than 100 rows: 585
diag_2 raw codes with fewer than 100 rows: 621
diag_3 raw codes with fewer than 100 rows: 654
```

Thinking during discussion:

- For `race`, missingness is small and the number of categories is small.
- For `diag_1`, `diag_2`, and `diag_3`, missingness is also small.
- The main diagnosis problem is not missingness; it is high cardinality from hundreds of ICD-9 codes.
- Therefore missing handling and diagnosis representation should be separate decisions.

Decision:
Keep all four columns and fill missing values with `Missing`.

Choice:

```text
race missing -> "Missing"
diag_1 missing -> "Missing"
diag_2 missing -> "Missing"
diag_3 missing -> "Missing"
```

Why this option was chosen:

The missing percentages are low enough that dropping rows or dropping these columns would be unnecessary. Explicitly using `Missing` preserves the fact that a value was not recorded and keeps preprocessing simple.

Implementation impact:

- Keep `race`.
- Keep `diag_1`, `diag_2`, and `diag_3` for now.
- Fill missing values in these columns with `Missing`.
- Decide diagnosis representation separately before final encoding.

Risks / assumptions:

- `Missing` may represent different documentation situations, but the counts are small enough for this to be acceptable.
- Raw diagnosis codes should not be one-hot encoded directly until the diagnosis representation decision is finalized.

## Decision 7 - Diagnosis Representation

Status: accepted

Question:
How should `diag_1`, `diag_2`, and `diag_3` be represented for modeling?

Context:
The raw diagnosis columns contain ICD-9 diagnosis codes. These codes are clinically meaningful, but there are hundreds of unique values and many rare codes.

Reference basis for grouping:

- The dataset paper included in this folder, `archive/description.pdf`, lists primary diagnosis groups using ICD-9 ranges in Table 2. It groups examples such as circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms, and other categories.
- The ICD-9 chapter structure also supports these ranges. For example, ICD-9 lists circulatory diseases as 390-459, respiratory diseases as 460-519, digestive diseases as 520-579, genitourinary diseases as 580-629, musculoskeletal/connective tissue diseases as 710-739, and injury/poisoning as 800-999.

Important note:
The grouping is not invented for this project. For the first modeling version, we will use the diagnosis grouping introduced in the dataset paper, based on the paper's ICD-9 ranges, with standard ICD-9 chapter ranges as supporting justification.

Options to compare:

1. Use raw ICD-9 codes directly with one-hot encoding.
2. Use raw ICD-9 codes but group rare codes into `Other`.
3. Convert ICD-9 codes into broader clinical diagnosis groups.
4. Use both broader diagnosis groups and selected common raw ICD-9 codes.

Candidate diagnosis groups:

```text
Circulatory
Respiratory
Digestive
Diabetes
Injury/Poisoning
Musculoskeletal
Genitourinary
Neoplasms
External/Supplemental
Other
Missing
```

Grouped category counts under the accepted modeling scope:

```text
diag_1_group groups: 11
Circulatory: 21,384
Other: 11,203
Respiratory: 9,486
Digestive: 6,487
Diabetes: 5,748
Injury/Poisoning: 4,694
Musculoskeletal: 4,064
Genitourinary: 3,440
Neoplasms: 2,538
External/Supplemental: 919
Missing: 10

diag_2_group groups: 11
Circulatory: 22,079
Other: 16,289
Diabetes: 9,700
Respiratory: 6,925
Genitourinary: 5,328
Digestive: 2,854
Injury/Poisoning: 1,824
External/Supplemental: 1,787
Neoplasms: 1,599
Musculoskeletal: 1,295
Missing: 293

diag_3_group groups: 11
Circulatory: 20,864
Other: 16,505
Diabetes: 12,546
Respiratory: 4,650
Genitourinary: 4,047
External/Supplemental: 3,516
Digestive: 2,699
Injury/Poisoning: 1,409
Musculoskeletal: 1,368
Missing: 1,224
Neoplasms: 1,145
```

Thinking during discussion:

- The raw diagnosis columns contain too many ICD-9 codes for a clean first model.
- Raw one-hot encoding would create hundreds of sparse indicators.
- The dataset paper itself introduced ICD-9 diagnosis groups for this dataset, so our grouping is grounded in the dataset documentation rather than being arbitrary.
- Grouping reduces diagnosis representation from 694, 723, and 756 raw-code possibilities to 11 categories per diagnosis column.
- This gives roughly 33 diagnosis group indicators after one-hot encoding, which is much more manageable and interpretable.

Decision:
Use Option 3 for the first modeling version: convert ICD-9 codes into broader clinical diagnosis groups.

Choice:

```text
Create:
- diag_1_group
- diag_2_group
- diag_3_group

Drop from the first modeling feature matrix:
- diag_1
- diag_2
- diag_3
```

Why this option was chosen:

Broader diagnosis groups reduce sparsity, are easier to explain in the report, and are clinically interpretable. They avoid creating hundreds of one-hot columns from raw ICD-9 codes, many of which have too few examples for stable learning. This approach uses the diagnosis grouping introduced in the original dataset paper.

Why we also follow the paper here:

The raw diagnosis codes are high-cardinality and clinically hard to interpret one code at a time. The paper already introduced a clinically meaningful ICD-9 grouping for this exact dataset, and the grouped distributions are large enough to be stable. Following the paper here reduces arbitrary feature engineering and gives us a defensible, cited representation for diagnosis.

Implementation impact:

- Fill missing diagnosis codes with `Missing`.
- Apply an ICD-9 grouping function to `diag_1`, `diag_2`, and `diag_3`.
- Keep the grouped diagnosis columns for modeling.
- Drop the raw diagnosis code columns in the first modeling version.

Risks / assumptions:

- Grouping loses detail from specific ICD-9 codes.
- Some clinically different conditions are combined into broad categories.
- A later model version can test adding selected common raw ICD-9 codes if diagnosis detail appears important.

## Decision 8 - Admission, Discharge, and Admission Source Codes

Status: accepted

Question:
How should the numeric-looking administrative code columns be represented for modeling?

Columns:

```text
admission_type_id
discharge_disposition_id
admission_source_id
```

Reference evidence from the dataset paper:

- The dataset paper included in this folder, `archive/description.pdf`, describes these three fields as nominal variables, not numeric measurements.
- `admission_type_id` is described as an integer identifier corresponding to 9 distinct values, with examples such as emergency, urgent, elective, newborn, and not available.
- `discharge_disposition_id` is described as an integer identifier corresponding to 29 distinct values, with examples such as discharged to home, expired, and not available.
- `admission_source_id` is described as an integer identifier corresponding to 21 distinct values, with examples such as physician referral, emergency room, and transfer from a hospital.
- The paper also states that admission source and discharge disposition were among the variables chosen to control for patient demographics and illness severity.
- The paper removed encounters ending in hospice or patient death to avoid biasing the analysis. That aligns with Decision 1 in this log.

How the dataset paper handled these fields in its analysis:

- `discharge_disposition_id` was simplified in the paper's reported tables into:

```text
Discharged to home
Otherwise
```

- `admission_source_id` was simplified in the paper's reported tables into:

```text
Admitted from emergency room
Admitted because of physician/clinic referral
Otherwise
```

- In the final logistic regression table, discharge was represented as `Home` versus `Other`.
- In the final logistic regression table, admission was represented as `Emergency`, `referral`, and `Other`.
- The paper describes `admission_type_id` as a nominal integer identifier with examples such as emergency, urgent, elective, newborn, and not available, but the paper's listed final control variables emphasize admission source and discharge disposition rather than separately reporting admission type.

Important limitation from the paper:

The paper confirms that these are categorical identifiers and gives examples of their meaning, but it does not provide a full code-to-label map for every ID value in the PDF text.

Reference from the old starting notebook:

The old notebook `prediction-on-hospital-readmission.ipynb` grouped these codes into fewer categories before modeling. For example:

```text
admission_type_id:
2 -> 1
7 -> 1
6 -> 5
8 -> 5

discharge_disposition_id:
6, 8, 9, 13 -> 1
3, 4, 5, 14, 22, 23, 24 -> 2
12, 15, 16, 17 -> 10
25, 26 -> 18

admission_source_id:
2, 3 -> 1
5, 6, 10, 22, 25 -> 4
15, 17, 20, 21 -> 9
13, 14 -> 11
```

Initial interpretation:

- These columns must not be treated as continuous numeric variables.
- They can either be one-hot encoded as raw IDs or grouped into broader categories before one-hot encoding.
- Grouping is likely easier to explain and can reduce sparse categories, but the grouping scheme must be documented clearly.

Options to compare:

1. Keep raw numeric IDs as numeric features.
2. Treat raw IDs as categorical and one-hot encode them directly.
3. Group IDs into broader categories, then one-hot encode.
4. Drop these columns.

Decision:
Use Option 3: group IDs into broader categories, then one-hot encode.

Final choice:

```text
discharge_disposition_id -> discharge_disposition_group
Home if discharged to home
Other otherwise

admission_source_id -> admission_source_group
Emergency room
Physician/clinic referral
Other

admission_type_id -> admission_type_group
Emergency/Urgent/Trauma
Elective
Unknown/Not available
Other/Newborn
```

Why this option was chosen:

The associated dataset/source paper is the source of truth for the meaning of `discharge_disposition_id` and `admission_source_id` as nominal identifiers. Its reported analysis simplified discharge to `Home` versus `Other`, and admission source to `Emergency room`, `Physician/clinic referral`, and `Other`. We follow that as a strong, dataset-grounded modeling reference, not because it is the only possible valid choice.

For `admission_type_id`, the paper confirms it is a nominal categorical identifier but does not provide the same final grouping in the reported model. We therefore group it into broad interpretable categories based on the examples given by the paper and the grouping pattern from the old starting notebook.

Why we also follow the paper here:

For discharge disposition and admission source, the paper's grouped categories are simple, clinically interpretable, and already shown to be meaningful in the associated analysis. They also prevent the model from treating arbitrary numeric IDs as ordered numbers. We follow the paper for these two fields because it gives us a defensible grouping while keeping the feature space small. `admission_type_id` is not copied exactly from the paper, so its grouping remains a pragmatic project choice.

Implementation impact:

- Create grouped columns for admission type, discharge disposition, and admission source.
- Drop the raw numeric ID columns from the first modeling feature matrix.
- One-hot encode the grouped columns later.

Risks / assumptions:

- Grouping loses detail from specific administrative IDs.
- For discharge disposition, the paper's `Home` versus `Other` grouping is intentionally simple; a later model can test a richer discharge grouping if needed.

## Decision 9 - Demographic Feature Representation

Status: accepted

Question:
How should age, race, and gender be represented?

Source-of-truth reference:

The dataset paper groups age into three broad categories after observing the relationship between age and readmission:

```text
30 years old or younger
30-60 years old
Older than 60
```

The paper kept race in its final model, with `Missing` treated as a category. The paper removed gender from the core model because it was not statistically significant.

Decision:

```text
age -> age_group_paper with values <=30, 30-60, >60
race -> keep as categorical, with Missing already handled by Decision 6
gender -> drop from first modeling version
```

Why this option was chosen:

This follows the dataset paper closely. Age grouping is explicitly motivated in the paper. Race was retained by the paper and has low missingness in our modeling scope. Gender shows almost no difference in our scoped EDA and was removed by the paper, so dropping it keeps the first model simpler and avoids relying on a low-signal demographic attribute.

Why we also follow the paper here:

The paper did not group age arbitrarily; it reported that the relationship between age and the logit of readmission showed three broad intervals. Using those age groups gives us a paper-supported transformation instead of inventing a new binning scheme. For gender, the paper removed it after finding it was not significant, and our EDA also shows little difference, so dropping it is consistent with both sources of evidence. Race is kept because the paper retained it and our missingness is low.

Implementation impact:

- Create `age_group_paper`.
- Drop raw `age` from the first feature matrix.
- Keep `race`.
- Drop `gender`.

Risks / assumptions:

- Dropping gender may remove a small amount of signal, but the paper and our EDA both suggest it is weak.
- A later sensitivity model can compare performance with gender included.

## Decision 10 - Lab Result Features

Status: accepted

Question:
How should `A1Cresult` and `max_glu_serum` be handled?

Source-of-truth reference:

The dataset paper emphasizes HbA1c measurement and explicitly treats "not measured" as meaningful. The data dictionary also defines `None` for these lab columns as "test was not taken," not as ordinary missing data.

Decision:

```text
Keep A1Cresult as categorical.
Keep max_glu_serum as categorical.
Do not convert "None" to missing.
Do not impute these columns.
```

Why this option was chosen:

`None` means the lab test was not performed, which is clinically and procedurally meaningful. One-hot encoding these categories lets the model learn whether a test was performed and what the result range was. This follows the dataset documentation and paper framing.

Why we also follow the paper here:

The paper's main research question is about HbA1c measurement, so whether a test was performed is itself part of the clinical signal. Treating `None` as missing would erase that information and contradict the dataset definition. Keeping `None` as a category preserves the paper's interpretation of lab measurement status.

Implementation impact:

- Preserve `None` as a category.
- One-hot encode `A1Cresult` and `max_glu_serum`.

Risks / assumptions:

- The model may learn hospital testing behavior as well as patient biology. This is acceptable for readmission prediction but should be discussed in limitations.

## Decision 11 - Medication Features

Status: accepted

Question:
How should the diabetes medication columns be represented?

Source-of-truth reference:

The dataset paper documents medication status columns and also defines medication change as an important derived concept. The paper focuses on whether diabetes medication was changed in response to HbA1c information.

Decision:

```text
Keep change.
Keep diabetesMed.
Create num_diabetes_meds_used.
Create num_diabetes_med_changes.
Drop medication columns with zero variance.
Drop original medication columns with fewer than 100 non-"No" cases in the accepted modeling scope.
Keep remaining common medication status columns as categorical.
```

Medication columns to drop in the first modeling version because they are zero-variance or extremely rare under the accepted row scope:

```text
chlorpropamide
acetohexamide
tolbutamide
miglitol
troglitazone
tolazamide
examide
citoglipton
glipizide-metformin
glimepiride-pioglitazone
metformin-rosiglitazone
metformin-pioglitazone
```

Why this option was chosen:

This keeps the medication signal emphasized by the dataset paper while avoiding sparse one-hot indicators for drugs that are almost never used. The summary features preserve general treatment intensity even when rare individual drug columns are dropped.

Paper-aligned part of this decision:

The paper defines medication change as an important concept and focuses on whether diabetes medication was changed during the encounter. We follow that by keeping `change` and creating medication-change summary features. The decision to drop extremely rare individual drug columns is our own ML practicality choice, because sparse drug indicators are unlikely to be stable in the first model.

Implementation impact:

- Build summary medication features before dropping rare medication columns.
- One-hot encode common retained medication status columns.
- Drop rare/zero-variance original medication columns from the first feature matrix.

Risks / assumptions:

- Rare medications may matter clinically for small subgroups, but they are too sparse for a stable first model.
- Later experiments can test whether keeping all medication columns improves performance.

## Decision 12 - Utilization Feature Engineering

Status: accepted

Question:
How should prior utilization variables be handled?

Source-of-truth reference:

The dataset includes prior outpatient, emergency, and inpatient visit counts in the year before the encounter. Our EDA showed prior utilization, especially prior inpatient visits, is one of the clearest signals for 30-day readmission.

Decision:

```text
Keep number_outpatient.
Keep number_emergency.
Keep number_inpatient.
Create service_utilization = number_outpatient + number_emergency + number_inpatient.
Do not bucket or cap utilization variables in the first modeling version.
```

Why this option was chosen:

The raw counts are meaningful and already available. The combined `service_utilization` feature is easy to explain and captures total recent healthcare use. We avoid capping or bucketing for the first training pass to keep preprocessing simple and preserve information.

Implementation impact:

- Add `service_utilization`.
- Scale numeric utilization features for linear models.
- Let tree-based models handle skew naturally.

Risks / assumptions:

- `service_utilization` is correlated with its component counts. This is acceptable for tree models and can be monitored for logistic regression.
- A later model can compare bucketed utilization features if needed.

## Decision 13 - Rare-Category Threshold

Status: accepted

Question:
What threshold should define rare categories for `medical_specialty` and `payer_code`?

Decision:

```text
Categories with fewer than 200 rows in the training data -> Other
Missing values -> Missing
Unseen values at inference time -> Other
```

Why this option was chosen:

A count threshold is simple to explain and avoids target-based leakage. The threshold should be learned from training data only, then applied unchanged to validation, test, and future data.

Implementation impact:

- Fit rare-category grouping only on training data.
- Use `Other` for rare or unseen categories.
- Use `Missing` only for missing values.

Risks / assumptions:

- The threshold is pragmatic. It can be tuned later, but 200 is a reasonable first value for a 69,973-row dataset.

## Decision 14 - Encoding, Scaling, Split, and Imbalance Strategy

Status: accepted

Question:
What should the first training pipeline use for encoding, scaling, splitting, and class imbalance?

Decision:

```text
Split:
Use stratified train/validation/test split.
Recommended split: 70% train, 15% validation, 15% test.
Because Decision 1 leaves one row per patient, a grouped split is not required.

Categorical encoding:
Use one-hot encoding for categorical/grouped features.
Use handle_unknown="ignore" or equivalent protection for unseen categories.

Numeric scaling:
Scale numeric features for Logistic Regression.
Scaling is not required for tree-based models, but model-specific pipelines can handle this.

Class imbalance:
Do not use SMOTE in the first baseline.
Use class_weight="balanced" where supported, and compare against an unweighted baseline if time permits.

Metrics:
Report majority-class baseline accuracy.
Use PR-AUC, ROC-AUC, recall, precision, F1-score, and confusion matrix.
Treat PR-AUC and recall as especially important because the positive class is only 8.97%.
```

Why this option was chosen:

This is a clean, reproducible first training setup. The stratified split preserves the minority-class proportion. Avoiding SMOTE initially keeps the baseline honest and easier to explain. Class weights are simpler and safer for the first pass. Accuracy alone is misleading because always predicting no 30-day readmission already performs well on accuracy, so imbalance-aware metrics are required.

Implementation impact:

- Build reproducible preprocessing pipelines.
- Train at least a majority baseline and Logistic Regression baseline.
- Then compare at least one tree-based model, such as Random Forest or Gradient Boosting.
- Keep the test set untouched until final model selection.

Risks / assumptions:

- Class weights may improve recall at the cost of precision.
- The best classification threshold may not be 0.5, so threshold tuning should happen on the validation set if time allows.

## Implementation - Modeling Notebook Version 1

Status: implemented

Notebook:

```text
hospital_readmission_modeling.ipynb
```

What was built:

- Loaded `archive/diabetic_data.csv`.
- Used `pd.read_csv(..., na_values="?", keep_default_na=False)` so raw `?` values become missing while lab-result strings such as `None` remain real categories.
- Applied the accepted row scope:
  - sort by `encounter_id`
  - keep the first encounter per `patient_nbr`
  - remove hospice/expired discharge disposition IDs `[11, 13, 14, 19, 20, 21]`
- Created the binary target:

```text
readmitted_30 = 1 if readmitted == "<30", else 0
```

- Built the first feature matrix using all accepted preprocessing decisions:
  - dropped `readmitted`, `readmitted_30`, `encounter_id`, and `patient_nbr` from `X`
  - dropped `weight`
  - kept `medical_specialty` and `payer_code`, filled missing values as `Missing`, and grouped categories with fewer than 200 training rows or unseen categories as `Other`
  - filled `race` and diagnosis missing values as `Missing`
  - converted `diag_1`, `diag_2`, and `diag_3` into paper-style ICD-9 diagnosis groups, then dropped the raw diagnosis codes
  - grouped `admission_type_id`, `discharge_disposition_id`, and `admission_source_id`, then dropped the raw ID columns
  - grouped age into paper age groups `<=30`, `30-60`, and `>60`
  - dropped `gender`
  - kept `A1Cresult` and `max_glu_serum`, preserving `None` as a true category
  - created `num_diabetes_meds_used` and `num_diabetes_med_changes`
  - kept common medication status columns plus `change` and `diabetesMed`
  - dropped the accepted rare/zero-variance medication columns
  - created `service_utilization`
  - one-hot encoded categorical features with unknown-category protection
  - scaled numeric features for Logistic Regression, Gaussian Naive Bayes, and optional KNN

Split:

```text
Train: 48,981 rows
Validation: 10,496 rows
Test: 10,496 rows
Positive rate in each split: about 8.97%
```

The split is stratified 70/15/15. The test set is not evaluated until after final model selection from validation results.

Models compared on validation:

- Majority-class baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- AdaBoost
- Gaussian Naive Bayes

KNN implementation choice:

KNN was included as optional code but skipped in this executed notebook because the training split has 48,981 rows and the one-hot encoded feature space makes full validation-time distance computation inefficient for the course-project comparison.

Class imbalance handling:

- No SMOTE was used.
- `class_weight="balanced"` or `class_weight="balanced_subsample"` was used for models that support class weights.
- Balanced `sample_weight` was passed to Gradient Boosting, AdaBoost, and Gaussian Naive Bayes.

Metrics used:

- PR-AUC
- ROC-AUC
- recall
- precision
- F1
- accuracy
- confusion matrix counts: TN, FP, FN, TP

Validation comparison summary:

```text
Top validation models by PR-AUC:
Random Forest: PR-AUC 0.1517, ROC-AUC 0.6464, recall 0.5600, F1 0.2237
HistGradientBoosting: PR-AUC 0.1512, ROC-AUC 0.6492, recall 0.5972, F1 0.2183
Logistic Regression: PR-AUC 0.1482, ROC-AUC 0.6380, recall 0.5824, F1 0.2135

Majority baseline:
PR-AUC 0.0897, ROC-AUC 0.5000, recall 0.0000, accuracy 0.9103
```

Final test evaluation:

The final test table was restricted to the majority baseline and the top three validation models by PR-AUC:

```text
HistGradientBoosting: PR-AUC 0.1457, ROC-AUC 0.6231, recall 0.5637, F1 0.2115
Logistic Regression: PR-AUC 0.1452, ROC-AUC 0.6248, recall 0.5552, F1 0.2058
Random Forest: PR-AUC 0.1430, ROC-AUC 0.6296, recall 0.5223, F1 0.2117
Majority baseline: PR-AUC 0.0897, ROC-AUC 0.5000, recall 0.0000, F1 0.0000
```

Visual outputs included:

- validation PR-AUC bar chart by model
- validation recall/F1 bar chart by model
- test confusion matrix for the best validation-selected model

Practical coding choices:

- Rare-category grouping is implemented as a custom sklearn transformer so it is fitted only on training data inside each model pipeline.
- Dense one-hot encoded matrices are used because all selected models are compatible at this dataset size, and Gaussian Naive Bayes and histogram gradient boosting require dense input.
- Default 0.5 classification thresholds are used for this first modeling version. Threshold tuning can be a later validation-only experiment if recall or precision needs to be adjusted.

Execution check:

The notebook was executed end to end successfully after correcting the CSV load step to preserve lab-result `None` categories.

## Implementation - Extended Modeling Search

Status: implemented

Reason for this phase:

After the first modeling notebook was completed, the project goal changed from a conservative first course-project baseline to an aggressive search for better predictive results. The accepted preprocessing decisions remain documented above, but this phase intentionally tested departures from those decisions where they could improve validation performance.

Files created:

```text
modeling_experiments.py
targeted_modeling_search.py
evaluate_targeted_selected.py
native_catboost_search.py
ensemble_search.py
modeling_experiment_report.md
experiment_results/
```

Dependency update:

`requirements.txt` now includes:

```text
xgboost
lightgbm
catboost
```

What was tried:

- Broad preprocessing/model search:
  - accepted preprocessing variants
  - rare-category thresholds 100, 200, and 500
  - raw administrative IDs as categorical features
  - detailed administrative grouping
  - raw age vs paper age groups
  - gender kept vs dropped
  - weight dropped, indicator, and category variants
  - diagnosis groups, raw diagnosis codes, and diagnosis groups plus raw codes
  - all medications, rare-medication dropping, and medication summaries only
  - raw utilization, log utilization, and utilization buckets
  - Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, GaussianNB, LightGBM, XGBoost, and CatBoost
- Targeted search:
  - focused on the best broad-search family: raw administrative IDs as categorical variables
  - tuned LightGBM, XGBoost, CatBoost, Extra Trees, Random Forest, and Logistic Regression variants
  - tested multiple rare thresholds and threshold strategies
- Native CatBoost search:
  - trained CatBoost directly on categorical features instead of one-hot encoded features
  - tested Balanced and SqrtBalanced class weighting
  - tested depth, learning rate, L2, medication-summary, age, weight, and rare-threshold variants
- Ensemble search:
  - averaged scores from strongest native CatBoost, one-hot CatBoost, XGBoost, and LightGBM candidates
  - tested simple average ensembles and a few hand-weighted blends

Threshold strategies evaluated:

```text
default 0.5
best validation F1
best validation F2
maximum validation recall with precision >= 0.12
maximum validation recall with precision >= 0.15
maximum validation recall with precision >= 0.20
```

Important practical fix:

During experimentation, the first version of the broad/targeted runner reused model instances across feature configurations. This made validation rows valid because each validation score was computed immediately after fitting, but it corrupted stored fitted estimators for later selected-test evaluation. The scripts were fixed to clone estimator objects before fitting, and selected test evaluation was rerun with fresh model instances.

Best broad-search result:

```text
Best broad validation PR-AUC:
raw_admin_raw_age_weight_category + XGBoost_depth4_aucpr
Validation PR-AUC 0.1854

Best broad selected test PR-AUC:
raw_admin_raw_age_weight_category + LightGBM_balanced_leaves31
Test PR-AUC 0.1834
```

Best targeted-search result:

```text
Best targeted validation PR-AUC:
target_raw_admin_age_paper_weight_category_rare100 + CatBoost_d5_lr0.03_SqrtBalanced
Validation PR-AUC 0.1956

Best targeted validation F1:
target_raw_admin_age_paper_weight_category_rare100 + XGBoost_d5_lr0.015_spw0.75
Validation F1 0.2582

Best targeted selected test PR-AUC:
CatBoost_d5_lr0.03_SqrtBalanced
Test PR-AUC 0.1887

Best targeted selected test F1:
XGBoost_d4_lr0.015_spw1 with max-recall precision >= 0.20 threshold
Test F1 0.2511
```

Best native CatBoost result:

```text
Best native CatBoost validation PR-AUC:
native_cat_raw_admin_age_paper_summaries_only_rare100
NativeCatBoost_d6_lr0.025_l27.0_SqrtBalanced
Validation PR-AUC 0.1978
Validation ROC-AUC 0.6648
Validation recall 0.3177
Validation precision 0.2078
Validation F1 0.2513
Validation accuracy 0.8302

Best native CatBoost selected test PR-AUC:
native_cat_raw_admin_age_paper_summaries_only_rare100
NativeCatBoost_d5_lr0.02_l28.0_SqrtBalanced
Threshold: max recall with precision >= 0.20
Test PR-AUC 0.1983
Test ROC-AUC 0.6542
Test recall 0.3100
Test precision 0.2025
Test F1 0.2450
Test accuracy 0.8285
```

Best ensemble result:

```text
Best ensemble validation PR-AUC:
avg_2_native_summary_d6_sqrt__xgb_rawage_weight_d5
Validation PR-AUC 0.2005
Validation F1 0.2564

Best ensemble selected test PR-AUC:
avg_3_native_summary_d6_sqrt__native_agepaper_weight_d6_sqrt__lgbm_agepaper_weight
Test PR-AUC 0.1973
```

Neural-network search:

```text
Script:
neural_network_search.py

Completed model families:
- PyTorch embedding MLPs for categorical + numeric tabular features.
- PyTorch TabNet via pytorch-tabnet.

Practical framework checks:
- PyTorch was available in the local environment.
- pytorch-tabnet was available in the local environment.
- TensorFlow was not used because the local import failed.
- A lightweight transformer-style prototype was attempted but was too slow on the CPU-only environment before producing useful validation output, so it was excluded from the completed comparison tables.

Feature families tried:
- raw administrative IDs + paper age + medication summaries only + weight category
- raw administrative IDs + raw age + weight category
- raw administrative IDs + paper age + medication summaries only + weight indicator

Neural hyperparameter choices tried:
- MLP hidden layers: 128-64, 256-128, 256-128-64, 512-256, 512-256-128
- dropout: 0.15, 0.20, 0.25, 0.30, 0.35
- positive-class weighting: 0.5x, 0.75x, and 1.0x empirical imbalance ratio
- TabNet widths/attention widths: 16/16 and 24/24
- TabNet steps: 4
- validation-selected threshold strategies reused from the tree/boosting search

Best neural validation PR-AUC:
nn_raw_admin_age_paper_weight_indicator_rare100
EmbeddingMLP_512_256_do0.30_pw0.75
Validation PR-AUC 0.1818
Validation ROC-AUC 0.6465
Validation recall 0.3188
Validation precision 0.1938
Validation F1 0.2411
Validation accuracy 0.8200

Best neural selected test PR-AUC:
nn_raw_admin_age_paper_summaries_only_rare100
EmbeddingMLP_256_128_64_do0.20_pw0.5
Threshold: best F1 selected on validation
Test PR-AUC 0.1824
Test ROC-AUC 0.6483
Test recall 0.3546
Test precision 0.1715
Test F1 0.2311
Test accuracy 0.7883

Best neural selected test F1:
nn_raw_admin_age_paper_weight_indicator_rare100
EmbeddingMLP_512_256_128_do0.35_pw1.0
Threshold: max recall with precision >= 0.20 selected on validation
Test PR-AUC 0.1775
Test ROC-AUC 0.6491
Test recall 0.3206
Test precision 0.1980
Test F1 0.2448
Test accuracy 0.8225
```

Neural-network conclusion:

The best neural networks improved over the first notebook baseline but did not beat tuned native CatBoost. TabNet was slower than the MLPs and had lower validation PR-AUC. The neural results are documented as an explored option, but they do not change the final model recommendation.

Imbalance-handling search:

```text
Scripts:
- imbalance_experiments.py
- imbalance_refinement_search.py
- imbalance_refined_ensemble.py

Methods tried:
- wider class-weight grids for XGBoost, LightGBM, Logistic Regression, and native CatBoost
- random oversampling on training only
- random undersampling on training only
- SMOTENC on training only
- Balanced Random Forest
- EasyEnsemble
- RUSBoost
- sigmoid calibration for XGBoost and LightGBM
- focal-loss PyTorch embedding MLPs
- refined score-average and rank-average ensembles
- lift tables for selected imbalance-handling candidates

Practical result:
- Resampling did not improve the best held-out PR-AUC.
- SMOTENC and RUSBoost often raised recall but hurt precision and PR-AUC.
- Balanced Random Forest was weaker than boosted-tree approaches.
- EasyEnsemble ran successfully in single-process mode but performed poorly.
- Focal-loss neural networks did not beat CatBoost/XGBoost.
- The best gains came from refined CatBoost class weighting, validation threshold tuning, and a small rank-average ensemble for F1.

Best refined validation PR-AUC:
imb_ref_age_paper_summaries_only_rare100
RefinedNativeCat_d6_lr0.015_l210.0_customPW0.25
Validation PR-AUC 0.1982
Validation ROC-AUC 0.6666
Validation recall 0.3518
Validation precision 0.1917
Validation F1 0.2481
Validation accuracy 0.8089

Best refined selected test PR-AUC:
imb_ref_age_paper_summaries_only_rare100
RefinedNativeCat_d6_lr0.015_l210.0_SqrtBalanced
Threshold: max recall with validation precision >= 0.20
Test PR-AUC 0.1991
Test ROC-AUC 0.6535
Test recall 0.3217
Test precision 0.1979
Test F1 0.2450
Test accuracy 0.8221

Best refined selected test F1:
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

Lift table for refined native CatBoost:
Top 1% highest-risk patients: precision 0.4667, recall 0.0520, lift 5.20x
Top 5% highest-risk patients: precision 0.2857, recall 0.1592, lift 3.18x
Top 10% highest-risk patients: precision 0.2305, recall 0.2569, lift 2.57x
Top 20% highest-risk patients: precision 0.1700, recall 0.3790, lift 1.89x
```

Imbalance-handling conclusion:

The strongest way to handle the dataset imbalance was to keep validation/test naturally imbalanced, use class-weighted boosting, tune thresholds on validation, and evaluate risk concentration with lift tables. Synthetic resampling did not improve the final held-out result.

Balanced-test sensitivity check:

```text
Script:
balanced_test_evaluation.py

Purpose:
Evaluate selected final models on an artificial 50/50 test subset created only from the original held-out test split.

Construction:
- kept all 942 positive readmitted_30 test cases
- sampled 942 negative test cases with random_state=42
- final balanced test rows: 1,884
- positive rate: 50%

Best balanced-test PR-AUC:
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

Best refined native CatBoost balanced-test operating point:
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

The balanced test is useful for understanding behavior when positives and negatives are equally common, but it is not the official final evaluation. PR-AUC, precision, F1, and accuracy rise strongly because the test prevalence changes from about 9% positive to 50% positive. ROC-AUC remains around 0.65, similar to the natural test result, which means the underlying ranking ability is about the same.

Feature-engineering and tuning loop:

```text
Scripts:
- feature_engineering_search.py
- feature_engineering_ensemble.py
- catboost_tuning_search.py

Additional features tried:
- ICD-9 chapter, prefix, and numeric diagnosis detail
- Elixhauser-like comorbidity flags and comorbidity count
- medication class summaries and insulin-change flags
- utilization flags and interactions
- A1C/glucose measurement and high-result flags
- administrative risk flags

Additional modeling choices tried:
- native CatBoost on engineered features
- score-average and rank-average ensembles
- validation early stopping for CatBoost
- deeper/regularized CatBoost variants
- custom positive-class weights
- underbagged CatBoost
```

Best first-encounter held-out PR-AUC after this loop:

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

Best first-encounter held-out F1 after this loop:

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

CatBoost tuning result:

```text
Best tuning-loop validation PR-AUC: 0.2057
Best tuning-loop selected test PR-AUC: 0.1971
Best tuning-loop selected test F1: 0.2509
```

Interpretation:

Feature engineering and small ensembles produced a very small improvement on the original first-encounter test set. The more aggressive CatBoost tuning improved validation results but did not improve the held-out test result, so it is documented as overfitting risk rather than a final-model improvement.

Alternate row-scope performance experiment:

```text
Script:
all_encounters_group_split_search.py

Scope:
- use all eligible encounters
- remove hospice/expired discharges
- split by patient_nbr so no patient appears in more than one split
- keep the same readmitted_30 target
```

This does not replace Decision 1 automatically. Decision 1 remains the accepted course-project scope unless the project is explicitly reframed around all eligible encounters. This alternate experiment was run because the first-encounter models plateaued near PR-AUC 0.20.

Patient-group split summary:

```text
Train rows 69,444, patients 48,992, positive rate 11.38%
Validation rows 15,071, patients 10,499, positive rate 11.80%
Test rows 14,828, patients 10,499, positive rate 11.03%
```

Best alternate-scope held-out PR-AUC:

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

Using all encounters with patient-level splitting gave the strongest performance found so far. This suggests that later encounters carry useful signal that is removed by the conservative first-encounter-only design. The tradeoff is explainability and project framing: the first-encounter scope is cleaner and closer to the original study while the all-encounter patient-group split is stronger for predictive performance.

Paper reproduction check:

```text
Scripts:
- paper_reproduction_search.py
- paper_rf_sensitivity_results.csv

Reference:
Bhuvan et al., "Identifying Diabetic Patients with High Risk of Readmission", arXiv:1602.04257.
```

The paper reported PR-AUC 0.242 for Random Forest on the `<30` vs `>30/NO` task. Their visible setup used all encounter rows after preprocessing, dropped `weight`, `payer_code`, and `medical_specialty`, removed missing race/diagnosis rows, grouped ICD-9 diagnosis codes, kept only insulin from individual medication columns used 22 risk factors, and used a random 75/25 train/test split.

Local reproduction:

```text
Paper-style rows after filtering: 98,053
Positive rate: 11.29%
Model: Random Forest, 250 trees, max_depth=5

All rows, random 75/25 split:
Test PR-AUC 0.2083
Test ROC-AUC 0.6564
Best-F1 recall 0.4203
Best-F1 precision 0.2079
Best-F1 F1 0.2782
Patient overlap between train and test: 7,906 patients

All rows, patient-group 75/25 split:
Test PR-AUC 0.2027
Test ROC-AUC 0.6510
Best-F1 recall 0.4889
Best-F1 precision 0.1849
Best-F1 F1 0.2683
Patient overlap between train and test: 0 patients

First encounter + hospice/expired removed + paper-style features:
Test PR-AUC 0.1655
Test ROC-AUC 0.6425
Best-F1 recall 0.3182
Best-F1 precision 0.1884
Best-F1 F1 0.2366
```

Seed sensitivity:

```text
20 random 75/25 seeds with the paper-style RF:
Mean PR-AUC 0.2172
Max PR-AUC 0.2242
Min PR-AUC 0.2096
```

Interpretation:

Copying the visible paper setup did not reproduce the paper's 0.242 PR-AUC. The reproduction supports the idea that all-encounter evaluation is easier than first-encounter evaluation but the current all-encounter CatBoost patient-group split still performs better than the local paper-style Random Forest reproduction.

Final modeling conclusion:

The best results came from moving beyond the original conservative preprocessing in three ways:

- using raw administrative IDs as categorical variables instead of only broad paper-style groups
- using CatBoost/XGBoost/LightGBM instead of only basic scikit-learn models
- tuning the classification threshold on validation data

The strongest single-model family is native CatBoost with:

```text
raw administrative IDs
paper age groups
medication summary features
rare/unseen category handling
refined class weighting
```

The final model is substantially better than the first modeling notebook:

```text
Majority baseline PR-AUC: 0.0897
First notebook best selected test PR-AUC: 0.1457
Broad search best selected test PR-AUC: 0.1834
Targeted search best selected test PR-AUC: 0.1887
Neural network best selected test PR-AUC: 0.1824
Native CatBoost best selected test PR-AUC: 0.1983
Imbalance-refined best selected test PR-AUC: 0.1991
Feature-engineered first-encounter ensemble best selected test PR-AUC: 0.2002
Feature-engineered first-encounter best selected test F1: 0.2531
Alternate all-encounter patient-group split best selected test PR-AUC: 0.2290
Alternate all-encounter patient-group split best selected test F1: 0.2816
```

Risk / interpretation:

- The test set has now been used across multiple validation-selected experiment families, so the cleanest result to report is the validation-selected native CatBoost family, not a post-hoc test-only winner.
- Accuracy remains misleading because the majority baseline has about 91% accuracy with zero recall.
- Precision remains modest around 0.20 at useful operating points, so the model is not clinically deployable as-is. It is, however, a much stronger course-project model than the initial baseline.

Notebook update:

`hospital_readmission_modeling.ipynb` now includes an extended experiment summary section that reads the saved CSV results and displays comparison tables with accuracy included. It also summarizes the neural-network and imbalance-handling searches from the saved CSV files without rerunning the full training loops.

## Implementation Update: Plateau Diagnostic and Patient-History Features

After the first all-encounter patient-safe model plateaued around PR-AUC 0.229, an additional loop tested feature groups, split effects, ensembling, and prior patient-history features.

New scripts:

```text
plateau_diagnostic_search.py
plateau_ensemble_search.py
patient_history_feature_search.py
patient_history_tuning_search.py
history_balanced_bagging_search.py
history_heterogeneous_search.py
history_catboost_seed_sweep.py
history_negative_ratio_refinement.py
history_catboost_order_sensitivity.py
history_catboost_bootstrap_search.py
```

New outputs:

```text
experiment_results/plateau_diagnostic_results.csv
experiment_results/plateau_diagnostic_lift_tables.csv
experiment_results/plateau_dataset_signal_summary.csv
experiment_results/plateau_ensemble_results.csv
experiment_results/plateau_ensemble_lift_tables.csv
experiment_results/plateau_ensemble_selected_by_validation.csv
experiment_results/patient_history_feature_results.csv
experiment_results/patient_history_feature_lift_tables.csv
experiment_results/patient_history_tuning_results.csv
experiment_results/patient_history_tuning_lift_tables.csv
experiment_results/history_balanced_bagging_results.csv
experiment_results/history_heterogeneous_results.csv
experiment_results/history_catboost_seed_sweep_results.csv
experiment_results/history_negative_ratio_refinement_results.csv
experiment_results/history_catboost_order_sensitivity_results.csv
experiment_results/history_catboost_bootstrap_results.csv
plateau_analysis_report.md
```

Practical feature findings:

- Raw diagnosis codes hurt performance and should not replace grouped diagnosis features.
- Administrative/discharge/source variables are among the strongest predictors.
- Diagnosis features matter.
- Medication and lab features add limited marginal signal.
- Broad ensembling did not break the plateau.
- Random row splits produce higher apparent PR-AUC than patient-safe splits, confirming that evaluation design strongly affects reported results.

Additional accepted-for-exploration feature family:

For the all-encounter framing only, create prior patient-history features using only earlier encounters for the same patient ordered by `encounter_id`. These include:

```text
patient_prior_encounters
patient_prior_readmit30_count
patient_prior_readmit_any_count
patient_prior_readmit30_rate
patient_prior_readmit_any_rate
patient_has_prior_readmit30
patient_has_prior_readmit_any
previous encounter utilization summaries
previous encounter readmission/admission/discharge/diagnosis/lab/medication categories
```

Important limitation:

These prior-history features are not part of the original first-encounter modeling scope. They are valid only if the project is framed as predicting risk for all eligible encounters, where earlier encounters for the same patient are historical information available before the current prediction.

Best observed patient-safe result after this loop:

```text
NegRefineCat_d6_lr002_neg8_seed202
Test PR-AUC 0.2415
Test ROC-AUC 0.6817
Test recall 0.4446
Test precision 0.2160
Test F1 0.2907
Test accuracy 0.7608
```

Most defensible validation-selected single history model:

```text
NegRefineCat_d6_lr002_neg7.5_seed37
Validation PR-AUC 0.2879
Test PR-AUC 0.2414
Test ROC-AUC 0.6827
Test recall 0.4226
Test precision 0.2223
Test F1 0.2913
Test accuracy 0.7733
```

Additional advanced-search loop:

- balanced CatBoost negative-subset bagging
- LightGBM, XGBoost, Random Forest, Extra Trees, target-encoded HistGradientBoosting, logistic stackers, and heterogeneous score/rank ensembles
- focused CatBoost depth/learning-rate/L2/seed sweeps
- near-full negative-ratio refinement around the best CatBoost setup
- CatBoost row-order sensitivity and separated row-order seed vs model seed
- CatBoost Bayesian, Bernoulli, MVS, no-bootstrap, and Ordered boosting variants

Practical conclusion from this loop:

The final gain from 0.2389 to 0.2414/0.2415 came from focused CatBoost ratio/row-order refinement around the patient-history feature set. More complex heterogeneous ensembles, deeper trees, bootstrap variants, and Ordered boosting did not beat the best default CatBoost model. Because the absolute improvement is small and seed/order-sensitive, report 0.2414 as the clean validation-selected result and 0.2415 as the best observed exploratory result.

Final interpretation:

The plateau appears mainly due to dataset limitations and target noise rather than lack of model complexity. The public UCI data contain useful administrative, diagnosis, utilization, and limited longitudinal signal, but they do not include richer clinical information such as vitals, continuous labs, medication doses, discharge plans, notes, social determinants, exact dates, or hospital/provider identifiers. With patient-safe evaluation, the practical ceiling appears to be around PR-AUC 0.24 to 0.242 for this feature set.
