# Hospital Readmission Prediction

This is an AI1215 course project on predicting 30-day hospital readmission for diabetic patient encounters using the UCI Diabetes 130-US Hospitals dataset.

The final model is a self-contained CatBoost risk-ranking pipeline. It is designed to help identify higher-risk patients for follow-up review, not to act as a standalone clinical decision system.

## Final Model Result

```text
Model family: CatBoost with all eligible encounters, patient-safe split, and prior-history features
Test PR-AUC: 0.2414
Test ROC-AUC: 0.6827
Validation-selected threshold:
  Recall: 0.3621
  Precision: 0.2416
  F1: 0.2898
  Accuracy: 0.8044
```

The saved output also includes a diagnostic test-best-F1 threshold row with recall `0.4226`, precision `0.2223`, F1 `0.2913`, and accuracy `0.7733`. That row is useful for interpretation, but the validation-selected threshold is the cleaner operating point.

Baseline PR-AUC is the positive rate: `0.1103`.

The final model more than doubles this baseline, so the strongest way to present it is as a patient risk-ranking tool.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

The raw dataset must be here:

```text
archive/diabetic_data.csv
```

Run the final pipeline without manual patient input:

```bash
python FINAL_MODEL_PIPELINE.py
```

Run the final pipeline and then enter a patient manually for a prediction:

```bash
python FINAL_MODEL_PIPELINE.py --interactive-predict
```

The interactive mode asks for patient encounter information one field at a time, then prints a 30-day readmission risk score and whether the patient is above or below the model's selected risk threshold. The score is for ranking; it is not a calibrated probability.

## Most Important Files

- `FINAL_MODEL_PIPELINE.py`: self-contained final model file. It contains the final preprocessing, feature engineering, CatBoost training, evaluation, and optional patient prediction demo.
- `final_report_package/AI1215_Final_Report_LaTeX.pdf`: final report PDF.
- `final_report_package/AI1215_Final_Report_LaTeX.tex`: final report LaTeX source.
- `final_report_package/PRESENTATION_SCRIPT.md`: presentation speaking script.
- `PROFESSOR_SUBMISSION_GUIDE.md`: short guide explaining what to inspect.
- `hospital_readmission_eda.ipynb`: EDA notebook.
- `hospital_readmission_modeling.ipynb`: modeling notebook with baseline models and experiment summaries.
- `experiment_results/`: saved results from the models and feature-engineering attempts.
- `experiments/`: experiment scripts used during the broader search history.

## Notes

- `requirements.txt` is pinned and minimal for the final pipeline.
- `requirements-experiments.txt` contains optional heavier packages used by the broader experiment history.
- `requirements-pinned.txt` mirrors the final-pipeline package versions for compatibility with earlier submission notes.
- `final_model_outputs/` is generated when the final pipeline runs and is ignored by Git.
- The current final result comes from `FINAL_MODEL_PIPELINE.py`; experiment scripts are history, not required for the final run.

## Limitation

The result plateaus because the dataset is imbalanced and does not include many useful clinical details, such as exact dates, vitals, continuous lab values, medication doses, discharge plans, hospital/provider identifiers, or social determinants of health.
