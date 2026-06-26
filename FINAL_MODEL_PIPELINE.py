"""Final hospital readmission model pipeline.

This is the single-file final model pipeline. It loads the raw UCI Diabetes
130-US Hospitals data, applies the final preprocessing and feature engineering,
trains the validation-selected CatBoost model, writes final outputs, and can
optionally prompt for one patient encounter after training.

Default full run:
    python FINAL_MODEL_PIPELINE.py

Fast wiring check without training:
    python FINAL_MODEL_PIPELINE.py --dry-run

Interactive prediction after training:
    python FINAL_MODEL_PIPELINE.py --interactive-predict
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "archive" / "diabetic_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "final_model_outputs"
SPLIT_RANDOM_STATE = 42

HOSPICE_OR_EXPIRED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]
ID_AND_TARGET_COLUMNS = ["readmitted", "readmitted_30", "encounter_id", "patient_nbr"]

MEDICATION_COLS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

AGE_CHOICES = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]
AGE_MIDPOINT = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}
RACE_CHOICES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"]
GENDER_CHOICES = ["Female", "Male", "Unknown/Invalid"]
LAB_GLUCOSE_CHOICES = ["None", "Norm", ">200", ">300"]
A1C_CHOICES = ["None", "Norm", ">7", ">8"]
MEDICATION_STATUS_CHOICES = ["No", "Steady", "Up", "Down"]
COMMON_MEDICATION_PROMPTS = [
    "insulin",
    "metformin",
    "glipizide",
    "glyburide",
    "glimepiride",
    "pioglitazone",
    "rosiglitazone",
    "repaglinide",
    "nateglinide",
    "acarbose",
]
ADMISSION_TYPE_OPTIONS = {
    "emergency": 1,
    "urgent": 2,
    "elective": 3,
    "trauma": 7,
    "unknown": 5,
}
DISCHARGE_OPTIONS = {
    "home": 1,
    "transfer": 2,
    "skilled_nursing": 3,
    "home_health": 6,
    "left_ama": 7,
    "other": 18,
}
ADMISSION_SOURCE_OPTIONS = {
    "physician_referral": 1,
    "clinic_referral": 2,
    "hospital_transfer": 4,
    "facility_transfer": 5,
    "emergency_room": 7,
    "other": 9,
}


class RareCategoryGrouper:
    """Fit train-only category frequency rules and map rare/unseen values."""

    def __init__(self, columns: list[str], min_count: int = 100):
        self.columns = columns
        self.min_count = min_count
        self.frequent_categories: dict[str, set[str]] = {}

    def fit(self, X: pd.DataFrame) -> "RareCategoryGrouper":
        self.frequent_categories = {}
        for col in self.columns:
            values = X[col].fillna("Missing").astype(str)
            counts = values.value_counts(dropna=False)
            keep = set(counts[counts >= self.min_count].index)
            keep.add("Missing")
            self.frequent_categories[col] = keep
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            values = X[col].fillna("Missing").astype(str)
            keep = self.frequent_categories[col]
            X[col] = np.where(values.isin(keep), values, "Other")
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final readmission CatBoost pipeline.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=37, help="Final model sampling/model seed.")
    parser.add_argument("--split-seed", type=int, default=SPLIT_RANDOM_STATE)
    parser.add_argument("--negative-ratio", type=float, default=7.5)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--od-wait", type=int, default=140)
    parser.add_argument("--dry-run", action="store_true", help="Build features/splits but do not train.")
    parser.add_argument("--quick", action="store_true", help="Train a short smoke-test model.")
    parser.add_argument(
        "--interactive-predict",
        action="store_true",
        help="After training, prompt for patient encounter details and print readmission-risk scores.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def final_model_name(args: argparse.Namespace) -> str:
    return (
        f"NegRefineCat_d{args.depth}_lr{float_token(args.learning_rate)}"
        f"_neg{float_token(args.negative_ratio)}_seed{args.seed}"
    )


def icd9_group(code) -> str:
    if pd.isna(code):
        return "Missing"
    text = str(code).strip()
    if text == "":
        return "Missing"
    if text.startswith(("V", "E")):
        return "External/Supplemental"
    try:
        value = float(text)
    except ValueError:
        return "Other"

    whole = int(value)
    if 390 <= value <= 459 or whole == 785:
        return "Circulatory"
    if 460 <= value <= 519 or whole == 786:
        return "Respiratory"
    if 520 <= value <= 579 or whole == 787:
        return "Digestive"
    if 250 <= value < 251:
        return "Diabetes"
    if 800 <= value <= 999:
        return "Injury/Poisoning"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or whole == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def numeric_icd(code):
    if pd.isna(code):
        return np.nan
    text = str(code).strip()
    if not text or text.startswith(("V", "E")):
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def icd_prefix(code) -> str:
    if pd.isna(code):
        return "Missing"
    text = str(code).strip()
    if not text:
        return "Missing"
    if text.startswith(("V", "E")):
        return text[:3]
    whole = text.split(".")[0]
    return whole[:3] if whole else "Missing"


def icd_chapter(code) -> str:
    if pd.isna(code):
        return "Missing"
    text = str(code).strip()
    if not text:
        return "Missing"
    if text.startswith("V"):
        return "Supplementary V"
    if text.startswith("E"):
        return "External E"
    value = numeric_icd(text)
    if pd.isna(value):
        return "Other"
    whole = int(value)
    if 1 <= value <= 139:
        return "Infectious"
    if 140 <= value <= 239:
        return "Neoplasms"
    if 240 <= value <= 279:
        return "Endocrine/Metabolic"
    if 280 <= value <= 289:
        return "Blood"
    if 290 <= value <= 319:
        return "Mental"
    if 320 <= value <= 389:
        return "Nervous/Sense"
    if 390 <= value <= 459 or whole == 785:
        return "Circulatory"
    if 460 <= value <= 519 or whole == 786:
        return "Respiratory"
    if 520 <= value <= 579 or whole == 787:
        return "Digestive"
    if 580 <= value <= 629 or whole == 788:
        return "Genitourinary"
    if 630 <= value <= 679:
        return "Pregnancy"
    if 680 <= value <= 709:
        return "Skin"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 740 <= value <= 759:
        return "Congenital"
    if 760 <= value <= 779:
        return "Perinatal"
    if 780 <= value <= 799:
        return "Symptoms"
    if 800 <= value <= 999:
        return "Injury/Poisoning"
    return "Other"


def group_age_paper(age_value) -> str:
    if age_value in ["[0-10)", "[10-20)", "[20-30)"]:
        return "<=30"
    if age_value in ["[30-40)", "[40-50)", "[50-60)"]:
        return "30-60"
    return ">60"


def in_ranges(value, ranges: list[tuple[float, float]]) -> bool:
    if pd.isna(value):
        return False
    for lo, hi in ranges:
        if lo <= value <= hi:
            return True
    return False


def any_diag_flag(
    source: pd.DataFrame,
    ranges: list[tuple[float, float]] | None = None,
    prefixes: list[str] | None = None,
    exact: list[str] | None = None,
) -> np.ndarray:
    ranges = ranges or []
    prefixes = prefixes or []
    exact_values = set(exact or [])
    out = np.zeros(len(source), dtype=int)
    for col in ["diag_1", "diag_2", "diag_3"]:
        text = source[col].fillna("").astype(str).str.strip()
        nums = text.map(numeric_icd)
        flag = nums.map(lambda x: in_ranges(x, ranges)).to_numpy(dtype=bool)
        if prefixes:
            flag |= text.str.startswith(tuple(prefixes)).to_numpy(dtype=bool)
        if exact_values:
            flag |= text.str.split(".").str[0].isin(exact_values).to_numpy(dtype=bool)
        out |= flag.astype(int)
    return out


def load_all_eligible_encounters(data_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_path, na_values="?", keep_default_na=False, low_memory=False)
    df = raw.copy()
    df["readmitted_30"] = df["readmitted"].eq("<30").astype(int)
    scoped = df[~df["discharge_disposition_id"].isin(HOSPICE_OR_EXPIRED_DISCHARGE_IDS)].copy()
    return scoped.reset_index(drop=True)


def patient_group_split(scoped: pd.DataFrame, seed: int):
    patient_labels = scoped.groupby("patient_nbr")["readmitted_30"].max().reset_index()
    train_val_pat, test_pat = train_test_split(
        patient_labels,
        test_size=0.15,
        stratify=patient_labels["readmitted_30"],
        random_state=seed,
    )
    val_relative = 0.15 / 0.85
    train_pat, val_pat = train_test_split(
        train_val_pat,
        test_size=val_relative,
        stratify=train_val_pat["readmitted_30"],
        random_state=seed,
    )

    patient = scoped["patient_nbr"]
    train_idx = np.flatnonzero(patient.isin(train_pat["patient_nbr"]).to_numpy())
    val_idx = np.flatnonzero(patient.isin(val_pat["patient_nbr"]).to_numpy())
    test_idx = np.flatnonzero(patient.isin(test_pat["patient_nbr"]).to_numpy())
    return train_idx, val_idx, test_idx


def assert_patient_safe_split(scoped: pd.DataFrame, train_idx, val_idx, test_idx) -> None:
    train_patients = set(scoped.iloc[train_idx]["patient_nbr"])
    val_patients = set(scoped.iloc[val_idx]["patient_nbr"])
    test_patients = set(scoped.iloc[test_idx]["patient_nbr"])
    if train_patients & val_patients:
        raise RuntimeError("Patient leakage detected between train and validation.")
    if train_patients & test_patients:
        raise RuntimeError("Patient leakage detected between train and test.")
    if val_patients & test_patients:
        raise RuntimeError("Patient leakage detected between validation and test.")


def add_diagnosis_detail(X: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    chapters = []
    for col in ["diag_1", "diag_2", "diag_3"]:
        X[f"{col}_chapter"] = source[col].fillna("Missing").map(icd_chapter)
        X[f"{col}_prefix3"] = source[col].fillna("Missing").map(icd_prefix)
        X[f"{col}_numeric"] = source[col].map(numeric_icd)
        chapters.append(X[f"{col}_chapter"])

    chapter_df = pd.concat(chapters, axis=1)
    for chapter in [
        "Circulatory",
        "Respiratory",
        "Genitourinary",
        "Endocrine/Metabolic",
        "Injury/Poisoning",
        "Mental",
        "Neoplasms",
    ]:
        key = chapter.lower().replace("/", "_").replace(" ", "_")
        X[f"diag_any_{key}"] = chapter_df.eq(chapter).any(axis=1).astype(int)
        X[f"diag_count_{key}"] = chapter_df.eq(chapter).sum(axis=1)
    X["diag_unique_chapters"] = chapter_df.nunique(axis=1)
    X["diag_primary_secondary_same_chapter"] = (chapter_df.iloc[:, 0] == chapter_df.iloc[:, 1]).astype(int)
    X["diag_all_same_chapter"] = (
        (chapter_df.iloc[:, 0] == chapter_df.iloc[:, 1])
        & (chapter_df.iloc[:, 1] == chapter_df.iloc[:, 2])
    ).astype(int)
    return X


def add_elixhauser_like_features(X: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    flags = {
        "cm_congestive_heart_failure": {"exact": ["428"]},
        "cm_arrhythmia": {"exact": ["426", "427"]},
        "cm_valvular": {"ranges": [(394, 397)]},
        "cm_pulmonary_circulation": {"ranges": [(415, 417)]},
        "cm_peripheral_vascular": {"ranges": [(440, 443)]},
        "cm_hypertension": {"ranges": [(401, 405)]},
        "cm_paralysis": {"ranges": [(342, 344)]},
        "cm_neurological": {"ranges": [(331, 337)]},
        "cm_chronic_pulmonary": {"ranges": [(490, 496)]},
        "cm_diabetes_uncomplicated": {"ranges": [(250.0, 250.3)]},
        "cm_diabetes_complicated": {"ranges": [(250.4, 250.9)]},
        "cm_hypothyroid": {"ranges": [(243, 244)]},
        "cm_renal_failure": {"exact": ["585", "586"]},
        "cm_liver_disease": {"ranges": [(570, 573)]},
        "cm_peptic_ulcer": {"ranges": [(531, 534)]},
        "cm_aids": {"exact": ["042", "043", "044"]},
        "cm_lymphoma": {"ranges": [(200, 202)]},
        "cm_metastatic_cancer": {"ranges": [(196, 199)]},
        "cm_solid_tumor": {"ranges": [(140, 172), (174, 195)]},
        "cm_rheumatoid_collagen": {"exact": ["701", "710", "714", "720", "725"]},
        "cm_coagulopathy": {"exact": ["286", "287"]},
        "cm_obesity": {"exact": ["278"]},
        "cm_weight_loss": {"ranges": [(260, 263)]},
        "cm_fluid_electrolyte": {"exact": ["276"]},
        "cm_deficiency_anemia": {"exact": ["280", "281"]},
        "cm_alcohol_abuse": {"exact": ["291", "303"], "prefixes": ["305.0"]},
        "cm_drug_abuse": {
            "exact": ["292", "304"],
            "prefixes": ["305.2", "305.3", "305.4", "305.5", "305.6", "305.7", "305.8", "305.9"],
        },
        "cm_psychoses": {"ranges": [(295, 298)]},
        "cm_depression": {"exact": ["311"], "prefixes": ["296.2", "296.3", "300.4"]},
    }
    for name, rule in flags.items():
        X[name] = any_diag_flag(
            source,
            ranges=rule.get("ranges"),
            prefixes=rule.get("prefixes"),
            exact=rule.get("exact"),
        )
    cm_cols = [c for c in X.columns if c.startswith("cm_")]
    X["comorbidity_count"] = X[cm_cols].sum(axis=1)
    X["has_major_comorbidity"] = X["comorbidity_count"].ge(2).astype(int)
    return X


def add_medication_detail(X: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    med = source[MEDICATION_COLS].fillna("No").astype(str)
    X["med_count_no"] = med.eq("No").sum(axis=1)
    X["med_count_steady"] = med.eq("Steady").sum(axis=1)
    X["med_count_up"] = med.eq("Up").sum(axis=1)
    X["med_count_down"] = med.eq("Down").sum(axis=1)
    X["any_med_up"] = X["med_count_up"].gt(0).astype(int)
    X["any_med_down"] = X["med_count_down"].gt(0).astype(int)
    X["any_med_intensification"] = X["any_med_up"]
    X["any_med_deintensification"] = X["any_med_down"]

    med_classes = {
        "biguanide": ["metformin"],
        "sulfonylurea": ["glimepiride", "glipizide", "glyburide"],
        "meglitinide": ["repaglinide", "nateglinide"],
        "thiazolidinedione": ["pioglitazone", "rosiglitazone"],
        "alpha_glucosidase": ["acarbose", "miglitol"],
        "combination": ["glyburide-metformin", "glipizide-metformin"],
    }
    for name, cols in med_classes.items():
        present = med[cols].ne("No").any(axis=1)
        changed = med[cols].isin(["Up", "Down"]).any(axis=1)
        X[f"med_class_{name}_used"] = present.astype(int)
        X[f"med_class_{name}_changed"] = changed.astype(int)

    X["insulin_used"] = med["insulin"].ne("No").astype(int)
    X["insulin_changed"] = med["insulin"].isin(["Up", "Down"]).astype(int)
    X["insulin_up"] = med["insulin"].eq("Up").astype(int)
    X["insulin_down"] = med["insulin"].eq("Down").astype(int)
    X["oral_med_classes_used"] = X[
        [c for c in X.columns if c.startswith("med_class_") and c.endswith("_used")]
    ].sum(axis=1)
    return X


def add_utilization_features(X: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    inpatient = source["number_inpatient"].astype(float)
    emergency = source["number_emergency"].astype(float)
    outpatient = source["number_outpatient"].astype(float)
    time_in_hospital = source["time_in_hospital"].astype(float)
    num_meds = source["num_medications"].astype(float)
    num_labs = source["num_lab_procedures"].astype(float)
    num_procedures = source["num_procedures"].astype(float)

    X["age_midpoint"] = source["age"].map(AGE_MIDPOINT).astype(float)
    X["elderly_70_plus"] = X["age_midpoint"].ge(70).astype(int)
    X["very_elderly_80_plus"] = X["age_midpoint"].ge(80).astype(int)
    X["has_prior_inpatient"] = inpatient.gt(0).astype(int)
    X["has_prior_emergency"] = emergency.gt(0).astype(int)
    X["has_prior_outpatient"] = outpatient.gt(0).astype(int)
    X["prior_acute_visits"] = inpatient + emergency
    X["weighted_prior_utilization"] = inpatient * 2.0 + emergency * 1.5 + outpatient
    X["log1p_prior_acute_visits"] = np.log1p(X["prior_acute_visits"])
    X["frequent_inpatient"] = inpatient.ge(2).astype(int)
    X["frequent_acute_user"] = X["prior_acute_visits"].ge(2).astype(int)
    X["long_stay_7_plus"] = time_in_hospital.ge(7).astype(int)
    X["very_long_stay_10_plus"] = time_in_hospital.ge(10).astype(int)
    X["polypharmacy_15_plus"] = num_meds.ge(15).astype(int)
    X["polypharmacy_20_plus"] = num_meds.ge(20).astype(int)
    X["many_labs_50_plus"] = num_labs.ge(50).astype(int)
    X["procedures_2_plus"] = num_procedures.ge(2).astype(int)
    X["labs_per_day"] = num_labs / (time_in_hospital + 1)
    X["meds_per_day"] = num_meds / (time_in_hospital + 1)
    X["procedures_per_day"] = num_procedures / (time_in_hospital + 1)
    X["utilization_per_day"] = X["weighted_prior_utilization"] / (time_in_hospital + 1)
    X["age_x_prior_acute"] = X["age_midpoint"] * X["prior_acute_visits"]
    X["age_x_inpatient"] = X["age_midpoint"] * inpatient
    X["long_stay_x_polypharmacy"] = X["long_stay_7_plus"] * X["polypharmacy_15_plus"]
    X["elderly_x_prior_inpatient"] = X["elderly_70_plus"] * X["has_prior_inpatient"]
    X["elderly_x_frequent_acute"] = X["elderly_70_plus"] * X["frequent_acute_user"]
    return X


def add_admin_features(X: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    discharge = source["discharge_disposition_id"]
    admission_type = source["admission_type_id"]
    admission_source = source["admission_source_id"]
    X["discharged_home"] = discharge.eq(1).astype(int)
    X["discharged_home_health"] = discharge.eq(6).astype(int)
    X["discharged_transfer_facility"] = discharge.isin([2, 3, 4, 5, 22, 23, 24, 27, 28, 29, 30]).astype(int)
    X["left_ama"] = discharge.eq(7).astype(int)
    X["emergency_or_urgent_admission"] = admission_type.isin([1, 2, 7]).astype(int)
    X["elective_admission"] = admission_type.eq(3).astype(int)
    X["emergency_room_source"] = admission_source.eq(7).astype(int)
    X["transfer_source"] = admission_source.isin([4, 5, 6, 10, 18, 22, 25, 26]).astype(int)
    X["transfer_or_facility_x_prior_inpatient"] = X["discharged_transfer_facility"] * X["has_prior_inpatient"]
    X["home_health_x_elderly"] = X["discharged_home_health"] * X["elderly_70_plus"]
    return X


def add_lab_features(X: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    a1c = source["A1Cresult"].fillna("None").astype(str)
    glu = source["max_glu_serum"].fillna("None").astype(str)
    X["a1c_measured"] = a1c.ne("None").astype(int)
    X["a1c_high"] = a1c.isin([">7", ">8"]).astype(int)
    X["a1c_very_high"] = a1c.eq(">8").astype(int)
    X["glucose_measured"] = glu.ne("None").astype(int)
    X["glucose_high"] = glu.isin([">200", ">300"]).astype(int)
    X["glucose_very_high"] = glu.eq(">300").astype(int)
    X["a1c_high_x_insulin_changed"] = X["a1c_high"] * X["insulin_changed"]
    X["glucose_high_x_insulin_changed"] = X["glucose_high"] * X["insulin_changed"]
    X["a1c_high_x_any_med_intensification"] = X["a1c_high"] * X["any_med_intensification"]
    return X


def add_categorical_interactions(X: pd.DataFrame) -> pd.DataFrame:
    X["diag12_chapter_pair"] = X["diag_1_chapter"].astype(str) + "__" + X["diag_2_chapter"].astype(str)
    X["age_diag1_chapter"] = X["age_group_paper"].astype(str) + "__" + X["diag_1_chapter"].astype(str)
    X["discharge_diag1_chapter"] = X["discharge_disposition_raw"].astype(str) + "__" + X["diag_1_chapter"].astype(str)
    X["source_diag1_chapter"] = X["admission_source_raw"].astype(str) + "__" + X["diag_1_chapter"].astype(str)
    return X


def _prior_cumulative_sum(sorted_df: pd.DataFrame, group_col: str, value_col: str):
    return sorted_df.groupby(group_col)[value_col].cumsum() - sorted_df[value_col]


def add_patient_history_features(X: pd.DataFrame, scoped: pd.DataFrame) -> pd.DataFrame:
    """Add only prior-within-patient features ordered by encounter_id."""
    source = scoped.loc[X.index].copy()
    source["_readmitted_30"] = source["readmitted"].eq("<30").astype(int)
    source["_readmitted_any"] = source["readmitted"].isin(["<30", ">30"]).astype(int)
    sorted_source = source.sort_values(["patient_nbr", "encounter_id"]).copy()
    g = sorted_source.groupby("patient_nbr", sort=False)

    hist = pd.DataFrame(index=sorted_source.index)
    hist["patient_prior_encounters"] = g.cumcount().astype(float)
    hist["patient_prior_readmit30_count"] = _prior_cumulative_sum(sorted_source, "patient_nbr", "_readmitted_30")
    hist["patient_prior_readmit_any_count"] = _prior_cumulative_sum(sorted_source, "patient_nbr", "_readmitted_any")
    hist["patient_prior_no_readmit_count"] = hist["patient_prior_encounters"] - hist["patient_prior_readmit_any_count"]
    denom = hist["patient_prior_encounters"].replace(0, np.nan)
    hist["patient_prior_readmit30_rate"] = (hist["patient_prior_readmit30_count"] / denom).fillna(0.0)
    hist["patient_prior_readmit_any_rate"] = (hist["patient_prior_readmit_any_count"] / denom).fillna(0.0)
    hist["patient_has_prior_encounter"] = hist["patient_prior_encounters"].gt(0).astype(int)
    hist["patient_has_prior_readmit30"] = hist["patient_prior_readmit30_count"].gt(0).astype(int)
    hist["patient_has_prior_readmit_any"] = hist["patient_prior_readmit_any_count"].gt(0).astype(int)
    hist["patient_prior_encounter_bucket"] = pd.cut(
        hist["patient_prior_encounters"],
        bins=[-1, 0, 1, 2, 4, 999],
        labels=["0", "1", "2", "3-4", "5+"],
    ).astype(str)

    numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_diagnoses",
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
    ]
    for col in numeric_cols:
        values = sorted_source[col].astype(float)
        prior_sum = sorted_source.assign(_value=values).groupby("patient_nbr")["_value"].cumsum() - values
        hist[f"patient_prior_avg_{col}"] = (prior_sum / denom).fillna(0.0)
        hist[f"patient_prev_{col}"] = g[col].shift(1).fillna(0).astype(float)

    previous_cats = [
        "readmitted",
        "discharge_disposition_id",
        "admission_source_id",
        "admission_type_id",
        "diag_1",
        "diag_2",
        "diag_3",
        "A1Cresult",
        "max_glu_serum",
        "change",
        "diabetesMed",
    ]
    for col in previous_cats:
        hist[f"patient_prev_{col}"] = g[col].shift(1).fillna("No prior").astype(str)

    hist = hist.sort_index()
    out = X.copy()
    for col in hist.columns:
        out[col] = hist[col]
    return out


def build_final_features(scoped: pd.DataFrame) -> pd.DataFrame:
    X = scoped.drop(columns=ID_AND_TARGET_COLUMNS).copy()

    for col in ["race", "medical_specialty", "payer_code"]:
        X[col] = X[col].fillna("Missing").astype(str)

    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        X[diag_col] = X[diag_col].fillna("Missing").astype(str)
        X[f"{diag_col}_group"] = X[diag_col].apply(icd9_group)

    X["admission_type_raw"] = "admission_type_" + X["admission_type_id"].astype(str)
    X["discharge_disposition_raw"] = "discharge_" + X["discharge_disposition_id"].astype(str)
    X["admission_source_raw"] = "source_" + X["admission_source_id"].astype(str)
    X["age_group_paper"] = X["age"].apply(group_age_paper)

    med = X[MEDICATION_COLS].fillna("No").astype(str)
    X["num_diabetes_meds_used"] = med.ne("No").sum(axis=1)
    X["num_diabetes_med_changes"] = med.isin(["Up", "Down"]).sum(axis=1)
    X["service_utilization"] = X["number_outpatient"] + X["number_emergency"] + X["number_inpatient"]
    for col in ["number_outpatient", "number_emergency", "number_inpatient", "service_utilization"]:
        X[f"log1p_{col}"] = np.log1p(X[col].astype(float))
    X["weight_recorded"] = X["weight"].notna().astype(int)

    drop_cols = [
        "weight",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "age",
        "diag_1",
        "diag_2",
        "diag_3",
        *MEDICATION_COLS,
    ]
    X = X.drop(columns=drop_cols, errors="ignore")

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    X[categorical_cols] = X[categorical_cols].fillna("Missing").astype(str)

    source = scoped.loc[X.index]
    X = add_diagnosis_detail(X, source)
    X = add_elixhauser_like_features(X, source)
    X = add_medication_detail(X, source)
    X = add_utilization_features(X, source)
    X = add_admin_features(X, source)
    X = add_lab_features(X, source)
    X = add_categorical_interactions(X)
    X = add_patient_history_features(X, scoped)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    X[categorical_cols] = X[categorical_cols].fillna("Missing").astype(str)
    return X


def all_categorical_columns(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=["object", "category"]).columns.tolist()


def prepare_native_frames(
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    min_count: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    cat_cols = all_categorical_columns(X_train)
    grouper = RareCategoryGrouper(columns=cat_cols, min_count=min_count)
    X_train = grouper.fit_transform(X_train.copy())
    X_eval = grouper.transform(X_eval.copy())

    cat_cols = all_categorical_columns(X_train)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Missing").astype(str)
        X_eval[col] = X_eval[col].fillna("Missing").astype(str)
    medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(medians)
    X_eval[num_cols] = X_eval[num_cols].fillna(medians)
    cat_features = [X_train.columns.get_loc(col) for col in cat_cols]
    return X_train, X_eval, cat_features


def threshold_metrics(y_true, y_score, threshold: float) -> dict:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def best_thresholds(y_true, y_score) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return {"default_0.5": 0.5}

    p = precision[:-1]
    r = recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    beta2 = 2.0
    f2 = np.divide(
        (1 + beta2**2) * p * r,
        beta2**2 * p + r,
        out=np.zeros_like(p),
        where=(beta2**2 * p + r) > 0,
    )

    choices = {
        "default_0.5": 0.5,
        "best_f1": float(thresholds[int(np.nanargmax(f1))]),
        "best_f2": float(thresholds[int(np.nanargmax(f2))]),
    }

    for min_precision in [0.12, 0.15, 0.20]:
        valid = np.where(p >= min_precision)[0]
        if len(valid):
            idx = valid[int(np.nanargmax(r[valid]))]
            choices[f"max_recall_precision_ge_{min_precision:.2f}"] = float(thresholds[idx])
    return choices


def lift_rows(y_true, y_score, metadata: dict, fractions=(0.01, 0.05, 0.10, 0.20)) -> list[dict]:
    y_array = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))
    total_pos = max(1, int(y_array.sum()))
    base_rate = float(y_array.mean())
    rows = []
    for frac in fractions:
        n_flagged = max(1, int(np.ceil(len(y_array) * frac)))
        selected = order[:n_flagged]
        positives = int(y_array[selected].sum())
        precision = positives / n_flagged
        recall = positives / total_pos
        rows.append(
            {
                **metadata,
                "top_fraction": frac,
                "n_flagged": n_flagged,
                "positives_captured": positives,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "base_rate": base_rate,
                "lift": precision / base_rate if base_rate > 0 else np.nan,
            }
        )
    return rows


def train_indices_for_ratio(y_train: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    n_neg = min(len(neg_idx), int(round(len(pos_idx) * ratio)))
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    selected = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(selected)
    return selected


def make_final_model(args: argparse.Namespace) -> CatBoostClassifier:
    iterations = 50 if args.quick else args.iterations
    od_wait = min(args.od_wait, 20) if args.quick else args.od_wait
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=args.seed,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=od_wait,
    )


def split_summary(scoped: pd.DataFrame, y: pd.Series, train_idx, val_idx, test_idx) -> pd.DataFrame:
    rows = []
    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        labels = y.iloc[idx]
        rows.append(
            {
                "split": name,
                "rows": len(idx),
                "patients": scoped.iloc[idx]["patient_nbr"].nunique(),
                "positive_count": int(labels.sum()),
                "positive_rate": float(labels.mean()),
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _clean_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} [{suffix}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please enter yes or no.")


def prompt_choice(label: str, choices: list[str], default: str, aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or {}
    lookup = {_clean_key(choice): choice for choice in choices}
    lookup.update({_clean_key(key): value for key, value in aliases.items()})
    choice_text = ", ".join(choices)
    while True:
        value = input(f"{label} [{default}] ({choice_text}): ").strip()
        if not value:
            return default
        key = _clean_key(value)
        if key in lookup:
            return lookup[key]
        print(f"Please enter one of: {choice_text}.")


def prompt_int(label: str, default: int, min_value: int = 0, max_value: int | None = None) -> int:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            parsed = int(value)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if parsed < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        if max_value is not None and parsed > max_value:
            print(f"Please enter a value <= {max_value}.")
            continue
        return parsed


def prompt_optional_int(label: str) -> int | None:
    while True:
        value = input(f"{label} [new patient/no prior history]: ").strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            print("Please enter a whole-number patient_nbr or leave blank.")


def prompt_code_choice(
    label: str,
    options: dict[str, int],
    default_key: str,
    reject_hospice_expired: bool = False,
) -> int:
    option_text = ", ".join(f"{name}={code}" for name, code in options.items())
    valid_codes = set(options.values())
    while True:
        value = input(f"{label} [{default_key}] ({option_text}): ").strip()
        if not value:
            code = options[default_key]
        elif value.isdigit():
            code = int(value)
        else:
            key = _clean_key(value)
            if key not in options:
                print(f"Please enter one of: {option_text}. You may also enter a numeric code.")
                continue
            code = options[key]
        if reject_hospice_expired and code in HOSPICE_OR_EXPIRED_DISCHARGE_IDS:
            print("That discharge code is out of scope because hospice/expired cases were removed.")
            continue
        if valid_codes and code not in valid_codes:
            print("Using a custom code. Make sure it matches the dataset codebook.")
        return code


def default_raw_record(scoped: pd.DataFrame) -> dict:
    record = {}
    for col in scoped.columns:
        if col == "readmitted_30":
            record[col] = 0
        elif col == "readmitted":
            record[col] = "NO"
        elif col in MEDICATION_COLS:
            record[col] = "No"
        elif pd.api.types.is_numeric_dtype(scoped[col]):
            value = scoped[col].median()
            record[col] = int(round(float(value))) if pd.notna(value) else 0
        else:
            mode = scoped[col].dropna().mode()
            record[col] = mode.iloc[0] if not mode.empty else "Missing"
    record["encounter_id"] = int(scoped["encounter_id"].max()) + 1
    record["patient_nbr"] = int(scoped["patient_nbr"].max()) + 1
    record["readmitted"] = "NO"
    record["readmitted_30"] = 0
    record["weight"] = np.nan
    for col in MEDICATION_COLS:
        record[col] = "No"
    return record


def collect_patient_record(scoped: pd.DataFrame) -> dict:
    record = default_raw_record(scoped)

    print("\nEnter one patient encounter. Press Enter to keep the default shown in brackets.")
    print("This is a course-project demo, not a clinical decision tool.\n")

    patient_nbr = prompt_optional_int("Existing patient_nbr, if this patient is already in the dataset")
    if patient_nbr is not None:
        record["patient_nbr"] = patient_nbr

    record["race"] = prompt_choice(
        "Race",
        RACE_CHOICES,
        str(record.get("race", "Caucasian")),
        aliases={"african_american": "AfricanAmerican", "unknown": "Missing"},
    )
    record["gender"] = prompt_choice("Gender", GENDER_CHOICES, str(record.get("gender", "Female")))
    record["age"] = prompt_choice("Age group", AGE_CHOICES, str(record.get("age", "[70-80)")))

    if prompt_yes_no("Was patient weight recorded?", default=False):
        record["weight"] = input("Weight category, for example [75-100) [Missing]: ").strip() or "Missing"
    else:
        record["weight"] = np.nan

    record["payer_code"] = input("Payer code [Missing]: ").strip() or "Missing"
    record["medical_specialty"] = input("Medical specialty [Missing]: ").strip() or "Missing"

    record["admission_type_id"] = prompt_code_choice("Admission type", ADMISSION_TYPE_OPTIONS, "emergency")
    record["discharge_disposition_id"] = prompt_code_choice(
        "Discharge disposition",
        DISCHARGE_OPTIONS,
        "home",
        reject_hospice_expired=True,
    )
    record["admission_source_id"] = prompt_code_choice("Admission source", ADMISSION_SOURCE_OPTIONS, "emergency_room")

    record["time_in_hospital"] = prompt_int("Days in hospital", int(record["time_in_hospital"]), 1, 14)
    record["num_lab_procedures"] = prompt_int("Number of lab procedures", int(record["num_lab_procedures"]), 0)
    record["num_procedures"] = prompt_int("Number of procedures", int(record["num_procedures"]), 0)
    record["num_medications"] = prompt_int("Number of medications", int(record["num_medications"]), 0)
    record["number_outpatient"] = prompt_int("Prior outpatient visits", int(record["number_outpatient"]), 0)
    record["number_emergency"] = prompt_int("Prior emergency visits", int(record["number_emergency"]), 0)
    record["number_inpatient"] = prompt_int("Prior inpatient visits", int(record["number_inpatient"]), 0)
    record["number_diagnoses"] = prompt_int("Number of diagnoses", int(record["number_diagnoses"]), 1)

    record["diag_1"] = input(f"Primary ICD-9 diagnosis code [{record['diag_1']}]: ").strip() or record["diag_1"]
    record["diag_2"] = input(f"Secondary ICD-9 diagnosis code [{record['diag_2']}]: ").strip() or record["diag_2"]
    record["diag_3"] = input(f"Additional ICD-9 diagnosis code [{record['diag_3']}]: ").strip() or record["diag_3"]

    record["max_glu_serum"] = prompt_choice("Max glucose serum result", LAB_GLUCOSE_CHOICES, "None")
    record["A1Cresult"] = prompt_choice("A1C result", A1C_CHOICES, "None")
    record["change"] = "Ch" if prompt_yes_no("Were diabetes medications changed?", default=False) else "No"
    has_diabetes_med = prompt_yes_no("Was any diabetes medication prescribed?", default=True)
    record["diabetesMed"] = "Yes" if has_diabetes_med else "No"

    if has_diabetes_med:
        for med in COMMON_MEDICATION_PROMPTS:
            record[med] = prompt_choice(f"{med} status", MEDICATION_STATUS_CHOICES, "No")
    for med in MEDICATION_COLS:
        record.setdefault(med, "No")

    return record


def predict_patient_record(
    model: CatBoostClassifier,
    scoped: pd.DataFrame,
    X_train_raw: pd.DataFrame,
    record: dict,
    final_threshold: float,
    reference_scores: np.ndarray,
) -> float:
    new_row = pd.DataFrame([record], columns=scoped.columns)
    augmented = pd.concat([scoped, new_row], ignore_index=True)
    X_augmented = build_final_features(augmented)
    X_new_raw = X_augmented.tail(1).copy()
    _, X_new, _ = prepare_native_frames(X_train_raw.copy(), X_new_raw, min_count=100)
    score = float(model.predict_proba(X_new)[:, 1][0])
    percentile = float((reference_scores <= score).mean() * 100.0)
    decision = "AT OR ABOVE" if score >= final_threshold else "below"

    print("\nPrediction")
    print(f"- 30-day readmission risk score: {score:.4f}")
    print("- This is not a calibrated probability; it is best used for ranking patients by risk.")
    print(f"- Validation-selected operating threshold: {final_threshold:.4f}")
    print(f"- Classification at that threshold: {decision} the risk-flag threshold")
    print(f"- Risk ranking: higher than about {percentile:.1f}% of held-out test encounters")
    print("Interpret as a risk-ranking aid for a course project, not as medical advice.\n")
    return score


def interactive_prediction_loop(
    model: CatBoostClassifier,
    scoped: pd.DataFrame,
    X_train_raw: pd.DataFrame,
    final_threshold: float,
    reference_scores: np.ndarray,
) -> None:
    while True:
        try:
            record = collect_patient_record(scoped)
        except EOFError:
            print("\nNo interactive input was received; patient prediction prompt cancelled.")
            break
        predict_patient_record(model, scoped, X_train_raw, record, final_threshold, reference_scores)
        try:
            enter_another = prompt_yes_no("Enter another patient?", default=False)
        except EOFError:
            enter_another = False
        if not enter_another:
            break


def maybe_run_interactive_prediction(
    args: argparse.Namespace,
    model: CatBoostClassifier,
    scoped: pd.DataFrame,
    X_train_raw: pd.DataFrame,
    final_threshold: float,
    reference_scores: np.ndarray,
) -> None:
    if args.interactive_predict:
        interactive_prediction_loop(model, scoped, X_train_raw, final_threshold, reference_scores)
        return
    if sys.stdin.isatty() and prompt_yes_no("\nStart interactive patient prediction now?", default=False):
        interactive_prediction_loop(model, scoped, X_train_raw, final_threshold, reference_scores)
    else:
        print("\nTo enter a patient manually after training, run: python FINAL_MODEL_PIPELINE.py --interactive-predict")


def run() -> None:
    args = parse_args()
    args.data_path = resolve_repo_path(args.data_path)
    args.output_dir = resolve_repo_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.split_seed)
    model_name = final_model_name(args)

    scoped = load_all_eligible_encounters(args.data_path)
    y = scoped["readmitted_30"].astype(int)

    train_idx, val_idx, test_idx = patient_group_split(scoped, args.split_seed)
    assert_patient_safe_split(scoped, train_idx, val_idx, test_idx)
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    X = build_final_features(scoped)
    X_train_raw = X.iloc[train_idx].copy()
    X_val_raw = X.iloc[val_idx].copy()
    X_test_raw = X.iloc[test_idx].copy()

    X_train, X_val, cat_features = prepare_native_frames(X_train_raw, X_val_raw, min_count=100)
    _, X_test, _ = prepare_native_frames(X_train_raw.copy(), X_test_raw, min_count=100)

    selected_train_idx = train_indices_for_ratio(y_train, args.negative_ratio, args.seed)

    summary = {
        "model_name": model_name,
        "feature_config": "all_eligible_patient_safe_history_weight_indicator_catboost",
        "data_path": str(args.data_path),
        "rows_after_hospice_expired_removal": int(len(scoped)),
        "patients_after_hospice_expired_removal": int(scoped["patient_nbr"].nunique()),
        "positive_rate_after_hospice_expired_removal": float(y.mean()),
        "n_features": int(X.shape[1]),
        "n_catboost_categorical_features": int(len(cat_features)),
        "train_subset_rows": int(len(selected_train_idx)),
        "train_subset_positive_rows": int(y_train[selected_train_idx].sum()),
        "train_subset_negative_rows": int((y_train[selected_train_idx] == 0).sum()),
        "negative_ratio": float(args.negative_ratio),
        "split_seed": int(args.split_seed),
        "seed": int(args.seed),
        "iterations": int(50 if args.quick else args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_strength": float(args.random_strength),
        "od_wait": int(min(args.od_wait, 20) if args.quick else args.od_wait),
        "dry_run": bool(args.dry_run),
        "quick": bool(args.quick),
    }

    split_summary(scoped, y, train_idx, val_idx, test_idx).to_csv(
        args.output_dir / "final_split_summary.csv", index=False
    )
    write_json(args.output_dir / "final_pipeline_summary.json", summary)

    print("Final model pipeline prepared:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        print(f"Dry run complete. Wrote summaries to {args.output_dir}.")
        return

    model = make_final_model(args)
    start = time.perf_counter()
    model.fit(
        X_train.iloc[selected_train_idx],
        y_train[selected_train_idx],
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    fit_seconds = time.perf_counter() - start

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    validation_thresholds = best_thresholds(y_val, val_score)
    final_threshold = validation_thresholds["best_f1"]
    best_iteration = model.get_best_iteration()
    if best_iteration is None:
        best_iteration = int(model.get_param("iterations"))

    rows = []
    for split_name, labels, scores in [
        ("validation", y_val, val_score),
        ("test", y_test, test_score),
    ]:
        row = threshold_metrics(labels, scores, final_threshold)
        row.update(
            {
                "model_name": model_name,
                "split": split_name,
                "threshold_strategy": "validation_best_f1",
                "threshold_source": "validation",
                "negative_ratio": args.negative_ratio,
                "seed": args.seed,
                "fit_seconds": fit_seconds,
                "best_iteration": int(best_iteration),
            }
        )
        rows.append(row)

    diagnostic_test_threshold = best_thresholds(y_test, test_score)["best_f1"]
    diagnostic_row = threshold_metrics(y_test, test_score, diagnostic_test_threshold)
    diagnostic_row.update(
        {
            "model_name": model_name,
            "split": "test",
            "threshold_strategy": "test_best_f1_diagnostic_only",
            "threshold_source": "test",
            "negative_ratio": args.negative_ratio,
            "seed": args.seed,
            "fit_seconds": fit_seconds,
            "best_iteration": int(best_iteration),
        }
    )
    rows.append(diagnostic_row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "final_model_metrics.csv", index=False)

    lift = pd.DataFrame(
        lift_rows(
            y_test,
            test_score,
            {
                "model_name": model_name,
                "split": "test",
                "threshold_strategy": "ranking",
                "negative_ratio": args.negative_ratio,
                "seed": args.seed,
            },
        )
    )
    lift.to_csv(args.output_dir / "final_model_lift_table.csv", index=False)

    np.save(args.output_dir / "final_validation_scores.npy", val_score)
    np.save(args.output_dir / "final_test_scores.npy", test_score)

    print("\nFinal model metrics:")
    print(metrics.to_string(index=False))
    print("\nFinal model lift table:")
    print(lift.to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir.resolve()}")

    maybe_run_interactive_prediction(args, model, scoped, X_train_raw, final_threshold, test_score)


if __name__ == "__main__":
    run()
