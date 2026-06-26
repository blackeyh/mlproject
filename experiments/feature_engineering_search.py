from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from modeling_experiments import (
    MEDICATION_COLS,
    RANDOM_STATE,
    RESULTS_DIR,
    FeatureConfig,
    RareCategoryGrouper,
    best_thresholds,
    build_feature_matrix,
    get_scores,
    load_scoped_data,
    threshold_metrics,
)
from imbalance_experiments import majority_row, lift_rows, select_validation_candidates


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(RANDOM_STATE)


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


@dataclass(frozen=True)
class EngineeredFeatureConfig:
    name: str
    base: FeatureConfig
    add_diagnosis_detail: bool = True
    add_elixhauser_flags: bool = True
    add_medication_detail: bool = True
    add_utilization_interactions: bool = True
    add_lab_interactions: bool = True
    add_admin_risk_flags: bool = True
    add_categorical_interactions: bool = False
    rare_min_count: int = 100


def split_indices(y):
    train_val_idx, test_idx, y_train_val, y_test = train_test_split(
        np.arange(len(y)),
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    val_relative = 0.15 / 0.85
    train_idx, val_idx, y_train, y_val = train_test_split(
        train_val_idx,
        y_train_val,
        test_size=val_relative,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )
    return train_idx, val_idx, test_idx, y_train, y_val, y_test


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


def icd_prefix(code):
    if pd.isna(code):
        return "Missing"
    text = str(code).strip()
    if not text:
        return "Missing"
    if text.startswith(("V", "E")):
        return text[:3]
    whole = text.split(".")[0]
    return whole[:3] if whole else "Missing"


def icd_chapter(code):
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


def in_ranges(value, ranges):
    if pd.isna(value):
        return False
    whole = int(value)
    for lo, hi in ranges:
        if lo <= value <= hi:
            return True
    return False


def any_diag_flag(source, ranges=None, prefixes=None, exact=None):
    ranges = ranges or []
    prefixes = prefixes or []
    exact = set(exact or [])
    out = np.zeros(len(source), dtype=int)
    for col in ["diag_1", "diag_2", "diag_3"]:
        text = source[col].fillna("").astype(str).str.strip()
        nums = text.map(numeric_icd)
        flag = nums.map(lambda x: in_ranges(x, ranges)).to_numpy(dtype=bool)
        if prefixes:
            flag |= text.str.startswith(tuple(prefixes)).to_numpy(dtype=bool)
        if exact:
            flag |= text.str.split(".").str[0].isin(exact).to_numpy(dtype=bool)
        out |= flag.astype(int)
    return out


def add_elixhauser_like_features(X, source):
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
        "cm_drug_abuse": {"exact": ["292", "304"], "prefixes": ["305.2", "305.3", "305.4", "305.5", "305.6", "305.7", "305.8", "305.9"]},
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


def add_diagnosis_detail(X, source):
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
        (chapter_df.iloc[:, 0] == chapter_df.iloc[:, 1]) & (chapter_df.iloc[:, 1] == chapter_df.iloc[:, 2])
    ).astype(int)
    return X


def add_medication_detail(X, source):
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


def add_utilization_features(X, source):
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


def add_lab_features(X, source):
    a1c = source["A1Cresult"].fillna("None").astype(str)
    glu = source["max_glu_serum"].fillna("None").astype(str)
    X["a1c_measured"] = a1c.ne("None").astype(int)
    X["a1c_high"] = a1c.isin([">7", ">8"]).astype(int)
    X["a1c_very_high"] = a1c.eq(">8").astype(int)
    X["glucose_measured"] = glu.ne("None").astype(int)
    X["glucose_high"] = glu.isin([">200", ">300"]).astype(int)
    X["glucose_very_high"] = glu.eq(">300").astype(int)
    if "insulin_changed" in X:
        X["a1c_high_x_insulin_changed"] = X["a1c_high"] * X["insulin_changed"]
        X["glucose_high_x_insulin_changed"] = X["glucose_high"] * X["insulin_changed"]
    if "any_med_intensification" in X:
        X["a1c_high_x_any_med_intensification"] = X["a1c_high"] * X["any_med_intensification"]
    return X


def add_admin_features(X, source):
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
    if "has_prior_inpatient" in X:
        X["transfer_or_facility_x_prior_inpatient"] = X["discharged_transfer_facility"] * X["has_prior_inpatient"]
        X["home_health_x_elderly"] = X["discharged_home_health"] * X.get("elderly_70_plus", 0)
    return X


def add_categorical_interactions(X):
    if {"diag_1_chapter", "diag_2_chapter"}.issubset(X.columns):
        X["diag12_chapter_pair"] = X["diag_1_chapter"].astype(str) + "__" + X["diag_2_chapter"].astype(str)
    if {"age_group_paper", "diag_1_chapter"}.issubset(X.columns):
        X["age_diag1_chapter"] = X["age_group_paper"].astype(str) + "__" + X["diag_1_chapter"].astype(str)
    if {"discharge_disposition_raw", "diag_1_chapter"}.issubset(X.columns):
        X["discharge_diag1_chapter"] = X["discharge_disposition_raw"].astype(str) + "__" + X["diag_1_chapter"].astype(str)
    if {"admission_source_raw", "diag_1_chapter"}.issubset(X.columns):
        X["source_diag1_chapter"] = X["admission_source_raw"].astype(str) + "__" + X["diag_1_chapter"].astype(str)
    return X


def build_engineered_matrix(scoped, cfg: EngineeredFeatureConfig):
    X, y = build_feature_matrix(scoped, cfg.base)
    source = scoped.loc[X.index]
    X = X.copy()
    if cfg.add_diagnosis_detail:
        X = add_diagnosis_detail(X, source)
    if cfg.add_elixhauser_flags:
        X = add_elixhauser_like_features(X, source)
    if cfg.add_medication_detail:
        X = add_medication_detail(X, source)
    if cfg.add_utilization_interactions:
        X = add_utilization_features(X, source)
    if cfg.add_admin_risk_flags:
        X = add_admin_features(X, source)
    if cfg.add_lab_interactions:
        X = add_lab_features(X, source)
    if cfg.add_categorical_interactions:
        X = add_categorical_interactions(X)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    X[categorical_cols] = X[categorical_cols].fillna("Missing").astype(str)
    return X, y


def engineered_configs():
    return [
        EngineeredFeatureConfig(
            name="fe_summary_core",
            base=FeatureConfig(
                name="fe_base_summary",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="paper",
                gender_mode="keep",
                weight_mode="category",
                medication_mode="summaries_only",
                utilization_mode="log_plus_raw",
            ),
            add_categorical_interactions=False,
        ),
        EngineeredFeatureConfig(
            name="fe_summary_cat_interactions",
            base=FeatureConfig(
                name="fe_base_summary_interactions",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="paper",
                gender_mode="keep",
                weight_mode="category",
                medication_mode="summaries_only",
                utilization_mode="log_plus_raw",
            ),
            add_categorical_interactions=True,
            rare_min_count=150,
        ),
        EngineeredFeatureConfig(
            name="fe_summary_diag_raw",
            base=FeatureConfig(
                name="fe_base_summary_diag_raw",
                rare_min_count=100,
                diagnosis_mode="groups_plus_raw",
                admin_mode="raw_ids",
                age_mode="paper",
                gender_mode="keep",
                weight_mode="category",
                medication_mode="summaries_only",
                utilization_mode="log_plus_raw",
            ),
            add_categorical_interactions=False,
        ),
        EngineeredFeatureConfig(
            name="fe_weight_indicator_core",
            base=FeatureConfig(
                name="fe_base_weight_indicator",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="paper",
                gender_mode="keep",
                weight_mode="indicator",
                medication_mode="summaries_only",
                utilization_mode="log_plus_raw",
            ),
            add_categorical_interactions=False,
        ),
        EngineeredFeatureConfig(
            name="fe_raw_age_weight_core",
            base=FeatureConfig(
                name="fe_base_raw_age",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="raw",
                gender_mode="keep",
                weight_mode="category",
                utilization_mode="log_plus_raw",
            ),
            add_categorical_interactions=False,
        ),
    ]


def all_categorical_columns(X):
    return X.select_dtypes(include=["object", "category"]).columns.tolist()


def prepare_native_frames(X_train, X_eval, min_count):
    rare_cols = all_categorical_columns(X_train)
    grouper = RareCategoryGrouper(columns=rare_cols, min_count=min_count)
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


def make_target_encoder_pipeline(model, X_train, min_count, smooth=20.0):
    categorical_cols = all_categorical_columns(X_train)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_step = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "target_encoder",
                TargetEncoder(
                    target_type="binary",
                    smooth=smooth,
                    cv=5,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    numeric_step = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_step, categorical_cols),
            ("numeric", numeric_step, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("rare_categories", RareCategoryGrouper(columns=categorical_cols, min_count=min_count)),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def make_ohe_pipeline(model, X_train, min_count):
    categorical_cols = all_categorical_columns(X_train)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_step = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_step = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_step, categorical_cols),
            ("numeric", numeric_step, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("rare_categories", RareCategoryGrouper(columns=categorical_cols, min_count=min_count)),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def native_model_specs(pos_weight):
    return [
        {
            "model_name": "FE_NativeCat_d6_lr0.015_l210_SqrtBalanced",
            "model": CatBoostClassifier(
                iterations=1600,
                learning_rate=0.015,
                depth=6,
                l2_leaf_reg=10.0,
                loss_function="Logloss",
                eval_metric="PRAUC",
                auto_class_weights="SqrtBalanced",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
        },
        {
            "model_name": "FE_NativeCat_d6_lr0.015_l210_customPW025",
            "model": CatBoostClassifier(
                iterations=1500,
                learning_rate=0.015,
                depth=6,
                l2_leaf_reg=10.0,
                loss_function="Logloss",
                eval_metric="PRAUC",
                class_weights=[1.0, pos_weight * 0.25],
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
        },
        {
            "model_name": "FE_NativeCat_d5_lr0.018_l28_SqrtBalanced",
            "model": CatBoostClassifier(
                iterations=1300,
                learning_rate=0.018,
                depth=5,
                l2_leaf_reg=8.0,
                loss_function="Logloss",
                eval_metric="PRAUC",
                auto_class_weights="SqrtBalanced",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
        },
    ]


def encoded_model_specs(pos_weight):
    return [
        {
            "model_name": "FE_TargetEnc_XGB_d4_spw050",
            "encoding": "target",
            "smooth": 20.0,
            "model": XGBClassifier(
                n_estimators=900,
                learning_rate=0.012,
                max_depth=4,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
                scale_pos_weight=pos_weight * 0.50,
                max_delta_step=1,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "model_name": "FE_TargetEnc_XGB_d5_spw050",
            "encoding": "target",
            "smooth": 20.0,
            "model": XGBClassifier(
                n_estimators=950,
                learning_rate=0.010,
                max_depth=5,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
                scale_pos_weight=pos_weight * 0.50,
                max_delta_step=1,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "model_name": "FE_TargetEnc_LGBM_l31_spw025",
            "encoding": "target",
            "smooth": 20.0,
            "model": LGBMClassifier(
                objective="binary",
                n_estimators=900,
                learning_rate=0.012,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                scale_pos_weight=pos_weight * 0.25,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        },
        {
            "model_name": "FE_OHE_XGB_d5_spw050",
            "encoding": "onehot",
            "model": XGBClassifier(
                n_estimators=900,
                learning_rate=0.010,
                max_depth=5,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
                scale_pos_weight=pos_weight * 0.50,
                max_delta_step=1,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
    ]


def add_threshold_rows(rows, y_true, score, metadata):
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(metadata)
        row.update({"threshold_strategy": strategy, "status": "ok", "error": ""})
        rows.append(row)


def run():
    scoped = load_scoped_data()
    _, y = build_feature_matrix(scoped, FeatureConfig(name="split_reference"))
    train_idx, val_idx, test_idx, y_train, y_val, y_test = split_indices(y)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    rows = [majority_row(y_train, y_val, "validation")]
    fitted = {}
    native_specs = native_model_specs(pos_weight)
    encoded_specs = encoded_model_specs(pos_weight)

    for cfg in engineered_configs():
        X_cfg, _ = build_engineered_matrix(scoped, cfg)
        X_train = X_cfg.iloc[train_idx].copy()
        X_val = X_cfg.iloc[val_idx].copy()
        X_test = X_cfg.iloc[test_idx].copy()
        n_cat = len(all_categorical_columns(X_train))
        print(f"\n=== Engineered config: {cfg.name} ({X_cfg.shape[1]} columns, {n_cat} categorical) ===")

        for spec in native_specs:
            start = time.perf_counter()
            try:
                X_train_cat, X_val_cat, cat_features = prepare_native_frames(
                    X_train, X_val, min_count=cfg.rare_min_count
                )
                _, X_test_cat, _ = prepare_native_frames(
                    X_train.copy(), X_test, min_count=cfg.rare_min_count
                )
                model = spec["model"].copy()
                model.fit(X_train_cat, y_train, cat_features=cat_features)
                fit_seconds = time.perf_counter() - start
                val_score = model.predict_proba(X_val_cat)[:, 1]
                test_score = model.predict_proba(X_test_cat)[:, 1]
                metadata = {
                    "split": "validation",
                    "experiment_family": "feature_engineering_native_catboost",
                    "feature_config": cfg.name,
                    "model_name": spec["model_name"],
                    "encoding": "native_catboost",
                    "fit_seconds": fit_seconds,
                    "n_columns": int(X_cfg.shape[1]),
                    "n_categorical": int(n_cat),
                    **asdict(cfg),
                }
                add_threshold_rows(rows, y_val, val_score, metadata)
                fitted[(cfg.name, spec["model_name"])] = {
                    "test_score": test_score,
                    "fit_seconds": fit_seconds,
                    "experiment_family": metadata["experiment_family"],
                }
                best = pd.DataFrame(
                    [r for r in rows if r.get("feature_config") == cfg.name and r.get("model_name") == spec["model_name"]]
                ).sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
                print(
                    f"{spec['model_name']}: PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, "
                    f"recall={best.recall:.4f}, precision={best.precision:.4f}, fit={fit_seconds:.1f}s"
                )
            except Exception as exc:
                rows.append(
                    {
                        "split": "validation",
                        "experiment_family": "feature_engineering_native_catboost",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "encoding": "native_catboost",
                        "threshold_strategy": "failed",
                        "fit_seconds": time.perf_counter() - start,
                        "status": "failed",
                        "error": str(exc),
                        **asdict(cfg),
                    }
                )
                print(f"{spec['model_name']}: FAILED: {exc}")

        for spec in encoded_specs:
            start = time.perf_counter()
            try:
                if spec["encoding"] == "target":
                    estimator = make_target_encoder_pipeline(
                        clone(spec["model"]),
                        X_train,
                        min_count=cfg.rare_min_count,
                        smooth=spec.get("smooth", 20.0),
                    )
                else:
                    estimator = make_ohe_pipeline(
                        clone(spec["model"]),
                        X_train,
                        min_count=cfg.rare_min_count,
                    )
                estimator.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - start
                val_score = get_scores(estimator, X_val)
                test_score = get_scores(estimator, X_test)
                metadata = {
                    "split": "validation",
                    "experiment_family": "feature_engineering_encoded_boosting",
                    "feature_config": cfg.name,
                    "model_name": spec["model_name"],
                    "encoding": spec["encoding"],
                    "fit_seconds": fit_seconds,
                    "n_columns": int(X_cfg.shape[1]),
                    "n_categorical": int(n_cat),
                    **asdict(cfg),
                }
                add_threshold_rows(rows, y_val, val_score, metadata)
                fitted[(cfg.name, spec["model_name"])] = {
                    "test_score": test_score,
                    "fit_seconds": fit_seconds,
                    "experiment_family": metadata["experiment_family"],
                }
                best = pd.DataFrame(
                    [r for r in rows if r.get("feature_config") == cfg.name and r.get("model_name") == spec["model_name"]]
                ).sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
                print(
                    f"{spec['model_name']}: PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, "
                    f"recall={best.recall:.4f}, precision={best.precision:.4f}, fit={fit_seconds:.1f}s"
                )
            except Exception as exc:
                rows.append(
                    {
                        "split": "validation",
                        "experiment_family": "feature_engineering_encoded_boosting",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "encoding": spec["encoding"],
                        "threshold_strategy": "failed",
                        "fit_seconds": time.perf_counter() - start,
                        "status": "failed",
                        "error": str(exc),
                        **asdict(cfg),
                    }
                )
                print(f"{spec['model_name']}: FAILED: {exc}")

        pd.DataFrame(rows).to_csv(RESULTS_DIR / "feature_engineering_validation_results.csv", index=False)

    validation = pd.DataFrame(rows)
    validation.to_csv(RESULTS_DIR / "feature_engineering_validation_results.csv", index=False)
    selected = select_validation_candidates(validation, max_rows=30)
    selected.to_csv(RESULTS_DIR / "feature_engineering_selected_for_test.csv", index=False)

    test_rows = [majority_row(y_train, y_test, "test")]
    lift = []
    for _, selected_row in selected.iterrows():
        key = (selected_row["feature_config"], selected_row["model_name"])
        item = fitted[key]
        score = item["test_score"]
        metrics = threshold_metrics(y_test, score, selected_row["threshold"])
        metrics.update(
            {
                "split": "test",
                "experiment_family": item["experiment_family"],
                "feature_config": selected_row["feature_config"],
                "model_name": selected_row["model_name"],
                "threshold_strategy": selected_row["threshold_strategy"],
                "fit_seconds": item["fit_seconds"],
                "status": "ok",
                "error": "",
                "selected_validation_pr_auc": selected_row["pr_auc"],
                "selected_validation_recall": selected_row["recall"],
                "selected_validation_precision": selected_row["precision"],
                "selected_validation_f1": selected_row["f1"],
            }
        )
        test_rows.append(metrics)
        lift.extend(
            lift_rows(
                y_test,
                score,
                {
                    "split": "test",
                    "experiment_family": item["experiment_family"],
                    "feature_config": selected_row["feature_config"],
                    "model_name": selected_row["model_name"],
                    "threshold_strategy": selected_row["threshold_strategy"],
                },
            )
        )

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(RESULTS_DIR / "feature_engineering_test_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "feature_engineering_lift_tables.csv", index=False)

    cols = [
        "experiment_family",
        "feature_config",
        "model_name",
        "threshold_strategy",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
    ]
    ok = validation[(validation["split"] == "validation") & (validation["status"] == "ok")].copy()
    print("\nFeature engineering top validation by PR-AUC:")
    print(ok.sort_values(["pr_auc", "f1"], ascending=False)[cols].head(30).to_string(index=False))
    print("\nFeature engineering top validation by F1:")
    print(ok.sort_values(["f1", "pr_auc"], ascending=False)[cols].head(30).to_string(index=False))
    print("\nFeature engineering selected test results:")
    print(test_results.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp"]].to_string(index=False))


if __name__ == "__main__":
    run()
