from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "archive" / "diabetic_data.csv"
RESULTS_DIR = PROJECT_ROOT / "experiment_results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
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

RARE_OR_ZERO_VARIANCE_MEDICATION_COLS = [
    "chlorpropamide",
    "acetohexamide",
    "tolbutamide",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    rare_min_count: int = 200
    diagnosis_mode: str = "groups_only"
    admin_mode: str = "paper"
    age_mode: str = "paper"
    gender_mode: str = "drop"
    weight_mode: str = "drop"
    medication_mode: str = "drop_rare"
    utilization_mode: str = "raw_plus_sum"
    payer_specialty_mode: str = "keep"


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, min_count=200, missing_label="Missing", other_label="Other"):
        self.columns = columns
        self.min_count = min_count
        self.missing_label = missing_label
        self.other_label = other_label

    def fit(self, X, y=None):
        X = X.copy()
        self.frequent_categories_ = {}
        for col in self.columns:
            values = X[col].fillna(self.missing_label).astype(str)
            counts = values.value_counts(dropna=False)
            keep = set(counts[counts >= self.min_count].index)
            keep.add(self.missing_label)
            self.frequent_categories_[col] = keep
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            values = X[col].fillna(self.missing_label).astype(str)
            keep = self.frequent_categories_[col]
            X[col] = np.where(values.isin(keep), values, self.other_label)
        return X


def icd9_group(code):
    if pd.isna(code):
        return "Missing"
    text = str(code).strip()
    if text == "":
        return "Missing"
    if text.startswith("V") or text.startswith("E"):
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


def group_admission_type_paper(x):
    if x in [1, 2, 7]:
        return "Emergency/Urgent/Trauma"
    if x == 3:
        return "Elective"
    if x in [5, 6, 8]:
        return "Unknown/Not available"
    return "Other/Newborn"


def group_discharge_paper(x):
    if x == 1:
        return "Home"
    return "Other"


def group_admission_source_paper(x):
    if x == 7:
        return "Emergency room"
    if x in [1, 2, 3]:
        return "Physician/clinic referral"
    return "Other"


def group_admission_type_detailed(x):
    if x in [1, 2]:
        return "Emergency/Urgent"
    if x == 3:
        return "Elective"
    if x == 4:
        return "Newborn"
    if x == 7:
        return "Trauma Center"
    if x in [5, 6, 8]:
        return "Unknown/Not mapped"
    return "Other"


def group_discharge_detailed(x):
    if x == 1:
        return "Home"
    if x == 6:
        return "Home health"
    if x == 7:
        return "Left AMA"
    if x in [11, 19, 20, 21]:
        return "Expired"
    if x in [13, 14]:
        return "Hospice"
    if x in [2, 3, 4, 5, 22, 23, 24, 27, 28, 29, 30]:
        return "Transfer/Facility"
    if x in [18, 25, 26]:
        return "Unknown/Not mapped"
    return "Other"


def group_admission_source_detailed(x):
    if x in [1, 2, 3]:
        return "Referral"
    if x == 7:
        return "Emergency Room"
    if x in [4, 5, 6, 10, 18, 22, 25, 26]:
        return "Transfer"
    if x in [11, 12, 13, 14, 23, 24]:
        return "Birth-related"
    if x in [9, 15, 17, 20, 21]:
        return "Unknown/Not mapped"
    if x == 8:
        return "Court/Law enforcement"
    return "Other"


def group_age_paper(age_value):
    if age_value in ["[0-10)", "[10-20)", "[20-30)"]:
        return "<=30"
    if age_value in ["[30-40)", "[40-50)", "[50-60)"]:
        return "30-60"
    return ">60"


def load_scoped_data():
    raw = pd.read_csv(DATA_PATH, na_values="?", keep_default_na=False, low_memory=False)
    df = raw.copy()
    df["readmitted_30"] = df["readmitted"].eq("<30").astype(int)
    first = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first").copy()
    scoped = first[~first["discharge_disposition_id"].isin(HOSPICE_OR_EXPIRED_DISCHARGE_IDS)].copy()
    return scoped


def build_feature_matrix(scoped_df, cfg: FeatureConfig):
    y = scoped_df["readmitted_30"].astype(int).copy()
    X = scoped_df.drop(columns=ID_AND_TARGET_COLUMNS).copy()

    for col in ["race", "medical_specialty", "payer_code"]:
        X[col] = X[col].fillna("Missing").astype(str)

    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        X[diag_col] = X[diag_col].fillna("Missing").astype(str)
        X[f"{diag_col}_group"] = X[diag_col].apply(icd9_group)

    if cfg.admin_mode == "paper":
        X["admission_type_group"] = X["admission_type_id"].apply(group_admission_type_paper)
        X["discharge_disposition_group"] = X["discharge_disposition_id"].apply(group_discharge_paper)
        X["admission_source_group"] = X["admission_source_id"].apply(group_admission_source_paper)
    elif cfg.admin_mode == "detailed":
        X["admission_type_group"] = X["admission_type_id"].apply(group_admission_type_detailed)
        X["discharge_disposition_group"] = X["discharge_disposition_id"].apply(group_discharge_detailed)
        X["admission_source_group"] = X["admission_source_id"].apply(group_admission_source_detailed)
    elif cfg.admin_mode == "raw_ids":
        X["admission_type_raw"] = "admission_type_" + X["admission_type_id"].astype(str)
        X["discharge_disposition_raw"] = "discharge_" + X["discharge_disposition_id"].astype(str)
        X["admission_source_raw"] = "source_" + X["admission_source_id"].astype(str)
    else:
        raise ValueError(f"Unknown admin_mode={cfg.admin_mode}")

    if cfg.age_mode == "paper":
        X["age_group_paper"] = X["age"].apply(group_age_paper)
    elif cfg.age_mode == "raw":
        X["age_raw"] = X["age"].astype(str)
    else:
        raise ValueError(f"Unknown age_mode={cfg.age_mode}")

    X["num_diabetes_meds_used"] = X[MEDICATION_COLS].ne("No").sum(axis=1)
    X["num_diabetes_med_changes"] = X[MEDICATION_COLS].isin(["Up", "Down"]).sum(axis=1)
    X["service_utilization"] = (
        X["number_outpatient"] + X["number_emergency"] + X["number_inpatient"]
    )

    if cfg.utilization_mode in ["log_plus_raw", "log_and_bucket"]:
        for col in ["number_outpatient", "number_emergency", "number_inpatient", "service_utilization"]:
            X[f"log1p_{col}"] = np.log1p(X[col])

    if cfg.utilization_mode == "log_and_bucket":
        X["prior_inpatient_bucket"] = pd.cut(
            X["number_inpatient"],
            bins=[-1, 0, 1, 2, 999],
            labels=["0", "1", "2", "3+"],
        ).astype(str)
        X["prior_emergency_bucket"] = pd.cut(
            X["number_emergency"],
            bins=[-1, 0, 1, 2, 999],
            labels=["0", "1", "2", "3+"],
        ).astype(str)
        X["service_utilization_bucket"] = pd.cut(
            X["service_utilization"],
            bins=[-1, 0, 1, 2, 4, 999],
            labels=["0", "1", "2", "3-4", "5+"],
        ).astype(str)

    if cfg.weight_mode == "indicator":
        X["weight_recorded"] = X["weight"].notna().astype(int)
    elif cfg.weight_mode == "category":
        X["weight_category"] = X["weight"].fillna("Missing").astype(str)
    elif cfg.weight_mode != "drop":
        raise ValueError(f"Unknown weight_mode={cfg.weight_mode}")

    drop_cols = [
        "weight",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "age",
    ]

    if cfg.diagnosis_mode == "groups_only":
        drop_cols += ["diag_1", "diag_2", "diag_3"]
    elif cfg.diagnosis_mode == "groups_plus_raw":
        pass
    elif cfg.diagnosis_mode == "raw_only":
        drop_cols += ["diag_1_group", "diag_2_group", "diag_3_group"]
    else:
        raise ValueError(f"Unknown diagnosis_mode={cfg.diagnosis_mode}")

    if cfg.gender_mode == "drop":
        drop_cols.append("gender")
    elif cfg.gender_mode == "keep":
        X["gender"] = X["gender"].fillna("Missing").astype(str)
    else:
        raise ValueError(f"Unknown gender_mode={cfg.gender_mode}")

    if cfg.medication_mode == "drop_rare":
        drop_cols += RARE_OR_ZERO_VARIANCE_MEDICATION_COLS
    elif cfg.medication_mode == "summaries_only":
        drop_cols += MEDICATION_COLS
    elif cfg.medication_mode == "keep_all":
        pass
    else:
        raise ValueError(f"Unknown medication_mode={cfg.medication_mode}")

    if cfg.payer_specialty_mode == "drop":
        drop_cols += ["payer_code", "medical_specialty"]
    elif cfg.payer_specialty_mode != "keep":
        raise ValueError(f"Unknown payer_specialty_mode={cfg.payer_specialty_mode}")

    X = X.drop(columns=drop_cols, errors="ignore")
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    X[categorical_cols] = X[categorical_cols].fillna("Missing").astype(str)
    return X, y


def make_preprocessor(categorical_cols, numeric_cols, scale_numeric=False):
    numeric_step = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if scale_numeric else "passthrough"),
        ]
    )
    categorical_step = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_step, categorical_cols),
            ("numeric", numeric_step, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model, X_train, rare_cols, rare_min_count, scale_numeric=False):
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    return Pipeline(
        steps=[
            (
                "rare_categories",
                RareCategoryGrouper(columns=rare_cols, min_count=rare_min_count),
            ),
            (
                "preprocess",
                make_preprocessor(categorical_cols, numeric_cols, scale_numeric=scale_numeric),
            ),
            ("model", model),
        ]
    )


def get_scores(estimator, X_eval):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_eval)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw_scores = estimator.decision_function(X_eval)
        return 1 / (1 + np.exp(-raw_scores))
    raise TypeError("Estimator does not expose predict_proba or decision_function.")


def threshold_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def best_thresholds(y_true, y_score):
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


def model_specs(pos_weight):
    specs = []
    specs.extend(
        [
            {
                "model_name": "LogisticRegression_C1_balanced",
                "model": LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced",
                    C=1.0,
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": True,
                "sample_weight": False,
            },
            {
                "model_name": "LogisticRegression_C0.2_balanced",
                "model": LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced",
                    C=0.2,
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": True,
                "sample_weight": False,
            },
            {
                "model_name": "DecisionTree_depth8_balanced",
                "model": DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_leaf=100,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": False,
                "sample_weight": False,
            },
            {
                "model_name": "RandomForest_depth14_leaf50",
                "model": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=14,
                    min_samples_leaf=50,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": False,
                "sample_weight": False,
            },
            {
                "model_name": "RandomForest_depth20_leaf20",
                "model": RandomForestClassifier(
                    n_estimators=250,
                    max_depth=20,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": False,
                "sample_weight": False,
            },
            {
                "model_name": "ExtraTrees_depth20_leaf20",
                "model": ExtraTreesClassifier(
                    n_estimators=250,
                    max_depth=20,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": False,
                "sample_weight": False,
            },
            {
                "model_name": "GradientBoosting_weighted",
                "model": GradientBoostingClassifier(
                    n_estimators=160,
                    learning_rate=0.04,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": False,
                "sample_weight": True,
            },
            {
                "model_name": "HistGradientBoosting_balanced",
                "model": HistGradientBoostingClassifier(
                    max_iter=220,
                    learning_rate=0.04,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": False,
                "sample_weight": False,
            },
            {
                "model_name": "HistGradientBoosting_deeper",
                "model": HistGradientBoostingClassifier(
                    max_iter=260,
                    learning_rate=0.035,
                    max_leaf_nodes=63,
                    min_samples_leaf=35,
                    l2_regularization=0.05,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": False,
                "sample_weight": False,
            },
            {
                "model_name": "AdaBoost_weighted",
                "model": AdaBoostClassifier(
                    n_estimators=180,
                    learning_rate=0.04,
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": False,
                "sample_weight": True,
            },
            {
                "model_name": "GaussianNB_weighted",
                "model": GaussianNB(),
                "scale_numeric": True,
                "sample_weight": True,
            },
        ]
    )

    if LGBMClassifier is not None:
        specs.extend(
            [
                {
                    "model_name": "LightGBM_balanced_depth3",
                    "model": LGBMClassifier(
                        objective="binary",
                        n_estimators=500,
                        learning_rate=0.025,
                        num_leaves=15,
                        max_depth=3,
                        min_child_samples=80,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    "scale_numeric": False,
                    "sample_weight": False,
                },
                {
                    "model_name": "LightGBM_balanced_leaves31",
                    "model": LGBMClassifier(
                        objective="binary",
                        n_estimators=600,
                        learning_rate=0.02,
                        num_leaves=31,
                        min_child_samples=60,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    "scale_numeric": False,
                    "sample_weight": False,
                },
            ]
        )

    if XGBClassifier is not None:
        specs.extend(
            [
                {
                    "model_name": "XGBoost_depth3_aucpr",
                    "model": XGBClassifier(
                        n_estimators=450,
                        learning_rate=0.025,
                        max_depth=3,
                        min_child_weight=8,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        scale_pos_weight=pos_weight,
                        eval_metric="aucpr",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    "scale_numeric": False,
                    "sample_weight": False,
                },
                {
                    "model_name": "XGBoost_depth4_aucpr",
                    "model": XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.02,
                        max_depth=4,
                        min_child_weight=10,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=3.0,
                        scale_pos_weight=pos_weight,
                        eval_metric="aucpr",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    "scale_numeric": False,
                    "sample_weight": False,
                },
            ]
        )

    if CatBoostClassifier is not None:
        specs.append(
            {
                "model_name": "CatBoost_ohe_balanced",
                "model": CatBoostClassifier(
                    iterations=450,
                    learning_rate=0.035,
                    depth=5,
                    l2_leaf_reg=5.0,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    auto_class_weights="Balanced",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                ),
                "scale_numeric": False,
                "sample_weight": False,
            }
        )
    return specs


def feature_configs():
    return [
        FeatureConfig(name="v1_accepted", rare_min_count=200),
        FeatureConfig(
            name="accepted_rare100",
            rare_min_count=100,
        ),
        FeatureConfig(
            name="accepted_rare500",
            rare_min_count=500,
        ),
        FeatureConfig(
            name="detailed_admin_raw_age_gender_weight_log",
            rare_min_count=100,
            admin_mode="detailed",
            age_mode="raw",
            gender_mode="keep",
            weight_mode="indicator",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="raw_admin_raw_age_weight_category",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="raw",
            gender_mode="keep",
            weight_mode="category",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="diag_groups_plus_raw_rare100",
            rare_min_count=100,
            diagnosis_mode="groups_plus_raw",
        ),
        FeatureConfig(
            name="diag_raw_only_rare100",
            rare_min_count=100,
            diagnosis_mode="raw_only",
        ),
        FeatureConfig(
            name="keep_all_meds_log_buckets",
            rare_min_count=100,
            medication_mode="keep_all",
            utilization_mode="log_and_bucket",
        ),
        FeatureConfig(
            name="med_summaries_only",
            rare_min_count=200,
            medication_mode="summaries_only",
        ),
        FeatureConfig(
            name="drop_payer_specialty",
            rare_min_count=200,
            payer_specialty_mode="drop",
        ),
    ]


def rare_columns_for(X):
    rare_cols = []
    for col in ["medical_specialty", "payer_code", "diag_1", "diag_2", "diag_3"]:
        if col in X.columns:
            rare_cols.append(col)
    return rare_cols


def evaluate_dummy(y_train, y_eval, split_name):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = dummy.predict(np.zeros((len(y_eval), 1)))
    y_score = np.full(len(y_eval), y_train.mean())
    metrics = threshold_metrics(y_eval, y_score, 0.5)
    metrics.update(
        {
            "split": split_name,
            "feature_config": "baseline_no_features",
            "model_name": "MajorityBaseline",
            "threshold_strategy": "most_frequent",
            "fit_seconds": 0.0,
            "status": "ok",
            "error": "",
        }
    )
    return metrics


def run():
    scoped = load_scoped_data()
    all_rows = []
    fitted = {}
    failures = []

    X_reference, y = build_feature_matrix(scoped, FeatureConfig(name="split_reference"))
    X_train_val_idx, X_test_idx, y_train_val, y_test = train_test_split(
        np.arange(len(y)),
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    val_relative = 0.15 / 0.85
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        X_train_val_idx,
        y_train_val,
        test_size=val_relative,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )
    del X_reference

    split_summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(y_train), "positive": int(y_train.sum()), "positive_rate": float(y_train.mean())},
            {"split": "validation", "rows": len(y_val), "positive": int(y_val.sum()), "positive_rate": float(y_val.mean())},
            {"split": "test", "rows": len(y_test), "positive": int(y_test.sum()), "positive_rate": float(y_test.mean())},
        ]
    )
    split_summary.to_csv(RESULTS_DIR / "split_summary.csv", index=False)

    all_rows.append(evaluate_dummy(y_train, y_val, "validation"))

    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    specs = model_specs(pos_weight=pos_weight)

    for cfg in feature_configs():
        X_cfg, y_cfg = build_feature_matrix(scoped, cfg)
        X_train = X_cfg.iloc[X_train_idx].copy()
        X_val = X_cfg.iloc[X_val_idx].copy()
        rare_cols = rare_columns_for(X_train)
        sample_weight_balanced = compute_sample_weight(class_weight="balanced", y=y_train)

        config_path = RESULTS_DIR / f"columns_{cfg.name}.json"
        config_path.write_text(
            json.dumps(
                {
                    "feature_config": asdict(cfg),
                    "columns": X_cfg.columns.tolist(),
                    "rare_columns": rare_cols,
                    "n_columns_before_encoding": int(X_cfg.shape[1]),
                },
                indent=2,
            )
        )

        print(f"\\n=== Feature config: {cfg.name} ({X_cfg.shape[1]} columns before encoding) ===")
        for spec in specs:
            model_name = spec["model_name"]
            estimator = make_pipeline(
                model=clone(spec["model"]),
                X_train=X_train,
                rare_cols=rare_cols,
                rare_min_count=cfg.rare_min_count,
                scale_numeric=spec["scale_numeric"],
            )
            start = time.perf_counter()
            try:
                if spec["sample_weight"]:
                    estimator.fit(X_train, y_train, model__sample_weight=sample_weight_balanced)
                else:
                    estimator.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - start
                y_score = get_scores(estimator, X_val)
                threshold_choices = best_thresholds(y_val, y_score)

                for strategy, threshold in threshold_choices.items():
                    row = threshold_metrics(y_val, y_score, threshold)
                    row.update(
                        {
                            "split": "validation",
                            "feature_config": cfg.name,
                            "model_name": model_name,
                            "threshold_strategy": strategy,
                            "fit_seconds": fit_seconds,
                            "status": "ok",
                            "error": "",
                            "n_columns_before_encoding": int(X_cfg.shape[1]),
                            "rare_min_count": cfg.rare_min_count,
                            "diagnosis_mode": cfg.diagnosis_mode,
                            "admin_mode": cfg.admin_mode,
                            "age_mode": cfg.age_mode,
                            "gender_mode": cfg.gender_mode,
                            "weight_mode": cfg.weight_mode,
                            "medication_mode": cfg.medication_mode,
                            "utilization_mode": cfg.utilization_mode,
                            "payer_specialty_mode": cfg.payer_specialty_mode,
                        }
                    )
                    all_rows.append(row)

                best_f1 = max(
                    (r for r in all_rows if r.get("feature_config") == cfg.name and r.get("model_name") == model_name),
                    key=lambda r: r["f1"],
                )
                print(
                    f"{model_name}: PR-AUC={best_f1['pr_auc']:.4f}, "
                    f"best F1={best_f1['f1']:.4f}, recall={best_f1['recall']:.4f}, "
                    f"precision={best_f1['precision']:.4f}, fit={fit_seconds:.1f}s"
                )
                fitted[(cfg.name, model_name)] = {
                    "estimator": estimator,
                    "config": cfg,
                    "threshold_choices": threshold_choices,
                }
            except Exception as exc:
                fit_seconds = time.perf_counter() - start
                err = str(exc)
                failures.append({"feature_config": cfg.name, "model_name": model_name, "error": err})
                all_rows.append(
                    {
                        "split": "validation",
                        "feature_config": cfg.name,
                        "model_name": model_name,
                        "threshold_strategy": "failed",
                        "threshold": np.nan,
                        "pr_auc": np.nan,
                        "roc_auc": np.nan,
                        "recall": np.nan,
                        "precision": np.nan,
                        "f1": np.nan,
                        "accuracy": np.nan,
                        "tn": np.nan,
                        "fp": np.nan,
                        "fn": np.nan,
                        "tp": np.nan,
                        "fit_seconds": fit_seconds,
                        "status": "failed",
                        "error": err,
                        "n_columns_before_encoding": int(X_cfg.shape[1]),
                        **asdict(cfg),
                    }
                )
                print(f"{model_name}: FAILED: {err}")

        pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "validation_experiment_results.csv", index=False)

    results = pd.DataFrame(all_rows)
    results.to_csv(RESULTS_DIR / "validation_experiment_results.csv", index=False)
    pd.DataFrame(failures).to_csv(RESULTS_DIR / "experiment_failures.csv", index=False)

    ok = results[(results["split"] == "validation") & (results["status"] == "ok")].copy()
    non_baseline = ok[ok["model_name"] != "MajorityBaseline"].copy()

    selection_frames = [
        non_baseline.sort_values(["pr_auc", "f1"], ascending=False).head(5),
        non_baseline[non_baseline["threshold_strategy"] == "best_f1"].sort_values(["f1", "pr_auc"], ascending=False).head(5),
        non_baseline[non_baseline["threshold_strategy"] == "best_f2"].sort_values(["recall", "precision"], ascending=False).head(5),
        non_baseline[non_baseline["threshold_strategy"] == "max_recall_precision_ge_0.15"].sort_values(
            ["recall", "f1"], ascending=False
        ).head(5),
    ]
    selected = pd.concat(selection_frames, ignore_index=True)
    selected = selected.drop_duplicates(["feature_config", "model_name", "threshold_strategy"]).head(12)
    selected.to_csv(RESULTS_DIR / "selected_for_test_from_validation.csv", index=False)

    test_rows = [evaluate_dummy(y_train, y_test, "test")]
    for _, row in selected.iterrows():
        cfg_name = row["feature_config"]
        model_name = row["model_name"]
        key = (cfg_name, model_name)
        if key not in fitted:
            continue
        cfg = fitted[key]["config"]
        X_cfg, _ = build_feature_matrix(scoped, cfg)
        X_test = X_cfg.iloc[X_test_idx].copy()
        estimator = fitted[key]["estimator"]
        y_score = get_scores(estimator, X_test)
        metrics = threshold_metrics(y_test, y_score, row["threshold"])
        metrics.update(
            {
                "split": "test",
                "feature_config": cfg_name,
                "model_name": model_name,
                "threshold_strategy": row["threshold_strategy"],
                "fit_seconds": row["fit_seconds"],
                "status": "ok",
                "error": "",
                "selected_validation_pr_auc": row["pr_auc"],
                "selected_validation_recall": row["recall"],
                "selected_validation_precision": row["precision"],
                "selected_validation_f1": row["f1"],
                "n_columns_before_encoding": row["n_columns_before_encoding"],
                "rare_min_count": row["rare_min_count"],
                "diagnosis_mode": row["diagnosis_mode"],
                "admin_mode": row["admin_mode"],
                "age_mode": row["age_mode"],
                "gender_mode": row["gender_mode"],
                "weight_mode": row["weight_mode"],
                "medication_mode": row["medication_mode"],
                "utilization_mode": row["utilization_mode"],
                "payer_specialty_mode": row["payer_specialty_mode"],
            }
        )
        test_rows.append(metrics)

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(RESULTS_DIR / "test_results_selected_models.csv", index=False)

    top_validation = ok.sort_values(["pr_auc", "f1"], ascending=False).head(30)
    top_f1 = ok.sort_values(["f1", "pr_auc"], ascending=False).head(30)
    print("\\nTop validation rows by PR-AUC:")
    print(top_validation[["feature_config", "model_name", "threshold_strategy", "pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy"]].to_string(index=False))
    print("\\nTop validation rows by F1:")
    print(top_f1[["feature_config", "model_name", "threshold_strategy", "pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy"]].to_string(index=False))
    print("\\nSelected test results:")
    print(test_results.sort_values(["pr_auc", "f1"], ascending=False)[["feature_config", "model_name", "threshold_strategy", "pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy", "tn", "fp", "fn", "tp"]].to_string(index=False))


if __name__ == "__main__":
    run()
