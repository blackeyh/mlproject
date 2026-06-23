from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modeling_experiments import (
    DATA_PATH,
    HOSPICE_OR_EXPIRED_DISCHARGE_IDS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_DIR,
    icd9_group,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(RANDOM_STATE)


PAPER_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
]


def load_paper_rows(remove_hospice_expired=False, first_encounter=False):
    df = pd.read_csv(DATA_PATH, na_values="?", keep_default_na=False, low_memory=False)
    df["readmitted_30"] = df["readmitted"].eq("<30").astype(int)

    if first_encounter:
        df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first").copy()
    if remove_hospice_expired:
        df = df[~df["discharge_disposition_id"].isin(HOSPICE_OR_EXPIRED_DISCHARGE_IDS)].copy()

    # The paper drops rows missing race or diagnosis and drops weight, payer_code,
    # and medical_specialty because of high missingness.
    required = ["race", "diag_1", "diag_2", "diag_3"]
    df = df.dropna(subset=required).copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].apply(icd9_group)

    for col in [
        "race",
        "gender",
        "age",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
        "max_glu_serum",
        "A1Cresult",
        "insulin",
        "change",
        "diabetesMed",
    ]:
        df[col] = df[col].fillna("Missing").astype(str)

    X = df[PAPER_FEATURES].copy()
    y = df["readmitted_30"].astype(int).copy()
    groups = df["patient_nbr"].copy()
    return X, y, groups


def split_indices(y, groups, mode):
    idx = np.arange(len(y))
    if mode == "paper_random_75_25":
        train_idx, test_idx = train_test_split(
            idx,
            test_size=0.25,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        return train_idx, test_idx

    if mode == "patient_group_75_25":
        # GroupShuffleSplit is patient-safe but cannot stratify exactly.
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(idx, y, groups=groups))
        return train_idx, test_idx

    raise ValueError(f"Unknown split mode: {mode}")


def best_threshold_metrics(y_true, score):
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    if len(thresholds) == 0:
        threshold = 0.5
    else:
        p = precision[:-1]
        r = recall[:-1]
        f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
        threshold = float(thresholds[int(np.nanargmax(f1))])
    return metrics_at_threshold(y_true, score, threshold, "best_f1")


def metrics_at_threshold(y_true, score, threshold, threshold_strategy):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold_strategy": threshold_strategy,
        "threshold": float(threshold),
        "pr_auc": average_precision_score(y_true, score),
        "roc_auc": roc_auc_score(y_true, score),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise TypeError("Model does not expose predict_proba.")


def run_one(dataset_name, X, y, groups, split_mode):
    train_idx, test_idx = split_indices(y, groups, split_mode)
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    preprocessor = make_preprocessor(X_train)
    models = [
        (
            "PaperRF_250_depth5_unweighted",
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=250,
                            max_depth=5,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "PaperRF_250_depth5_balanced",
            Pipeline(
                steps=[
                    ("preprocess", make_preprocessor(X_train)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=250,
                            max_depth=5,
                            class_weight="balanced_subsample",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "PaperRF_250_depth8_balanced",
            Pipeline(
                steps=[
                    ("preprocess", make_preprocessor(X_train)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=250,
                            max_depth=8,
                            min_samples_leaf=20,
                            class_weight="balanced_subsample",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "PaperGaussianNB",
            Pipeline(
                steps=[
                    ("preprocess", make_preprocessor(X_train)),
                    ("model", GaussianNB()),
                ]
            ),
        ),
    ]

    rows = []
    split_info = {
        "dataset_name": dataset_name,
        "split_mode": split_mode,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "train_patients": int(groups.iloc[train_idx].nunique()),
        "test_patients": int(groups.iloc[test_idx].nunique()),
        "patient_overlap": int(
            len(set(groups.iloc[train_idx].unique()).intersection(set(groups.iloc[test_idx].unique())))
        ),
    }

    for model_name, model in models:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        score = get_score(model, X_test)
        for threshold_strategy, threshold in [
            ("default_0.5", 0.5),
            ("best_f1", None),
        ]:
            if threshold is None:
                row = best_threshold_metrics(y_test, score)
            else:
                row = metrics_at_threshold(y_test, score, threshold, threshold_strategy)
            row.update(split_info)
            row.update(
                {
                    "model_name": model_name,
                    "fit_seconds": fit_seconds,
                    "status": "ok",
                }
            )
            rows.append(row)
        print(
            f"{dataset_name} | {split_mode} | {model_name}: "
            f"PR-AUC={average_precision_score(y_test, score):.4f}, "
            f"ROC-AUC={roc_auc_score(y_test, score):.4f}, fit={fit_seconds:.1f}s",
            flush=True,
        )
    return rows


def run():
    experiments = [
        ("paper_all_rows", False, False, "paper_random_75_25"),
        ("paper_all_rows_patient_group", False, False, "patient_group_75_25"),
        ("first_encounter_hospice_removed", True, True, "paper_random_75_25"),
        ("first_encounter_hospice_removed_patient_group", True, True, "patient_group_75_25"),
    ]
    rows = []
    for dataset_name, remove_hospice, first_encounter, split_mode in experiments:
        X, y, groups = load_paper_rows(
            remove_hospice_expired=remove_hospice,
            first_encounter=first_encounter,
        )
        print(
            f"\n=== {dataset_name}: rows={len(y)}, positive_rate={y.mean():.4f}, "
            f"patients={groups.nunique()} ===",
            flush=True,
        )
        rows.extend(run_one(dataset_name, X, y, groups, split_mode))

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "paper_reproduction_results.csv", index=False)
    cols = [
        "dataset_name",
        "split_mode",
        "model_name",
        "threshold_strategy",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
        "train_rows",
        "test_rows",
        "patient_overlap",
    ]
    print("\nPaper reproduction results sorted by PR-AUC:")
    print(results.sort_values(["pr_auc", "f1"], ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    run()
