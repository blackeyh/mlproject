from __future__ import annotations

import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from all_encounters_group_split_search import load_all_eligible_encounters
from feature_engineering_search import (
    EngineeredFeatureConfig,
    build_engineered_matrix,
    prepare_native_frames,
)
from imbalance_experiments import lift_rows
from modeling_experiments import (
    RANDOM_STATE,
    RESULTS_DIR,
    FeatureConfig,
    best_thresholds,
    threshold_metrics,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.random.seed(RANDOM_STATE)


def patient_group_split(scoped: pd.DataFrame, seed: int = RANDOM_STATE):
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


def random_row_split(y: pd.Series, seed: int = RANDOM_STATE):
    train_val_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.15,
        stratify=y,
        random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15 / 0.85,
        stratify=y.iloc[train_val_idx],
        random_state=seed,
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def base_config(name: str, weight_mode: str = "category", diagnosis_mode: str = "groups_only"):
    return EngineeredFeatureConfig(
        name=name,
        base=FeatureConfig(
            name=f"{name}_base",
            rare_min_count=100,
            diagnosis_mode=diagnosis_mode,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode=weight_mode,
            medication_mode="summaries_only",
            utilization_mode="log_plus_raw",
        ),
        rare_min_count=100,
    )


def diagnostic_configs():
    full_summary = base_config("full_summary")
    full_indicator = base_config("full_indicator", weight_mode="indicator")
    return [
        replace(
            full_summary,
            name="base_no_extra_engineering",
            add_diagnosis_detail=False,
            add_elixhauser_flags=False,
            add_medication_detail=False,
            add_utilization_interactions=False,
            add_lab_interactions=False,
            add_admin_risk_flags=False,
            add_categorical_interactions=False,
        ),
        full_summary,
        full_indicator,
        replace(full_summary, name="full_plus_categorical_interactions", add_categorical_interactions=True),
        base_config("full_plus_raw_diag_codes", diagnosis_mode="groups_plus_raw"),
        replace(full_summary, name="no_diagnosis_detail", add_diagnosis_detail=False, add_elixhauser_flags=False),
        replace(full_summary, name="no_medication_detail", add_medication_detail=False),
        replace(full_summary, name="no_lab_interactions", add_lab_interactions=False),
    ]


DROP_GROUP_PATTERNS = {
    "drop_prior_utilization": [
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "service_utilization",
        "prior_",
        "utilization",
        "frequent_acute",
        "frequent_inpatient",
        "has_prior_",
    ],
    "drop_admin_discharge_source": [
        "admission_type",
        "discharge",
        "admission_source",
        "emergency_or_urgent_admission",
        "elective_admission",
        "emergency_room_source",
        "transfer_source",
        "home_health",
        "left_ama",
    ],
    "drop_diagnosis_features": ["diag_", "cm_", "comorbidity"],
    "drop_medication_features": ["med_", "insulin", "diabetesMed", "change", "num_diabetes_meds", "num_diabetes_med"],
    "drop_lab_features": ["A1C", "a1c", "max_glu", "glucose"],
}


def feature_group_views(X: pd.DataFrame):
    views = {"all_features": X}
    for name, patterns in DROP_GROUP_PATTERNS.items():
        drop_cols = [
            col
            for col in X.columns
            if any(pattern.lower() in col.lower() for pattern in patterns)
        ]
        keep = X.drop(columns=drop_cols, errors="ignore")
        views[name] = keep
    views["admin_util_only"] = X[
        [
            col
            for col in X.columns
            if any(
                pattern.lower() in col.lower()
                for pattern in [
                    "admission",
                    "discharge",
                    "source",
                    "time_in_hospital",
                    "number_outpatient",
                    "number_emergency",
                    "number_inpatient",
                    "service_utilization",
                    "prior_",
                    "age",
                    "race",
                    "gender",
                    "payer",
                    "specialty",
                    "weight",
                ]
            )
        ]
    ]
    return views


def make_model(pos_weight: float, seed: int = RANDOM_STATE, iterations: int = 1200):
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=10.0,
        random_strength=1.0,
        loss_function="Logloss",
        eval_metric="PRAUC",
        class_weights=[1.0, pos_weight * 0.25],
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=100,
    )


def add_eval_rows(rows, y_true, score, metadata):
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(metadata)
        row["threshold_strategy"] = strategy
        rows.append(row)


def fit_eval_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx,
    val_idx,
    test_idx,
    *,
    config_name: str,
    experiment_type: str,
    split_name: str,
    seed: int = RANDOM_STATE,
    iterations: int = 1200,
):
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    X_train_raw = X.iloc[train_idx].copy()
    X_val_raw = X.iloc[val_idx].copy()
    X_test_raw = X.iloc[test_idx].copy()
    X_train, X_val, cat_features = prepare_native_frames(X_train_raw, X_val_raw, min_count=100)
    _, X_test, _ = prepare_native_frames(X_train_raw.copy(), X_test_raw, min_count=100)

    model = make_model(pos_weight, seed=seed, iterations=iterations)
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    fit_seconds = time.perf_counter() - start

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    rows = []
    common = {
        "experiment_type": experiment_type,
        "split_name": split_name,
        "feature_config": config_name,
        "model_name": "PlateauCatBoost_d6_lr002_custom025",
        "seed": seed,
        "n_columns": X.shape[1],
        "best_iteration": int(model.get_best_iteration() or iterations),
        "fit_seconds": fit_seconds,
    }
    add_eval_rows(rows, y_val, val_score, {**common, "split": "validation"})
    add_eval_rows(rows, y_test, test_score, {**common, "split": "test"})

    lift = lift_rows(
        y_test,
        test_score,
        {**common, "split": "test", "threshold_strategy": "ranking"},
    )
    return rows, lift, val_score, test_score


def summarize_dataset_signal(scoped: pd.DataFrame):
    rows = []
    y = scoped["readmitted_30"].astype(int)
    for col in [
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
        "time_in_hospital",
        "num_medications",
        "num_lab_procedures",
        "num_procedures",
        "number_diagnoses",
    ]:
        score = scoped[col].astype(float)
        rows.append(
            {
                "feature": col,
                "positive_rate_lowest_quartile": float(y[score <= score.quantile(0.25)].mean()),
                "positive_rate_highest_quartile": float(y[score >= score.quantile(0.75)].mean()),
                "average_precision_single_feature": float(average_precision_score(y, score)),
                "roc_auc_single_feature": float(roc_auc_score(y, score)),
            }
        )
    for col in ["discharge_disposition_id", "admission_source_id", "admission_type_id", "age"]:
        rates = (
            scoped.assign(target=y)
            .groupby(col)["target"]
            .agg(["count", "mean"])
            .sort_values("mean", ascending=False)
            .head(8)
        )
        for value, row in rates.iterrows():
            rows.append(
                {
                    "feature": f"{col}={value}",
                    "positive_rate_lowest_quartile": np.nan,
                    "positive_rate_highest_quartile": float(row["mean"]),
                    "average_precision_single_feature": np.nan,
                    "roc_auc_single_feature": np.nan,
                    "count": int(row["count"]),
                }
            )
    return pd.DataFrame(rows)


def run():
    scoped = load_all_eligible_encounters()
    y = scoped["readmitted_30"].astype(int)
    patient_train, patient_val, patient_test = patient_group_split(scoped, RANDOM_STATE)
    random_train, random_val, random_test = random_row_split(y, RANDOM_STATE)

    all_rows = []
    lift_all = []

    print("Plateau diagnostic search", flush=True)
    print(f"All eligible rows: {len(scoped):,}; positive rate: {y.mean():.4f}", flush=True)

    # 1) Feature engineering variants on the patient-safe split.
    for cfg in diagnostic_configs():
        X, _ = build_engineered_matrix(scoped, cfg)
        rows, lift, _, _ = fit_eval_catboost(
            X,
            y,
            patient_train,
            patient_val,
            patient_test,
            config_name=cfg.name,
            experiment_type="feature_engineering_variant",
            split_name="patient_group_70_15_15",
        )
        all_rows.extend(rows)
        lift_all.extend(lift)
        best_test = (
            pd.DataFrame(rows)
            .query("split == 'test'")
            .sort_values(["pr_auc", "f1"], ascending=False)
            .iloc[0]
        )
        print(
            f"{cfg.name}: test PR-AUC={best_test.pr_auc:.4f}, "
            f"ROC-AUC={best_test.roc_auc:.4f}, F1={best_test.f1:.4f}, "
            f"columns={X.shape[1]}",
            flush=True,
        )

    # 2) Feature group ablations from the best current full feature family.
    full_cfg = base_config("feature_group_full_summary")
    X_full, _ = build_engineered_matrix(scoped, full_cfg)
    for view_name, X_view in feature_group_views(X_full).items():
        rows, lift, _, _ = fit_eval_catboost(
            X_view,
            y,
            patient_train,
            patient_val,
            patient_test,
            config_name=view_name,
            experiment_type="feature_group_ablation",
            split_name="patient_group_70_15_15",
            iterations=1000,
        )
        all_rows.extend(rows)
        lift_all.extend(lift)
        best_test = (
            pd.DataFrame(rows)
            .query("split == 'test'")
            .sort_values(["pr_auc", "f1"], ascending=False)
            .iloc[0]
        )
        print(
            f"{view_name}: test PR-AUC={best_test.pr_auc:.4f}, "
            f"ROC-AUC={best_test.roc_auc:.4f}, F1={best_test.f1:.4f}, "
            f"columns={X_view.shape[1]}",
            flush=True,
        )

    # 3) Same model/features under patient-safe vs random encounter split.
    for split_name, split in [
        ("patient_group_70_15_15", (patient_train, patient_val, patient_test)),
        ("random_row_70_15_15", (random_train, random_val, random_test)),
    ]:
        rows, lift, _, _ = fit_eval_catboost(
            X_full,
            y,
            *split,
            config_name="full_summary_split_sensitivity",
            experiment_type="split_sensitivity",
            split_name=split_name,
            iterations=1200,
        )
        all_rows.extend(rows)
        lift_all.extend(lift)
        best_test = (
            pd.DataFrame(rows)
            .query("split == 'test'")
            .sort_values(["pr_auc", "f1"], ascending=False)
            .iloc[0]
        )
        print(
            f"{split_name}: test PR-AUC={best_test.pr_auc:.4f}, "
            f"ROC-AUC={best_test.roc_auc:.4f}, F1={best_test.f1:.4f}",
            flush=True,
        )

    results = pd.DataFrame(all_rows)
    results.to_csv(RESULTS_DIR / "plateau_diagnostic_results.csv", index=False)
    pd.DataFrame(lift_all).to_csv(RESULTS_DIR / "plateau_diagnostic_lift_tables.csv", index=False)

    signal = summarize_dataset_signal(scoped)
    signal.to_csv(RESULTS_DIR / "plateau_dataset_signal_summary.csv", index=False)

    print("\nTop diagnostic test rows by PR-AUC:")
    cols = [
        "experiment_type",
        "split_name",
        "feature_config",
        "threshold_strategy",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
    ]
    print(
        results.query("split == 'test'")
        .sort_values(["pr_auc", "f1"], ascending=False)[cols]
        .head(20)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    run()
