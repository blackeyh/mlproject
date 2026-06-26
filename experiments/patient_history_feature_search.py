from __future__ import annotations

import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from all_encounters_group_split_search import load_all_eligible_encounters
from feature_engineering_search import build_engineered_matrix, prepare_native_frames
from imbalance_experiments import lift_rows
from modeling_experiments import RESULTS_DIR, RANDOM_STATE, best_thresholds, threshold_metrics
from plateau_diagnostic_search import base_config, patient_group_split


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.random.seed(RANDOM_STATE)


def _prior_cumulative_sum(sorted_df: pd.DataFrame, group_col: str, value_col: str):
    return sorted_df.groupby(group_col)[value_col].cumsum() - sorted_df[value_col]


def add_patient_history_features(X: pd.DataFrame, scoped: pd.DataFrame):
    """Add only prior-within-patient features ordered by encounter_id.

    These are valid only for the all-encounter framing. They should not be used
    for the stricter first-encounter-only project setup.
    """
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


def configs():
    full = base_config("history_full_indicator", weight_mode="indicator")
    return [
        full,
        replace(full, name="history_full_indicator_cat_interactions", add_categorical_interactions=True),
        replace(full, name="history_no_medication_detail", add_medication_detail=False),
    ]


def model_specs(pos_weight):
    base = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 120,
    }
    return [
        (
            "HistoryCat_d6_lr0015_custom025",
            CatBoostClassifier(
                iterations=1900,
                learning_rate=0.015,
                depth=6,
                l2_leaf_reg=10.0,
                random_strength=1.0,
                class_weights=[1.0, pos_weight * 0.25],
                **base,
            ),
        ),
        (
            "HistoryCat_d6_lr002_custom025",
            CatBoostClassifier(
                iterations=1300,
                learning_rate=0.02,
                depth=6,
                l2_leaf_reg=10.0,
                random_strength=1.0,
                class_weights=[1.0, pos_weight * 0.25],
                **base,
            ),
        ),
    ]


def add_rows(rows, y_true, score, metadata):
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(metadata)
        row["threshold_strategy"] = strategy
        rows.append(row)


def run():
    scoped = load_all_eligible_encounters()
    y = scoped["readmitted_30"].astype(int)
    train_idx, val_idx, test_idx = patient_group_split(scoped, RANDOM_STATE)
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    rows = []
    lift = []
    for cfg in configs():
        X_base, _ = build_engineered_matrix(scoped, cfg)
        X = add_patient_history_features(X_base, scoped)
        X_train_raw = X.iloc[train_idx].copy()
        X_val_raw = X.iloc[val_idx].copy()
        X_test_raw = X.iloc[test_idx].copy()
        X_train, X_val, cat_features = prepare_native_frames(X_train_raw, X_val_raw, min_count=100)
        _, X_test, _ = prepare_native_frames(X_train_raw.copy(), X_test_raw, min_count=100)

        for model_name, model in model_specs(pos_weight):
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
            common = {
                "experiment_type": "patient_history_features",
                "feature_config": cfg.name,
                "model_name": model_name,
                "n_columns": X.shape[1],
                "fit_seconds": fit_seconds,
                "best_iteration": int(model.get_best_iteration() or model.get_param("iterations")),
            }
            add_rows(rows, y_val, val_score, {**common, "split": "validation"})
            add_rows(rows, y_test, test_score, {**common, "split": "test"})
            lift.extend(
                lift_rows(
                    y_test,
                    test_score,
                    {**common, "split": "test", "threshold_strategy": "ranking"},
                )
            )
            print(
                f"{cfg.name} + {model_name}: "
                f"val PR-AUC={threshold_metrics(y_val, val_score, 0.5)['pr_auc']:.4f}, "
                f"test PR-AUC={threshold_metrics(y_test, test_score, 0.5)['pr_auc']:.4f}, "
                f"fit={fit_seconds:.1f}s",
                flush=True,
            )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "patient_history_feature_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "patient_history_feature_lift_tables.csv", index=False)

    cols = [
        "split",
        "feature_config",
        "model_name",
        "threshold_strategy",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
        "n_columns",
    ]
    print("\nTop test rows:")
    print(
        results.query("split == 'test'")
        .sort_values(["pr_auc", "f1"], ascending=False)[cols]
        .head(20)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    run()
