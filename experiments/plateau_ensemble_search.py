from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from feature_engineering_search import build_engineered_matrix, prepare_native_frames
from imbalance_experiments import lift_rows
from modeling_experiments import RESULTS_DIR, RANDOM_STATE, FeatureConfig, best_thresholds, threshold_metrics
from plateau_diagnostic_search import base_config, load_all_eligible_encounters, patient_group_split


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.random.seed(RANDOM_STATE)


def selected_configs():
    full_summary = base_config("ens_full_summary")
    full_indicator = base_config("ens_full_indicator", weight_mode="indicator")
    return [
        full_summary,
        full_indicator,
        replace(full_summary, name="ens_no_medication_detail", add_medication_detail=False),
        replace(full_summary, name="ens_no_lab_interactions", add_lab_interactions=False),
        replace(full_summary, name="ens_cat_interactions", add_categorical_interactions=True),
    ]


def model_specs(pos_weight: float):
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
            "Cat_d6_lr0015_l210_custom025",
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
            "Cat_d6_lr002_l210_custom025",
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
        (
            "Cat_d6_lr0015_l210_sqrt",
            CatBoostClassifier(
                iterations=1900,
                learning_rate=0.015,
                depth=6,
                l2_leaf_reg=10.0,
                random_strength=1.0,
                auto_class_weights="SqrtBalanced",
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

    score_bank = {}
    rows = []
    lift = []

    for cfg in selected_configs():
        X, _ = build_engineered_matrix(scoped, cfg)
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
            key = f"{cfg.name}__{model_name}"
            score_bank[key] = {"val": val_score, "test": test_score}
            metadata = {
                "split": "validation",
                "experiment_type": "plateau_ensemble_member",
                "ensemble_name": "",
                "member_key": key,
                "feature_config": cfg.name,
                "model_name": model_name,
                "n_members": 1,
                "fit_seconds": fit_seconds,
                "best_iteration": int(model.get_best_iteration() or model.get_param("iterations")),
            }
            add_rows(rows, y_val, val_score, metadata)
            add_rows(rows, y_test, test_score, {**metadata, "split": "test"})
            print(
                f"{key}: val PR-AUC={rows[-12]['pr_auc']:.4f}, "
                f"test PR-AUC={threshold_metrics(y_test, test_score, 0.5)['pr_auc']:.4f}, "
                f"fit={fit_seconds:.1f}s",
                flush=True,
            )

    member_summary = (
        pd.DataFrame(rows)
        .query("split == 'validation' and threshold_strategy == 'best_f1'")
        .sort_values(["pr_auc", "f1"], ascending=False)
    )
    top_keys = member_summary["member_key"].drop_duplicates().head(8).tolist()
    print("\nTop validation members for ensembling:")
    print(member_summary[["member_key", "pr_auc", "roc_auc", "f1"]].head(8).to_string(index=False), flush=True)

    ensemble_rows = []
    for size in [2, 3, 4, 5]:
        for keys in itertools.combinations(top_keys, size):
            val_scores = np.vstack([score_bank[k]["val"] for k in keys])
            test_scores = np.vstack([score_bank[k]["test"] for k in keys])
            blends = {
                "score_average": (val_scores.mean(axis=0), test_scores.mean(axis=0)),
                "rank_average": (
                    np.vstack([pd.Series(s).rank(pct=True).to_numpy() for s in val_scores]).mean(axis=0),
                    np.vstack([pd.Series(s).rank(pct=True).to_numpy() for s in test_scores]).mean(axis=0),
                ),
            }
            for blend_type, (val_score, test_score) in blends.items():
                name = f"{blend_type}__" + "__".join(keys)
                metadata = {
                    "split": "validation",
                    "experiment_type": "plateau_probability_ensemble",
                    "ensemble_name": name,
                    "member_key": "|".join(keys),
                    "feature_config": "multiple",
                    "model_name": blend_type,
                    "n_members": size,
                    "fit_seconds": 0.0,
                    "best_iteration": np.nan,
                }
                add_rows(ensemble_rows, y_val, val_score, metadata)
                add_rows(ensemble_rows, y_test, test_score, {**metadata, "split": "test"})
                lift.extend(
                    lift_rows(
                        y_test,
                        test_score,
                        {**metadata, "split": "test", "threshold_strategy": "ranking"},
                    )
                )

    all_rows = pd.concat([pd.DataFrame(rows), pd.DataFrame(ensemble_rows)], ignore_index=True)
    all_rows.to_csv(RESULTS_DIR / "plateau_ensemble_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "plateau_ensemble_lift_tables.csv", index=False)

    selected = (
        all_rows.query("split == 'validation' and experiment_type == 'plateau_probability_ensemble'")
        .sort_values(["pr_auc", "f1"], ascending=False)
        .head(20)
    )
    selected.to_csv(RESULTS_DIR / "plateau_ensemble_selected_by_validation.csv", index=False)

    print("\nTop validation ensembles:")
    cols = [
        "ensemble_name",
        "threshold_strategy",
        "n_members",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
    ]
    print(selected[cols].to_string(index=False), flush=True)

    selected_names = selected["ensemble_name"].head(10).tolist()
    test_selected = all_rows[
        (all_rows["split"] == "test") & (all_rows["ensemble_name"].isin(selected_names))
    ].copy()
    print("\nTest rows for top validation ensembles:")
    print(
        test_selected.sort_values(["pr_auc", "f1"], ascending=False)[
            ["ensemble_name", "threshold_strategy", "n_members", "pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy"]
        ]
        .head(20)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    run()
