from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier

from modeling_experiments import (
    RANDOM_STATE,
    RESULTS_DIR,
    FeatureConfig,
    RareCategoryGrouper,
    best_thresholds,
    build_feature_matrix,
    load_scoped_data,
    threshold_metrics,
)
from native_catboost_search import prepare_catboost_frames
from imbalance_experiments import lift_rows
from imbalance_refined_ensemble import percentile_rank
from feature_engineering_search import (
    EngineeredFeatureConfig,
    build_engineered_matrix,
    prepare_native_frames,
)


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


def fe_summary_config(name):
    return EngineeredFeatureConfig(
        name=name,
        base=FeatureConfig(
            name=name + "_base",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="category",
            medication_mode="summaries_only",
            utilization_mode="log_plus_raw",
        ),
    )


def fe_indicator_config(name):
    return EngineeredFeatureConfig(
        name=name,
        base=FeatureConfig(
            name=name + "_base",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="indicator",
            medication_mode="summaries_only",
            utilization_mode="log_plus_raw",
        ),
    )


def fe_rawage_config(name):
    return EngineeredFeatureConfig(
        name=name,
        base=FeatureConfig(
            name=name + "_base",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="raw",
            gender_mode="keep",
            weight_mode="category",
            utilization_mode="log_plus_raw",
        ),
    )


def native_summary_config(name):
    return FeatureConfig(
        name=name,
        rare_min_count=100,
        admin_mode="raw_ids",
        age_mode="paper",
        gender_mode="keep",
        weight_mode="category",
        medication_mode="summaries_only",
        utilization_mode="log_plus_raw",
    )


def native_indicator_config(name):
    return FeatureConfig(
        name=name,
        rare_min_count=100,
        admin_mode="raw_ids",
        age_mode="paper",
        gender_mode="keep",
        weight_mode="indicator",
        medication_mode="summaries_only",
        utilization_mode="log_plus_raw",
    )


def candidates(pos_weight):
    return [
        {
            "candidate": "fe_indicator_d6_sqrt",
            "kind": "feature_engineered",
            "config": fe_indicator_config("fe_ens_indicator"),
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
            "candidate": "fe_summary_d6_custom025",
            "kind": "feature_engineered",
            "config": fe_summary_config("fe_ens_summary"),
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
            "candidate": "fe_rawage_d6_custom025",
            "kind": "feature_engineered",
            "config": fe_rawage_config("fe_ens_rawage"),
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
            "candidate": "old_summary_d6_sqrt",
            "kind": "native",
            "config": native_summary_config("old_ens_summary"),
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
            "candidate": "old_summary_d6_custom025",
            "kind": "native",
            "config": native_summary_config("old_ens_summary_custom"),
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
            "candidate": "old_indicator_d6_sqrt",
            "kind": "native",
            "config": native_indicator_config("old_ens_indicator"),
            "model": CatBoostClassifier(
                iterations=1300,
                learning_rate=0.018,
                depth=6,
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


def fit_candidate(spec, scoped, train_idx, val_idx, test_idx, y_train):
    start = time.perf_counter()
    cfg = spec["config"]
    if spec["kind"] == "feature_engineered":
        X_cfg, _ = build_engineered_matrix(scoped, cfg)
        X_train_raw = X_cfg.iloc[train_idx].copy()
        X_val_raw = X_cfg.iloc[val_idx].copy()
        X_test_raw = X_cfg.iloc[test_idx].copy()
        X_train, X_val, cat_features = prepare_native_frames(
            X_train_raw,
            X_val_raw,
            min_count=cfg.rare_min_count,
        )
        _, X_test, _ = prepare_native_frames(
            X_train_raw.copy(),
            X_test_raw,
            min_count=cfg.rare_min_count,
        )
    else:
        X_cfg, _ = build_feature_matrix(scoped, cfg)
        X_train_raw = X_cfg.iloc[train_idx].copy()
        X_val_raw = X_cfg.iloc[val_idx].copy()
        X_test_raw = X_cfg.iloc[test_idx].copy()
        X_train, X_val, cat_features = prepare_catboost_frames(
            X_train_raw,
            X_val_raw,
            rare_min_count=cfg.rare_min_count,
        )
        _, X_test, _ = prepare_catboost_frames(
            X_train_raw.copy(),
            X_test_raw,
            rare_min_count=cfg.rare_min_count,
        )
    model = spec["model"].copy()
    model.fit(X_train, y_train, cat_features=cat_features)
    return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1], time.perf_counter() - start


def rows_for_scores(name, members, y_true, score, split, blend_type):
    rows = []
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(
            {
                "split": split,
                "ensemble_name": name,
                "members": "|".join(members),
                "blend_type": blend_type,
                "threshold_strategy": strategy,
                "status": "ok",
            }
        )
        rows.append(row)
    return rows


def run():
    scoped = load_scoped_data()
    _, y = build_feature_matrix(scoped, FeatureConfig(name="split_reference"))
    train_idx, val_idx, test_idx, y_train, y_val, y_test = split_indices(y)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    scores = {}
    candidate_rows = []
    for spec in candidates(pos_weight):
        val_score, test_score, fit_seconds = fit_candidate(spec, scoped, train_idx, val_idx, test_idx, y_train)
        scores[spec["candidate"]] = {
            "val": val_score,
            "test": test_score,
            "val_rank": percentile_rank(val_score),
            "test_rank": percentile_rank(test_score),
            "fit_seconds": fit_seconds,
        }
        candidate_rows.extend(
            rows_for_scores(spec["candidate"], [spec["candidate"]], y_val, val_score, "validation", "single")
        )
        print(f"Fit {spec['candidate']} in {fit_seconds:.1f}s")

    ensemble_rows = []
    names = list(scores)
    for size in [2, 3, 4, 5, 6]:
        for combo in itertools.combinations(names, size):
            avg_score = np.mean([scores[name]["val"] for name in combo], axis=0)
            rank_score = np.mean([scores[name]["val_rank"] for name in combo], axis=0)
            ensemble_rows.extend(
                rows_for_scores(
                    f"avg_{size}_" + "__".join(combo),
                    combo,
                    y_val,
                    avg_score,
                    "validation",
                    "score_average",
                )
            )
            ensemble_rows.extend(
                rows_for_scores(
                    f"rankavg_{size}_" + "__".join(combo),
                    combo,
                    y_val,
                    rank_score,
                    "validation",
                    "rank_average",
                )
            )

    validation = pd.DataFrame(candidate_rows + ensemble_rows)
    validation.to_csv(RESULTS_DIR / "feature_engineering_ensemble_validation_results.csv", index=False)
    non_single = validation[validation["blend_type"] != "single"].copy()
    selected = pd.concat(
        [
            non_single.sort_values(["pr_auc", "f1"], ascending=False).head(10),
            non_single[non_single["threshold_strategy"] == "best_f1"]
            .sort_values(["f1", "pr_auc"], ascending=False)
            .head(10),
            non_single[non_single["threshold_strategy"] == "max_recall_precision_ge_0.20"]
            .sort_values(["f1", "pr_auc"], ascending=False)
            .head(10),
            non_single[non_single["threshold_strategy"] == "max_recall_precision_ge_0.15"]
            .sort_values(["recall", "pr_auc"], ascending=False)
            .head(10),
        ],
        ignore_index=True,
    ).drop_duplicates(["ensemble_name", "threshold_strategy"]).head(30)
    selected.to_csv(RESULTS_DIR / "feature_engineering_ensemble_selected_for_test.csv", index=False)

    test_rows = []
    lift = []
    for _, row in selected.iterrows():
        members = row["members"].split("|")
        if row["blend_type"] == "rank_average":
            test_score = np.mean([scores[name]["test_rank"] for name in members], axis=0)
        else:
            test_score = np.mean([scores[name]["test"] for name in members], axis=0)
        metrics = threshold_metrics(y_test, test_score, row["threshold"])
        metrics.update(
            {
                "split": "test",
                "ensemble_name": row["ensemble_name"],
                "members": row["members"],
                "blend_type": row["blend_type"],
                "threshold_strategy": row["threshold_strategy"],
                "status": "ok",
                "selected_validation_pr_auc": row["pr_auc"],
                "selected_validation_f1": row["f1"],
                "selected_validation_recall": row["recall"],
                "selected_validation_precision": row["precision"],
            }
        )
        test_rows.append(metrics)
        lift.extend(
            lift_rows(
                y_test,
                test_score,
                {
                    "split": "test",
                    "ensemble_name": row["ensemble_name"],
                    "members": row["members"],
                    "blend_type": row["blend_type"],
                    "threshold_strategy": row["threshold_strategy"],
                },
            )
        )
    test = pd.DataFrame(test_rows)
    test.to_csv(RESULTS_DIR / "feature_engineering_ensemble_test_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "feature_engineering_ensemble_lift_tables.csv", index=False)

    cols = [
        "ensemble_name",
        "blend_type",
        "threshold_strategy",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
    ]
    print("\nTop FE ensemble validation by PR-AUC:")
    print(validation.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["members"]].head(30).to_string(index=False))
    print("\nTop FE ensemble validation by F1:")
    print(validation.sort_values(["f1", "pr_auc"], ascending=False)[cols + ["members"]].head(30).to_string(index=False))
    print("\nSelected FE ensemble test results:")
    print(test.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp", "selected_validation_pr_auc", "selected_validation_f1"]].to_string(index=False))


if __name__ == "__main__":
    run()
