from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from modeling_experiments import (
    RANDOM_STATE,
    RESULTS_DIR,
    FeatureConfig,
    best_thresholds,
    build_feature_matrix,
    get_scores,
    load_scoped_data,
    make_pipeline,
    rare_columns_for,
    threshold_metrics,
)
from native_catboost_search import prepare_catboost_frames
from imbalance_experiments import lift_rows


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


def cfg_summary(name):
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


def cfg_indicator(name):
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


def cfg_raw_weight(name):
    return FeatureConfig(
        name=name,
        rare_min_count=100,
        admin_mode="raw_ids",
        age_mode="raw",
        gender_mode="keep",
        weight_mode="category",
        utilization_mode="log_plus_raw",
    )


def candidate_specs(pos_weight):
    summary = cfg_summary("ens_ref_age_paper_summaries_only_rare100")
    indicator = cfg_indicator("ens_ref_age_paper_weight_indicator_rare100")
    raw_weight = cfg_raw_weight("ens_ref_raw_age_weight_category_rare100")

    return [
        {
            "candidate": "old_native_summary_d5_sqrt",
            "kind": "native",
            "config": summary,
            "model": CatBoostClassifier(
                iterations=1200,
                learning_rate=0.020,
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
        {
            "candidate": "old_native_summary_d6_sqrt",
            "kind": "native",
            "config": summary,
            "model": CatBoostClassifier(
                iterations=1000,
                learning_rate=0.025,
                depth=6,
                l2_leaf_reg=7.0,
                loss_function="Logloss",
                eval_metric="PRAUC",
                auto_class_weights="SqrtBalanced",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
        },
        {
            "candidate": "ref_native_summary_d6_sqrt_lowlr",
            "kind": "native",
            "config": summary,
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
            "candidate": "ref_native_summary_d6_custom025",
            "kind": "native",
            "config": summary,
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
            "candidate": "ref_native_indicator_d6_sqrt",
            "kind": "native",
            "config": indicator,
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
        {
            "candidate": "ref_xgb_summary_d5_spw050",
            "kind": "ohe",
            "config": summary,
            "scale_numeric": False,
            "model": XGBClassifier(
                n_estimators=1100,
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
            "candidate": "ref_lgbm_summary_l31_spw025",
            "kind": "ohe",
            "config": summary,
            "scale_numeric": False,
            "model": LGBMClassifier(
                objective="binary",
                n_estimators=1000,
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
            "candidate": "ref_xgb_indicator_d5_spw150",
            "kind": "ohe",
            "config": indicator,
            "scale_numeric": False,
            "model": XGBClassifier(
                n_estimators=1100,
                learning_rate=0.010,
                max_depth=5,
                min_child_weight=12,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                scale_pos_weight=pos_weight * 1.50,
                max_delta_step=1,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "candidate": "ref_xgb_raw_weight_d5_spw050",
            "kind": "ohe",
            "config": raw_weight,
            "scale_numeric": False,
            "model": XGBClassifier(
                n_estimators=1100,
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


def fit_candidate(spec, scoped, train_idx, val_idx, test_idx, y_train):
    cfg = spec["config"]
    X_cfg, _ = build_feature_matrix(scoped, cfg)
    X_train_raw = X_cfg.iloc[train_idx].copy()
    X_val_raw = X_cfg.iloc[val_idx].copy()
    X_test_raw = X_cfg.iloc[test_idx].copy()
    start = time.perf_counter()

    if spec["kind"] == "native":
        X_train, X_val, cat_features = prepare_catboost_frames(
            X_train_raw, X_val_raw, rare_min_count=cfg.rare_min_count
        )
        _, X_test, _ = prepare_catboost_frames(
            X_train_raw.copy(), X_test_raw, rare_min_count=cfg.rare_min_count
        )
        model = spec["model"].copy()
        model.fit(X_train, y_train, cat_features=cat_features)
        val_score = model.predict_proba(X_val)[:, 1]
        test_score = model.predict_proba(X_test)[:, 1]
    else:
        rare_cols = rare_columns_for(X_train_raw)
        estimator = make_pipeline(
            model=clone(spec["model"]),
            X_train=X_train_raw,
            rare_cols=rare_cols,
            rare_min_count=cfg.rare_min_count,
            scale_numeric=spec["scale_numeric"],
        )
        estimator.fit(X_train_raw, y_train)
        val_score = get_scores(estimator, X_val_raw)
        test_score = get_scores(estimator, X_test_raw)

    return val_score, test_score, time.perf_counter() - start


def percentile_rank(score):
    order = np.argsort(np.asarray(score))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(score), endpoint=True)
    return ranks


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

    candidate_scores = {}
    candidate_rows = []
    for spec in candidate_specs(pos_weight):
        val_score, test_score, fit_seconds = fit_candidate(
            spec, scoped, train_idx, val_idx, test_idx, y_train
        )
        candidate_scores[spec["candidate"]] = {
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

    names = list(candidate_scores)
    ensemble_rows = []
    for size in [2, 3, 4, 5]:
        for combo in itertools.combinations(names, size):
            avg_score = np.mean([candidate_scores[name]["val"] for name in combo], axis=0)
            rank_score = np.mean([candidate_scores[name]["val_rank"] for name in combo], axis=0)
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

    weighted_blends = [
        {
            "name": "weighted_ref_native_trio",
            "weights": {
                "ref_native_summary_d6_custom025": 0.45,
                "ref_native_summary_d6_sqrt_lowlr": 0.35,
                "ref_native_indicator_d6_sqrt": 0.20,
            },
        },
        {
            "name": "weighted_old_new_native",
            "weights": {
                "old_native_summary_d5_sqrt": 0.30,
                "old_native_summary_d6_sqrt": 0.25,
                "ref_native_summary_d6_custom025": 0.25,
                "ref_native_indicator_d6_sqrt": 0.20,
            },
        },
        {
            "name": "weighted_native_xgb_lgbm",
            "weights": {
                "ref_native_summary_d6_custom025": 0.45,
                "ref_native_summary_d6_sqrt_lowlr": 0.25,
                "ref_xgb_summary_d5_spw050": 0.20,
                "ref_lgbm_summary_l31_spw025": 0.10,
            },
        },
    ]
    for blend in weighted_blends:
        total = sum(blend["weights"].values())
        score = sum(candidate_scores[name]["val"] * weight / total for name, weight in blend["weights"].items())
        ensemble_rows.extend(
            rows_for_scores(blend["name"], list(blend["weights"]), y_val, score, "validation", "weighted_score")
        )

    validation = pd.DataFrame(candidate_rows + ensemble_rows)
    validation.to_csv(RESULTS_DIR / "imbalance_ensemble_validation_results.csv", index=False)

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
    selected.to_csv(RESULTS_DIR / "imbalance_ensemble_selected_for_test.csv", index=False)

    test_rows = []
    lift = []
    for _, row in selected.iterrows():
        members = row["members"].split("|")
        if row["blend_type"] == "rank_average":
            test_score = np.mean([candidate_scores[name]["test_rank"] for name in members], axis=0)
        elif row["blend_type"] == "weighted_score":
            weights = next(b["weights"] for b in weighted_blends if b["name"] == row["ensemble_name"])
            total = sum(weights.values())
            test_score = sum(candidate_scores[name]["test"] * weight / total for name, weight in weights.items())
        else:
            test_score = np.mean([candidate_scores[name]["test"] for name in members], axis=0)

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
    test.to_csv(RESULTS_DIR / "imbalance_ensemble_test_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "imbalance_ensemble_lift_tables.csv", index=False)

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
    print("\nTop refined ensemble validation by PR-AUC:")
    print(validation.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["members"]].head(30).to_string(index=False))
    print("\nTop refined ensemble validation by F1:")
    print(validation.sort_values(["f1", "pr_auc"], ascending=False)[cols + ["members"]].head(30).to_string(index=False))
    print("\nSelected refined ensemble test results:")
    print(test.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp", "selected_validation_pr_auc", "selected_validation_f1"]].to_string(index=False))


if __name__ == "__main__":
    run()
