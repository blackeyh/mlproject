from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

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
from imbalance_experiments import majority_row
from imbalance_refined_ensemble import percentile_rank


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


def summary_config(name):
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


def indicator_config(name):
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


def balanced_subset_indices(test_idx, y_test):
    y_array = y_test.to_numpy()
    pos_rel = np.where(y_array == 1)[0]
    neg_rel = np.where(y_array == 0)[0]
    rng = np.random.default_rng(RANDOM_STATE)
    sampled_neg_rel = rng.choice(neg_rel, size=len(pos_rel), replace=False)
    balanced_rel = np.concatenate([pos_rel, sampled_neg_rel])
    rng.shuffle(balanced_rel)
    return test_idx[balanced_rel], y_test.iloc[balanced_rel].reset_index(drop=True)


def fit_native(scoped, cfg, model, train_idx, val_idx, test_idx, y_train):
    X_cfg, _ = build_feature_matrix(scoped, cfg)
    X_train_raw = X_cfg.iloc[train_idx].copy()
    X_val_raw = X_cfg.iloc[val_idx].copy()
    X_test_raw = X_cfg.iloc[test_idx].copy()
    start = time.perf_counter()
    X_train, X_val, cat_features = prepare_catboost_frames(
        X_train_raw, X_val_raw, rare_min_count=cfg.rare_min_count
    )
    _, X_test, _ = prepare_catboost_frames(
        X_train_raw.copy(), X_test_raw, rare_min_count=cfg.rare_min_count
    )
    fitted = model.copy()
    fitted.fit(X_train, y_train, cat_features=cat_features)
    return fitted.predict_proba(X_val)[:, 1], fitted.predict_proba(X_test)[:, 1], time.perf_counter() - start


def fit_ohe(scoped, cfg, model, train_idx, val_idx, test_idx, y_train):
    X_cfg, _ = build_feature_matrix(scoped, cfg)
    X_train = X_cfg.iloc[train_idx].copy()
    X_val = X_cfg.iloc[val_idx].copy()
    X_test = X_cfg.iloc[test_idx].copy()
    rare_cols = rare_columns_for(X_train)
    start = time.perf_counter()
    estimator = make_pipeline(
        model=clone(model),
        X_train=X_train,
        rare_cols=rare_cols,
        rare_min_count=cfg.rare_min_count,
        scale_numeric=False,
    )
    estimator.fit(X_train, y_train)
    return get_scores(estimator, X_val), get_scores(estimator, X_test), time.perf_counter() - start


def rows_for_model(model_name, y_val, val_score, y_balanced, balanced_score, chosen_strategies, fit_seconds):
    thresholds = best_thresholds(y_val, val_score)
    rows = []
    for strategy in chosen_strategies:
        threshold = thresholds[strategy]
        row = threshold_metrics(y_balanced, balanced_score, threshold)
        row.update(
            {
                "split": "balanced_test_50_50",
                "model_name": model_name,
                "threshold_strategy": strategy,
                "validation_threshold": threshold,
                "fit_seconds": fit_seconds,
                "status": "ok",
            }
        )
        rows.append(row)
    return rows


def run():
    scoped = load_scoped_data()
    _, y = build_feature_matrix(scoped, FeatureConfig(name="split_reference"))
    train_idx, val_idx, test_idx, y_train, y_val, y_test = split_indices(y)
    balanced_idx, y_balanced = balanced_subset_indices(test_idx, y_test)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    rows = []
    baseline = majority_row(y_train, y_balanced, "balanced_test_50_50")
    baseline["model_name"] = "MajorityBaseline"
    baseline["test_set"] = "balanced_50_50"
    baseline["n_rows"] = len(y_balanced)
    baseline["positive_rate"] = float(y_balanced.mean())
    rows.append(baseline)

    cfg_summary = summary_config("balanced_eval_age_paper_summaries_only_rare100")
    cfg_indicator = indicator_config("balanced_eval_age_paper_weight_indicator_rare100")

    # Best selected PR-AUC model from the imbalance refinement search.
    pr_model = CatBoostClassifier(
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
    )
    val_score, balanced_score, fit_seconds = fit_native(
        scoped, cfg_summary, pr_model, train_idx, val_idx, balanced_idx, y_train
    )
    for row in rows_for_model(
        "RefinedNativeCat_d6_lr0.015_l210.0_SqrtBalanced",
        y_val,
        val_score,
        y_balanced,
        balanced_score,
        ["default_0.5", "best_f1", "max_recall_precision_ge_0.20", "max_recall_precision_ge_0.15"],
        fit_seconds,
    ):
        row["model_family"] = "best_pr_auc_single_model"
        row["test_set"] = "balanced_50_50"
        row["n_rows"] = len(y_balanced)
        row["positive_rate"] = float(y_balanced.mean())
        rows.append(row)

    # Best selected F1 model from the imbalance ensemble search.
    custom_model = CatBoostClassifier(
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
    )
    indicator_model = CatBoostClassifier(
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
    )
    xgb_model = XGBClassifier(
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
    )

    custom_val, custom_bal, custom_seconds = fit_native(
        scoped, cfg_summary, custom_model, train_idx, val_idx, balanced_idx, y_train
    )
    indicator_val, indicator_bal, indicator_seconds = fit_native(
        scoped, cfg_indicator, indicator_model, train_idx, val_idx, balanced_idx, y_train
    )
    xgb_val, xgb_bal, xgb_seconds = fit_ohe(
        scoped, cfg_summary, xgb_model, train_idx, val_idx, balanced_idx, y_train
    )
    ensemble_val = np.mean(
        [percentile_rank(custom_val), percentile_rank(indicator_val), percentile_rank(xgb_val)],
        axis=0,
    )
    ensemble_balanced = np.mean(
        [percentile_rank(custom_bal), percentile_rank(indicator_bal), percentile_rank(xgb_bal)],
        axis=0,
    )
    ensemble_seconds = custom_seconds + indicator_seconds + xgb_seconds
    for row in rows_for_model(
        "RankAvg_ref_native_custom025__ref_native_indicator_sqrt__ref_xgb_summary_spw050",
        y_val,
        ensemble_val,
        y_balanced,
        ensemble_balanced,
        ["default_0.5", "best_f1", "max_recall_precision_ge_0.20", "max_recall_precision_ge_0.15"],
        ensemble_seconds,
    ):
        row["model_family"] = "best_f1_rank_average_ensemble"
        row["test_set"] = "balanced_50_50"
        row["n_rows"] = len(y_balanced)
        row["positive_rate"] = float(y_balanced.mean())
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "balanced_test_results.csv", index=False)

    cols = [
        "model_family",
        "model_name",
        "threshold_strategy",
        "n_rows",
        "positive_rate",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    print(results[cols].sort_values(["pr_auc", "f1"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    run()
