from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split

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
from native_catboost_search import (
    feature_configs as native_feature_configs,
    model_specs as native_model_specs,
    prepare_catboost_frames,
)
from targeted_modeling_search import targeted_feature_configs, targeted_model_specs


def candidate_specs(pos_weight):
    targeted_configs = {cfg.name: cfg for cfg in targeted_feature_configs()}
    targeted_models = {spec["model_name"]: spec for spec in targeted_model_specs(pos_weight)}
    native_configs = {cfg.name: cfg for cfg in native_feature_configs()}
    native_models = {spec["model_name"]: spec for spec in native_model_specs()}

    return [
        {
            "candidate": "native_summary_d6_sqrt",
            "kind": "native",
            "config": native_configs["native_cat_raw_admin_age_paper_summaries_only_rare100"],
            "model_spec": native_models["NativeCatBoost_d6_lr0.025_l27.0_SqrtBalanced"],
        },
        {
            "candidate": "native_summary_d5_sqrt",
            "kind": "native",
            "config": native_configs["native_cat_raw_admin_age_paper_summaries_only_rare100"],
            "model_spec": native_models["NativeCatBoost_d5_lr0.02_l28.0_SqrtBalanced"],
        },
        {
            "candidate": "native_rawage_d6_sqrt",
            "kind": "native",
            "config": native_configs["native_cat_raw_admin_raw_age_weight_category_rare100"],
            "model_spec": native_models["NativeCatBoost_d6_lr0.025_l27.0_SqrtBalanced"],
        },
        {
            "candidate": "native_agepaper_weight_d6_sqrt",
            "kind": "native",
            "config": native_configs["native_cat_raw_admin_age_paper_weight_category_rare100"],
            "model_spec": native_models["NativeCatBoost_d6_lr0.025_l27.0_SqrtBalanced"],
        },
        {
            "candidate": "ohe_cat_agepaper_weight_d5_sqrt",
            "kind": "ohe",
            "config": targeted_configs["target_raw_admin_age_paper_weight_category_rare100"],
            "model_spec": targeted_models["CatBoost_d5_lr0.03_SqrtBalanced"],
        },
        {
            "candidate": "xgb_agepaper_weight_d5",
            "kind": "ohe",
            "config": targeted_configs["target_raw_admin_age_paper_weight_category_rare100"],
            "model_spec": targeted_models["XGBoost_d5_lr0.015_spw0.75"],
        },
        {
            "candidate": "xgb_rawage_weight_d5",
            "kind": "ohe",
            "config": targeted_configs["target_raw_admin_weight_category_rare100"],
            "model_spec": targeted_models["XGBoost_d5_lr0.015_spw0.75"],
        },
        {
            "candidate": "xgb_weight_indicator_d4",
            "kind": "ohe",
            "config": targeted_configs["target_raw_admin_weight_indicator_rare100"],
            "model_spec": targeted_models["XGBoost_d4_lr0.015_spw1"],
        },
        {
            "candidate": "lgbm_agepaper_weight",
            "kind": "ohe",
            "config": targeted_configs["target_raw_admin_age_paper_weight_category_rare100"],
            "model_spec": targeted_models["LightGBM_l31_d-1_lr0.02_balanced"],
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
            X_train_raw,
            X_val_raw,
            rare_min_count=cfg.rare_min_count,
        )
        _, X_test, _ = prepare_catboost_frames(
            X_train_raw.copy(),
            X_test_raw,
            rare_min_count=cfg.rare_min_count,
        )
        model = spec["model_spec"]["model"].copy()
        model.fit(X_train, y_train, cat_features=cat_features)
        val_score = model.predict_proba(X_val)[:, 1]
        test_score = model.predict_proba(X_test)[:, 1]
    else:
        rare_cols = rare_columns_for(X_train_raw)
        model_spec = spec["model_spec"]
        estimator = make_pipeline(
            model=clone(model_spec["model"]),
            X_train=X_train_raw,
            rare_cols=rare_cols,
            rare_min_count=cfg.rare_min_count,
            scale_numeric=model_spec["scale_numeric"],
        )
        estimator.fit(X_train_raw, y_train)
        val_score = get_scores(estimator, X_val_raw)
        test_score = get_scores(estimator, X_test_raw)

    fit_seconds = time.perf_counter() - start
    return val_score, test_score, fit_seconds


def rows_for_scores(name, members, y_true, score, split):
    rows = []
    thresholds = best_thresholds(y_true, score)
    for strategy, threshold in thresholds.items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(
            {
                "split": split,
                "ensemble_name": name,
                "members": "|".join(members),
                "threshold_strategy": strategy,
                "status": "ok",
            }
        )
        rows.append(row)
    return rows


def run():
    scoped = load_scoped_data()
    _, y = build_feature_matrix(scoped, FeatureConfig(name="split_reference"))
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
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    candidate_scores = {}
    candidate_rows = []
    for spec in candidate_specs(pos_weight):
        val_score, test_score, fit_seconds = fit_candidate(
            spec,
            scoped,
            train_idx,
            val_idx,
            test_idx,
            y_train,
        )
        candidate_scores[spec["candidate"]] = {
            "val": val_score,
            "test": test_score,
            "fit_seconds": fit_seconds,
        }
        for row in rows_for_scores(spec["candidate"], [spec["candidate"]], y_val, val_score, "validation"):
            row["fit_seconds"] = fit_seconds
            candidate_rows.append(row)
        print(f"Fit candidate {spec['candidate']} in {fit_seconds:.1f}s")

    ensemble_rows = []
    names = list(candidate_scores)
    for size in [2, 3, 4, 5]:
        for combo in itertools.combinations(names, size):
            val_score = np.mean([candidate_scores[name]["val"] for name in combo], axis=0)
            ensemble_name = f"avg_{size}_" + "__".join(combo)
            ensemble_rows.extend(rows_for_scores(ensemble_name, combo, y_val, val_score, "validation"))

    # A few hand-weighted blends centered on the strongest validation PR-AUC candidates.
    weighted_blends = [
        {
            "name": "weighted_native_summary_xgb_cat",
            "weights": {
                "native_summary_d6_sqrt": 0.45,
                "xgb_agepaper_weight_d5": 0.30,
                "ohe_cat_agepaper_weight_d5_sqrt": 0.25,
            },
        },
        {
            "name": "weighted_native_pair_xgb",
            "weights": {
                "native_summary_d6_sqrt": 0.40,
                "native_summary_d5_sqrt": 0.25,
                "xgb_rawage_weight_d5": 0.20,
                "xgb_agepaper_weight_d5": 0.15,
            },
        },
        {
            "name": "weighted_catboost_only",
            "weights": {
                "native_summary_d6_sqrt": 0.45,
                "native_rawage_d6_sqrt": 0.25,
                "native_agepaper_weight_d6_sqrt": 0.15,
                "ohe_cat_agepaper_weight_d5_sqrt": 0.15,
            },
        },
    ]
    for blend in weighted_blends:
        total = sum(blend["weights"].values())
        val_score = sum(
            candidate_scores[name]["val"] * weight / total
            for name, weight in blend["weights"].items()
        )
        ensemble_rows.extend(rows_for_scores(blend["name"], list(blend["weights"]), y_val, val_score, "validation"))

    validation = pd.DataFrame(candidate_rows + ensemble_rows)
    validation.to_csv(RESULTS_DIR / "ensemble_validation_results.csv", index=False)

    non_single = validation[validation["ensemble_name"].str.startswith(("avg_", "weighted_"))].copy()
    selected = pd.concat(
        [
            non_single.sort_values(["pr_auc", "f1"], ascending=False).head(8),
            non_single[non_single["threshold_strategy"] == "best_f1"]
            .sort_values(["f1", "pr_auc"], ascending=False)
            .head(8),
            non_single[non_single["threshold_strategy"] == "max_recall_precision_ge_0.20"]
            .sort_values(["f1", "pr_auc"], ascending=False)
            .head(8),
            non_single[non_single["threshold_strategy"] == "max_recall_precision_ge_0.15"]
            .sort_values(["recall", "pr_auc"], ascending=False)
            .head(8),
        ],
        ignore_index=True,
    ).drop_duplicates(["ensemble_name", "threshold_strategy"]).head(24)
    selected.to_csv(RESULTS_DIR / "ensemble_selected_for_test.csv", index=False)

    test_rows = []
    for _, row in selected.iterrows():
        members = row["members"].split("|")
        test_score = np.mean([candidate_scores[name]["test"] for name in members], axis=0)
        metrics = threshold_metrics(y_test, test_score, row["threshold"])
        metrics.update(
            {
                "split": "test",
                "ensemble_name": row["ensemble_name"],
                "members": row["members"],
                "threshold_strategy": row["threshold_strategy"],
                "status": "ok",
                "selected_validation_pr_auc": row["pr_auc"],
                "selected_validation_f1": row["f1"],
                "selected_validation_recall": row["recall"],
                "selected_validation_precision": row["precision"],
            }
        )
        test_rows.append(metrics)
    test = pd.DataFrame(test_rows)
    test.to_csv(RESULTS_DIR / "ensemble_test_results.csv", index=False)

    cols = [
        "ensemble_name",
        "threshold_strategy",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
    ]
    print("\\nTop ensemble validation by PR-AUC:")
    print(validation.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["members"]].head(30).to_string(index=False))
    print("\\nTop ensemble validation by F1:")
    print(validation.sort_values(["f1", "pr_auc"], ascending=False)[cols + ["members"]].head(30).to_string(index=False))
    print("\\nSelected ensemble test results:")
    print(test.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp", "selected_validation_pr_auc", "selected_validation_f1"]].to_string(index=False))


if __name__ == "__main__":
    run()
