from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from modeling_experiments import (
    RANDOM_STATE,
    RESULTS_DIR,
    FeatureConfig,
    build_feature_matrix,
    evaluate_dummy,
    get_scores,
    load_scoped_data,
    make_pipeline,
    rare_columns_for,
    threshold_metrics,
)
from targeted_modeling_search import targeted_feature_configs, targeted_model_specs


def run():
    selected_path = RESULTS_DIR / "targeted_selected_for_test_from_validation.csv"
    selected = pd.read_csv(selected_path)

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
    train_idx, _, y_train, _ = train_test_split(
        train_val_idx,
        y_train_val,
        test_size=val_relative,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    config_by_name = {cfg.name: cfg for cfg in targeted_feature_configs()}
    spec_by_name = {spec["model_name"]: spec for spec in targeted_model_specs(pos_weight)}
    fitted = {}

    test_rows = [evaluate_dummy(y_train, y_test, "test")]
    for _, selected_row in selected.iterrows():
        key = (selected_row["feature_config"], selected_row["model_name"])
        if key not in fitted:
            cfg = config_by_name[selected_row["feature_config"]]
            spec = spec_by_name[selected_row["model_name"]]
            X_cfg, _ = build_feature_matrix(scoped, cfg)
            X_train = X_cfg.iloc[train_idx].copy()
            rare_cols = rare_columns_for(X_train)
            estimator = make_pipeline(
                model=clone(spec["model"]),
                X_train=X_train,
                rare_cols=rare_cols,
                rare_min_count=cfg.rare_min_count,
                scale_numeric=spec["scale_numeric"],
            )
            start = time.perf_counter()
            if spec["sample_weight"]:
                sample_weight_balanced = compute_sample_weight(class_weight="balanced", y=y_train)
                estimator.fit(X_train, y_train, model__sample_weight=sample_weight_balanced)
            else:
                estimator.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            fitted[key] = {
                "estimator": estimator,
                "config": cfg,
                "fit_seconds": fit_seconds,
                "X_cfg": X_cfg,
            }
            print(f"Fit {key[1]} on {key[0]} in {fit_seconds:.1f}s")

        item = fitted[key]
        X_test = item["X_cfg"].iloc[test_idx].copy()
        y_score = get_scores(item["estimator"], X_test)
        metrics = threshold_metrics(y_test, y_score, selected_row["threshold"])
        metrics.update(
            {
                "split": "test",
                "feature_config": selected_row["feature_config"],
                "model_name": selected_row["model_name"],
                "threshold_strategy": selected_row["threshold_strategy"],
                "fit_seconds": item["fit_seconds"],
                "status": "ok",
                "error": "",
                "selected_validation_pr_auc": selected_row["pr_auc"],
                "selected_validation_roc_auc": selected_row["roc_auc"],
                "selected_validation_recall": selected_row["recall"],
                "selected_validation_precision": selected_row["precision"],
                "selected_validation_f1": selected_row["f1"],
                "selected_validation_accuracy": selected_row["accuracy"],
                "selected_validation_threshold": selected_row["threshold"],
            }
        )
        test_rows.append(metrics)

    test_results = pd.DataFrame(test_rows)
    out_path = RESULTS_DIR / "targeted_test_results_selected_models.csv"
    test_results.to_csv(out_path, index=False)

    cols = [
        "feature_config",
        "model_name",
        "threshold_strategy",
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
        "selected_validation_pr_auc",
        "selected_validation_f1",
    ]
    print("\nTargeted selected test results by PR-AUC:")
    print(test_results.sort_values(["pr_auc", "f1"], ascending=False)[cols].to_string(index=False))
    print("\nTargeted selected test results by F1:")
    print(test_results.sort_values(["f1", "pr_auc"], ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    run()
