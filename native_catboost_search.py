from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
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
    rare_columns_for,
    threshold_metrics,
)


def feature_configs():
    return [
        FeatureConfig(
            name="native_cat_raw_admin_age_paper_weight_category_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="category",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="native_cat_raw_admin_raw_age_weight_category_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="raw",
            gender_mode="keep",
            weight_mode="category",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="native_cat_raw_admin_weight_drop_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="raw",
            gender_mode="keep",
            weight_mode="drop",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="native_cat_raw_admin_age_paper_summaries_only_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="category",
            medication_mode="summaries_only",
            utilization_mode="log_plus_raw",
        ),
    ]


def model_specs():
    specs = []
    for depth, lr, iterations, l2, weight in [
        (4, 0.035, 700, 5.0, "Balanced"),
        (5, 0.030, 900, 5.0, "Balanced"),
        (6, 0.025, 900, 7.0, "Balanced"),
        (5, 0.030, 900, 5.0, "SqrtBalanced"),
        (6, 0.025, 1000, 7.0, "SqrtBalanced"),
        (4, 0.025, 1000, 8.0, "SqrtBalanced"),
        (5, 0.020, 1200, 8.0, "SqrtBalanced"),
    ]:
        specs.append(
            {
                "model_name": f"NativeCatBoost_d{depth}_lr{lr}_l2{l2}_{weight}",
                "model": CatBoostClassifier(
                    iterations=iterations,
                    learning_rate=lr,
                    depth=depth,
                    l2_leaf_reg=l2,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    auto_class_weights=weight,
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                ),
            }
        )
    return specs


def prepare_catboost_frames(X_train, X_eval, rare_min_count):
    rare_cols = rare_columns_for(X_train)
    grouper = RareCategoryGrouper(columns=rare_cols, min_count=rare_min_count)
    X_train = grouper.fit_transform(X_train)
    X_eval = grouper.transform(X_eval)

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Missing").astype(str)
        X_eval[col] = X_eval[col].fillna("Missing").astype(str)

    medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(medians)
    X_eval[num_cols] = X_eval[num_cols].fillna(medians)

    cat_features = [X_train.columns.get_loc(col) for col in cat_cols]
    return X_train, X_eval, cat_features


def dummy_row(y_train, y_eval, split):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = dummy.predict(np.zeros((len(y_eval), 1)))
    y_score = np.full(len(y_eval), y_train.mean())
    metrics = threshold_metrics(y_eval, y_score, 0.5)
    metrics.update(
        {
            "split": split,
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

    rows = [dummy_row(y_train, y_val, "validation")]
    fitted = {}

    for cfg in feature_configs():
        X_cfg, _ = build_feature_matrix(scoped, cfg)
        X_train_raw = X_cfg.iloc[train_idx].copy()
        X_val_raw = X_cfg.iloc[val_idx].copy()
        X_train, X_val, cat_features = prepare_catboost_frames(
            X_train_raw,
            X_val_raw,
            rare_min_count=cfg.rare_min_count,
        )
        print(f"\\n=== Native CatBoost feature config: {cfg.name} ({X_cfg.shape[1]} columns, {len(cat_features)} categorical) ===")
        for spec in model_specs():
            model = spec["model"].copy()
            start = time.perf_counter()
            try:
                model.fit(X_train, y_train, cat_features=cat_features)
                fit_seconds = time.perf_counter() - start
                y_score = model.predict_proba(X_val)[:, 1]
                thresholds = best_thresholds(y_val, y_score)
                for strategy, threshold in thresholds.items():
                    row = threshold_metrics(y_val, y_score, threshold)
                    row.update(
                        {
                            "split": "validation",
                            "feature_config": cfg.name,
                            "model_name": spec["model_name"],
                            "threshold_strategy": strategy,
                            "fit_seconds": fit_seconds,
                            "status": "ok",
                            "error": "",
                            "n_columns_before_encoding": int(X_cfg.shape[1]),
                            **asdict(cfg),
                        }
                    )
                    rows.append(row)
                best_row = max(
                    (
                        r
                        for r in rows
                        if r.get("feature_config") == cfg.name
                        and r.get("model_name") == spec["model_name"]
                    ),
                    key=lambda r: r["f1"],
                )
                print(
                    f"{spec['model_name']}: PR-AUC={best_row['pr_auc']:.4f}, "
                    f"best F1={best_row['f1']:.4f}, recall={best_row['recall']:.4f}, "
                    f"precision={best_row['precision']:.4f}, fit={fit_seconds:.1f}s"
                )
                fitted[(cfg.name, spec["model_name"])] = {
                    "model": model,
                    "config": cfg,
                    "cat_features": cat_features,
                    "X_cfg": X_cfg,
                    "fit_seconds": fit_seconds,
                }
            except Exception as exc:
                rows.append(
                    {
                        "split": "validation",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "threshold_strategy": "failed",
                        "status": "failed",
                        "error": str(exc),
                        **asdict(cfg),
                    }
                )
                print(f"{spec['model_name']}: FAILED: {exc}")

        pd.DataFrame(rows).to_csv(RESULTS_DIR / "native_catboost_validation_results.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "native_catboost_validation_results.csv", index=False)
    ok = results[(results["split"] == "validation") & (results["status"] == "ok")].copy()
    non_baseline = ok[ok["model_name"] != "MajorityBaseline"].copy()
    selected = pd.concat(
        [
            non_baseline.sort_values(["pr_auc", "f1"], ascending=False).head(6),
            non_baseline[non_baseline["threshold_strategy"] == "best_f1"]
            .sort_values(["f1", "pr_auc"], ascending=False)
            .head(6),
            non_baseline[non_baseline["threshold_strategy"] == "max_recall_precision_ge_0.20"]
            .sort_values(["f1", "pr_auc"], ascending=False)
            .head(6),
            non_baseline[non_baseline["threshold_strategy"] == "max_recall_precision_ge_0.15"]
            .sort_values(["recall", "pr_auc"], ascending=False)
            .head(6),
        ],
        ignore_index=True,
    ).drop_duplicates(["feature_config", "model_name", "threshold_strategy"]).head(18)
    selected.to_csv(RESULTS_DIR / "native_catboost_selected_for_test.csv", index=False)

    test_rows = [dummy_row(y_train, y_test, "test")]
    for _, selected_row in selected.iterrows():
        key = (selected_row["feature_config"], selected_row["model_name"])
        item = fitted[key]
        cfg = item["config"]
        X_cfg = item["X_cfg"]
        X_train_raw = X_cfg.iloc[train_idx].copy()
        X_test_raw = X_cfg.iloc[test_idx].copy()
        _, X_test, _ = prepare_catboost_frames(
            X_train_raw,
            X_test_raw,
            rare_min_count=cfg.rare_min_count,
        )
        y_score = item["model"].predict_proba(X_test)[:, 1]
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
                "selected_validation_f1": selected_row["f1"],
                "selected_validation_recall": selected_row["recall"],
                "selected_validation_precision": selected_row["precision"],
            }
        )
        test_rows.append(metrics)

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(RESULTS_DIR / "native_catboost_test_results.csv", index=False)

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
    ]
    print("\\nNative CatBoost top validation by PR-AUC:")
    print(ok.sort_values(["pr_auc", "f1"], ascending=False)[cols].head(25).to_string(index=False))
    print("\\nNative CatBoost top validation by F1:")
    print(ok.sort_values(["f1", "pr_auc"], ascending=False)[cols].head(25).to_string(index=False))
    print("\\nNative CatBoost selected test results:")
    print(test_results.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp"]].to_string(index=False))


if __name__ == "__main__":
    run()
