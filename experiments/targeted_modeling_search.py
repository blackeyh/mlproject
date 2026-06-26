from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from modeling_experiments import (
    RANDOM_STATE,
    RESULTS_DIR,
    FeatureConfig,
    best_thresholds,
    build_feature_matrix,
    evaluate_dummy,
    get_scores,
    load_scoped_data,
    make_pipeline,
    rare_columns_for,
    threshold_metrics,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def targeted_feature_configs():
    configs = []
    for rare in [50, 100, 200, 500]:
        configs.append(
            FeatureConfig(
                name=f"target_raw_admin_weight_category_rare{rare}",
                rare_min_count=rare,
                admin_mode="raw_ids",
                age_mode="raw",
                gender_mode="keep",
                weight_mode="category",
                utilization_mode="log_plus_raw",
            )
        )
    configs.extend(
        [
            FeatureConfig(
                name="target_raw_admin_weight_indicator_rare100",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="raw",
                gender_mode="keep",
                weight_mode="indicator",
                utilization_mode="log_plus_raw",
            ),
            FeatureConfig(
                name="target_raw_admin_weight_drop_rare100",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="raw",
                gender_mode="keep",
                weight_mode="drop",
                utilization_mode="log_plus_raw",
            ),
            FeatureConfig(
                name="target_raw_admin_age_paper_weight_category_rare100",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="paper",
                gender_mode="keep",
                weight_mode="category",
                utilization_mode="log_plus_raw",
            ),
            FeatureConfig(
                name="target_raw_admin_gender_drop_weight_category_rare100",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="raw",
                gender_mode="drop",
                weight_mode="category",
                utilization_mode="log_plus_raw",
            ),
            FeatureConfig(
                name="target_raw_admin_weight_category_no_log_rare100",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="raw",
                gender_mode="keep",
                weight_mode="category",
                utilization_mode="raw_plus_sum",
            ),
            FeatureConfig(
                name="target_raw_admin_weight_category_log_buckets_rare100",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="raw",
                gender_mode="keep",
                weight_mode="category",
                utilization_mode="log_and_bucket",
            ),
            FeatureConfig(
                name="target_raw_admin_weight_category_summaries_only_rare100",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="raw",
                gender_mode="keep",
                weight_mode="category",
                medication_mode="summaries_only",
                utilization_mode="log_plus_raw",
            ),
        ]
    )
    return configs


def targeted_model_specs(pos_weight):
    specs = [
        {
            "model_name": "Logistic_C0.5_balanced",
            "model": LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                C=0.5,
                random_state=RANDOM_STATE,
            ),
            "scale_numeric": True,
            "sample_weight": False,
        },
        {
            "model_name": "ExtraTrees_400_depth24_leaf10",
            "model": ExtraTreesClassifier(
                n_estimators=400,
                max_depth=24,
                min_samples_leaf=10,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "scale_numeric": False,
            "sample_weight": False,
        },
        {
            "model_name": "RandomForest_400_depth24_leaf10",
            "model": RandomForestClassifier(
                n_estimators=400,
                max_depth=24,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "scale_numeric": False,
            "sample_weight": False,
        },
    ]

    if LGBMClassifier is not None:
        for leaves, depth, lr, estimators, child, weight_mode, spw_factor in [
            (15, 3, 0.025, 700, 80, "balanced", None),
            (31, -1, 0.020, 800, 60, "balanced", None),
            (63, -1, 0.015, 900, 50, "balanced", None),
            (31, -1, 0.020, 800, 80, None, 0.5),
            (31, -1, 0.020, 800, 80, None, 1.0),
            (31, -1, 0.020, 800, 80, None, 1.5),
            (15, 3, 0.020, 900, 100, None, 0.75),
        ]:
            kwargs = {
                "objective": "binary",
                "n_estimators": estimators,
                "learning_rate": lr,
                "num_leaves": leaves,
                "max_depth": depth,
                "min_child_samples": child,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 1.5,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "verbose": -1,
            }
            if weight_mode == "balanced":
                kwargs["class_weight"] = "balanced"
                name_weight = "balanced"
            else:
                kwargs["scale_pos_weight"] = pos_weight * spw_factor
                name_weight = f"spw{spw_factor:g}"
            specs.append(
                {
                    "model_name": f"LightGBM_l{leaves}_d{depth}_lr{lr}_{name_weight}",
                    "model": LGBMClassifier(**kwargs),
                    "scale_numeric": False,
                    "sample_weight": False,
                }
            )

    if XGBClassifier is not None:
        for depth, lr, estimators, child, reg, spw_factor in [
            (2, 0.030, 650, 8, 2.0, 0.5),
            (3, 0.025, 700, 8, 2.0, 0.75),
            (3, 0.020, 850, 10, 3.0, 1.0),
            (4, 0.020, 750, 10, 3.0, 0.75),
            (4, 0.015, 900, 12, 4.0, 1.0),
            (5, 0.015, 700, 15, 5.0, 0.75),
            (3, 0.015, 1000, 15, 5.0, 0.5),
        ]:
            specs.append(
                {
                    "model_name": f"XGBoost_d{depth}_lr{lr}_spw{spw_factor:g}",
                    "model": XGBClassifier(
                        n_estimators=estimators,
                        learning_rate=lr,
                        max_depth=depth,
                        min_child_weight=child,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=reg,
                        scale_pos_weight=pos_weight * spw_factor,
                        eval_metric="aucpr",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    "scale_numeric": False,
                    "sample_weight": False,
                }
            )

    if CatBoostClassifier is not None:
        for depth, lr, iterations, weight_mode in [
            (4, 0.035, 600, "Balanced"),
            (5, 0.030, 700, "Balanced"),
            (6, 0.025, 700, "Balanced"),
            (5, 0.030, 700, "SqrtBalanced"),
        ]:
            specs.append(
                {
                    "model_name": f"CatBoost_d{depth}_lr{lr}_{weight_mode}",
                    "model": CatBoostClassifier(
                        iterations=iterations,
                        learning_rate=lr,
                        depth=depth,
                        l2_leaf_reg=5.0,
                        loss_function="Logloss",
                        eval_metric="PRAUC",
                        auto_class_weights=weight_mode,
                        random_seed=RANDOM_STATE,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "scale_numeric": False,
                    "sample_weight": False,
                }
            )
    return specs


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
    specs = targeted_model_specs(pos_weight)
    rows = [evaluate_dummy(y_train, y_val, "validation")]
    fitted = {}

    for cfg in targeted_feature_configs():
        X_cfg, _ = build_feature_matrix(scoped, cfg)
        X_train = X_cfg.iloc[train_idx].copy()
        X_val = X_cfg.iloc[val_idx].copy()
        rare_cols = rare_columns_for(X_train)
        sample_weight_balanced = compute_sample_weight(class_weight="balanced", y=y_train)

        (RESULTS_DIR / f"targeted_columns_{cfg.name}.json").write_text(
            json.dumps(
                {
                    "feature_config": asdict(cfg),
                    "columns": X_cfg.columns.tolist(),
                    "rare_columns": rare_cols,
                    "n_columns_before_encoding": int(X_cfg.shape[1]),
                },
                indent=2,
            )
        )

        print(f"\\n=== Targeted feature config: {cfg.name} ({X_cfg.shape[1]} columns) ===")
        for spec in specs:
            estimator = make_pipeline(
                model=clone(spec["model"]),
                X_train=X_train,
                rare_cols=rare_cols,
                rare_min_count=cfg.rare_min_count,
                scale_numeric=spec["scale_numeric"],
            )
            start = time.perf_counter()
            try:
                if spec["sample_weight"]:
                    estimator.fit(X_train, y_train, model__sample_weight=sample_weight_balanced)
                else:
                    estimator.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - start
                y_score = get_scores(estimator, X_val)
                threshold_choices = best_thresholds(y_val, y_score)
                for strategy, threshold in threshold_choices.items():
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
                fitted[(cfg.name, spec["model_name"])] = {"estimator": estimator, "config": cfg}
            except Exception as exc:
                fit_seconds = time.perf_counter() - start
                rows.append(
                    {
                        "split": "validation",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "threshold_strategy": "failed",
                        "threshold": np.nan,
                        "pr_auc": np.nan,
                        "roc_auc": np.nan,
                        "recall": np.nan,
                        "precision": np.nan,
                        "f1": np.nan,
                        "accuracy": np.nan,
                        "tn": np.nan,
                        "fp": np.nan,
                        "fn": np.nan,
                        "tp": np.nan,
                        "fit_seconds": fit_seconds,
                        "status": "failed",
                        "error": str(exc),
                        "n_columns_before_encoding": int(X_cfg.shape[1]),
                        **asdict(cfg),
                    }
                )
                print(f"{spec['model_name']}: FAILED: {exc}")

        pd.DataFrame(rows).to_csv(RESULTS_DIR / "targeted_validation_results.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "targeted_validation_results.csv", index=False)
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
    )
    selected = selected.drop_duplicates(["feature_config", "model_name", "threshold_strategy"]).head(18)
    selected.to_csv(RESULTS_DIR / "targeted_selected_for_test_from_validation.csv", index=False)

    test_rows = [evaluate_dummy(y_train, y_test, "test")]
    for _, selected_row in selected.iterrows():
        key = (selected_row["feature_config"], selected_row["model_name"])
        if key not in fitted:
            continue
        cfg = fitted[key]["config"]
        X_cfg, _ = build_feature_matrix(scoped, cfg)
        X_test = X_cfg.iloc[test_idx].copy()
        y_score = get_scores(fitted[key]["estimator"], X_test)
        metrics = threshold_metrics(y_test, y_score, selected_row["threshold"])
        metrics.update(
            {
                "split": "test",
                "feature_config": selected_row["feature_config"],
                "model_name": selected_row["model_name"],
                "threshold_strategy": selected_row["threshold_strategy"],
                "fit_seconds": selected_row["fit_seconds"],
                "status": "ok",
                "error": "",
                "selected_validation_pr_auc": selected_row["pr_auc"],
                "selected_validation_recall": selected_row["recall"],
                "selected_validation_precision": selected_row["precision"],
                "selected_validation_f1": selected_row["f1"],
                "n_columns_before_encoding": selected_row["n_columns_before_encoding"],
                "rare_min_count": selected_row["rare_min_count"],
                "diagnosis_mode": selected_row["diagnosis_mode"],
                "admin_mode": selected_row["admin_mode"],
                "age_mode": selected_row["age_mode"],
                "gender_mode": selected_row["gender_mode"],
                "weight_mode": selected_row["weight_mode"],
                "medication_mode": selected_row["medication_mode"],
                "utilization_mode": selected_row["utilization_mode"],
                "payer_specialty_mode": selected_row["payer_specialty_mode"],
            }
        )
        test_rows.append(metrics)

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(RESULTS_DIR / "targeted_test_results_selected_models.csv", index=False)

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
    print("\\nTargeted top validation by PR-AUC:")
    print(ok.sort_values(["pr_auc", "f1"], ascending=False)[cols].head(30).to_string(index=False))
    print("\\nTargeted top validation by F1:")
    print(ok.sort_values(["f1", "pr_auc"], ascending=False)[cols].head(30).to_string(index=False))
    print("\\nTargeted selected test results:")
    print(test_results.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp"]].to_string(index=False))


if __name__ == "__main__":
    run()
