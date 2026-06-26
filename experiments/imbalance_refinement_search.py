from __future__ import annotations

import time
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from imblearn.ensemble import EasyEnsembleClassifier

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
from imbalance_experiments import majority_row, select_validation_candidates, lift_rows


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(RANDOM_STATE)


def feature_configs():
    return [
        FeatureConfig(
            name="imb_ref_age_paper_summaries_only_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="category",
            medication_mode="summaries_only",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="imb_ref_age_paper_weight_indicator_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="indicator",
            medication_mode="summaries_only",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="imb_ref_raw_age_weight_category_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="raw",
            gender_mode="keep",
            weight_mode="category",
            utilization_mode="log_plus_raw",
        ),
    ]


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


def add_rows(rows, y_true, y_score, metadata):
    for strategy, threshold in best_thresholds(y_true, y_score).items():
        row = threshold_metrics(y_true, y_score, threshold)
        row.update(metadata)
        row.update({"threshold_strategy": strategy, "status": "ok", "error": ""})
        rows.append(row)


def ohe_weighted_specs(pos_weight):
    specs = []
    for depth, lr, n_estimators, child, reg_lambda, factor in [
        (3, 0.014, 900, 6, 2.0, 0.25),
        (3, 0.014, 900, 8, 3.0, 0.50),
        (3, 0.014, 900, 10, 4.0, 0.75),
        (4, 0.012, 1000, 6, 2.0, 0.25),
        (4, 0.012, 1000, 8, 3.0, 0.50),
        (4, 0.012, 1000, 10, 4.0, 0.75),
        (4, 0.012, 1000, 10, 4.0, 1.00),
        (4, 0.012, 1000, 12, 5.0, 1.50),
        (5, 0.010, 1100, 8, 3.0, 0.50),
        (5, 0.010, 1100, 10, 4.0, 1.00),
        (5, 0.010, 1100, 12, 5.0, 1.50),
    ]:
        specs.append(
            {
                "model_name": f"RefinedXGB_d{depth}_lr{lr}_child{child}_spw{factor:.2f}",
                "model": XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=lr,
                    max_depth=depth,
                    min_child_weight=child,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=reg_lambda,
                    scale_pos_weight=pos_weight * factor,
                    max_delta_step=1,
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                "scale_numeric": False,
            }
        )

    for leaves, lr, n_estimators, child, reg_lambda, factor in [
        (15, 0.014, 900, 60, 1.0, 0.25),
        (15, 0.014, 900, 80, 2.0, 0.50),
        (31, 0.012, 1000, 40, 1.0, 0.25),
        (31, 0.012, 1000, 50, 1.0, 0.50),
        (31, 0.012, 1000, 60, 2.0, 0.75),
        (31, 0.012, 1000, 70, 3.0, 1.00),
        (63, 0.010, 1100, 60, 2.0, 0.50),
        (63, 0.010, 1100, 80, 3.0, 0.75),
        (63, 0.010, 1100, 100, 4.0, 1.00),
    ]:
        specs.append(
            {
                "model_name": f"RefinedLGBM_l{leaves}_lr{lr}_child{child}_spw{factor:.2f}",
                "model": LGBMClassifier(
                    objective="binary",
                    n_estimators=n_estimators,
                    learning_rate=lr,
                    num_leaves=leaves,
                    min_child_samples=child,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=reg_lambda,
                    scale_pos_weight=pos_weight * factor,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                ),
                "scale_numeric": False,
            }
        )

    specs.append(
        {
            "model_name": "EasyEnsemble_singleprocess_10x60",
            "model": EasyEnsembleClassifier(
                n_estimators=10,
                estimator=AdaBoostClassifier(
                    n_estimators=60,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE,
                ),
                sampling_strategy="auto",
                replacement=False,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "scale_numeric": False,
        }
    )
    return specs


def native_specs(pos_weight):
    specs = []
    for depth, lr, iterations, l2, weight_name, weight_kwargs in [
        (5, 0.018, 1300, 8.0, "SqrtBalanced", {"auto_class_weights": "SqrtBalanced"}),
        (6, 0.018, 1300, 8.0, "SqrtBalanced", {"auto_class_weights": "SqrtBalanced"}),
        (5, 0.015, 1600, 10.0, "SqrtBalanced", {"auto_class_weights": "SqrtBalanced"}),
        (6, 0.015, 1600, 10.0, "SqrtBalanced", {"auto_class_weights": "SqrtBalanced"}),
        (4, 0.018, 1400, 12.0, "Balanced", {"auto_class_weights": "Balanced"}),
        (5, 0.018, 1400, 12.0, "Balanced", {"auto_class_weights": "Balanced"}),
        (5, 0.018, 1300, 8.0, "customPW0.25", {"class_weights": [1.0, pos_weight * 0.25]}),
        (5, 0.018, 1300, 8.0, "customPW0.50", {"class_weights": [1.0, pos_weight * 0.50]}),
        (6, 0.015, 1500, 10.0, "customPW0.25", {"class_weights": [1.0, pos_weight * 0.25]}),
        (6, 0.015, 1500, 10.0, "customPW0.50", {"class_weights": [1.0, pos_weight * 0.50]}),
    ]:
        specs.append(
            {
                "model_name": f"RefinedNativeCat_d{depth}_lr{lr}_l2{l2}_{weight_name}",
                "model": CatBoostClassifier(
                    iterations=iterations,
                    learning_rate=lr,
                    depth=depth,
                    l2_leaf_reg=l2,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    random_strength=1.0,
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                    **weight_kwargs,
                ),
            }
        )
    return specs


def run():
    scoped = load_scoped_data()
    _, y = build_feature_matrix(scoped, FeatureConfig(name="split_reference"))
    train_idx, val_idx, test_idx, y_train, y_val, y_test = split_indices(y)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    rows = [majority_row(y_train, y_val, "validation")]
    fitted = {}
    lift = []
    ohe_specs = ohe_weighted_specs(pos_weight)
    cat_specs = native_specs(pos_weight)

    for cfg in feature_configs():
        X_cfg, _ = build_feature_matrix(scoped, cfg)
        X_train = X_cfg.iloc[train_idx].copy()
        X_val = X_cfg.iloc[val_idx].copy()
        X_test = X_cfg.iloc[test_idx].copy()
        rare_cols = rare_columns_for(X_train)
        print(f"\n=== Refinement feature config: {cfg.name} ({X_cfg.shape[1]} columns) ===")

        for spec in ohe_specs:
            start = time.perf_counter()
            try:
                estimator = make_pipeline(
                    model=clone(spec["model"]),
                    X_train=X_train,
                    rare_cols=rare_cols,
                    rare_min_count=cfg.rare_min_count,
                    scale_numeric=spec["scale_numeric"],
                )
                estimator.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - start
                val_score = get_scores(estimator, X_val)
                test_score = get_scores(estimator, X_test)
                metadata = {
                    "split": "validation",
                    "experiment_family": "imbalance_refined_ohe",
                    "feature_config": cfg.name,
                    "model_name": spec["model_name"],
                    "fit_seconds": fit_seconds,
                    **asdict(cfg),
                }
                add_rows(rows, y_val, val_score, metadata)
                fitted[(cfg.name, spec["model_name"])] = {
                    "test_score": test_score,
                    "fit_seconds": fit_seconds,
                    "experiment_family": "imbalance_refined_ohe",
                }
                best = pd.DataFrame([r for r in rows if r.get("feature_config") == cfg.name and r.get("model_name") == spec["model_name"]]).sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
                print(f"{spec['model_name']}: PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, recall={best.recall:.4f}, precision={best.precision:.4f}, fit={fit_seconds:.1f}s")
            except Exception as exc:
                rows.append(
                    {
                        "split": "validation",
                        "experiment_family": "imbalance_refined_ohe",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "threshold_strategy": "failed",
                        "fit_seconds": time.perf_counter() - start,
                        "status": "failed",
                        "error": str(exc),
                        **asdict(cfg),
                    }
                )
                print(f"{spec['model_name']}: FAILED: {exc}")

        if "summaries_only" in cfg.name or "weight_indicator" in cfg.name:
            X_train_cat, X_val_cat, cat_features = prepare_catboost_frames(
                X_train.copy(), X_val.copy(), rare_min_count=cfg.rare_min_count
            )
            _, X_test_cat, _ = prepare_catboost_frames(
                X_train.copy(), X_test.copy(), rare_min_count=cfg.rare_min_count
            )
            for spec in cat_specs:
                start = time.perf_counter()
                try:
                    model = spec["model"].copy()
                    model.fit(X_train_cat, y_train, cat_features=cat_features)
                    fit_seconds = time.perf_counter() - start
                    val_score = model.predict_proba(X_val_cat)[:, 1]
                    test_score = model.predict_proba(X_test_cat)[:, 1]
                    metadata = {
                        "split": "validation",
                        "experiment_family": "imbalance_refined_native_catboost",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "fit_seconds": fit_seconds,
                        **asdict(cfg),
                    }
                    add_rows(rows, y_val, val_score, metadata)
                    fitted[(cfg.name, spec["model_name"])] = {
                        "test_score": test_score,
                        "fit_seconds": fit_seconds,
                        "experiment_family": "imbalance_refined_native_catboost",
                    }
                    best = pd.DataFrame([r for r in rows if r.get("feature_config") == cfg.name and r.get("model_name") == spec["model_name"]]).sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
                    print(f"{spec['model_name']}: PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, recall={best.recall:.4f}, precision={best.precision:.4f}, fit={fit_seconds:.1f}s")
                except Exception as exc:
                    rows.append(
                        {
                            "split": "validation",
                            "experiment_family": "imbalance_refined_native_catboost",
                            "feature_config": cfg.name,
                            "model_name": spec["model_name"],
                            "threshold_strategy": "failed",
                            "fit_seconds": time.perf_counter() - start,
                            "status": "failed",
                            "error": str(exc),
                            **asdict(cfg),
                        }
                    )
                    print(f"{spec['model_name']}: FAILED: {exc}")

        pd.DataFrame(rows).to_csv(RESULTS_DIR / "imbalance_refinement_validation_results.csv", index=False)

    validation = pd.DataFrame(rows)
    validation.to_csv(RESULTS_DIR / "imbalance_refinement_validation_results.csv", index=False)
    selected = select_validation_candidates(validation, max_rows=30)
    selected.to_csv(RESULTS_DIR / "imbalance_refinement_selected_for_test.csv", index=False)

    test_rows = [majority_row(y_train, y_test, "test")]
    for _, selected_row in selected.iterrows():
        key = (selected_row["feature_config"], selected_row["model_name"])
        item = fitted[key]
        test_score = item["test_score"]
        metrics = threshold_metrics(y_test, test_score, selected_row["threshold"])
        metrics.update(
            {
                "split": "test",
                "experiment_family": item["experiment_family"],
                "feature_config": selected_row["feature_config"],
                "model_name": selected_row["model_name"],
                "threshold_strategy": selected_row["threshold_strategy"],
                "fit_seconds": item["fit_seconds"],
                "status": "ok",
                "error": "",
                "selected_validation_pr_auc": selected_row["pr_auc"],
                "selected_validation_recall": selected_row["recall"],
                "selected_validation_precision": selected_row["precision"],
                "selected_validation_f1": selected_row["f1"],
            }
        )
        test_rows.append(metrics)
        lift.extend(
            lift_rows(
                y_test,
                test_score,
                {
                    "split": "test",
                    "experiment_family": item["experiment_family"],
                    "feature_config": selected_row["feature_config"],
                    "model_name": selected_row["model_name"],
                    "threshold_strategy": selected_row["threshold_strategy"],
                },
            )
        )

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(RESULTS_DIR / "imbalance_refinement_test_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "imbalance_refinement_lift_tables.csv", index=False)

    cols = [
        "experiment_family",
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
    ok = validation[(validation["split"] == "validation") & (validation["status"] == "ok")].copy()
    print("\nRefinement top validation by PR-AUC:")
    print(ok.sort_values(["pr_auc", "f1"], ascending=False)[cols].head(30).to_string(index=False))
    print("\nRefinement top validation by F1:")
    print(ok.sort_values(["f1", "pr_auc"], ascending=False)[cols].head(30).to_string(index=False))
    print("\nRefinement selected test results:")
    print(test_results.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp"]].to_string(index=False))


if __name__ == "__main__":
    run()
