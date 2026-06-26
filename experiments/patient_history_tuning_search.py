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
from patient_history_feature_search import add_patient_history_features
from plateau_diagnostic_search import base_config, patient_group_split


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.random.seed(RANDOM_STATE)


def tuned_models(pos_weight):
    base = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 160,
    }
    specs = []
    for name, params in [
        ("d5_lr002_l210_cw025", dict(iterations=1500, learning_rate=0.02, depth=5, l2_leaf_reg=10.0, class_weights=[1.0, pos_weight * 0.25])),
        ("d5_lr0015_l220_cw025", dict(iterations=2100, learning_rate=0.015, depth=5, l2_leaf_reg=20.0, class_weights=[1.0, pos_weight * 0.25])),
        ("d6_lr0015_l210_cw015", dict(iterations=1900, learning_rate=0.015, depth=6, l2_leaf_reg=10.0, class_weights=[1.0, pos_weight * 0.15])),
        ("d6_lr0015_l210_cw025", dict(iterations=2100, learning_rate=0.015, depth=6, l2_leaf_reg=10.0, class_weights=[1.0, pos_weight * 0.25])),
        ("d6_lr0015_l210_cw035", dict(iterations=1900, learning_rate=0.015, depth=6, l2_leaf_reg=10.0, class_weights=[1.0, pos_weight * 0.35])),
        ("d6_lr002_l220_cw025", dict(iterations=1500, learning_rate=0.02, depth=6, l2_leaf_reg=20.0, class_weights=[1.0, pos_weight * 0.25])),
        ("d7_lr0012_l220_cw025", dict(iterations=2200, learning_rate=0.012, depth=7, l2_leaf_reg=20.0, class_weights=[1.0, pos_weight * 0.25])),
        ("d6_lr0015_l210_sqrt", dict(iterations=2100, learning_rate=0.015, depth=6, l2_leaf_reg=10.0, auto_class_weights="SqrtBalanced")),
        ("d5_lr002_l230_sqrt", dict(iterations=1600, learning_rate=0.02, depth=5, l2_leaf_reg=30.0, auto_class_weights="SqrtBalanced")),
    ]:
        specs.append(
            (
                f"HistoryTuneCat_{name}",
                CatBoostClassifier(random_strength=1.0, **params, **base),
            )
        )
    return specs


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

    cfg = replace(
        base_config("history_tuned_indicator_cat_interactions", weight_mode="indicator"),
        add_categorical_interactions=True,
    )
    X_base, _ = build_engineered_matrix(scoped, cfg)
    X = add_patient_history_features(X_base, scoped)
    X_train_raw = X.iloc[train_idx].copy()
    X_val_raw = X.iloc[val_idx].copy()
    X_test_raw = X.iloc[test_idx].copy()
    X_train, X_val, cat_features = prepare_native_frames(X_train_raw, X_val_raw, min_count=100)
    _, X_test, _ = prepare_native_frames(X_train_raw.copy(), X_test_raw, min_count=100)

    rows = []
    lift = []
    for model_name, model in tuned_models(pos_weight):
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
            "experiment_type": "patient_history_tuning",
            "feature_config": cfg.name,
            "model_name": model_name,
            "n_columns": X.shape[1],
            "fit_seconds": fit_seconds,
            "best_iteration": int(model.get_best_iteration() or model.get_param("iterations")),
        }
        add_rows(rows, y_val, val_score, {**common, "split": "validation"})
        add_rows(rows, y_test, test_score, {**common, "split": "test"})
        lift.extend(lift_rows(y_test, test_score, {**common, "split": "test", "threshold_strategy": "ranking"}))
        print(
            f"{model_name}: val PR-AUC={threshold_metrics(y_val, val_score, 0.5)['pr_auc']:.4f}, "
            f"test PR-AUC={threshold_metrics(y_test, test_score, 0.5)['pr_auc']:.4f}, "
            f"best_iter={common['best_iteration']}, fit={fit_seconds:.1f}s",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "patient_history_tuning_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "patient_history_tuning_lift_tables.csv", index=False)

    cols = [
        "split",
        "model_name",
        "threshold_strategy",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
        "best_iteration",
    ]
    print("\nTop validation rows:")
    print(
        results.query("split == 'validation'")
        .sort_values(["pr_auc", "f1"], ascending=False)[cols]
        .head(15)
        .to_string(index=False),
        flush=True,
    )
    print("\nTop test rows:")
    print(
        results.query("split == 'test'")
        .sort_values(["pr_auc", "f1"], ascending=False)[cols]
        .head(15)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    run()
