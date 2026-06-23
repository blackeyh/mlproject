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


def model_grid():
    # Focused around the best observed unweighted history CatBoost result.
    configs = []
    seeds = [7, 42, 101, 202, 303, 404, 707, 909]
    for seed in seeds:
        configs.append(
            {
                "name": f"SeedSweep_d6_lr002_l210_seed{seed}",
                "seed": seed,
                "depth": 6,
                "learning_rate": 0.02,
                "l2_leaf_reg": 10.0,
                "iterations": 1400,
                "random_strength": 1.0,
                "bootstrap_type": None,
                "subsample": None,
            }
        )
    for seed in [202, 303, 707]:
        for l2 in [5.0, 20.0, 30.0]:
            configs.append(
                {
                    "name": f"SeedSweep_d6_lr002_l2{int(l2)}_seed{seed}",
                    "seed": seed,
                    "depth": 6,
                    "learning_rate": 0.02,
                    "l2_leaf_reg": l2,
                    "iterations": 1500,
                    "random_strength": 1.0,
                    "bootstrap_type": None,
                    "subsample": None,
                }
            )
    for seed in [202, 303, 707]:
        configs.append(
            {
                "name": f"SeedSweep_d6_lr0015_l220_seed{seed}",
                "seed": seed,
                "depth": 6,
                "learning_rate": 0.015,
                "l2_leaf_reg": 20.0,
                "iterations": 2200,
                "random_strength": 1.0,
                "bootstrap_type": None,
                "subsample": None,
            }
        )
        configs.append(
            {
                "name": f"SeedSweep_d7_lr0012_l220_seed{seed}",
                "seed": seed,
                "depth": 7,
                "learning_rate": 0.012,
                "l2_leaf_reg": 20.0,
                "iterations": 2200,
                "random_strength": 1.0,
                "bootstrap_type": None,
                "subsample": None,
            }
        )
    for seed in [202, 303, 707]:
        configs.append(
            {
                "name": f"SeedSweep_d6_lr002_l210_bayes_seed{seed}",
                "seed": seed,
                "depth": 6,
                "learning_rate": 0.02,
                "l2_leaf_reg": 10.0,
                "iterations": 1600,
                "random_strength": 2.0,
                "bootstrap_type": "Bayesian",
                "subsample": None,
            }
        )
    return configs


def make_model(cfg):
    params = {
        "iterations": cfg["iterations"],
        "learning_rate": cfg["learning_rate"],
        "depth": cfg["depth"],
        "l2_leaf_reg": cfg["l2_leaf_reg"],
        "random_strength": cfg["random_strength"],
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": cfg["seed"],
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 160,
    }
    if cfg["bootstrap_type"] is not None:
        params["bootstrap_type"] = cfg["bootstrap_type"]
    if cfg["subsample"] is not None:
        params["subsample"] = cfg["subsample"]
    return CatBoostClassifier(**params)


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

    cfg_features = replace(
        base_config("history_seed_sweep_indicator_cat_interactions", weight_mode="indicator"),
        add_categorical_interactions=True,
    )
    X_base, _ = build_engineered_matrix(scoped, cfg_features)
    X = add_patient_history_features(X_base, scoped)
    X_train_raw = X.iloc[train_idx].copy()
    X_val_raw = X.iloc[val_idx].copy()
    X_test_raw = X.iloc[test_idx].copy()
    X_train, X_val, cat_features = prepare_native_frames(X_train_raw, X_val_raw, min_count=100)
    _, X_test, _ = prepare_native_frames(X_train_raw.copy(), X_test_raw, min_count=100)

    rows = []
    lift = []
    for cfg in model_grid():
        start = time.perf_counter()
        model = make_model(cfg)
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
            "experiment_type": "history_catboost_seed_sweep",
            "feature_config": cfg_features.name,
            "model_name": cfg["name"],
            "seed": cfg["seed"],
            "depth": cfg["depth"],
            "learning_rate": cfg["learning_rate"],
            "l2_leaf_reg": cfg["l2_leaf_reg"],
            "iterations": cfg["iterations"],
            "random_strength": cfg["random_strength"],
            "bootstrap_type": cfg["bootstrap_type"] or "Default",
            "n_columns": X.shape[1],
            "fit_seconds": fit_seconds,
            "best_iteration": int(model.get_best_iteration() or cfg["iterations"]),
        }
        add_rows(rows, y_val, val_score, {**common, "split": "validation"})
        add_rows(rows, y_test, test_score, {**common, "split": "test"})
        lift.extend(lift_rows(y_test, test_score, {**common, "split": "test", "threshold_strategy": "ranking"}))
        print(
            f"{cfg['name']}: val PR-AUC={threshold_metrics(y_val, val_score, 0.5)['pr_auc']:.4f}, "
            f"test PR-AUC={threshold_metrics(y_test, test_score, 0.5)['pr_auc']:.4f}, "
            f"best_iter={common['best_iteration']}, fit={fit_seconds:.1f}s",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "history_catboost_seed_sweep_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "history_catboost_seed_sweep_lift_tables.csv", index=False)

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
    print(results.query("split == 'validation'").sort_values(["pr_auc", "f1"], ascending=False)[cols].head(20).to_string(index=False), flush=True)
    print("\nTop test rows:")
    print(results.query("split == 'test'").sort_values(["pr_auc", "f1"], ascending=False)[cols].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
