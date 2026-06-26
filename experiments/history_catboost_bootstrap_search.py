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


def add_rows(rows, y_true, score, metadata):
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(metadata)
        row["threshold_strategy"] = strategy
        rows.append(row)


def shuffled_indices(y_train: np.ndarray, *, ratio: float | None, seed: int):
    rng = np.random.default_rng(seed)
    if ratio is None:
        selected = np.arange(len(y_train))
    else:
        pos_idx = np.flatnonzero(y_train == 1)
        neg_idx = np.flatnonzero(y_train == 0)
        n_neg = min(len(neg_idx), int(round(len(pos_idx) * ratio)))
        selected = np.concatenate([pos_idx, rng.choice(neg_idx, size=n_neg, replace=False)])
    rng.shuffle(selected)
    return selected


def make_model(cfg):
    params = {
        "iterations": 1500,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": cfg.get("l2_leaf_reg", 10.0),
        "random_strength": cfg.get("random_strength", 1.0),
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": cfg["model_seed"],
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 140,
    }
    for key in ["bootstrap_type", "bagging_temperature", "subsample", "boosting_type"]:
        if cfg.get(key) is not None:
            params[key] = cfg[key]
    return CatBoostClassifier(**params)


def candidate_configs():
    base = {
        "row_order_seed": 202,
        "model_seed": 202,
        "ratio": None,
        "bootstrap_type": None,
        "bagging_temperature": None,
        "subsample": None,
        "boosting_type": None,
        "random_strength": 1.0,
        "l2_leaf_reg": 10.0,
    }
    configs = [dict(base, name="full_default_row202_model202")]
    for temp in [0.0, 0.5, 2.0, 5.0]:
        configs.append(dict(base, name=f"full_bayesian_temp{temp:g}", bootstrap_type="Bayesian", bagging_temperature=temp))
    for subsample in [0.66, 0.8, 0.9]:
        configs.append(dict(base, name=f"full_bernoulli_sub{subsample:g}", bootstrap_type="Bernoulli", subsample=subsample))
    for subsample in [0.8, 0.9]:
        configs.append(dict(base, name=f"full_mvs_sub{subsample:g}", bootstrap_type="MVS", subsample=subsample))
    configs.append(dict(base, name="full_no_bootstrap", bootstrap_type="No"))
    configs.append(dict(base, name="full_ordered_boosting", boosting_type="Ordered"))

    ratio75 = dict(base, row_order_seed=37, model_seed=37, ratio=7.5)
    configs.append(dict(ratio75, name="ratio75_default_row37_model37"))
    for temp in [0.0, 0.5, 2.0]:
        configs.append(dict(ratio75, name=f"ratio75_bayesian_temp{temp:g}", bootstrap_type="Bayesian", bagging_temperature=temp))
    for subsample in [0.8, 0.9]:
        configs.append(dict(ratio75, name=f"ratio75_bernoulli_sub{subsample:g}", bootstrap_type="Bernoulli", subsample=subsample))
    return configs


def run():
    scoped = load_all_eligible_encounters()
    y = scoped["readmitted_30"].astype(int)
    train_idx, val_idx, test_idx = patient_group_split(scoped, RANDOM_STATE)
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    cfg = replace(
        base_config("history_bootstrap_indicator_cat_interactions", weight_mode="indicator"),
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
    print(
        f"CatBoost bootstrap search: train rows={len(y_train):,}, "
        f"validation={len(y_val):,}, test={len(y_test):,}, candidates={len(candidate_configs())}",
        flush=True,
    )
    for search_cfg in candidate_configs():
        model_name = f"BootstrapCat_{search_cfg['name']}"
        metadata = {
            "split": "",
            "experiment_type": "history_catboost_bootstrap_search",
            "feature_config": cfg.name,
            "model_name": model_name,
            "row_order_seed": search_cfg["row_order_seed"],
            "model_seed": search_cfg["model_seed"],
            "negative_ratio": search_cfg["ratio"] if search_cfg["ratio"] is not None else -1,
            "bootstrap_type": search_cfg.get("bootstrap_type") or "Default",
            "bagging_temperature": search_cfg.get("bagging_temperature"),
            "subsample": search_cfg.get("subsample"),
            "boosting_type": search_cfg.get("boosting_type") or "Default",
            "random_strength": search_cfg.get("random_strength", 1.0),
            "l2_leaf_reg": search_cfg.get("l2_leaf_reg", 10.0),
            "train_subset_rows": np.nan,
            "n_columns": X.shape[1],
            "fit_seconds": np.nan,
            "best_iteration": np.nan,
            "status": "ok",
            "error": "",
        }
        order = shuffled_indices(y_train, ratio=search_cfg["ratio"], seed=search_cfg["row_order_seed"])
        metadata["train_subset_rows"] = len(order)
        model = make_model(search_cfg)
        start = time.perf_counter()
        try:
            model.fit(
                X_train.iloc[order],
                y_train[order],
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )
            fit_seconds = time.perf_counter() - start
            val_score = model.predict_proba(X_val)[:, 1]
            test_score = model.predict_proba(X_test)[:, 1]
            metadata["fit_seconds"] = fit_seconds
            metadata["best_iteration"] = int(model.get_best_iteration() or model.get_param("iterations"))
            add_rows(rows, y_val, val_score, {**metadata, "split": "validation"})
            add_rows(rows, y_test, test_score, {**metadata, "split": "test"})
            lift.extend(
                lift_rows(
                    y_test,
                    test_score,
                    {**metadata, "split": "test", "threshold_strategy": "ranking"},
                )
            )
            val_pr_auc = threshold_metrics(y_val, val_score, 0.5)["pr_auc"]
            test_pr_auc = threshold_metrics(y_test, test_score, 0.5)["pr_auc"]
            print(
                f"{model_name}: val PR-AUC={val_pr_auc:.4f}, test PR-AUC={test_pr_auc:.4f}, "
                f"bootstrap={metadata['bootstrap_type']}, temp={metadata['bagging_temperature']}, "
                f"subsample={metadata['subsample']}, boosting={metadata['boosting_type']}, "
                f"fit={fit_seconds:.1f}s",
                flush=True,
            )
        except Exception as exc:
            fail = dict(metadata)
            fail.update({"split": "validation", "threshold_strategy": "failed", "status": "failed", "error": repr(exc)})
            rows.append(fail)
            print(f"{model_name}: failed: {exc!r}", flush=True)

        pd.DataFrame(rows).to_csv(RESULTS_DIR / "history_catboost_bootstrap_results.csv", index=False)
        pd.DataFrame(lift).to_csv(RESULTS_DIR / "history_catboost_bootstrap_lift_tables.csv", index=False)

    results = pd.DataFrame(rows)
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
        "negative_ratio",
        "bootstrap_type",
        "bagging_temperature",
        "subsample",
        "boosting_type",
        "best_iteration",
    ]
    print("\nTop validation rows:")
    print(
        results.query("split == 'validation' and status == 'ok'")
        .sort_values(["pr_auc", "f1"], ascending=False)[cols]
        .head(20)
        .to_string(index=False),
        flush=True,
    )
    print("\nTop test rows:")
    print(
        results.query("split == 'test' and status == 'ok'")
        .sort_values(["pr_auc", "f1"], ascending=False)[cols]
        .head(30)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    run()
