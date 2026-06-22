from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import replace

import numpy as np
import optuna
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
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(RANDOM_STATE)


RESULTS_PATH = RESULTS_DIR / "history_catboost_optuna_results.csv"
LIFT_PATH = RESULTS_DIR / "history_catboost_optuna_lift_tables.csv"
STUDY_PATH = RESULTS_DIR / "history_catboost_optuna_study.json"


def add_rows(rows, y_true, score, metadata):
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(metadata)
        row["threshold_strategy"] = strategy
        rows.append(row)


def train_indices_for_ratio(y_train: np.ndarray, ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    n_neg = min(len(neg_idx), int(round(len(pos_idx) * ratio)))
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    selected = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(selected)
    return selected


def suggest_config(trial: optuna.Trial):
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Default", "Bayesian", "Bernoulli"])
    class_weight_mode = trial.suggest_categorical("class_weight_mode", ["none", "sqrt", "custom"])
    cfg = {
        "negative_ratio": trial.suggest_float("negative_ratio", 6.5, 8.05),
        "row_order_seed": trial.suggest_categorical(
            "row_order_seed",
            [17, 37, 73, 107, 149, 181, 202, 257, 293, 313, 404, 515, 707, 919],
        ),
        "model_seed": trial.suggest_categorical(
            "model_seed",
            [17, 37, 73, 107, 149, 181, 202, 257, 293, 313, 404, 515, 707, 919],
        ),
        "iterations": trial.suggest_int("iterations", 1000, 2200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.012, 0.035, log=True),
        "depth": trial.suggest_int("depth", 5, 7),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 4.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.4, 2.4),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "bootstrap_type": bootstrap_type,
        "class_weight_mode": class_weight_mode,
    }
    if bootstrap_type == "Bayesian":
        cfg["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
        cfg["subsample"] = None
    elif bootstrap_type == "Bernoulli":
        cfg["bagging_temperature"] = None
        cfg["subsample"] = trial.suggest_float("subsample", 0.75, 0.98)
    else:
        cfg["bagging_temperature"] = None
        cfg["subsample"] = None
    if class_weight_mode == "custom":
        cfg["pos_weight_multiplier"] = trial.suggest_float("pos_weight_multiplier", 0.08, 0.40)
    else:
        cfg["pos_weight_multiplier"] = None
    return cfg


def make_model(cfg, pos_weight):
    params = {
        "iterations": int(cfg["iterations"]),
        "learning_rate": float(cfg["learning_rate"]),
        "depth": int(cfg["depth"]),
        "l2_leaf_reg": float(cfg["l2_leaf_reg"]),
        "random_strength": float(cfg["random_strength"]),
        "border_count": int(cfg["border_count"]),
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": int(cfg["model_seed"]),
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 140,
    }
    if cfg["bootstrap_type"] == "Bayesian":
        params["bootstrap_type"] = "Bayesian"
        params["bagging_temperature"] = float(cfg["bagging_temperature"])
    elif cfg["bootstrap_type"] == "Bernoulli":
        params["bootstrap_type"] = "Bernoulli"
        params["subsample"] = float(cfg["subsample"])

    if cfg["class_weight_mode"] == "sqrt":
        params["auto_class_weights"] = "SqrtBalanced"
    elif cfg["class_weight_mode"] == "custom":
        params["class_weights"] = [1.0, float(pos_weight * cfg["pos_weight_multiplier"])]
    return CatBoostClassifier(**params)


def load_frames():
    scoped = load_all_eligible_encounters()
    y = scoped["readmitted_30"].astype(int)
    train_idx, val_idx, test_idx = patient_group_split(scoped, RANDOM_STATE)
    cfg = replace(
        base_config("history_optuna_indicator_cat_interactions", weight_mode="indicator"),
        add_categorical_interactions=True,
    )
    X_base, _ = build_engineered_matrix(scoped, cfg)
    X = add_patient_history_features(X_base, scoped)
    X_train_raw = X.iloc[train_idx].copy()
    X_val_raw = X.iloc[val_idx].copy()
    X_test_raw = X.iloc[test_idx].copy()
    X_train, X_val, cat_features = prepare_native_frames(X_train_raw, X_val_raw, min_count=100)
    _, X_test, _ = prepare_native_frames(X_train_raw.copy(), X_test_raw, min_count=100)
    return {
        "feature_config": cfg.name,
        "X": X,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y.iloc[train_idx].to_numpy(),
        "y_val": y.iloc[val_idx].to_numpy(),
        "y_test": y.iloc[test_idx].to_numpy(),
        "cat_features": cat_features,
    }


def run(n_trials: int):
    data = load_frames()
    y_train = data["y_train"]
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    print(
        f"Optuna CatBoost search: trials={n_trials}, train={len(y_train):,}, "
        f"positives={int(y_train.sum()):,}, validation={len(data['y_val']):,}, test={len(data['y_test']):,}",
        flush=True,
    )

    rows = []
    lift = []
    seen_trial_rows = set()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True, group=True),
    )
    study.enqueue_trial(
        {
            "negative_ratio": 7.5,
            "row_order_seed": 37,
            "model_seed": 37,
            "iterations": 1500,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 10.0,
            "random_strength": 1.0,
            "border_count": 254,
            "bootstrap_type": "Default",
            "class_weight_mode": "none",
        }
    )
    study.enqueue_trial(
        {
            "negative_ratio": 8.0,
            "row_order_seed": 202,
            "model_seed": 202,
            "iterations": 1500,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 10.0,
            "random_strength": 1.0,
            "border_count": 254,
            "bootstrap_type": "Default",
            "class_weight_mode": "none",
        }
    )

    def objective(trial: optuna.Trial):
        cfg = suggest_config(trial)
        subset_idx = train_indices_for_ratio(y_train, cfg["negative_ratio"], int(cfg["row_order_seed"]))
        model = make_model(cfg, pos_weight)
        start = time.perf_counter()
        model.fit(
            data["X_train"].iloc[subset_idx],
            y_train[subset_idx],
            cat_features=data["cat_features"],
            eval_set=(data["X_val"], data["y_val"]),
            use_best_model=True,
        )
        fit_seconds = time.perf_counter() - start
        val_score = model.predict_proba(data["X_val"])[:, 1]
        test_score = model.predict_proba(data["X_test"])[:, 1]
        val_pr_auc = threshold_metrics(data["y_val"], val_score, 0.5)["pr_auc"]
        test_pr_auc = threshold_metrics(data["y_test"], test_score, 0.5)["pr_auc"]
        trial.set_user_attr("test_pr_auc", float(test_pr_auc))
        trial.set_user_attr("fit_seconds", float(fit_seconds))
        trial.set_user_attr("best_iteration", int(model.get_best_iteration() or cfg["iterations"]))
        trial.set_user_attr("train_subset_rows", int(len(subset_idx)))

        common = {
            "experiment_type": "history_catboost_optuna",
            "feature_config": data["feature_config"],
            "model_name": f"OptunaCat_trial{trial.number:03d}",
            "trial_number": trial.number,
            "negative_ratio": cfg["negative_ratio"],
            "row_order_seed": cfg["row_order_seed"],
            "model_seed": cfg["model_seed"],
            "iterations": cfg["iterations"],
            "learning_rate": cfg["learning_rate"],
            "depth": cfg["depth"],
            "l2_leaf_reg": cfg["l2_leaf_reg"],
            "random_strength": cfg["random_strength"],
            "border_count": cfg["border_count"],
            "bootstrap_type": cfg["bootstrap_type"],
            "bagging_temperature": cfg["bagging_temperature"],
            "subsample": cfg["subsample"],
            "class_weight_mode": cfg["class_weight_mode"],
            "pos_weight_multiplier": cfg["pos_weight_multiplier"],
            "n_columns": data["X"].shape[1],
            "train_subset_rows": len(subset_idx),
            "fit_seconds": fit_seconds,
            "best_iteration": int(model.get_best_iteration() or cfg["iterations"]),
            "objective": "validation_pr_auc",
        }
        if trial.number not in seen_trial_rows:
            add_rows(rows, data["y_val"], val_score, {**common, "split": "validation"})
            add_rows(rows, data["y_test"], test_score, {**common, "split": "test"})
            lift.extend(
                lift_rows(
                    data["y_test"],
                    test_score,
                    {**common, "split": "test", "threshold_strategy": "ranking"},
                )
            )
            seen_trial_rows.add(trial.number)
            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
            pd.DataFrame(lift).to_csv(LIFT_PATH, index=False)
            summary = {"completed_trials": len([t for t in study.trials if t.state.name == "COMPLETE"])}
            try:
                summary.update(
                    {
                        "best_value": study.best_value,
                        "best_trial": study.best_trial.number,
                        "best_params": study.best_params,
                    }
                )
            except ValueError:
                pass
            with open(STUDY_PATH, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        print(
            f"trial={trial.number:03d} val PR-AUC={val_pr_auc:.5f} test PR-AUC={test_pr_auc:.5f} "
            f"depth={cfg['depth']} lr={cfg['learning_rate']:.4f} l2={cfg['l2_leaf_reg']:.2f} "
            f"ratio={cfg['negative_ratio']:.3f} row_seed={cfg['row_order_seed']} model_seed={cfg['model_seed']} "
            f"cw={cfg['class_weight_mode']} boot={cfg['bootstrap_type']} best_iter={common['best_iteration']} "
            f"fit={fit_seconds:.1f}s",
            flush=True,
        )
        return val_pr_auc

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    pd.DataFrame(lift).to_csv(LIFT_PATH, index=False)
    with open(STUDY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_value": study.best_value,
                "best_trial": study.best_trial.number,
                "best_params": study.best_params,
                "best_user_attrs": study.best_trial.user_attrs,
            },
            f,
            indent=2,
        )

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
        "trial_number",
        "negative_ratio",
        "row_order_seed",
        "model_seed",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "class_weight_mode",
        "bootstrap_type",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=24)
    args = parser.parse_args()
    run(args.trials)
