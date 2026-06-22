from __future__ import annotations

import itertools
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


def make_model(seed: int, depth: int = 6, lr: float = 0.02, iterations: int = 1400):
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=10.0,
        random_strength=1.0,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=120,
    )


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


def rank_average(scores):
    return np.vstack([pd.Series(s).rank(pct=True).to_numpy() for s in scores]).mean(axis=0)


def run():
    scoped = load_all_eligible_encounters()
    y = scoped["readmitted_30"].astype(int)
    train_idx, val_idx, test_idx = patient_group_split(scoped, RANDOM_STATE)
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    cfg = replace(
        base_config("history_bagging_indicator_cat_interactions", weight_mode="indicator"),
        add_categorical_interactions=True,
    )
    X_base, _ = build_engineered_matrix(scoped, cfg)
    X = add_patient_history_features(X_base, scoped)
    X_train_raw = X.iloc[train_idx].copy()
    X_val_raw = X.iloc[val_idx].copy()
    X_test_raw = X.iloc[test_idx].copy()
    X_train, X_val, cat_features = prepare_native_frames(X_train_raw, X_val_raw, min_count=100)
    _, X_test, _ = prepare_native_frames(X_train_raw.copy(), X_test_raw, min_count=100)

    print(
        f"History balanced bagging: train rows={len(y_train):,}, "
        f"positives={int(y_train.sum()):,}, validation={len(y_val):,}, test={len(y_test):,}",
        flush=True,
    )

    ratios = [1.0, 2.0, 4.0, 8.0]
    seeds = [101, 202, 303, 404]
    model_shapes = [
        ("d6_lr002", 6, 0.02, 1200),
        ("d5_lr002", 5, 0.02, 1200),
    ]

    rows = []
    lift = []
    score_bank = {}
    for ratio in ratios:
        for seed in seeds:
            for shape_name, depth, lr, iterations in model_shapes:
                subset_idx = train_indices_for_ratio(y_train, ratio=ratio, seed=seed)
                model_name = f"BagCat_{shape_name}_neg{ratio:g}_seed{seed}"
                model = make_model(seed=seed, depth=depth, lr=lr, iterations=iterations)
                start = time.perf_counter()
                model.fit(
                    X_train.iloc[subset_idx],
                    y_train[subset_idx],
                    cat_features=cat_features,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                )
                fit_seconds = time.perf_counter() - start
                val_score = model.predict_proba(X_val)[:, 1]
                test_score = model.predict_proba(X_test)[:, 1]
                score_bank[model_name] = {
                    "ratio": ratio,
                    "seed": seed,
                    "val": val_score,
                    "test": test_score,
                }
                common = {
                    "experiment_type": "history_balanced_bagging_member",
                    "ensemble_name": "",
                    "member_key": model_name,
                    "feature_config": cfg.name,
                    "model_name": model_name,
                    "negative_ratio": ratio,
                    "seed": seed,
                    "n_members": 1,
                    "train_subset_rows": len(subset_idx),
                    "n_columns": X.shape[1],
                    "fit_seconds": fit_seconds,
                    "best_iteration": int(model.get_best_iteration() or iterations),
                }
                add_rows(rows, y_val, val_score, {**common, "split": "validation"})
                add_rows(rows, y_test, test_score, {**common, "split": "test"})
                print(
                    f"{model_name}: val PR-AUC={threshold_metrics(y_val, val_score, 0.5)['pr_auc']:.4f}, "
                    f"test PR-AUC={threshold_metrics(y_test, test_score, 0.5)['pr_auc']:.4f}, "
                    f"subset={len(subset_idx):,}, fit={fit_seconds:.1f}s",
                    flush=True,
                )

    member_df = pd.DataFrame(rows)
    validation_members = (
        member_df.query("split == 'validation' and threshold_strategy == 'best_f1'")
        .sort_values(["pr_auc", "f1"], ascending=False)
    )
    top_keys = validation_members["member_key"].head(12).tolist()
    print("\nTop bagging validation members:")
    print(validation_members[["member_key", "pr_auc", "roc_auc", "f1"]].head(12).to_string(index=False), flush=True)

    ensemble_rows = []
    # Ratio-specific ensembles.
    for ratio in ratios:
        keys = [k for k, v in score_bank.items() if v["ratio"] == ratio]
        if len(keys) < 2:
            continue
        for blend_type in ["score_average", "rank_average"]:
            val_scores = [score_bank[k]["val"] for k in keys]
            test_scores = [score_bank[k]["test"] for k in keys]
            val_blend = np.mean(val_scores, axis=0) if blend_type == "score_average" else rank_average(val_scores)
            test_blend = np.mean(test_scores, axis=0) if blend_type == "score_average" else rank_average(test_scores)
            name = f"{blend_type}_ratio_{ratio:g}_{len(keys)}members"
            common = {
                "experiment_type": "history_balanced_bagging_ensemble",
                "ensemble_name": name,
                "member_key": "|".join(keys),
                "feature_config": cfg.name,
                "model_name": blend_type,
                "negative_ratio": ratio,
                "seed": -1,
                "n_members": len(keys),
                "train_subset_rows": np.nan,
                "n_columns": X.shape[1],
                "fit_seconds": 0.0,
                "best_iteration": np.nan,
            }
            add_rows(ensemble_rows, y_val, val_blend, {**common, "split": "validation"})
            add_rows(ensemble_rows, y_test, test_blend, {**common, "split": "test"})
            lift.extend(lift_rows(y_test, test_blend, {**common, "split": "test", "threshold_strategy": "ranking"}))

    # Validation-selected top-member ensembles.
    for size in [3, 5, 8, 10, 12]:
        keys = top_keys[:size]
        if len(keys) < 2:
            continue
        for blend_type in ["score_average", "rank_average"]:
            val_scores = [score_bank[k]["val"] for k in keys]
            test_scores = [score_bank[k]["test"] for k in keys]
            val_blend = np.mean(val_scores, axis=0) if blend_type == "score_average" else rank_average(val_scores)
            test_blend = np.mean(test_scores, axis=0) if blend_type == "score_average" else rank_average(test_scores)
            name = f"{blend_type}_top{size}_validation_members"
            common = {
                "experiment_type": "history_balanced_bagging_ensemble",
                "ensemble_name": name,
                "member_key": "|".join(keys),
                "feature_config": cfg.name,
                "model_name": blend_type,
                "negative_ratio": -1,
                "seed": -1,
                "n_members": len(keys),
                "train_subset_rows": np.nan,
                "n_columns": X.shape[1],
                "fit_seconds": 0.0,
                "best_iteration": np.nan,
            }
            add_rows(ensemble_rows, y_val, val_blend, {**common, "split": "validation"})
            add_rows(ensemble_rows, y_test, test_blend, {**common, "split": "test"})
            lift.extend(lift_rows(y_test, test_blend, {**common, "split": "test", "threshold_strategy": "ranking"}))

    all_rows = pd.concat([member_df, pd.DataFrame(ensemble_rows)], ignore_index=True)
    all_rows.to_csv(RESULTS_DIR / "history_balanced_bagging_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "history_balanced_bagging_lift_tables.csv", index=False)

    print("\nTop validation rows:")
    cols = ["experiment_type", "ensemble_name", "member_key", "threshold_strategy", "pr_auc", "roc_auc", "recall", "precision", "f1"]
    print(
        all_rows.query("split == 'validation'")
        .sort_values(["pr_auc", "f1"], ascending=False)[cols]
        .head(20)
        .to_string(index=False),
        flush=True,
    )
    print("\nTop test rows:")
    test_cols = cols + ["accuracy", "n_members", "negative_ratio"]
    print(
        all_rows.query("split == 'test'")
        .sort_values(["pr_auc", "f1"], ascending=False)[test_cols]
        .head(30)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    run()
