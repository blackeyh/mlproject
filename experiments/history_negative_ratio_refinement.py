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


def make_model(seed: int, *, iterations: int = 1500):
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=10.0,
        random_strength=1.0,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=140,
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


def weighted_average(scores, weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return np.average(np.vstack(scores), axis=0, weights=weights)


def run():
    scoped = load_all_eligible_encounters()
    y = scoped["readmitted_30"].astype(int)
    train_idx, val_idx, test_idx = patient_group_split(scoped, RANDOM_STATE)
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    cfg = replace(
        base_config("history_neg_ratio_refine_indicator_cat_interactions", weight_mode="indicator"),
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
        f"Negative-ratio refinement: train rows={len(y_train):,}, "
        f"positives={int(y_train.sum()):,}, negatives={int((y_train == 0).sum()):,}, "
        f"validation={len(y_val):,}, test={len(y_test):,}",
        flush=True,
    )

    candidates = []
    for ratio in [7.0, 7.5, 8.0, 8.25, 8.5, 9.0]:
        for seed in [37, 202, 404, 808]:
            candidates.append((ratio, seed))
    for seed in [11, 73, 131, 313, 515, 919]:
        candidates.append((8.0, seed))

    rows = []
    lift = []
    score_bank = {}
    for ratio, seed in candidates:
        subset_idx = train_indices_for_ratio(y_train, ratio=ratio, seed=seed)
        model_name = f"NegRefineCat_d6_lr002_neg{ratio:g}_seed{seed}"
        model = make_model(seed=seed)
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
            "val_pr_auc": threshold_metrics(y_val, val_score, 0.5)["pr_auc"],
        }
        common = {
            "experiment_type": "history_negative_ratio_refinement_member",
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
            "best_iteration": int(model.get_best_iteration() or model.get_param("iterations")),
        }
        add_rows(rows, y_val, val_score, {**common, "split": "validation"})
        add_rows(rows, y_test, test_score, {**common, "split": "test"})
        lift.extend(
            lift_rows(
                y_test,
                test_score,
                {**common, "split": "test", "threshold_strategy": "ranking"},
            )
        )
        print(
            f"{model_name}: val PR-AUC={score_bank[model_name]['val_pr_auc']:.4f}, "
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

    ensemble_rows = []
    for size in [3, 5, 8, 12]:
        keys = top_keys[:size]
        if len(keys) < 2:
            continue
        val_scores = [score_bank[k]["val"] for k in keys]
        test_scores = [score_bank[k]["test"] for k in keys]
        val_weights = [score_bank[k]["val_pr_auc"] for k in keys]
        blends = {
            f"score_average_top{size}_validation_members": (
                np.mean(val_scores, axis=0),
                np.mean(test_scores, axis=0),
            ),
            f"rank_average_top{size}_validation_members": (
                rank_average(val_scores),
                rank_average(test_scores),
            ),
            f"val_pr_weighted_top{size}_validation_members": (
                weighted_average(val_scores, val_weights),
                weighted_average(test_scores, val_weights),
            ),
        }
        for name, (val_blend, test_blend) in blends.items():
            common = {
                "experiment_type": "history_negative_ratio_refinement_ensemble",
                "ensemble_name": name,
                "member_key": "|".join(keys),
                "feature_config": cfg.name,
                "model_name": name,
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
            lift.extend(
                lift_rows(
                    y_test,
                    test_blend,
                    {**common, "split": "test", "threshold_strategy": "ranking"},
                )
            )

    all_rows = pd.concat([member_df, pd.DataFrame(ensemble_rows)], ignore_index=True)
    all_rows.to_csv(RESULTS_DIR / "history_negative_ratio_refinement_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "history_negative_ratio_refinement_lift_tables.csv", index=False)

    cols = [
        "split",
        "experiment_type",
        "model_name",
        "threshold_strategy",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
        "negative_ratio",
        "seed",
        "n_members",
        "best_iteration",
    ]
    print("\nTop validation rows:")
    print(
        all_rows.query("split == 'validation'")
        .sort_values(["pr_auc", "f1"], ascending=False)[cols]
        .head(20)
        .to_string(index=False),
        flush=True,
    )
    print("\nTop test rows:")
    print(
        all_rows.query("split == 'test'")
        .sort_values(["pr_auc", "f1"], ascending=False)[cols]
        .head(30)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    run()
