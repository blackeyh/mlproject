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


RESULTS_PATH = RESULTS_DIR / "history_catboost_order_sensitivity_results.csv"
LIFT_PATH = RESULTS_DIR / "history_catboost_order_sensitivity_lift_tables.csv"


def make_model(
    *,
    model_seed: int,
    random_strength: float = 1.0,
    l2_leaf_reg: float = 10.0,
    iterations: int = 1500,
):
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=model_seed,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=140,
    )


def shuffled_order(n_rows: int, row_order_seed: int):
    rng = np.random.default_rng(row_order_seed)
    order = np.arange(n_rows)
    rng.shuffle(order)
    return order


def add_rows(rows, y_true, score, metadata):
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(metadata)
        row["threshold_strategy"] = strategy
        rows.append(row)


def rank_average(scores):
    return np.vstack([pd.Series(s).rank(pct=True).to_numpy() for s in scores]).mean(axis=0)


def weighted_average(scores, weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return np.average(np.vstack(scores), axis=0, weights=weights)


def candidate_configs():
    candidates = []
    for seed in [5, 17, 29, 41, 53, 67, 89, 107, 149, 181, 223, 257, 293, 331]:
        candidates.append(
            {
                "name": f"same_seed_{seed}",
                "row_order_seed": seed,
                "model_seed": seed,
                "random_strength": 1.0,
                "l2_leaf_reg": 10.0,
            }
        )
    for row_seed in [37, 73, 131, 313, 404, 515]:
        candidates.append(
            {
                "name": f"row{row_seed}_model202",
                "row_order_seed": row_seed,
                "model_seed": 202,
                "random_strength": 1.0,
                "l2_leaf_reg": 10.0,
            }
        )
    for model_seed in [37, 73, 131, 313, 404, 515]:
        candidates.append(
            {
                "name": f"row202_model{model_seed}",
                "row_order_seed": 202,
                "model_seed": model_seed,
                "random_strength": 1.0,
                "l2_leaf_reg": 10.0,
            }
        )
    for random_strength in [0.5, 1.5, 2.0, 3.0]:
        candidates.append(
            {
                "name": f"row202_model202_rs{random_strength:g}",
                "row_order_seed": 202,
                "model_seed": 202,
                "random_strength": random_strength,
                "l2_leaf_reg": 10.0,
            }
        )
    for l2 in [5.0, 15.0, 20.0]:
        candidates.append(
            {
                "name": f"row202_model202_l2{l2:g}",
                "row_order_seed": 202,
                "model_seed": 202,
                "random_strength": 1.0,
                "l2_leaf_reg": l2,
            }
        )
    return candidates


def run():
    scoped = load_all_eligible_encounters()
    y = scoped["readmitted_30"].astype(int)
    train_idx, val_idx, test_idx = patient_group_split(scoped, RANDOM_STATE)
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    cfg = replace(
        base_config("history_order_sensitivity_indicator_cat_interactions", weight_mode="indicator"),
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
        f"CatBoost order sensitivity: train rows={len(y_train):,}, "
        f"validation={len(y_val):,}, test={len(y_test):,}, candidates={len(candidate_configs())}",
        flush=True,
    )

    rows = []
    lift = []
    score_bank = {}
    for cfg_model in candidate_configs():
        order = shuffled_order(len(y_train), cfg_model["row_order_seed"])
        model_name = f"OrderCat_{cfg_model['name']}"
        model = make_model(
            model_seed=cfg_model["model_seed"],
            random_strength=cfg_model["random_strength"],
            l2_leaf_reg=cfg_model["l2_leaf_reg"],
        )
        start = time.perf_counter()
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
        val_pr_auc = threshold_metrics(y_val, val_score, 0.5)["pr_auc"]
        test_pr_auc = threshold_metrics(y_test, test_score, 0.5)["pr_auc"]
        score_bank[model_name] = {
            "val": val_score,
            "test": test_score,
            "val_pr_auc": val_pr_auc,
        }
        common = {
            "experiment_type": "history_catboost_order_sensitivity_member",
            "ensemble_name": "",
            "member_key": model_name,
            "feature_config": cfg.name,
            "model_name": model_name,
            "row_order_seed": cfg_model["row_order_seed"],
            "model_seed": cfg_model["model_seed"],
            "random_strength": cfg_model["random_strength"],
            "l2_leaf_reg": cfg_model["l2_leaf_reg"],
            "n_members": 1,
            "train_subset_rows": len(order),
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
        pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
        pd.DataFrame(lift).to_csv(LIFT_PATH, index=False)
        print(
            f"{model_name}: val PR-AUC={val_pr_auc:.4f}, test PR-AUC={test_pr_auc:.4f}, "
            f"row_seed={cfg_model['row_order_seed']}, model_seed={cfg_model['model_seed']}, "
            f"rs={cfg_model['random_strength']}, l2={cfg_model['l2_leaf_reg']}, "
            f"best_iter={common['best_iteration']}, fit={fit_seconds:.1f}s",
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
                "experiment_type": "history_catboost_order_sensitivity_ensemble",
                "ensemble_name": name,
                "member_key": "|".join(keys),
                "feature_config": cfg.name,
                "model_name": name,
                "row_order_seed": -1,
                "model_seed": -1,
                "random_strength": np.nan,
                "l2_leaf_reg": np.nan,
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
    all_rows.to_csv(RESULTS_PATH, index=False)
    pd.DataFrame(lift).to_csv(LIFT_PATH, index=False)

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
        "row_order_seed",
        "model_seed",
        "random_strength",
        "l2_leaf_reg",
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
