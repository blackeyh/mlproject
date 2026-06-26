from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from feature_engineering_search import (
    EngineeredFeatureConfig,
    build_engineered_matrix,
    prepare_native_frames,
)
from imbalance_experiments import lift_rows, majority_row, select_validation_candidates
from imbalance_refined_ensemble import percentile_rank
from modeling_experiments import (
    RANDOM_STATE,
    RESULTS_DIR,
    FeatureConfig,
    best_thresholds,
    build_feature_matrix,
    load_scoped_data,
    threshold_metrics,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(RANDOM_STATE)


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


def feature_configs():
    return [
        {
            "kind": "feature_engineered",
            "name": "tuned_fe_indicator",
            "config": EngineeredFeatureConfig(
                name="tuned_fe_indicator",
                base=FeatureConfig(
                    name="tuned_fe_indicator_base",
                    rare_min_count=100,
                    admin_mode="raw_ids",
                    age_mode="paper",
                    gender_mode="keep",
                    weight_mode="indicator",
                    medication_mode="summaries_only",
                    utilization_mode="log_plus_raw",
                ),
            ),
        },
        {
            "kind": "feature_engineered",
            "name": "tuned_fe_summary",
            "config": EngineeredFeatureConfig(
                name="tuned_fe_summary",
                base=FeatureConfig(
                    name="tuned_fe_summary_base",
                    rare_min_count=100,
                    admin_mode="raw_ids",
                    age_mode="paper",
                    gender_mode="keep",
                    weight_mode="category",
                    medication_mode="summaries_only",
                    utilization_mode="log_plus_raw",
                ),
            ),
        },
        {
            "kind": "native",
            "name": "tuned_old_summary",
            "config": FeatureConfig(
                name="tuned_old_summary",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="paper",
                gender_mode="keep",
                weight_mode="category",
                medication_mode="summaries_only",
                utilization_mode="log_plus_raw",
            ),
        },
    ]


def build_matrix(scoped, cfg_spec):
    if cfg_spec["kind"] == "feature_engineered":
        return build_engineered_matrix(scoped, cfg_spec["config"])
    return build_feature_matrix(scoped, cfg_spec["config"])


def base_params(seed=RANDOM_STATE):
    return {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 140,
    }


def tuned_specs(pos_weight):
    return [
        (
            "TunedCat_d6_lr0015_l210_sqrt_rs1",
            {
                "iterations": 2400,
                "learning_rate": 0.015,
                "depth": 6,
                "l2_leaf_reg": 10.0,
                "random_strength": 1.0,
                "auto_class_weights": "SqrtBalanced",
            },
        ),
        (
            "TunedCat_d5_lr0018_l28_sqrt_rs2",
            {
                "iterations": 2200,
                "learning_rate": 0.018,
                "depth": 5,
                "l2_leaf_reg": 8.0,
                "random_strength": 2.0,
                "auto_class_weights": "SqrtBalanced",
            },
        ),
        (
            "TunedCat_d7_lr0010_l220_sqrt_rs3",
            {
                "iterations": 2800,
                "learning_rate": 0.010,
                "depth": 7,
                "l2_leaf_reg": 20.0,
                "random_strength": 3.0,
                "auto_class_weights": "SqrtBalanced",
            },
        ),
        (
            "TunedCat_d6_lr0015_l210_custom025_rs1",
            {
                "iterations": 2300,
                "learning_rate": 0.015,
                "depth": 6,
                "l2_leaf_reg": 10.0,
                "random_strength": 1.0,
                "class_weights": [1.0, pos_weight * 0.25],
            },
        ),
    ]


def underbag_model(seed):
    return CatBoostClassifier(
        iterations=1300,
        learning_rate=0.020,
        depth=6,
        l2_leaf_reg=10.0,
        random_strength=2.0,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        **base_params(seed),
    )


def add_threshold_rows(rows, y_true, score, metadata):
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(metadata)
        row.update({"threshold_strategy": strategy, "status": "ok", "error": ""})
        rows.append(row)


def fit_predict(model, X_train, y_train, X_val, y_val, X_test, cat_features):
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]


def fit_full_models(scoped, cfg_spec, train_idx, val_idx, test_idx, y_train, y_val, rows, scores):
    X_cfg, _ = build_matrix(scoped, cfg_spec)
    X_train_raw = X_cfg.iloc[train_idx].copy()
    X_val_raw = X_cfg.iloc[val_idx].copy()
    X_test_raw = X_cfg.iloc[test_idx].copy()
    X_train, X_val, cat_features = prepare_native_frames(
        X_train_raw,
        X_val_raw,
        min_count=cfg_spec["config"].rare_min_count,
    )
    _, X_test, _ = prepare_native_frames(
        X_train_raw.copy(),
        X_test_raw,
        min_count=cfg_spec["config"].rare_min_count,
    )

    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"\n=== Tuned full CatBoost: {cfg_spec['name']} ({X_cfg.shape[1]} columns) ===", flush=True)
    for model_name, params in tuned_specs(pos_weight):
        start = time.perf_counter()
        try:
            model = CatBoostClassifier(**base_params(), **params)
            val_score, test_score = fit_predict(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                cat_features,
            )
            fit_seconds = time.perf_counter() - start
            metadata = {
                "split": "validation",
                "experiment_family": "catboost_tuning_full",
                "feature_config": cfg_spec["name"],
                "model_name": model_name,
                "fit_seconds": fit_seconds,
                "best_iteration": int(model.get_best_iteration() or params["iterations"]),
                "n_columns": int(X_cfg.shape[1]),
            }
            add_threshold_rows(rows, y_val, val_score, metadata)
            scores[(cfg_spec["name"], model_name)] = {
                "val_score": val_score,
                "test_score": test_score,
                "metadata": metadata,
            }
            best = (
                pd.DataFrame(
                    [
                        r
                        for r in rows
                        if r.get("feature_config") == cfg_spec["name"]
                        and r.get("model_name") == model_name
                    ]
                )
                .sort_values(["pr_auc", "f1"], ascending=False)
                .iloc[0]
            )
            print(
                f"{model_name}: val PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, "
                f"best_iter={metadata['best_iteration']}, fit={fit_seconds:.1f}s",
                flush=True,
            )
        except Exception as exc:
            rows.append(
                {
                    "split": "validation",
                    "experiment_family": "catboost_tuning_full",
                    "feature_config": cfg_spec["name"],
                    "model_name": model_name,
                    "threshold_strategy": "failed",
                    "fit_seconds": time.perf_counter() - start,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"{model_name}: FAILED: {exc}", flush=True)


def fit_underbag(scoped, cfg_spec, train_idx, val_idx, test_idx, y_train, y_val, rows, scores):
    X_cfg, _ = build_matrix(scoped, cfg_spec)
    X_train_full = X_cfg.iloc[train_idx].copy()
    X_val_raw = X_cfg.iloc[val_idx].copy()
    X_test_raw = X_cfg.iloc[test_idx].copy()
    y_train_array = np.asarray(y_train)
    pos_idx = np.where(y_train_array == 1)[0]
    neg_idx = np.where(y_train_array == 0)[0]

    print(f"\n=== Underbagged CatBoost: {cfg_spec['name']} ===", flush=True)
    for ratio in [2, 4]:
        seed_scores_val = []
        seed_scores_test = []
        start_ratio = time.perf_counter()
        for seed in [42, 101, 202, 303]:
            rng = np.random.default_rng(seed)
            n_neg = min(len(neg_idx), len(pos_idx) * ratio)
            sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            sampled_local = np.concatenate([pos_idx, sampled_neg])
            rng.shuffle(sampled_local)

            X_sample_raw = X_train_full.iloc[sampled_local].copy()
            y_sample = y_train_array[sampled_local]
            X_sample, X_val, cat_features = prepare_native_frames(
                X_sample_raw,
                X_val_raw.copy(),
                min_count=cfg_spec["config"].rare_min_count,
            )
            _, X_test, _ = prepare_native_frames(
                X_sample_raw.copy(),
                X_test_raw.copy(),
                min_count=cfg_spec["config"].rare_min_count,
            )
            model = underbag_model(seed)
            val_score, test_score = fit_predict(
                model,
                X_sample,
                y_sample,
                X_val,
                y_val,
                X_test,
                cat_features,
            )
            seed_scores_val.append(val_score)
            seed_scores_test.append(test_score)

        val_avg = np.mean(seed_scores_val, axis=0)
        test_avg = np.mean(seed_scores_test, axis=0)
        fit_seconds = time.perf_counter() - start_ratio
        model_name = f"UnderbagCat_ratio{ratio}_4seed_avg"
        metadata = {
            "split": "validation",
            "experiment_family": "catboost_tuning_underbag",
            "feature_config": cfg_spec["name"],
            "model_name": model_name,
            "fit_seconds": fit_seconds,
            "best_iteration": np.nan,
            "n_columns": int(X_cfg.shape[1]),
        }
        add_threshold_rows(rows, y_val, val_avg, metadata)
        scores[(cfg_spec["name"], model_name)] = {
            "val_score": val_avg,
            "test_score": test_avg,
            "metadata": metadata,
        }
        best = (
            pd.DataFrame(
                [
                    r
                    for r in rows
                    if r.get("feature_config") == cfg_spec["name"]
                    and r.get("model_name") == model_name
                ]
            )
            .sort_values(["pr_auc", "f1"], ascending=False)
            .iloc[0]
        )
        print(
            f"{model_name}: val PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, fit={fit_seconds:.1f}s",
            flush=True,
        )


def add_ensembles(rows, scores, y_val):
    validation = pd.DataFrame(rows)
    ok = validation[(validation["split"] == "validation") & (validation["status"] == "ok")].copy()
    ok = ok[ok["model_name"] != "MajorityBaseline"].copy()
    single_top = (
        ok.sort_values(["pr_auc", "f1"], ascending=False)
        .drop_duplicates(["feature_config", "model_name"])
        .head(10)
    )
    candidate_keys = [(r.feature_config, r.model_name) for r in single_top.itertuples()]
    print(f"\n=== Building ensembles from {len(candidate_keys)} validation-top candidates ===", flush=True)

    for size in [2, 3, 4]:
        for combo in itertools.combinations(candidate_keys, size):
            names = [key[1].replace("TunedCat_", "").replace("UnderbagCat_", "Underbag_") for key in combo]
            short_name = "__".join(names)[:180]
            val_stack = np.vstack([scores[key]["val_score"] for key in combo])
            test_stack = np.vstack([scores[key]["test_score"] for key in combo])
            for blend_type, val_score, test_score in [
                ("score_average", val_stack.mean(axis=0), test_stack.mean(axis=0)),
                (
                    "rank_average",
                    np.vstack([percentile_rank(s) for s in val_stack]).mean(axis=0),
                    np.vstack([percentile_rank(s) for s in test_stack]).mean(axis=0),
                ),
            ]:
                feature_config = "catboost_tuning_ensemble"
                model_name = f"{blend_type}_{size}_{short_name}"
                metadata = {
                    "split": "validation",
                    "experiment_family": "catboost_tuning_ensemble",
                    "feature_config": feature_config,
                    "model_name": model_name,
                    "fit_seconds": float(
                        sum(scores[key]["metadata"]["fit_seconds"] for key in combo)
                    ),
                    "best_iteration": np.nan,
                    "n_columns": np.nan,
                    "members": "|".join([f"{key[0]}:{key[1]}" for key in combo]),
                    "blend_type": blend_type,
                }
                add_threshold_rows(rows, y_val, val_score, metadata)
                scores[(feature_config, model_name)] = {
                    "val_score": val_score,
                    "test_score": test_score,
                    "metadata": metadata,
                }


def run():
    scoped = load_scoped_data()
    _, y = build_feature_matrix(scoped, FeatureConfig(name="split_reference"))
    train_idx, val_idx, test_idx, y_train, y_val, y_test = split_indices(y)

    rows = [majority_row(y_train, y_val, "validation")]
    scores = {}

    configs = feature_configs()
    for cfg_spec in configs:
        fit_full_models(scoped, cfg_spec, train_idx, val_idx, test_idx, y_train, y_val, rows, scores)

    for cfg_spec in [configs[0], configs[2]]:
        fit_underbag(scoped, cfg_spec, train_idx, val_idx, test_idx, y_train, y_val, rows, scores)

    add_ensembles(rows, scores, y_val)

    validation = pd.DataFrame(rows)
    validation.to_csv(RESULTS_DIR / "catboost_tuning_validation_results.csv", index=False)
    selected = select_validation_candidates(validation, max_rows=35)
    selected.to_csv(RESULTS_DIR / "catboost_tuning_selected_for_test.csv", index=False)

    test_rows = [majority_row(y_train, y_test, "test")]
    lift = []
    for selected_row in selected.itertuples():
        key = (selected_row.feature_config, selected_row.model_name)
        item = scores[key]
        metrics = threshold_metrics(y_test, item["test_score"], selected_row.threshold)
        metrics.update(
            {
                "split": "test",
                "experiment_family": item["metadata"]["experiment_family"],
                "feature_config": selected_row.feature_config,
                "model_name": selected_row.model_name,
                "threshold_strategy": selected_row.threshold_strategy,
                "fit_seconds": item["metadata"]["fit_seconds"],
                "status": "ok",
                "error": "",
                "selected_validation_pr_auc": selected_row.pr_auc,
                "selected_validation_recall": selected_row.recall,
                "selected_validation_precision": selected_row.precision,
                "selected_validation_f1": selected_row.f1,
            }
        )
        if "members" in item["metadata"]:
            metrics["members"] = item["metadata"]["members"]
            metrics["blend_type"] = item["metadata"]["blend_type"]
        test_rows.append(metrics)
        lift.extend(
            lift_rows(
                y_test,
                item["test_score"],
                {
                    "split": "test",
                    "experiment_family": item["metadata"]["experiment_family"],
                    "feature_config": selected_row.feature_config,
                    "model_name": selected_row.model_name,
                    "threshold_strategy": selected_row.threshold_strategy,
                },
            )
        )

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(RESULTS_DIR / "catboost_tuning_test_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "catboost_tuning_lift_tables.csv", index=False)

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
    print("\nCatBoost tuning top validation by PR-AUC:")
    print(ok.sort_values(["pr_auc", "f1"], ascending=False)[cols].head(30).to_string(index=False))
    print("\nCatBoost tuning top validation by F1:")
    print(ok.sort_values(["f1", "pr_auc"], ascending=False)[cols].head(30).to_string(index=False))
    print("\nCatBoost tuning selected test results:")
    print(test_results.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp"]].head(35).to_string(index=False))


if __name__ == "__main__":
    run()
