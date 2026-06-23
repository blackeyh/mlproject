from __future__ import annotations

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
from modeling_experiments import (
    DATA_PATH,
    HOSPICE_OR_EXPIRED_DISCHARGE_IDS,
    RANDOM_STATE,
    RESULTS_DIR,
    FeatureConfig,
    best_thresholds,
    threshold_metrics,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.random.seed(RANDOM_STATE)


def load_all_eligible_encounters():
    raw = pd.read_csv(DATA_PATH, na_values="?", keep_default_na=False, low_memory=False)
    df = raw.copy()
    df["readmitted_30"] = df["readmitted"].eq("<30").astype(int)
    scoped = df[~df["discharge_disposition_id"].isin(HOSPICE_OR_EXPIRED_DISCHARGE_IDS)].copy()
    return scoped.reset_index(drop=True)


def split_patient_groups(scoped):
    patient_labels = scoped.groupby("patient_nbr")["readmitted_30"].max().reset_index()
    train_val_pat, test_pat = train_test_split(
        patient_labels,
        test_size=0.15,
        stratify=patient_labels["readmitted_30"],
        random_state=RANDOM_STATE,
    )
    val_relative = 0.15 / 0.85
    train_pat, val_pat = train_test_split(
        train_val_pat,
        test_size=val_relative,
        stratify=train_val_pat["readmitted_30"],
        random_state=RANDOM_STATE,
    )

    patient = scoped["patient_nbr"]
    train_idx = np.flatnonzero(patient.isin(train_pat["patient_nbr"]).to_numpy())
    val_idx = np.flatnonzero(patient.isin(val_pat["patient_nbr"]).to_numpy())
    test_idx = np.flatnonzero(patient.isin(test_pat["patient_nbr"]).to_numpy())
    return train_idx, val_idx, test_idx


def feature_configs():
    return [
        EngineeredFeatureConfig(
            name="all_enc_fe_indicator",
            base=FeatureConfig(
                name="all_enc_fe_indicator_base",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="paper",
                gender_mode="keep",
                weight_mode="indicator",
                medication_mode="summaries_only",
                utilization_mode="log_plus_raw",
            ),
        ),
        EngineeredFeatureConfig(
            name="all_enc_fe_summary",
            base=FeatureConfig(
                name="all_enc_fe_summary_base",
                rare_min_count=100,
                admin_mode="raw_ids",
                age_mode="paper",
                gender_mode="keep",
                weight_mode="category",
                medication_mode="summaries_only",
                utilization_mode="log_plus_raw",
            ),
        ),
    ]


def model_specs(pos_weight):
    base = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 120,
    }
    return [
        (
            "AllEncCat_d6_lr0015_l210_sqrt",
            CatBoostClassifier(
                iterations=2000,
                learning_rate=0.015,
                depth=6,
                l2_leaf_reg=10.0,
                random_strength=1.0,
                auto_class_weights="SqrtBalanced",
                **base,
            ),
        ),
        (
            "AllEncCat_d6_lr0015_l210_custom025",
            CatBoostClassifier(
                iterations=1900,
                learning_rate=0.015,
                depth=6,
                l2_leaf_reg=10.0,
                random_strength=1.0,
                class_weights=[1.0, pos_weight * 0.25],
                **base,
            ),
        ),
    ]


def add_threshold_rows(rows, y_true, score, metadata):
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(metadata)
        row.update({"threshold_strategy": strategy, "status": "ok", "error": ""})
        rows.append(row)


def run():
    scoped = load_all_eligible_encounters()
    train_idx, val_idx, test_idx = split_patient_groups(scoped)
    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_idx),
                "patients": scoped.iloc[train_idx]["patient_nbr"].nunique(),
                "positive_rate": scoped.iloc[train_idx]["readmitted_30"].mean(),
            },
            {
                "split": "validation",
                "rows": len(val_idx),
                "patients": scoped.iloc[val_idx]["patient_nbr"].nunique(),
                "positive_rate": scoped.iloc[val_idx]["readmitted_30"].mean(),
            },
            {
                "split": "test",
                "rows": len(test_idx),
                "patients": scoped.iloc[test_idx]["patient_nbr"].nunique(),
                "positive_rate": scoped.iloc[test_idx]["readmitted_30"].mean(),
            },
        ]
    )
    split_summary.to_csv(RESULTS_DIR / "all_encounters_group_split_summary.csv", index=False)
    print("All-encounter patient-group split:")
    print(split_summary.to_string(index=False), flush=True)

    y_all = scoped["readmitted_30"].astype(int)
    y_train = y_all.iloc[train_idx].to_numpy()
    y_val = y_all.iloc[val_idx].to_numpy()
    y_test = y_all.iloc[test_idx].to_numpy()
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    rows = [majority_row(y_train, y_val, "validation")]
    scores = {}

    for cfg in feature_configs():
        X_cfg, _ = build_engineered_matrix(scoped, cfg)
        X_train_raw = X_cfg.iloc[train_idx].copy()
        X_val_raw = X_cfg.iloc[val_idx].copy()
        X_test_raw = X_cfg.iloc[test_idx].copy()
        X_train, X_val, cat_features = prepare_native_frames(
            X_train_raw,
            X_val_raw,
            min_count=cfg.rare_min_count,
        )
        _, X_test, _ = prepare_native_frames(
            X_train_raw.copy(),
            X_test_raw,
            min_count=cfg.rare_min_count,
        )
        print(f"\n=== All encounters config: {cfg.name} ({X_cfg.shape[1]} columns) ===", flush=True)
        for model_name, model in model_specs(pos_weight):
            start = time.perf_counter()
            try:
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
                metadata = {
                    "split": "validation",
                    "experiment_family": "all_encounters_group_split",
                    "feature_config": cfg.name,
                    "model_name": model_name,
                    "fit_seconds": fit_seconds,
                    "best_iteration": int(model.get_best_iteration() or model.get_param("iterations")),
                    "n_columns": int(X_cfg.shape[1]),
                }
                add_threshold_rows(rows, y_val, val_score, metadata)
                scores[(cfg.name, model_name)] = {
                    "test_score": test_score,
                    "fit_seconds": fit_seconds,
                }
                best = (
                    pd.DataFrame(
                        [
                            r
                            for r in rows
                            if r.get("feature_config") == cfg.name
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
                        "experiment_family": "all_encounters_group_split",
                        "feature_config": cfg.name,
                        "model_name": model_name,
                        "threshold_strategy": "failed",
                        "fit_seconds": time.perf_counter() - start,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                print(f"{model_name}: FAILED: {exc}", flush=True)

    validation = pd.DataFrame(rows)
    validation.to_csv(RESULTS_DIR / "all_encounters_group_split_validation_results.csv", index=False)
    selected = select_validation_candidates(validation, max_rows=12)
    selected.to_csv(RESULTS_DIR / "all_encounters_group_split_selected_for_test.csv", index=False)

    test_rows = [majority_row(y_train, y_test, "test")]
    lift = []
    for row in selected.itertuples():
        key = (row.feature_config, row.model_name)
        item = scores[key]
        metrics = threshold_metrics(y_test, item["test_score"], row.threshold)
        metrics.update(
            {
                "split": "test",
                "experiment_family": "all_encounters_group_split",
                "feature_config": row.feature_config,
                "model_name": row.model_name,
                "threshold_strategy": row.threshold_strategy,
                "fit_seconds": item["fit_seconds"],
                "status": "ok",
                "error": "",
                "selected_validation_pr_auc": row.pr_auc,
                "selected_validation_recall": row.recall,
                "selected_validation_precision": row.precision,
                "selected_validation_f1": row.f1,
            }
        )
        test_rows.append(metrics)
        lift.extend(
            lift_rows(
                y_test,
                item["test_score"],
                {
                    "split": "test",
                    "experiment_family": "all_encounters_group_split",
                    "feature_config": row.feature_config,
                    "model_name": row.model_name,
                    "threshold_strategy": row.threshold_strategy,
                },
            )
        )

    test = pd.DataFrame(test_rows)
    test.to_csv(RESULTS_DIR / "all_encounters_group_split_test_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "all_encounters_group_split_lift_tables.csv", index=False)

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
    print("\nAll-encounter validation top:")
    print(validation.sort_values(["pr_auc", "f1"], ascending=False)[cols].head(12).to_string(index=False))
    print("\nAll-encounter selected test results:")
    print(test.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp"]].to_string(index=False))


if __name__ == "__main__":
    run()
