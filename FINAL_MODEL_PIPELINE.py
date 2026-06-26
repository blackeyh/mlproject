"""Final hospital readmission model pipeline.

This is the single obvious file for the final model. It loads the raw UCI
Diabetes 130-US Hospitals data, applies the final preprocessing/feature
engineering choices, trains the validation-selected CatBoost model, and writes
metrics/lift outputs.

Default full run:
    python FINAL_MODEL_PIPELINE.py

Fast wiring check without training:
    python FINAL_MODEL_PIPELINE.py --dry-run

Quick training smoke test:
    python FINAL_MODEL_PIPELINE.py --quick

The broader experiment history remains in the other scripts and in
experiment_results/. This file is intentionally only the final model pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mlproject_mplconfig")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from all_encounters_group_split_search import load_all_eligible_encounters
from feature_engineering_search import build_engineered_matrix, prepare_native_frames
from imbalance_experiments import lift_rows
from modeling_experiments import RANDOM_STATE, best_thresholds, threshold_metrics
from patient_history_feature_search import add_patient_history_features
from plateau_diagnostic_search import base_config, patient_group_split


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


FINAL_MODEL_NAME = "NegRefineCat_d6_lr002_neg7.5_seed37"
OUTPUT_DIR = Path("final_model_outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final readmission CatBoost pipeline.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=37, help="Final model sampling/model seed.")
    parser.add_argument("--negative-ratio", type=float, default=7.5)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--od-wait", type=int, default=140)
    parser.add_argument("--dry-run", action="store_true", help="Build features/splits but do not train.")
    parser.add_argument("--quick", action="store_true", help="Train a short smoke-test model.")
    return parser.parse_args()


def assert_patient_safe_split(scoped: pd.DataFrame, train_idx, val_idx, test_idx) -> None:
    train_patients = set(scoped.iloc[train_idx]["patient_nbr"])
    val_patients = set(scoped.iloc[val_idx]["patient_nbr"])
    test_patients = set(scoped.iloc[test_idx]["patient_nbr"])
    if train_patients & val_patients:
        raise RuntimeError("Patient leakage detected between train and validation.")
    if train_patients & test_patients:
        raise RuntimeError("Patient leakage detected between train and test.")
    if val_patients & test_patients:
        raise RuntimeError("Patient leakage detected between validation and test.")


def final_feature_config():
    """Return the exact feature setup used for the final CatBoost model."""
    return replace(
        base_config("final_model_indicator_cat_interactions", weight_mode="indicator"),
        add_categorical_interactions=True,
    )


def build_final_features(scoped: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Build final feature matrix.

    Major preprocessing choices are implemented in the imported feature builder:
    - remove target/id columns from X
    - keep categorical Missing values as Missing
    - use raw administrative IDs as categories
    - use paper age grouping
    - use weight missingness indicator
    - summarize/detail medication features
    - add utilization, lab, diagnosis, admin, categorical-interaction features
    - add safe prior patient-history features using only earlier encounters
    """
    cfg = final_feature_config()
    X_base, _ = build_engineered_matrix(scoped, cfg)
    X = add_patient_history_features(X_base, scoped)
    return X, cfg.name


def train_indices_for_ratio(y_train: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Use all positives plus a fixed ratio of negatives for final training."""
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    n_neg = min(len(neg_idx), int(round(len(pos_idx) * ratio)))
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    selected = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(selected)
    return selected


def make_final_model(args: argparse.Namespace) -> CatBoostClassifier:
    iterations = 50 if args.quick else args.iterations
    od_wait = min(args.od_wait, 20) if args.quick else args.od_wait
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=args.seed,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=od_wait,
    )


def split_summary(scoped: pd.DataFrame, y: pd.Series, train_idx, val_idx, test_idx) -> pd.DataFrame:
    rows = []
    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        labels = y.iloc[idx]
        rows.append(
            {
                "split": name,
                "rows": len(idx),
                "patients": scoped.iloc[idx]["patient_nbr"].nunique(),
                "positive_count": int(labels.sum()),
                "positive_rate": float(labels.mean()),
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def run() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)

    # 1. Load raw data, create readmitted_30 target, remove hospice/expired rows.
    scoped = load_all_eligible_encounters()
    y = scoped["readmitted_30"].astype(int)

    # 2. Patient-safe 70/15/15-style train/validation/test split.
    train_idx, val_idx, test_idx = patient_group_split(scoped, RANDOM_STATE)
    assert_patient_safe_split(scoped, train_idx, val_idx, test_idx)
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    # 3. Final feature engineering.
    X, feature_config_name = build_final_features(scoped)
    X_train_raw = X.iloc[train_idx].copy()
    X_val_raw = X.iloc[val_idx].copy()
    X_test_raw = X.iloc[test_idx].copy()

    # 4. Train-only rare/unseen category handling for native CatBoost frames.
    X_train, X_val, cat_features = prepare_native_frames(X_train_raw, X_val_raw, min_count=100)
    _, X_test, _ = prepare_native_frames(X_train_raw.copy(), X_test_raw, min_count=100)

    selected_train_idx = train_indices_for_ratio(y_train, args.negative_ratio, args.seed)

    summary = {
        "model_name": FINAL_MODEL_NAME,
        "feature_config": feature_config_name,
        "rows_after_hospice_expired_removal": int(len(scoped)),
        "patients_after_hospice_expired_removal": int(scoped["patient_nbr"].nunique()),
        "positive_rate_after_hospice_expired_removal": float(y.mean()),
        "n_features": int(X.shape[1]),
        "n_catboost_categorical_features": int(len(cat_features)),
        "train_subset_rows": int(len(selected_train_idx)),
        "train_subset_positive_rows": int(y_train[selected_train_idx].sum()),
        "train_subset_negative_rows": int((y_train[selected_train_idx] == 0).sum()),
        "negative_ratio": float(args.negative_ratio),
        "seed": int(args.seed),
        "iterations": int(50 if args.quick else args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_strength": float(args.random_strength),
        "od_wait": int(min(args.od_wait, 20) if args.quick else args.od_wait),
        "dry_run": bool(args.dry_run),
        "quick": bool(args.quick),
    }

    split_summary(scoped, y, train_idx, val_idx, test_idx).to_csv(
        args.output_dir / "final_split_summary.csv", index=False
    )
    write_json(args.output_dir / "final_pipeline_summary.json", summary)

    print("Final model pipeline prepared:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        print(f"Dry run complete. Wrote summaries to {args.output_dir}.")
        return

    # 5. Fit final validation-selected CatBoost configuration.
    model = make_final_model(args)
    start = time.perf_counter()
    model.fit(
        X_train.iloc[selected_train_idx],
        y_train[selected_train_idx],
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    fit_seconds = time.perf_counter() - start

    # 6. Select operating threshold on validation only, then apply to test.
    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    validation_thresholds = best_thresholds(y_val, val_score)
    final_threshold = validation_thresholds["best_f1"]

    rows = []
    for split_name, labels, scores in [
        ("validation", y_val, val_score),
        ("test", y_test, test_score),
    ]:
        row = threshold_metrics(labels, scores, final_threshold)
        row.update(
            {
                "model_name": FINAL_MODEL_NAME,
                "split": split_name,
                "threshold_strategy": "validation_best_f1",
                "threshold_source": "validation",
                "negative_ratio": args.negative_ratio,
                "seed": args.seed,
                "fit_seconds": fit_seconds,
                "best_iteration": int(model.get_best_iteration() or model.get_param("iterations")),
            }
        )
        rows.append(row)

    # Diagnostic-only row: useful for matching earlier saved experiment tables, but
    # not used for model selection because it chooses the threshold from test.
    diagnostic_test_threshold = best_thresholds(y_test, test_score)["best_f1"]
    diagnostic_row = threshold_metrics(y_test, test_score, diagnostic_test_threshold)
    diagnostic_row.update(
        {
            "model_name": FINAL_MODEL_NAME,
            "split": "test",
            "threshold_strategy": "test_best_f1_diagnostic_only",
            "threshold_source": "test",
            "negative_ratio": args.negative_ratio,
            "seed": args.seed,
            "fit_seconds": fit_seconds,
            "best_iteration": int(model.get_best_iteration() or model.get_param("iterations")),
        }
    )
    rows.append(diagnostic_row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "final_model_metrics.csv", index=False)

    lift = pd.DataFrame(
        lift_rows(
            y_test,
            test_score,
            {
                "model_name": FINAL_MODEL_NAME,
                "split": "test",
                "threshold_strategy": "ranking",
                "negative_ratio": args.negative_ratio,
                "seed": args.seed,
            },
        )
    )
    lift.to_csv(args.output_dir / "final_model_lift_table.csv", index=False)

    np.save(args.output_dir / "final_validation_scores.npy", val_score)
    np.save(args.output_dir / "final_test_scores.npy", test_score)

    print("\nFinal model metrics:")
    print(metrics.to_string(index=False))
    print("\nFinal model lift table:")
    print(lift.to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    run()
