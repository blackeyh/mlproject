from __future__ import annotations

import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier

from all_encounters_group_split_search import load_all_eligible_encounters
from feature_engineering_search import build_engineered_matrix
from imbalance_experiments import lift_rows
from modeling_experiments import RESULTS_DIR, RANDOM_STATE, RareCategoryGrouper, best_thresholds, threshold_metrics
from patient_history_feature_search import add_patient_history_features
from plateau_diagnostic_search import base_config, patient_group_split


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.random.seed(RANDOM_STATE)


def categorical_cols(X):
    return X.select_dtypes(include=["object", "category"]).columns.tolist()


def numeric_cols(X):
    return X.select_dtypes(include=[np.number]).columns.tolist()


def prepare_category_native_frames(X_train, X_eval, min_count=100):
    cat_cols = categorical_cols(X_train)
    num_cols = numeric_cols(X_train)
    grouper = RareCategoryGrouper(columns=cat_cols, min_count=min_count)
    train = grouper.fit_transform(X_train.copy())
    eval_ = grouper.transform(X_eval.copy())
    medians = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(medians)
    eval_[num_cols] = eval_[num_cols].fillna(medians)
    for col in cat_cols:
        train[col] = train[col].fillna("Missing").astype(str).astype("category")
        categories = train[col].cat.categories
        eval_[col] = pd.Categorical(eval_[col].fillna("Missing").astype(str), categories=categories)
    return train, eval_, cat_cols


def make_ordinal_pipeline(model, X_train):
    cat_cols = categorical_cols(X_train)
    num_cols = numeric_cols(X_train)
    return Pipeline(
        steps=[
            ("rare", RareCategoryGrouper(columns=cat_cols, min_count=100)),
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                                    (
                                        "encode",
                                        OrdinalEncoder(
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1,
                                            encoded_missing_value=-1,
                                        ),
                                    ),
                                ]
                            ),
                            cat_cols,
                        ),
                        ("num", SimpleImputer(strategy="median"), num_cols),
                    ],
                    verbose_feature_names_out=False,
                ),
            ),
            ("model", model),
        ]
    )


def make_target_encoder_pipeline(model, X_train):
    cat_cols = categorical_cols(X_train)
    num_cols = numeric_cols(X_train)
    return Pipeline(
        steps=[
            ("rare", RareCategoryGrouper(columns=cat_cols, min_count=100)),
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                                    ("encode", TargetEncoder(target_type="binary", smooth=20.0, cv=5, random_state=RANDOM_STATE)),
                                ]
                            ),
                            cat_cols,
                        ),
                        ("num", SimpleImputer(strategy="median"), num_cols),
                    ],
                    verbose_feature_names_out=False,
                ),
            ),
            ("model", model),
        ]
    )


def add_rows(rows, y_true, score, metadata):
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(metadata)
        row["threshold_strategy"] = strategy
        rows.append(row)


def rank_average(scores):
    return np.vstack([pd.Series(s).rank(pct=True).to_numpy() for s in scores]).mean(axis=0)


def run():
    scoped = load_all_eligible_encounters()
    y = scoped["readmitted_30"].astype(int)
    train_idx, val_idx, test_idx = patient_group_split(scoped, RANDOM_STATE)
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    cfg = replace(
        base_config("history_heterogeneous_indicator_cat_interactions", weight_mode="indicator"),
        add_categorical_interactions=True,
    )
    X_base, _ = build_engineered_matrix(scoped, cfg)
    X = add_patient_history_features(X_base, scoped)
    X_train_raw = X.iloc[train_idx].copy()
    X_val_raw = X.iloc[val_idx].copy()
    X_test_raw = X.iloc[test_idx].copy()

    rows = []
    lift = []
    score_bank = {}

    native_train, native_val, cat_cols = prepare_category_native_frames(X_train_raw, X_val_raw)
    _, native_test, _ = prepare_category_native_frames(X_train_raw.copy(), X_test_raw)

    model_specs = [
        (
            "LightGBM_leaves31_spw1",
            "native",
            LGBMClassifier(
                objective="binary",
                n_estimators=1600,
                learning_rate=0.015,
                num_leaves=31,
                min_child_samples=80,
                reg_lambda=5.0,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        (
            "LightGBM_leaves31_spw025",
            "native",
            LGBMClassifier(
                objective="binary",
                n_estimators=1600,
                learning_rate=0.015,
                num_leaves=31,
                min_child_samples=80,
                reg_lambda=5.0,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=pos_weight * 0.25,
                random_state=RANDOM_STATE + 1,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        (
            "LightGBM_leaves63_spw015",
            "native",
            LGBMClassifier(
                objective="binary",
                n_estimators=1400,
                learning_rate=0.015,
                num_leaves=63,
                min_child_samples=120,
                reg_lambda=10.0,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight * 0.15,
                random_state=RANDOM_STATE + 2,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        (
            "XGBoost_depth4_spw1",
            "native",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                n_estimators=900,
                learning_rate=0.025,
                max_depth=4,
                min_child_weight=20,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=10.0,
                tree_method="hist",
                enable_categorical=True,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost_depth3_spw025",
            "native",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                n_estimators=1100,
                learning_rate=0.02,
                max_depth=3,
                min_child_weight=30,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=15.0,
                scale_pos_weight=pos_weight * 0.25,
                tree_method="hist",
                enable_categorical=True,
                random_state=RANDOM_STATE + 3,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees_ordinal_balanced",
            "ordinal",
            ExtraTreesClassifier(
                n_estimators=700,
                max_depth=24,
                min_samples_leaf=15,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "RandomForest_ordinal_balanced",
            "ordinal",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_leaf=25,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "HistGB_targetenc_balanced",
            "target",
            HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.025,
                max_leaf_nodes=31,
                min_samples_leaf=60,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "LogReg_targetenc_balanced",
            "target",
            LogisticRegression(
                max_iter=1000,
                C=0.25,
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    for model_name, prep_kind, estimator in model_specs:
        start = time.perf_counter()
        try:
            if prep_kind == "native":
                if model_name.startswith("LightGBM"):
                    estimator.fit(
                        native_train,
                        y_train,
                        eval_set=[(native_val, y_val)],
                        categorical_feature=cat_cols,
                        callbacks=[early_stopping(100, verbose=False)],
                    )
                else:
                    estimator.fit(
                        native_train,
                        y_train,
                        eval_set=[(native_val, y_val)],
                        verbose=False,
                    )
                val_score = estimator.predict_proba(native_val)[:, 1]
                test_score = estimator.predict_proba(native_test)[:, 1]
            elif prep_kind == "ordinal":
                model = make_ordinal_pipeline(estimator, X_train_raw)
                model.fit(X_train_raw, y_train)
                val_score = model.predict_proba(X_val_raw)[:, 1]
                test_score = model.predict_proba(X_test_raw)[:, 1]
            elif prep_kind == "target":
                model = make_target_encoder_pipeline(estimator, X_train_raw)
                model.fit(X_train_raw, y_train)
                val_score = model.predict_proba(X_val_raw)[:, 1]
                test_score = model.predict_proba(X_test_raw)[:, 1]
            else:
                raise ValueError(prep_kind)
            fit_seconds = time.perf_counter() - start
            score_bank[model_name] = {"val": val_score, "test": test_score}
            common = {
                "experiment_type": "history_heterogeneous_member",
                "ensemble_name": "",
                "member_key": model_name,
                "feature_config": cfg.name,
                "model_name": model_name,
                "prep_kind": prep_kind,
                "n_members": 1,
                "n_columns": X.shape[1],
                "fit_seconds": fit_seconds,
                "status": "ok",
                "error": "",
            }
            add_rows(rows, y_val, val_score, {**common, "split": "validation"})
            add_rows(rows, y_test, test_score, {**common, "split": "test"})
            print(
                f"{model_name}: val PR-AUC={average_precision_score(y_val, val_score):.4f}, "
                f"test PR-AUC={average_precision_score(y_test, test_score):.4f}, fit={fit_seconds:.1f}s",
                flush=True,
            )
        except Exception as exc:
            fit_seconds = time.perf_counter() - start
            rows.append(
                {
                    "experiment_type": "history_heterogeneous_member",
                    "split": "validation",
                    "ensemble_name": "",
                    "member_key": model_name,
                    "feature_config": cfg.name,
                    "model_name": model_name,
                    "prep_kind": prep_kind,
                    "threshold_strategy": "failed",
                    "n_members": 1,
                    "n_columns": X.shape[1],
                    "fit_seconds": fit_seconds,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"{model_name}: FAILED {exc}", flush=True)

    member_df = pd.DataFrame(rows)
    ok_validation = (
        member_df.query("split == 'validation' and threshold_strategy == 'best_f1' and status == 'ok'")
        .sort_values(["pr_auc", "f1"], ascending=False)
    )
    top_keys = ok_validation["member_key"].head(8).tolist()
    print("\nTop heterogeneous validation members:")
    print(ok_validation[["member_key", "pr_auc", "roc_auc", "f1"]].head(10).to_string(index=False), flush=True)

    ensemble_rows = []
    for size in [2, 3, 5, 8]:
        keys = top_keys[:size]
        if len(keys) < 2:
            continue
        for blend_type in ["score_average", "rank_average"]:
            val_scores = [score_bank[k]["val"] for k in keys]
            test_scores = [score_bank[k]["test"] for k in keys]
            val_blend = np.mean(val_scores, axis=0) if blend_type == "score_average" else rank_average(val_scores)
            test_blend = np.mean(test_scores, axis=0) if blend_type == "score_average" else rank_average(test_scores)
            name = f"{blend_type}_top{size}_heterogeneous"
            common = {
                "experiment_type": "history_heterogeneous_ensemble",
                "ensemble_name": name,
                "member_key": "|".join(keys),
                "feature_config": cfg.name,
                "model_name": blend_type,
                "prep_kind": "mixed",
                "n_members": len(keys),
                "n_columns": X.shape[1],
                "fit_seconds": 0.0,
                "status": "ok",
                "error": "",
            }
            add_rows(ensemble_rows, y_val, val_blend, {**common, "split": "validation"})
            add_rows(ensemble_rows, y_test, test_blend, {**common, "split": "test"})
            lift.extend(lift_rows(y_test, test_blend, {**common, "split": "test", "threshold_strategy": "ranking"}))

    # Logistic stacker trained on validation predictions, exploratory because it
    # uses the validation labels directly as meta-training data.
    if len(top_keys) >= 3:
        for size in [3, 5, min(8, len(top_keys))]:
            keys = top_keys[:size]
            val_stack = np.column_stack([score_bank[k]["val"] for k in keys])
            test_stack = np.column_stack([score_bank[k]["test"] for k in keys])
            for c in [0.1, 1.0]:
                meta = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
                meta.fit(val_stack, y_val)
                val_score = meta.predict_proba(val_stack)[:, 1]
                test_score = meta.predict_proba(test_stack)[:, 1]
                name = f"logistic_stack_top{size}_C{c:g}"
                common = {
                    "experiment_type": "history_heterogeneous_stacker",
                    "ensemble_name": name,
                    "member_key": "|".join(keys),
                    "feature_config": cfg.name,
                    "model_name": "LogisticStacker",
                    "prep_kind": "stacked",
                    "n_members": len(keys),
                    "n_columns": X.shape[1],
                    "fit_seconds": 0.0,
                    "status": "ok",
                    "error": "",
                }
                add_rows(ensemble_rows, y_val, val_score, {**common, "split": "validation"})
                add_rows(ensemble_rows, y_test, test_score, {**common, "split": "test"})
                lift.extend(lift_rows(y_test, test_score, {**common, "split": "test", "threshold_strategy": "ranking"}))

    all_rows = pd.concat([member_df, pd.DataFrame(ensemble_rows)], ignore_index=True)
    all_rows.to_csv(RESULTS_DIR / "history_heterogeneous_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "history_heterogeneous_lift_tables.csv", index=False)

    cols = ["experiment_type", "ensemble_name", "member_key", "threshold_strategy", "pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy"]
    print("\nTop validation rows:")
    print(all_rows.query("split == 'validation' and status == 'ok'").sort_values(["pr_auc", "f1"], ascending=False)[cols].head(20).to_string(index=False), flush=True)
    print("\nTop test rows:")
    print(all_rows.query("split == 'test' and status == 'ok'").sort_values(["pr_auc", "f1"], ascending=False)[cols].head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
