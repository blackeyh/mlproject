from __future__ import annotations

import time
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modeling_experiments import (
    RANDOM_STATE,
    RESULTS_DIR,
    FeatureConfig,
    RareCategoryGrouper,
    best_thresholds,
    build_feature_matrix,
    get_scores,
    load_scoped_data,
    make_pipeline,
    rare_columns_for,
    threshold_metrics,
)
from native_catboost_search import prepare_catboost_frames
from neural_network_search import EmbeddingMLP, TabularEncoder


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def feature_configs():
    return [
        FeatureConfig(
            name="imb_raw_admin_age_paper_summaries_only_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="category",
            medication_mode="summaries_only",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="imb_raw_admin_age_paper_weight_indicator_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="indicator",
            medication_mode="summaries_only",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="imb_raw_admin_age_paper_weight_category_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="category",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="imb_raw_admin_raw_age_weight_category_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="raw",
            gender_mode="keep",
            weight_mode="category",
            utilization_mode="log_plus_raw",
        ),
    ]


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


def majority_row(y_train, y_eval, split):
    y_score = np.full(len(y_eval), y_train.mean())
    row = threshold_metrics(y_eval, y_score, 0.5)
    row.update(
        {
            "split": split,
            "experiment_family": "baseline",
            "feature_config": "baseline_no_features",
            "model_name": "MajorityBaseline",
            "threshold_strategy": "most_frequent",
            "fit_seconds": 0.0,
            "status": "ok",
            "error": "",
        }
    )
    return row


def add_threshold_rows(rows, y_true, y_score, metadata):
    for strategy, threshold in best_thresholds(y_true, y_score).items():
        row = threshold_metrics(y_true, y_score, threshold)
        row.update(metadata)
        row.update(
            {
                "threshold_strategy": strategy,
                "status": "ok",
                "error": "",
            }
        )
        rows.append(row)


def select_validation_candidates(results, max_rows=30):
    ok = results[(results["split"] == "validation") & (results["status"] == "ok")].copy()
    non_baseline = ok[ok["model_name"] != "MajorityBaseline"].copy()
    frames = [
        non_baseline.sort_values(["pr_auc", "f1"], ascending=False).head(10),
        non_baseline[non_baseline["threshold_strategy"] == "best_f1"]
        .sort_values(["f1", "pr_auc"], ascending=False)
        .head(10),
        non_baseline[non_baseline["threshold_strategy"] == "max_recall_precision_ge_0.20"]
        .sort_values(["f1", "pr_auc"], ascending=False)
        .head(10),
        non_baseline[non_baseline["threshold_strategy"] == "max_recall_precision_ge_0.15"]
        .sort_values(["recall", "pr_auc"], ascending=False)
        .head(10),
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["feature_config", "model_name", "threshold_strategy"])
        .head(max_rows)
    )


def lift_rows(y_true, y_score, metadata, fractions=(0.01, 0.05, 0.10, 0.20)):
    y_array = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))
    total_pos = max(1, int(y_array.sum()))
    base_rate = float(y_array.mean())
    rows = []
    for frac in fractions:
        n_flagged = max(1, int(np.ceil(len(y_array) * frac)))
        selected = order[:n_flagged]
        positives = int(y_array[selected].sum())
        precision = positives / n_flagged
        recall = positives / total_pos
        row = {
            **metadata,
            "top_fraction": frac,
            "n_flagged": n_flagged,
            "positives_captured": positives,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "base_rate": base_rate,
            "lift": precision / base_rate if base_rate > 0 else np.nan,
        }
        rows.append(row)
    return rows


class MixedOneHotEncoder:
    def __init__(self, rare_min_count=100):
        self.rare_min_count = rare_min_count

    def fit(self, X):
        X = X.copy()
        self.rare_cols_ = rare_columns_for(X)
        self.grouper_ = RareCategoryGrouper(columns=self.rare_cols_, min_count=self.rare_min_count)
        X = self.grouper_.fit_transform(X)
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.num_medians_ = X[self.num_cols_].median()
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.num_cols_].fillna(self.num_medians_))
        self.ordinal_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
        X_cat = X[self.cat_cols_].fillna("Missing").astype(str)
        self.ordinal_.fit(X_cat)
        X_cat_ord = self.ordinal_.transform(X_cat).astype(np.int64)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.ohe_.fit(X_cat_ord)
        return self

    def transform_parts(self, X):
        X = self.grouper_.transform(X.copy())
        X_cat = X[self.cat_cols_].fillna("Missing").astype(str)
        X_cat_ord = self.ordinal_.transform(X_cat).astype(np.int64)
        X_num = self.scaler_.transform(X[self.num_cols_].fillna(self.num_medians_)).astype(np.float32)
        return X_cat_ord, X_num

    def transform_ohe(self, X):
        X_cat_ord, X_num = self.transform_parts(X)
        return sparse.hstack([self.ohe_.transform(X_cat_ord), sparse.csr_matrix(X_num)], format="csr")

    def combine_resampled(self, X_cat_ord, X_num):
        X_cat_ord = np.rint(X_cat_ord).astype(np.int64)
        return sparse.hstack([self.ohe_.transform(X_cat_ord), sparse.csr_matrix(X_num)], format="csr")


def sampled_model_specs():
    return [
        {
            "model_name": "RandomOver035_Logistic_C0.2",
            "sampler": RandomOverSampler(sampling_strategy=0.35, random_state=RANDOM_STATE),
            "sampler_kind": "ohe",
            "model": LogisticRegression(max_iter=1200, C=0.2, solver="liblinear", random_state=RANDOM_STATE),
        },
        {
            "model_name": "RandomOver050_XGBoost_d4",
            "sampler": RandomOverSampler(sampling_strategy=0.50, random_state=RANDOM_STATE),
            "sampler_kind": "ohe",
            "model": XGBClassifier(
                n_estimators=650,
                learning_rate=0.018,
                max_depth=4,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
                scale_pos_weight=1.0,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "model_name": "RandomUnder035_LightGBM",
            "sampler": RandomUnderSampler(sampling_strategy=0.35, random_state=RANDOM_STATE),
            "sampler_kind": "ohe",
            "model": LGBMClassifier(
                objective="binary",
                n_estimators=650,
                learning_rate=0.018,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        },
        {
            "model_name": "RandomUnder050_XGBoost_d4",
            "sampler": RandomUnderSampler(sampling_strategy=0.50, random_state=RANDOM_STATE),
            "sampler_kind": "ohe",
            "model": XGBClassifier(
                n_estimators=650,
                learning_rate=0.018,
                max_depth=4,
                min_child_weight=6,
                subsample=0.90,
                colsample_bytree=0.90,
                reg_lambda=2.0,
                scale_pos_weight=1.0,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "model_name": "SMOTENC025_Logistic_C0.2",
            "sampler": SMOTENC(
                categorical_features="auto",
                sampling_strategy=0.25,
                k_neighbors=5,
                random_state=RANDOM_STATE,
            ),
            "sampler_kind": "smotenc",
            "model": LogisticRegression(max_iter=1200, C=0.2, solver="liblinear", random_state=RANDOM_STATE),
        },
        {
            "model_name": "SMOTENC035_XGBoost_d4",
            "sampler": SMOTENC(
                categorical_features="auto",
                sampling_strategy=0.35,
                k_neighbors=5,
                random_state=RANDOM_STATE,
            ),
            "sampler_kind": "smotenc",
            "model": XGBClassifier(
                n_estimators=650,
                learning_rate=0.018,
                max_depth=4,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
                scale_pos_weight=1.0,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "model_name": "SMOTENC035_LightGBM",
            "sampler": SMOTENC(
                categorical_features="auto",
                sampling_strategy=0.35,
                k_neighbors=5,
                random_state=RANDOM_STATE,
            ),
            "sampler_kind": "smotenc",
            "model": LGBMClassifier(
                objective="binary",
                n_estimators=650,
                learning_rate=0.018,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        },
    ]


def make_smote_input(encoder, X):
    X_cat_ord, X_num = encoder.transform_parts(X)
    X_cat_df = pd.DataFrame(
        X_cat_ord,
        columns=[f"cat_{i}" for i in range(X_cat_ord.shape[1])],
    ).astype("category")
    X_num_df = pd.DataFrame(
        X_num,
        columns=[f"num_{i}" for i in range(X_num.shape[1])],
    )
    return pd.concat([X_cat_df, X_num_df], axis=1)


def sampled_fit_predict(X_train, y_train, X_val, X_test, cfg, spec):
    encoder = MixedOneHotEncoder(rare_min_count=cfg.rare_min_count).fit(X_train)
    sampler = spec["sampler"]
    model = clone(spec["model"])

    if spec["sampler_kind"] == "smotenc":
        X_train_mix = make_smote_input(encoder, X_train)
        X_res, y_res = sampler.fit_resample(X_train_mix, y_train.to_numpy())
        n_cat = len(encoder.cat_cols_)
        X_cat_res = X_res.iloc[:, :n_cat].to_numpy(dtype=np.int64)
        X_num_res = X_res.iloc[:, n_cat:].to_numpy(dtype=np.float32)
        X_fit = encoder.combine_resampled(X_cat_res, X_num_res)
    else:
        X_fit_base = encoder.transform_ohe(X_train)
        X_fit, y_res = sampler.fit_resample(X_fit_base, y_train.to_numpy())

    model.fit(X_fit, y_res)
    val_score = model.predict_proba(encoder.transform_ohe(X_val))[:, 1]
    test_score = model.predict_proba(encoder.transform_ohe(X_test))[:, 1]
    return val_score, test_score


def weighted_specs(pos_weight):
    specs = []
    for factor in [0.50, 0.75, 1.00, 1.25, 1.50]:
        spw = pos_weight * factor
        specs.append(
            {
                "model_name": f"XGBoost_d4_spw{factor:.2f}",
                "model": XGBClassifier(
                    n_estimators=750,
                    learning_rate=0.016,
                    max_depth=4,
                    min_child_weight=8,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=3.0,
                    scale_pos_weight=spw,
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                "scale_numeric": False,
                "calibrated": False,
            }
        )
        specs.append(
            {
                "model_name": f"LightGBM_spw{factor:.2f}",
                "model": LGBMClassifier(
                    objective="binary",
                    n_estimators=800,
                    learning_rate=0.016,
                    num_leaves=31,
                    min_child_samples=50,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    scale_pos_weight=spw,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                ),
                "scale_numeric": False,
                "calibrated": False,
            }
        )
    for weight in [np.sqrt(pos_weight), pos_weight, pos_weight * 1.5]:
        specs.append(
            {
                "model_name": f"Logistic_C0.2_class_weight_{weight:.2f}",
                "model": LogisticRegression(
                    max_iter=1200,
                    solver="liblinear",
                    C=0.2,
                    class_weight={0: 1.0, 1: float(weight)},
                    random_state=RANDOM_STATE,
                ),
                "scale_numeric": True,
                "calibrated": False,
            }
        )
    base_xgb = XGBClassifier(
        n_estimators=550,
        learning_rate=0.018,
        max_depth=4,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    specs.append(
        {
            "model_name": "CalibratedSigmoid_XGBoost_d4_spw1.00",
            "model": CalibratedClassifierCV(estimator=base_xgb, method="sigmoid", cv=3, n_jobs=None),
            "scale_numeric": False,
            "calibrated": True,
        }
    )
    base_lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=550,
        learning_rate=0.018,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    specs.append(
        {
            "model_name": "CalibratedSigmoid_LightGBM_spw1.00",
            "model": CalibratedClassifierCV(estimator=base_lgbm, method="sigmoid", cv=3, n_jobs=None),
            "scale_numeric": False,
            "calibrated": True,
        }
    )
    return specs


def balanced_ensemble_specs():
    return [
        {
            "model_name": "BalancedRandomForest_depth12_leaf30",
            "model": BalancedRandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=30,
                sampling_strategy="all",
                replacement=True,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale_numeric": False,
        },
        {
            "model_name": "BalancedRandomForest_depth18_leaf20",
            "model": BalancedRandomForestClassifier(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=20,
                sampling_strategy="all",
                replacement=True,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale_numeric": False,
        },
        {
            "model_name": "EasyEnsemble_AdaBoost_12x80",
            "model": EasyEnsembleClassifier(
                n_estimators=12,
                estimator=AdaBoostClassifier(
                    n_estimators=80,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE,
                ),
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale_numeric": False,
        },
        {
            "model_name": "RUSBoost_260_lr0.04",
            "model": RUSBoostClassifier(
                n_estimators=260,
                learning_rate=0.04,
                random_state=RANDOM_STATE,
            ),
            "scale_numeric": False,
        },
    ]


def native_catboost_custom_specs(pos_weight):
    specs = []
    for factor in [0.50, 0.75, 1.00, 1.25, 1.50]:
        weight = pos_weight * factor
        specs.append(
            {
                "model_name": f"NativeCatBoost_d5_customPW{factor:.2f}",
                "model": CatBoostClassifier(
                    iterations=1000,
                    learning_rate=0.022,
                    depth=5,
                    l2_leaf_reg=8.0,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    class_weights=[1.0, float(weight)],
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                ),
            }
        )
    return specs


def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    alpha_t = torch.where(targets == 1, torch.tensor(alpha, device=targets.device), torch.tensor(1 - alpha, device=targets.device))
    return (alpha_t * (1 - pt).pow(gamma) * bce).mean()


def train_focal_mlp(model, Xn_train, Xc_train, y_train, Xn_val, Xc_val, y_val, cfg):
    device = torch.device("cpu")
    model.to(device)
    train_ds = TensorDataset(
        torch.tensor(Xn_train, dtype=torch.float32),
        torch.tensor(Xc_train, dtype=torch.long),
        torch.tensor(y_train.to_numpy(), dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    Xn_val_t = torch.tensor(Xn_val, dtype=torch.float32, device=device)
    Xc_val_t = torch.tensor(Xc_val, dtype=torch.long, device=device)
    best_score = -np.inf
    best_state = None
    bad_epochs = 0
    start = time.perf_counter()
    for _ in range(cfg["max_epochs"]):
        model.train()
        for xb_num, xb_cat, yb in loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = focal_loss(model(xb_num, xb_cat), yb, alpha=cfg["alpha"], gamma=cfg["gamma"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xn_val_t, Xc_val_t).cpu().numpy()
            val_score = 1 / (1 + np.exp(-logits))
        pr_auc = threshold_metrics(y_val, val_score, 0.5)["pr_auc"]
        if pr_auc > best_score + 1e-5:
            best_score = pr_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= cfg["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, time.perf_counter() - start


def torch_scores(model, X_num, X_cat):
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(X_num, dtype=torch.float32),
            torch.tensor(X_cat, dtype=torch.long),
        ).cpu().numpy()
    return 1 / (1 + np.exp(-logits))


def focal_specs():
    return [
        {
            "model_name": "FocalEmbeddingMLP_512_256_gamma2_alpha0.75",
            "hidden": (512, 256),
            "dropout": 0.30,
            "alpha": 0.75,
            "gamma": 2.0,
            "lr": 8e-4,
            "weight_decay": 1e-4,
            "batch_size": 1024,
            "max_epochs": 40,
            "patience": 6,
        },
        {
            "model_name": "FocalEmbeddingMLP_512_256_128_gamma2_alpha0.80",
            "hidden": (512, 256, 128),
            "dropout": 0.35,
            "alpha": 0.80,
            "gamma": 2.0,
            "lr": 7e-4,
            "weight_decay": 1e-4,
            "batch_size": 1024,
            "max_epochs": 40,
            "patience": 6,
        },
    ]


def run():
    scoped = load_scoped_data()
    _, y = build_feature_matrix(scoped, FeatureConfig(name="split_reference"))
    train_idx, val_idx, test_idx, y_train, y_val, y_test = split_indices(y)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    all_rows = [majority_row(y_train, y_val, "validation")]
    fitted = {}
    lift = []

    cfgs = feature_configs()
    sampled_specs = sampled_model_specs()
    weighted_model_specs = weighted_specs(pos_weight)
    balanced_specs = balanced_ensemble_specs()
    native_specs = native_catboost_custom_specs(pos_weight)
    focal_model_specs = focal_specs()

    for cfg in cfgs:
        X_cfg, _ = build_feature_matrix(scoped, cfg)
        X_train = X_cfg.iloc[train_idx].copy()
        X_val = X_cfg.iloc[val_idx].copy()
        X_test = X_cfg.iloc[test_idx].copy()
        rare_cols = rare_columns_for(X_train)
        print(f"\n=== Imbalance feature config: {cfg.name} ({X_cfg.shape[1]} columns) ===")

        for spec in weighted_model_specs:
            start = time.perf_counter()
            try:
                estimator = make_pipeline(
                    model=clone(spec["model"]),
                    X_train=X_train,
                    rare_cols=rare_cols,
                    rare_min_count=cfg.rare_min_count,
                    scale_numeric=spec["scale_numeric"],
                )
                estimator.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - start
                val_score = get_scores(estimator, X_val)
                test_score = get_scores(estimator, X_test)
                metadata = {
                    "split": "validation",
                    "experiment_family": "weighted_or_calibrated",
                    "feature_config": cfg.name,
                    "model_name": spec["model_name"],
                    "fit_seconds": fit_seconds,
                    **asdict(cfg),
                }
                add_threshold_rows(all_rows, y_val, val_score, metadata)
                fitted[(cfg.name, spec["model_name"])] = {
                    "test_score": test_score,
                    "fit_seconds": fit_seconds,
                    "experiment_family": "weighted_or_calibrated",
                }
                best = pd.DataFrame([r for r in all_rows if r.get("feature_config") == cfg.name and r.get("model_name") == spec["model_name"]]).sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
                print(f"{spec['model_name']}: PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, recall={best.recall:.4f}, precision={best.precision:.4f}, fit={fit_seconds:.1f}s")
            except Exception as exc:
                all_rows.append(
                    {
                        "split": "validation",
                        "experiment_family": "weighted_or_calibrated",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "threshold_strategy": "failed",
                        "fit_seconds": time.perf_counter() - start,
                        "status": "failed",
                        "error": str(exc),
                        **asdict(cfg),
                    }
                )
                print(f"{spec['model_name']}: FAILED: {exc}")

        for spec in balanced_specs:
            start = time.perf_counter()
            try:
                estimator = make_pipeline(
                    model=clone(spec["model"]),
                    X_train=X_train,
                    rare_cols=rare_cols,
                    rare_min_count=cfg.rare_min_count,
                    scale_numeric=spec["scale_numeric"],
                )
                estimator.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - start
                val_score = get_scores(estimator, X_val)
                test_score = get_scores(estimator, X_test)
                metadata = {
                    "split": "validation",
                    "experiment_family": "balanced_ensemble",
                    "feature_config": cfg.name,
                    "model_name": spec["model_name"],
                    "fit_seconds": fit_seconds,
                    **asdict(cfg),
                }
                add_threshold_rows(all_rows, y_val, val_score, metadata)
                fitted[(cfg.name, spec["model_name"])] = {
                    "test_score": test_score,
                    "fit_seconds": fit_seconds,
                    "experiment_family": "balanced_ensemble",
                }
                best = pd.DataFrame([r for r in all_rows if r.get("feature_config") == cfg.name and r.get("model_name") == spec["model_name"]]).sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
                print(f"{spec['model_name']}: PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, recall={best.recall:.4f}, precision={best.precision:.4f}, fit={fit_seconds:.1f}s")
            except Exception as exc:
                all_rows.append(
                    {
                        "split": "validation",
                        "experiment_family": "balanced_ensemble",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "threshold_strategy": "failed",
                        "fit_seconds": time.perf_counter() - start,
                        "status": "failed",
                        "error": str(exc),
                        **asdict(cfg),
                    }
                )
                print(f"{spec['model_name']}: FAILED: {exc}")

        for spec in sampled_specs:
            start = time.perf_counter()
            try:
                val_score, test_score = sampled_fit_predict(X_train, y_train, X_val, X_test, cfg, spec)
                fit_seconds = time.perf_counter() - start
                metadata = {
                    "split": "validation",
                    "experiment_family": "training_resampling",
                    "feature_config": cfg.name,
                    "model_name": spec["model_name"],
                    "fit_seconds": fit_seconds,
                    **asdict(cfg),
                }
                add_threshold_rows(all_rows, y_val, val_score, metadata)
                fitted[(cfg.name, spec["model_name"])] = {
                    "test_score": test_score,
                    "fit_seconds": fit_seconds,
                    "experiment_family": "training_resampling",
                }
                best = pd.DataFrame([r for r in all_rows if r.get("feature_config") == cfg.name and r.get("model_name") == spec["model_name"]]).sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
                print(f"{spec['model_name']}: PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, recall={best.recall:.4f}, precision={best.precision:.4f}, fit={fit_seconds:.1f}s")
            except Exception as exc:
                all_rows.append(
                    {
                        "split": "validation",
                        "experiment_family": "training_resampling",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "threshold_strategy": "failed",
                        "fit_seconds": time.perf_counter() - start,
                        "status": "failed",
                        "error": str(exc),
                        **asdict(cfg),
                    }
                )
                print(f"{spec['model_name']}: FAILED: {exc}")

        if "summaries_only" in cfg.name or "weight_indicator" in cfg.name:
            X_train_cat, X_val_cat, cat_features = prepare_catboost_frames(
                X_train.copy(), X_val.copy(), rare_min_count=cfg.rare_min_count
            )
            _, X_test_cat, _ = prepare_catboost_frames(
                X_train.copy(), X_test.copy(), rare_min_count=cfg.rare_min_count
            )
            for spec in native_specs:
                start = time.perf_counter()
                try:
                    model = spec["model"].copy()
                    model.fit(X_train_cat, y_train, cat_features=cat_features)
                    fit_seconds = time.perf_counter() - start
                    val_score = model.predict_proba(X_val_cat)[:, 1]
                    test_score = model.predict_proba(X_test_cat)[:, 1]
                    metadata = {
                        "split": "validation",
                        "experiment_family": "native_catboost_custom_weights",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "fit_seconds": fit_seconds,
                        **asdict(cfg),
                    }
                    add_threshold_rows(all_rows, y_val, val_score, metadata)
                    fitted[(cfg.name, spec["model_name"])] = {
                        "test_score": test_score,
                        "fit_seconds": fit_seconds,
                        "experiment_family": "native_catboost_custom_weights",
                    }
                    best = pd.DataFrame([r for r in all_rows if r.get("feature_config") == cfg.name and r.get("model_name") == spec["model_name"]]).sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
                    print(f"{spec['model_name']}: PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, recall={best.recall:.4f}, precision={best.precision:.4f}, fit={fit_seconds:.1f}s")
                except Exception as exc:
                    all_rows.append(
                        {
                            "split": "validation",
                            "experiment_family": "native_catboost_custom_weights",
                            "feature_config": cfg.name,
                            "model_name": spec["model_name"],
                            "threshold_strategy": "failed",
                            "fit_seconds": time.perf_counter() - start,
                            "status": "failed",
                            "error": str(exc),
                            **asdict(cfg),
                        }
                    )
                    print(f"{spec['model_name']}: FAILED: {exc}")

        if "summaries_only" in cfg.name or "weight_indicator" in cfg.name:
            encoder = TabularEncoder(rare_min_count=cfg.rare_min_count).fit(X_train)
            Xn_train, Xc_train = encoder.transform(X_train)
            Xn_val, Xc_val = encoder.transform(X_val)
            Xn_test, Xc_test = encoder.transform(X_test)
            for spec in focal_model_specs:
                start = time.perf_counter()
                try:
                    model = EmbeddingMLP(
                        encoder.cat_dims,
                        Xn_train.shape[1],
                        hidden=spec["hidden"],
                        dropout=spec["dropout"],
                    )
                    model, fit_seconds = train_focal_mlp(
                        model,
                        Xn_train,
                        Xc_train,
                        y_train,
                        Xn_val,
                        Xc_val,
                        y_val,
                        spec,
                    )
                    val_score = torch_scores(model, Xn_val, Xc_val)
                    test_score = torch_scores(model, Xn_test, Xc_test)
                    metadata = {
                        "split": "validation",
                        "experiment_family": "focal_neural_network",
                        "feature_config": cfg.name,
                        "model_name": spec["model_name"],
                        "fit_seconds": fit_seconds,
                        **asdict(cfg),
                    }
                    add_threshold_rows(all_rows, y_val, val_score, metadata)
                    fitted[(cfg.name, spec["model_name"])] = {
                        "test_score": test_score,
                        "fit_seconds": fit_seconds,
                        "experiment_family": "focal_neural_network",
                    }
                    best = pd.DataFrame([r for r in all_rows if r.get("feature_config") == cfg.name and r.get("model_name") == spec["model_name"]]).sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
                    print(f"{spec['model_name']}: PR-AUC={best.pr_auc:.4f}, F1={best.f1:.4f}, recall={best.recall:.4f}, precision={best.precision:.4f}, fit={fit_seconds:.1f}s")
                except Exception as exc:
                    all_rows.append(
                        {
                            "split": "validation",
                            "experiment_family": "focal_neural_network",
                            "feature_config": cfg.name,
                            "model_name": spec["model_name"],
                            "threshold_strategy": "failed",
                            "fit_seconds": time.perf_counter() - start,
                            "status": "failed",
                            "error": str(exc),
                            **asdict(cfg),
                        }
                    )
                    print(f"{spec['model_name']}: FAILED: {exc}")

        pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "imbalance_validation_results.csv", index=False)

    validation = pd.DataFrame(all_rows)
    validation.to_csv(RESULTS_DIR / "imbalance_validation_results.csv", index=False)
    selected = select_validation_candidates(validation, max_rows=30)
    selected.to_csv(RESULTS_DIR / "imbalance_selected_for_test.csv", index=False)

    test_rows = [majority_row(y_train, y_test, "test")]
    for _, selected_row in selected.iterrows():
        key = (selected_row["feature_config"], selected_row["model_name"])
        item = fitted[key]
        test_score = item["test_score"]
        metrics = threshold_metrics(y_test, test_score, selected_row["threshold"])
        metrics.update(
            {
                "split": "test",
                "experiment_family": item["experiment_family"],
                "feature_config": selected_row["feature_config"],
                "model_name": selected_row["model_name"],
                "threshold_strategy": selected_row["threshold_strategy"],
                "fit_seconds": item["fit_seconds"],
                "status": "ok",
                "error": "",
                "selected_validation_pr_auc": selected_row["pr_auc"],
                "selected_validation_recall": selected_row["recall"],
                "selected_validation_precision": selected_row["precision"],
                "selected_validation_f1": selected_row["f1"],
            }
        )
        test_rows.append(metrics)
        lift.extend(
            lift_rows(
                y_test,
                test_score,
                {
                    "split": "test",
                    "experiment_family": item["experiment_family"],
                    "feature_config": selected_row["feature_config"],
                    "model_name": selected_row["model_name"],
                    "threshold_strategy": selected_row["threshold_strategy"],
                },
            )
        )

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(RESULTS_DIR / "imbalance_test_results.csv", index=False)
    pd.DataFrame(lift).to_csv(RESULTS_DIR / "imbalance_lift_tables.csv", index=False)

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
    print("\nImbalance top validation by PR-AUC:")
    print(ok.sort_values(["pr_auc", "f1"], ascending=False)[cols].head(30).to_string(index=False))
    print("\nImbalance top validation by F1:")
    print(ok.sort_values(["f1", "pr_auc"], ascending=False)[cols].head(30).to_string(index=False))
    print("\nImbalance selected test results:")
    print(test_results.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp"]].to_string(index=False))


if __name__ == "__main__":
    run()
