from __future__ import annotations

import time
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception:  # pragma: no cover
    TabNetClassifier = None

from modeling_experiments import (
    RANDOM_STATE,
    RESULTS_DIR,
    FeatureConfig,
    RareCategoryGrouper,
    best_thresholds,
    build_feature_matrix,
    load_scoped_data,
    rare_columns_for,
    threshold_metrics,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def feature_configs():
    return [
        FeatureConfig(
            name="nn_raw_admin_age_paper_summaries_only_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="category",
            medication_mode="summaries_only",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="nn_raw_admin_raw_age_weight_category_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="raw",
            gender_mode="keep",
            weight_mode="category",
            utilization_mode="log_plus_raw",
        ),
        FeatureConfig(
            name="nn_raw_admin_age_paper_weight_indicator_rare100",
            rare_min_count=100,
            admin_mode="raw_ids",
            age_mode="paper",
            gender_mode="keep",
            weight_mode="indicator",
            medication_mode="summaries_only",
            utilization_mode="log_plus_raw",
        ),
    ]


class TabularEncoder:
    def __init__(self, rare_min_count=100):
        self.rare_min_count = rare_min_count

    def fit(self, X):
        X = X.copy()
        self.rare_cols_ = rare_columns_for(X)
        self.grouper_ = RareCategoryGrouper(columns=self.rare_cols_, min_count=self.rare_min_count)
        X = self.grouper_.fit_transform(X)
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.vocabs_ = {}
        for col in self.cat_cols_:
            values = X[col].fillna("Missing").astype(str)
            cats = ["__UNK__"] + sorted(values.unique().tolist())
            self.vocabs_[col] = {cat: i for i, cat in enumerate(cats)}
        self.num_medians_ = X[self.num_cols_].median()
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.num_cols_].fillna(self.num_medians_))
        return self

    def transform(self, X):
        X = self.grouper_.transform(X.copy())
        cat_arrays = []
        for col in self.cat_cols_:
            vocab = self.vocabs_[col]
            ids = X[col].fillna("Missing").astype(str).map(vocab).fillna(0).astype("int64").to_numpy()
            cat_arrays.append(ids)
        X_cat = np.vstack(cat_arrays).T if cat_arrays else np.zeros((len(X), 0), dtype="int64")
        X_num = self.scaler_.transform(X[self.num_cols_].fillna(self.num_medians_)).astype("float32")
        return X_num, X_cat

    @property
    def cat_dims(self):
        return [len(self.vocabs_[col]) for col in self.cat_cols_]


class EmbeddingMLP(nn.Module):
    def __init__(self, cat_dims, n_num, hidden=(256, 128), dropout=0.25):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, min(50, max(4, int(round(dim**0.5 * 2))))) for dim in cat_dims]
        )
        emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        layers = []
        in_dim = emb_dim + n_num
        for h in hidden:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        if len(self.embeddings):
            emb = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
            x = torch.cat([x_num, emb], dim=1)
        else:
            x = x_num
        return self.net(x).squeeze(1)


class FTTransformerLite(nn.Module):
    def __init__(self, cat_dims, n_num, d_model=64, n_heads=4, n_layers=2, dropout=0.15):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(dim, d_model) for dim in cat_dims])
        self.num_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_num)])
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_num, x_cat):
        tokens = []
        if len(self.cat_embeddings):
            tokens.extend([emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)])
        tokens.extend([layer(x_num[:, i : i + 1]).unsqueeze(1) for i, layer in enumerate(self.num_embeddings)])
        x = torch.cat([self.cls.expand(x_num.shape[0], -1, -1)] + tokens, dim=1)
        x = self.encoder(x)
        return self.head(x[:, 0]).squeeze(1)


def dummy_row(y_train, y_eval, split):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = dummy.predict(np.zeros((len(y_eval), 1)))
    y_score = np.full(len(y_eval), y_train.mean())
    metrics = threshold_metrics(y_eval, y_score, 0.5)
    metrics.update(
        {
            "split": split,
            "feature_config": "baseline_no_features",
            "model_name": "MajorityBaseline",
            "threshold_strategy": "most_frequent",
            "fit_seconds": 0.0,
            "status": "ok",
            "error": "",
        }
    )
    return metrics


def train_torch_model(model, Xn_train, Xc_train, y_train, Xn_val, Xc_val, y_val, cfg):
    device = torch.device("cpu")
    model.to(device)
    batch_size = cfg["batch_size"]
    ds = TensorDataset(
        torch.tensor(Xn_train, dtype=torch.float32),
        torch.tensor(Xc_train, dtype=torch.long),
        torch.tensor(y_train.to_numpy(), dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    pos_weight = ((y_train == 0).sum() / (y_train == 1).sum()) * cfg["pos_weight_factor"]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    Xn_val_t = torch.tensor(Xn_val, dtype=torch.float32, device=device)
    Xc_val_t = torch.tensor(Xc_val, dtype=torch.long, device=device)
    best_score = -np.inf
    best_state = None
    bad_epochs = 0
    start = time.perf_counter()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for xb_num, xb_cat, yb in loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb_num, xb_cat), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xn_val_t, Xc_val_t).cpu().numpy()
            val_score = 1 / (1 + np.exp(-val_logits))
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
    fit_seconds = time.perf_counter() - start
    return model, fit_seconds, best_score


def torch_scores(model, X_num, X_cat):
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(X_num, dtype=torch.float32),
            torch.tensor(X_cat, dtype=torch.long),
        ).cpu().numpy()
    return 1 / (1 + np.exp(-logits))


def torch_specs(cat_dims, n_num):
    return [
        {
            "name": "EmbeddingMLP_256_128_do0.25_pw1.0",
            "kind": "torch",
            "model": lambda: EmbeddingMLP(cat_dims, n_num, hidden=(256, 128), dropout=0.25),
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 1024,
            "max_epochs": 45,
            "patience": 7,
            "pos_weight_factor": 1.0,
        },
        {
            "name": "EmbeddingMLP_512_256_do0.30_pw0.75",
            "kind": "torch",
            "model": lambda: EmbeddingMLP(cat_dims, n_num, hidden=(512, 256), dropout=0.30),
            "lr": 8e-4,
            "weight_decay": 2e-4,
            "batch_size": 1024,
            "max_epochs": 50,
            "patience": 8,
            "pos_weight_factor": 0.75,
        },
        {
            "name": "EmbeddingMLP_256_128_64_do0.20_pw0.5",
            "kind": "torch",
            "model": lambda: EmbeddingMLP(cat_dims, n_num, hidden=(256, 128, 64), dropout=0.20),
            "lr": 8e-4,
            "weight_decay": 1e-4,
            "batch_size": 1024,
            "max_epochs": 50,
            "patience": 8,
            "pos_weight_factor": 0.5,
        },
        {
            "name": "EmbeddingMLP_128_64_do0.15_pw0.5",
            "kind": "torch",
            "model": lambda: EmbeddingMLP(cat_dims, n_num, hidden=(128, 64), dropout=0.15),
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 1024,
            "max_epochs": 45,
            "patience": 7,
            "pos_weight_factor": 0.5,
        },
        {
            "name": "EmbeddingMLP_512_256_128_do0.35_pw1.0",
            "kind": "torch",
            "model": lambda: EmbeddingMLP(cat_dims, n_num, hidden=(512, 256, 128), dropout=0.35),
            "lr": 6e-4,
            "weight_decay": 3e-4,
            "batch_size": 1024,
            "max_epochs": 55,
            "patience": 8,
            "pos_weight_factor": 1.0,
        },
    ]


def tabnet_specs(cat_dims, n_num):
    if TabNetClassifier is None:
        return []
    return [
        {
            "name": "TabNet_d16_a16_steps4_gamma1.3_pw1.0",
            "kind": "tabnet",
            "n_d": 16,
            "n_a": 16,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 1e-4,
            "lr": 2e-2,
            "max_epochs": 60,
            "patience": 10,
            "batch_size": 1024,
            "pos_weight_factor": 1.0,
        },
        {
            "name": "TabNet_d24_a24_steps4_gamma1.5_pw0.75",
            "kind": "tabnet",
            "n_d": 24,
            "n_a": 24,
            "n_steps": 4,
            "gamma": 1.5,
            "lambda_sparse": 5e-5,
            "lr": 1e-2,
            "max_epochs": 70,
            "patience": 10,
            "batch_size": 1024,
            "pos_weight_factor": 0.75,
        },
    ]


def rows_for_scores(feature_name, model_name, y_true, score, split, fit_seconds, extra=None):
    rows = []
    for strategy, threshold in best_thresholds(y_true, score).items():
        row = threshold_metrics(y_true, score, threshold)
        row.update(
            {
                "split": split,
                "feature_config": feature_name,
                "model_name": model_name,
                "threshold_strategy": strategy,
                "fit_seconds": fit_seconds,
                "status": "ok",
                "error": "",
            }
        )
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def tabnet_matrix(X_num, X_cat):
    return np.concatenate([X_num.astype("float32"), X_cat.astype("float32")], axis=1)


def run():
    scoped = load_scoped_data()
    _, y = build_feature_matrix(scoped, FeatureConfig(name="split_reference"))
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

    rows = [dummy_row(y_train, y_val, "validation")]
    fitted = {}

    for feat_cfg in feature_configs():
        X_cfg, _ = build_feature_matrix(scoped, feat_cfg)
        X_train_raw = X_cfg.iloc[train_idx].copy()
        X_val_raw = X_cfg.iloc[val_idx].copy()
        X_test_raw = X_cfg.iloc[test_idx].copy()
        encoder = TabularEncoder(rare_min_count=feat_cfg.rare_min_count).fit(X_train_raw)
        Xn_train, Xc_train = encoder.transform(X_train_raw)
        Xn_val, Xc_val = encoder.transform(X_val_raw)
        Xn_test, Xc_test = encoder.transform(X_test_raw)
        print(f"\\n=== NN feature config: {feat_cfg.name} ({X_cfg.shape[1]} columns, {len(encoder.cat_dims)} categorical) ===")

        specs = torch_specs(encoder.cat_dims, Xn_train.shape[1]) + tabnet_specs(encoder.cat_dims, Xn_train.shape[1])
        for spec in specs:
            start = time.perf_counter()
            try:
                if spec["kind"] == "torch":
                    model = spec["model"]()
                    model, fit_seconds, best_epoch_pr = train_torch_model(
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
                    fitted[(feat_cfg.name, spec["name"])] = {
                        "val_score": val_score,
                        "test_score": test_score,
                        "fit_seconds": fit_seconds,
                    }
                    extra = {"best_epoch_pr_auc": best_epoch_pr, **asdict(feat_cfg)}
                else:
                    cat_idxs = list(range(Xn_train.shape[1], Xn_train.shape[1] + Xc_train.shape[1]))
                    cat_dims = encoder.cat_dims
                    model = TabNetClassifier(
                        n_d=spec["n_d"],
                        n_a=spec["n_a"],
                        n_steps=spec["n_steps"],
                        gamma=spec["gamma"],
                        lambda_sparse=spec["lambda_sparse"],
                        cat_idxs=cat_idxs,
                        cat_dims=cat_dims,
                        cat_emb_dim=[min(16, max(2, int(round(dim**0.25 * 4)))) for dim in cat_dims],
                        optimizer_fn=torch.optim.Adam,
                        optimizer_params={"lr": spec["lr"]},
                        seed=RANDOM_STATE,
                        verbose=0,
                    )
                    weights = np.where(
                        y_train.to_numpy() == 1,
                        ((y_train == 0).sum() / (y_train == 1).sum()) * spec["pos_weight_factor"],
                        1.0,
                    )
                    model.fit(
                        tabnet_matrix(Xn_train, Xc_train),
                        y_train.to_numpy(),
                        eval_set=[(tabnet_matrix(Xn_val, Xc_val), y_val.to_numpy())],
                        eval_name=["val"],
                        eval_metric=["auc"],
                        max_epochs=spec["max_epochs"],
                        patience=spec["patience"],
                        batch_size=spec["batch_size"],
                        virtual_batch_size=128,
                        weights=weights,
                    )
                    fit_seconds = time.perf_counter() - start
                    val_score = model.predict_proba(tabnet_matrix(Xn_val, Xc_val))[:, 1]
                    test_score = model.predict_proba(tabnet_matrix(Xn_test, Xc_test))[:, 1]
                    fitted[(feat_cfg.name, spec["name"])] = {
                        "val_score": val_score,
                        "test_score": test_score,
                        "fit_seconds": fit_seconds,
                    }
                    extra = {"best_epoch_pr_auc": np.nan, **asdict(feat_cfg)}

                model_rows = rows_for_scores(
                    feat_cfg.name,
                    spec["name"],
                    y_val,
                    val_score,
                    "validation",
                    fit_seconds,
                    extra=extra,
                )
                rows.extend(model_rows)
                best = max(model_rows, key=lambda r: r["f1"])
                print(
                    f"{spec['name']}: PR-AUC={best['pr_auc']:.4f}, best F1={best['f1']:.4f}, "
                    f"recall={best['recall']:.4f}, precision={best['precision']:.4f}, fit={fit_seconds:.1f}s"
                )
            except Exception as exc:
                fit_seconds = time.perf_counter() - start
                rows.append(
                    {
                        "split": "validation",
                        "feature_config": feat_cfg.name,
                        "model_name": spec["name"],
                        "threshold_strategy": "failed",
                        "threshold": np.nan,
                        "pr_auc": np.nan,
                        "roc_auc": np.nan,
                        "recall": np.nan,
                        "precision": np.nan,
                        "f1": np.nan,
                        "accuracy": np.nan,
                        "tn": np.nan,
                        "fp": np.nan,
                        "fn": np.nan,
                        "tp": np.nan,
                        "fit_seconds": fit_seconds,
                        "status": "failed",
                        "error": str(exc),
                        **asdict(feat_cfg),
                    }
                )
                print(f"{spec['name']}: FAILED: {exc}")

            pd.DataFrame(rows).to_csv(RESULTS_DIR / "nn_validation_results.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "nn_validation_results.csv", index=False)
    ok = results[(results["split"] == "validation") & (results["status"] == "ok")].copy()
    non_baseline = ok[ok["model_name"] != "MajorityBaseline"].copy()
    selected = pd.concat(
        [
            non_baseline.sort_values(["pr_auc", "f1"], ascending=False).head(8),
            non_baseline[non_baseline["threshold_strategy"] == "best_f1"]
            .sort_values(["f1", "pr_auc"], ascending=False)
            .head(8),
            non_baseline[non_baseline["threshold_strategy"] == "max_recall_precision_ge_0.20"]
            .sort_values(["f1", "pr_auc"], ascending=False)
            .head(8),
            non_baseline[non_baseline["threshold_strategy"] == "max_recall_precision_ge_0.15"]
            .sort_values(["recall", "pr_auc"], ascending=False)
            .head(8),
        ],
        ignore_index=True,
    ).drop_duplicates(["feature_config", "model_name", "threshold_strategy"]).head(24)
    selected.to_csv(RESULTS_DIR / "nn_selected_for_test.csv", index=False)

    test_rows = [dummy_row(y_train, y_test, "test")]
    for _, row in selected.iterrows():
        key = (row["feature_config"], row["model_name"])
        item = fitted[key]
        metrics = threshold_metrics(y_test, item["test_score"], row["threshold"])
        metrics.update(
            {
                "split": "test",
                "feature_config": row["feature_config"],
                "model_name": row["model_name"],
                "threshold_strategy": row["threshold_strategy"],
                "fit_seconds": item["fit_seconds"],
                "status": "ok",
                "error": "",
                "selected_validation_pr_auc": row["pr_auc"],
                "selected_validation_recall": row["recall"],
                "selected_validation_precision": row["precision"],
                "selected_validation_f1": row["f1"],
            }
        )
        test_rows.append(metrics)
    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(RESULTS_DIR / "nn_test_results.csv", index=False)

    cols = [
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
    print("\\nNeural top validation by PR-AUC:")
    print(ok.sort_values(["pr_auc", "f1"], ascending=False)[cols].head(30).to_string(index=False))
    print("\\nNeural top validation by F1:")
    print(ok.sort_values(["f1", "pr_auc"], ascending=False)[cols].head(30).to_string(index=False))
    print("\\nNeural selected test results:")
    print(test_results.sort_values(["pr_auc", "f1"], ascending=False)[cols + ["tn", "fp", "fn", "tp"]].to_string(index=False))


if __name__ == "__main__":
    run()
