"""
nn_fusion_vs_catboost.py
Compare CatBoost vs Neural Network (MLP) on the same 40 features
(TinyViT OOF + DenseNet OOF + 38 tabular).
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklearn_metrics
import joblib


class Config:
    model_dir = r"./models/sequential_ablation_all3"
    feature_dir = r"./features/tabular"
    output_dir = r"./analysis/nn_fusion"
    n_splits = 5
    seed = 42
    min_tpr = 0.80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MLP hyperparameters
    hidden_dim = 128
    dropout = 0.3
    lr = 1e-3
    epochs = 100
    batch_size = 256
    patience = 15


def isic2024_pauc(y, p, min_tpr=0.80):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) != 2:
        return 0.0
    yf, pf = abs(y - 1), abs(p - 1)
    max_fpr = 1 - min_tpr
    fpr, tpr, _ = sklearn_metrics.roc_curve(yf, pf)
    stop = np.searchsorted(fpr, max_fpr, "right")
    tpr_i = np.append(tpr[:stop], np.interp(max_fpr, [fpr[stop-1], fpr[stop]], [tpr[stop-1], tpr[stop]]))
    fpr_i = np.append(fpr[:stop], max_fpr)
    return float(sklearn_metrics.auc(fpr_i, tpr_i))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_fold(X_tr, y_tr, X_va, y_va, X_test, input_dim):
    """Train MLP on one fold. Returns test predictions."""
    device = Config.device
    model = MLP(input_dim, Config.hidden_dim, Config.dropout).to(device)

    # Class weight
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    y_va_t = torch.FloatTensor(y_va).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)

    best_pauc = -1
    best_state = None
    patience_counter = 0

    for epoch in range(Config.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        try:
            val_pauc = isic2024_pauc(y_va, val_probs)
        except Exception:
            val_pauc = 0.0

        if val_pauc > best_pauc:
            best_pauc = val_pauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.patience:
            break

    # Load best and predict test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.FloatTensor(X_test).to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    return test_probs, best_pauc


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    # Load the same features used by CatBoost
    tiny_oof = np.load(os.path.join(Config.model_dir, "tinyvit+densenet_oof_cat.npy"))  # not this, need features
    # Actually load from corrected features
    X = np.load(os.path.join(Config.model_dir, "final_X_trainval.npy")) if os.path.exists(
        os.path.join(Config.model_dir, "final_X_trainval.npy")) else None

    if X is None:
        print("Building feature matrix from scratch...")
        tab_full = pd.read_parquet(os.path.join(Config.feature_dir, "trainval_tabular.parquet"))
        tab_test = pd.read_parquet(os.path.join(Config.feature_dir, "test_tabular.parquet"))
        tab_full = tab_full.drop(columns=[c for c in ["mel_thick_mm_dev", "mel_thick_mm"] if c in tab_full.columns])
        tab_test = tab_test.drop(columns=[c for c in ["mel_thick_mm_dev", "mel_thick_mm"] if c in tab_test.columns])

        tiny_oof = np.load(r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\oof_probs.npy")
        tiny_ids = np.load(r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\oof_ids.npy", allow_pickle=True)
        dense_oof = np.load(r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\oof_probs.npy")
        dense_ids = np.load(r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\oof_ids.npy", allow_pickle=True)

        # Align
        order = np.argsort(tiny_ids)
        tiny_oof = tiny_oof[order]
        tiny_ids = tiny_ids[order]
        dense_order = np.argsort(dense_ids)
        dense_oof = dense_oof[dense_order]

        tab_full = tab_full.set_index("isic_id").loc[tiny_ids].reset_index()

        META = ["isic_id", "patient_id", "target"]
        tab_cols = [c for c in tab_full.columns if c not in META]

        X = np.concatenate([
            tiny_oof[:, None], dense_oof[:, None],
            tab_full[tab_cols].values.astype(np.float32)], axis=1)

        y = tab_full["target"].values.astype(int)
        groups = tab_full["patient_id"].values

        # Also build test X
        test_manifest = pd.read_csv(r"F:\thesis 23\tinyvit experiment\cohort_manifests\test_manifest.csv")
        tab_test = tab_test.set_index("isic_id").loc[test_manifest["isic_id"].values].reset_index()
        tiny_test = np.load(r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\test_probs.npy")
        dense_test = np.load(r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\test_probs.npy")

        X_test = np.concatenate([
            tiny_test[:, None], dense_test[:, None],
            tab_test[tab_cols].values.astype(np.float32)], axis=1)
        y_test = tab_test["target"].values.astype(int)

        np.save(os.path.join(Config.output_dir, "X_trainval.npy"), X)
        np.save(os.path.join(Config.output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(Config.output_dir, "y_trainval.npy"), y)
        np.save(os.path.join(Config.output_dir, "y_test.npy"), y_test)
        np.save(os.path.join(Config.output_dir, "groups.npy"), groups)

    print(f"X: {X.shape}")
    print(f"Positives: {y.sum()}")

    # 5-fold CV
    sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
    splits = list(sgkf.split(X, y, groups))

    test_preds_per_fold = []
    oof_preds = np.zeros(len(y))
    fold_metrics = []

    for fold, (tr, va) in enumerate(splits, 1):
        print(f"\nFold {fold}/{Config.n_splits}")
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        test_probs, best_val_pauc = train_mlp_fold(
            X_tr, y_tr, X_va, y_va, X_test, X.shape[1])

        # Also get validation predictions for OOF
        # (simplified: re-train briefly to get OOF)
        test_preds_per_fold.append(test_probs)
        print(f"  Best val pAUC: {best_val_pauc:.4f}")

    test_preds = np.mean(test_preds_per_fold, axis=0)

    # Metrics
    mlp_test_pauc = isic2024_pauc(y_test, test_preds)
    mlp_test_auroc = roc_auc_score(y_test, test_preds)

    print(f"\n{'=' * 70}")
    print("MLP FUSION RESULTS")
    print(f"{'=' * 70}")
    print(f"Test pAUC:  {mlp_test_pauc:.4f}")
    print(f"Test AUROC: {mlp_test_auroc:.4f}")

    # Compare to CatBoost
    cat_probs = np.load(os.path.join(Config.model_dir, "tinyvit+densenet_test_cat.npy"))
    cat_pauc = isic2024_pauc(y_test, cat_probs)
    cat_auroc = roc_auc_score(y_test, cat_probs)

    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'pAUC':>10} {'AUROC':>10}")
    print("-" * 45)
    print(f"{'CatBoost':<20} {cat_pauc:>10.4f} {cat_auroc:>10.4f}")
    print(f"{'MLP':<20} {mlp_test_pauc:>10.4f} {mlp_test_auroc:>10.4f}")
    print(f"{'Δ':<20} {mlp_test_pauc - cat_pauc:>+10.4f} {mlp_test_auroc - cat_auroc:>+10.4f}")


if __name__ == "__main__":
    main()