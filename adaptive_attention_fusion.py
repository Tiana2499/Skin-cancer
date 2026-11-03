"""
adaptive_attention_fusion.py
Replace CatBoost with an adaptive attention fusion network.
The network learns per-sample attention weights over the 40 input features.
No CatBoost is used — the network is the final classifier.
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


class Config:
    model_dir = r"./models/sequential_ablation_all3"
    feature_dir = r"./features/tabular"
    output_dir = r"./analysis/adaptive_attention"
    n_splits = 5
    seed = 42
    min_tpr = 0.80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adaptive attention hyperparameters
    d_model = 64
    n_heads = 4
    dropout = 0.3
    lr = 1e-3
    epochs = 100
    batch_size = 64
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


# ============================================================================
# ADAPTIVE ATTENTION FUSION
# ============================================================================
class AdaptiveAttentionFusion(nn.Module):
    """
    Adaptive attention fusion:
      1. Embed each of the 40 features into d_model dims.
      2. Compute per-sample attention weights via an MLP.
      3. Weighted sum of embeddings.
      4. MLP classifier on the fused representation.
    """
    def __init__(self, n_features=40, d_model=64, dropout=0.3):
        super().__init__()
        # Feature embedding: (B, 40) -> (B, 40, d_model)
        self.feature_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
        )

        # Attention scoring: (B, 40, d_model) -> (B, 40, 1)
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False),
        )

        # Classifier on weighted sum: (B, d_model) -> (B,)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        """
        x: (B, 40)
        returns: (B,) logits
        """
        B = x.shape[0]
        F = x.shape[1]

        # Embed each feature
        x_3d = x.unsqueeze(-1)              # (B, 40, 1)
        h = self.feature_embed(x_3d)        # (B, 40, d_model)

        # Attention weights
        attn_scores = self.attention(h).squeeze(-1)   # (B, 40)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, 40)

        # Weighted sum
        fused = (h * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        # Classify
        return self.classifier(fused).squeeze(-1)


def train_adaptive_fold(X_tr, y_tr, X_va, y_va, X_test):
    device = Config.device
    model = AdaptiveAttentionFusion(
        n_features=X_tr.shape[1],
        d_model=Config.d_model,
        dropout=Config.dropout,
    ).to(device)

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=Config.batch_size, shuffle=True)

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

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_va_t)).cpu().numpy()

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

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(model(X_test_t)).cpu().numpy()

    return test_probs, best_pauc


def load_or_build_features():
    X_path = os.path.join(Config.output_dir, "X_trainval.npy")
    if os.path.exists(X_path):
        X = np.load(X_path)
        X_test = np.load(os.path.join(Config.output_dir, "X_test.npy"))
        y = np.load(os.path.join(Config.output_dir, "y_trainval.npy"))
        y_test = np.load(os.path.join(Config.output_dir, "y_test.npy"))
        groups = np.load(os.path.join(Config.output_dir, "groups.npy"), allow_pickle=True)
        print(f"Loaded cached features: {X.shape}")
        return X, X_test, y, y_test, groups

    print("Building feature matrix...")
    tab_full = pd.read_parquet(os.path.join(Config.feature_dir, "trainval_tabular.parquet"))
    tab_test = pd.read_parquet(os.path.join(Config.feature_dir, "test_tabular.parquet"))
    drop = [c for c in ["mel_thick_mm_dev", "mel_thick_mm"] if c in tab_full.columns]
    tab_full = tab_full.drop(columns=drop)
    tab_test = tab_test.drop(columns=drop, errors="ignore")

    tiny_oof = np.load(r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\oof_probs.npy")
    tiny_ids = np.load(r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\oof_ids.npy", allow_pickle=True)
    dense_oof = np.load(r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\oof_probs.npy")
    dense_ids = np.load(r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\oof_ids.npy", allow_pickle=True)

    order = np.argsort(tiny_ids)
    tiny_oof = tiny_oof[order]
    tiny_ids = tiny_ids[order]
    dense_oof = dense_oof[np.argsort(dense_ids)]

    tab_full = tab_full.set_index("isic_id").loc[tiny_ids].reset_index()
    META = ["isic_id", "patient_id", "target"]
    tab_cols = [c for c in tab_full.columns if c not in META]

    X = np.concatenate([tiny_oof[:, None], dense_oof[:, None],
                        tab_full[tab_cols].values.astype(np.float32)], axis=1)
    y = tab_full["target"].values.astype(int)
    groups = tab_full["patient_id"].values

    test_manifest = pd.read_csv(r"F:\thesis 23\tinyvit experiment\cohort_manifests\test_manifest.csv")
    tab_test = tab_test.set_index("isic_id").loc[test_manifest["isic_id"].values].reset_index()
    tiny_test = np.load(r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\test_probs.npy")
    dense_test = np.load(r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\test_probs.npy")

    X_test = np.concatenate([tiny_test[:, None], dense_test[:, None],
                             tab_test[tab_cols].values.astype(np.float32)], axis=1)
    y_test = tab_test["target"].values.astype(int)

    os.makedirs(Config.output_dir, exist_ok=True)
    np.save(os.path.join(Config.output_dir, "X_trainval.npy"), X)
    np.save(os.path.join(Config.output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(Config.output_dir, "y_trainval.npy"), y)
    np.save(os.path.join(Config.output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(Config.output_dir, "groups.npy"), groups)

    return X, X_test, y, y_test, groups


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    X, X_test, y, y_test, groups = load_or_build_features()
    print(f"X: {X.shape}  Positives: {y.sum()}")
    print(f"X_test: {X_test.shape}  Positives: {y_test.sum()}")

    sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
    splits = list(sgkf.split(X, y, groups))

    test_preds_per_fold = []
    fold_results = []

    for fold, (tr, va) in enumerate(splits, 1):
        print(f"\nFold {fold}/{Config.n_splits}")
        test_probs, best_val_pauc = train_adaptive_fold(
            X[tr], y[tr], X[va], y[va], X_test)
        test_preds_per_fold.append(test_probs)
        fold_results.append(best_val_pauc)
        print(f"  Best val pAUC: {best_val_pauc:.4f}")

    test_preds = np.mean(test_preds_per_fold, axis=0)

    att_pauc = isic2024_pauc(y_test, test_preds)
    att_auroc = roc_auc_score(y_test, test_preds)

    print(f"\n{'=' * 70}")
    print("ADAPTIVE ATTENTION FUSION RESULTS")
    print(f"{'=' * 70}")
    print(f"Test pAUC:  {att_pauc:.4f}")
    print(f"Test AUROC: {att_auroc:.4f}")
    print(f"Per-fold:   {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

    cat_path = os.path.join(Config.model_dir, "tinyvit+densenet_test_cat.npy")
    if os.path.exists(cat_path):
        cat_probs = np.load(cat_path)
        cat_pauc = isic2024_pauc(y_test, cat_probs)
        cat_auroc = roc_auc_score(y_test, cat_probs)

        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")
        print(f"{'Model':<30} {'pAUC':>10} {'AUROC':>10}")
        print("-" * 55)
        print(f"{'CatBoost (raw 40 features)':<30} {cat_pauc:>10.4f} {cat_auroc:>10.4f}")
        print(f"{'Adaptive Attention Fusion':<30} {att_pauc:>10.4f} {att_auroc:>10.4f}")
        print(f"{'Δ':<30} {att_pauc - cat_pauc:>+10.4f} {att_auroc - cat_auroc:>+10.4f}")

    np.save(os.path.join(Config.output_dir, "adaptive_test_probs.npy"), test_preds)


if __name__ == "__main__":
    main()