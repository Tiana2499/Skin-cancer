"""
attention_fusion.py
Attention-based fusion of 40 features (2 image OOF + 38 tabular).
Learns per-sample softmax weights over features.
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
    output_dir = r"./analysis/attention_fusion"
    n_splits = 5
    seed = 42
    min_tpr = 0.80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = 64
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
# ATTENTION FUSION
# ============================================================================
class AttentionFusion(nn.Module):
    """
    Learned attention-weighted fusion of features.

    Each feature is projected to hidden_dim. Then attention weights
    are computed per-sample via a small MLP. The weighted sum is classified.
    """
    def __init__(self, n_features, hidden_dim=64, dropout=0.3):
        super().__init__()
        # Project each feature to hidden_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )
        # Attention scoring: (B, n_features, hidden) -> (B, n_features, 1)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        # Classifier on the weighted sum
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """
        x: (B, n_features)
        returns: (B,) logits
        """
        B, F = x.shape
        # Expand each feature: (B, F, 1) -> (B, F, hidden)
        feat_3d = x.unsqueeze(-1)  # (B, F, 1)
        proj = self.feature_proj(feat_3d)  # (B, F, hidden)

        # Attention weights: (B, F, 1) -> (B, F)
        attn_logits = self.attention(proj).squeeze(-1)  # (B, F)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, F)

        # Weighted sum: (B, F, hidden) * (B, F, 1) -> (B, hidden)
        weighted = (proj * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden)

        # Classify
        return self.classifier(weighted).squeeze(-1)


def train_attention_fold(X_tr, y_tr, X_va, y_va, X_test, n_features):
    """Train AttentionFusion on one fold."""
    device = Config.device
    model = AttentionFusion(n_features, Config.hidden_dim, Config.dropout).to(device)

    # Class-weighted loss
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)

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

        # Validate
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
        test_probs = torch.sigmoid(model(torch.FloatTensor(X_test).to(device))).cpu().numpy()

    return test_probs, best_pauc


def load_or_build_features():
    """Load saved features or rebuild from scratch."""
    X_path = os.path.join(Config.output_dir, "X_trainval.npy")

    if os.path.exists(X_path):
        X = np.load(X_path)
        X_test = np.load(os.path.join(Config.output_dir, "X_test.npy"))
        y = np.load(os.path.join(Config.output_dir, "y_trainval.npy"))
        y_test = np.load(os.path.join(Config.output_dir, "y_test.npy"))
        groups = np.load(os.path.join(Config.output_dir, "groups.npy"), allow_pickle=True)
        print(f"Loaded cached features: {X.shape}")
        return X, X_test, y, y_test, groups

    # Build from scratch (same as FT-Transformer script)
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
        test_probs, best_val_pauc = train_attention_fold(
            X[tr], y[tr], X[va], y[va], X_test, X.shape[1])
        test_preds_per_fold.append(test_probs)
        fold_results.append(best_val_pauc)
        print(f"  Best val pAUC: {best_val_pauc:.4f}")

    test_preds = np.mean(test_preds_per_fold, axis=0)

    att_pauc = isic2024_pauc(y_test, test_preds)
    att_auroc = roc_auc_score(y_test, test_preds)

    print(f"\n{'=' * 70}")
    print("ATTENTION FUSION RESULTS")
    print(f"{'=' * 70}")
    print(f"Test pAUC:  {att_pauc:.4f}")
    print(f"Test AUROC: {att_auroc:.4f}")
    print(f"Per-fold:   {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

    # Compare to CatBoost
    cat_path = os.path.join(Config.model_dir, "tinyvit+densenet_test_cat.npy")
    if os.path.exists(cat_path):
        cat_probs = np.load(cat_path)
        cat_pauc = isic2024_pauc(y_test, cat_probs)
        cat_auroc = roc_auc_score(y_test, cat_probs)

        print(f"\n{'=' * 70}")
        print("COMPARISON: CatBoost vs Attention Fusion")
        print(f"{'=' * 70}")
        print(f"{'Model':<25} {'pAUC':>10} {'AUROC':>10}")
        print("-" * 50)
        print(f"{'CatBoost':<25} {cat_pauc:>10.4f} {cat_auroc:>10.4f}")
        print(f"{'Attention Fusion':<25} {att_pauc:>10.4f} {att_auroc:>10.4f}")
        print(f"{'Δ (Att - Cat)':<25} {att_pauc - cat_pauc:>+10.4f} {att_auroc - cat_auroc:>+10.4f}")

    np.save(os.path.join(Config.output_dir, "attention_test_probs.npy"), test_preds)


if __name__ == "__main__":
    main()