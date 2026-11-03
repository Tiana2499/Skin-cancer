"""
fusion_benchmark.py
Compare concatenation vs. learned fusion strategies:
  1. Concatenation (baseline)
  2. Gated Multimodal Fusion (GMF)
  3. Cross-Modal Attention
  4. Bilinear Pooling
  5. FiLM (Feature-wise Linear Modulation)

Each fusion produces a 64-dim (or 128-dim) vector that feeds CatBoost.
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
from catboost import CatBoostClassifier


class Config:
    model_dir = r"./models/sequential_ablation_all3"
    feature_dir = r"./features/tabular"
    output_dir = r"./analysis/spe_catboost"
    n_splits = 5
    seed = 42
    min_tpr = 0.80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion_dim = 64
    dropout = 0.2
    lr = 1e-3
    epochs = 5
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
# FUSION MODULES
# ============================================================================
class GatedFusion(nn.Module):
    """Gated Multimodal Fusion: g * image + (1-g) * tabular."""
    def __init__(self, n_image=2, n_tabular=38, fusion_dim=64, dropout=0.2):
        super().__init__()
        self.img_proj = nn.Linear(n_image, fusion_dim)
        self.tab_proj = nn.Linear(n_tabular, fusion_dim)
        self.gate = nn.Linear(n_image + n_tabular, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_img = x[:, :2]
        x_tab = x[:, 2:]
        h_img = self.img_proj(x_img)
        h_tab = self.tab_proj(x_tab)
        g = torch.sigmoid(self.gate(x))
        out = g * h_img + (1 - g) * h_tab
        return self.dropout(self.norm(out))


class CrossModalFusion(nn.Module):
    """Image queries tabular features."""
    def __init__(self, n_image=2, n_tabular=38, d_model=32, n_heads=4, dropout=0.2):
        super().__init__()
        self.img_proj = nn.Linear(1, d_model)
        self.tab_proj = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model * n_image, 64)

    def forward(self, x):
        x_img = x[:, :2].unsqueeze(-1)
        x_tab = x[:, 2:].unsqueeze(-1)
        img_tokens = self.img_proj(x_img)
        tab_tokens = self.tab_proj(x_tab)
        attn_out, _ = self.attn(img_tokens, tab_tokens, tab_tokens)
        attn_out = self.norm(img_tokens + attn_out)
        flat = attn_out.reshape(attn_out.shape[0], -1)
        return self.out(flat)


class BilinearFusion(nn.Module):
    """Outer product of image and tabular embeddings."""
    def __init__(self, n_image=2, n_tabular=38, img_dim=16, tab_dim=16, dropout=0.2):
        super().__init__()
        self.img_proj = nn.Linear(n_image, img_dim)
        self.tab_proj = nn.Linear(n_tabular, tab_dim)
        self.out = nn.Linear(img_dim * tab_dim, 64)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_img = x[:, :2]
        x_tab = x[:, 2:]
        h_img = torch.tanh(self.img_proj(x_img))
        h_tab = torch.tanh(self.tab_proj(x_tab))
        outer = torch.bmm(h_img.unsqueeze(2), h_tab.unsqueeze(1))
        return self.dropout(self.out(outer.reshape(outer.shape[0], -1)))


class FiLMFusion(nn.Module):
    """Image modulates tabular via scale/shift."""
    def __init__(self, n_image=2, n_tabular=38, fusion_dim=64, dropout=0.2):
        super().__init__()
        self.tab_proj = nn.Linear(n_tabular, fusion_dim)
        self.film = nn.Linear(n_image, fusion_dim * 2)
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_img = x[:, :2]
        x_tab = x[:, 2:]
        h_tab = self.tab_proj(x_tab)
        gamma_beta = self.film(x_img)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        out = gamma * h_tab + beta
        return self.dropout(self.norm(out))


class ConcatFusion(nn.Module):
    """Baseline: simple concatenation."""
    def __init__(self, n_features=40, fusion_dim=64, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_features, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(self.proj(x)))


FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "gated": GatedFusion,
    "cross_modal": CrossModalFusion,
    "bilinear": BilinearFusion,
    "film": FiLMFusion,
}


def train_fusion_fold(fusion_type, X_tr, y_tr, X_va, y_va, X_test):
    """Train fusion module + CatBoost on one fold."""
    device = Config.device
    model = FUSION_REGISTRY[fusion_type](fusion_dim=Config.fusion_dim) if fusion_type == "concat" else FUSION_REGISTRY[fusion_type]()
    model = model.to(device)

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=Config.batch_size, shuffle=True)

    best_pauc = -1
    best_state = None
    patience = 0

    for epoch in range(Config.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb).mean(dim=-1) if hasattr(model(xb), "dim") and model(xb).dim() > 1 else model(xb)
            # For fusion modules that output (B, fusion_dim), we need a linear head
            # Actually, we pass the fused vector to CatBoost, not a classifier
            # So this loop trains the fusion to minimize a supervised objective
            # We'll use a linear head for training the fusion module
            # (see train_fusion_with_head below)
            pass
        break  # placeholder

    return None, None


def train_fusion_with_head(fusion_type, X_tr, y_tr, X_va, y_va, X_test):
    """
    Train fusion module + linear head end-to-end.
    Then extract fused features for CatBoost.
    """
    device = Config.device
    if fusion_type == "concat":
        fusion = ConcatFusion(fusion_dim=Config.fusion_dim).to(device)
    elif fusion_type == "gated":
        fusion = GatedFusion(fusion_dim=Config.fusion_dim).to(device)
    elif fusion_type == "cross_modal":
        fusion = CrossModalFusion().to(device)
    elif fusion_type == "bilinear":
        fusion = BilinearFusion().to(device)
    elif fusion_type == "film":
        fusion = FiLMFusion(fusion_dim=Config.fusion_dim).to(device)

    head = nn.Linear(64, 1).to(device)

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(list(fusion.parameters()) + list(head.parameters()), lr=Config.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=Config.batch_size, shuffle=True)

    best_pauc = -1
    best_state = None
    patience = 0

    for epoch in range(Config.epochs):
        fusion.train()
        head.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            fused = fusion(xb)
            logits = head(fused).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(fusion.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
        scheduler.step()

        fusion.eval()
        head.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(head(fusion(X_va_t)).squeeze(-1)).cpu().numpy()

        val_pauc = isic2024_pauc(y_va, val_probs) if len(np.unique(y_va)) == 2 else 0.0

        if val_pauc > best_pauc:
            best_pauc = val_pauc
            best_state = {
                "fusion": {k: v.cpu().clone() for k, v in fusion.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= Config.patience:
                break

    # Restore best
    fusion.load_state_dict(best_state["fusion"])
    head.load_state_dict(best_state["head"])
    fusion.eval()
    head.eval()

    # Extract fused features for train, val, test
    with torch.no_grad():
        X_tr_fused = fusion(X_tr_t).cpu().numpy()
        X_va_fused = fusion(X_va_t).cpu().numpy()
        X_test_fused = fusion(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    return X_tr_fused, X_va_fused, X_test_fused, best_pauc


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    # Load features
    X_path = os.path.join(Config.output_dir, "X_trainval.npy")
    if not os.path.exists(X_path):
        print("Feature matrix not found. Run ft_transformer_fusion.py first to build it.")
        return

    X = np.load(X_path)
    X_test = np.load(os.path.join(Config.output_dir, "X_test.npy"))
    y = np.load(os.path.join(Config.output_dir, "y_trainval.npy"))
    y_test = np.load(os.path.join(Config.output_dir, "y_test.npy"))
    groups = np.load(os.path.join(Config.output_dir, "groups.npy"), allow_pickle=True)

    print(f"X: {X.shape}  Positives: {y.sum()}")
    print(f"X_test: {X_test.shape}  Positives: {y_test.sum()}")

    sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
    splits = list(sgkf.split(X, y, groups))

    results = {}

    for fusion_type in FUSION_REGISTRY.keys():
        print(f"\n{'=' * 70}")
        print(f"Fusion: {fusion_type}")
        print(f"{'=' * 70}")

        test_preds_per_fold = []
        fold_metrics = []

        for fold, (tr, va) in enumerate(splits, 1):
            X_tr_fused, X_va_fused, X_test_fused, val_pauc = train_fusion_with_head(
                fusion_type, X[tr], y[tr], X[va], y[va], X_test)

            # Train CatBoost on fused features
            cat = CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                scale_pos_weight=(y[tr] == 0).sum() / max(y[tr].sum(), 1),
                random_seed=Config.seed, verbose=0, thread_count=-1)
            cat.fit(X_tr_fused, y[tr])

            # Predict test
            test_probs = cat.predict_proba(X_test_fused)[:, 1]
            test_preds_per_fold.append(test_probs)

            # Compute fold val pAUC using CatBoost (not the linear head)
            val_probs = cat.predict_proba(X_va_fused)[:, 1]
            fold_metrics.append(isic2024_pauc(y[va], val_probs))

        test_preds = np.mean(test_preds_per_fold, axis=0)
        cat_pauc = isic2024_pauc(y_test, test_preds)
        cat_auroc = roc_auc_score(y_test, test_preds)

        results[fusion_type] = {
            "test_pAUC": cat_pauc,
            "test_AUROC": cat_auroc,
            "per_fold_pAUC": np.mean(fold_metrics),
            "per_fold_std": np.std(fold_metrics),
        }

        print(f"Test pAUC:  {cat_pauc:.4f}")
        print(f"Test AUROC: {cat_auroc:.4f}")
        print(f"Per-fold:   {np.mean(fold_metrics):.4f} ± {np.std(fold_metrics):.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("FUSION COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Fusion':<20} {'Test pAUC':>12} {'Test AUROC':>12}")
    print("-" * 45)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_pAUC"]):
        print(f"{name:<20} {r['test_pAUC']:>12.4f} {r['test_AUROC']:>12.4f}")

    # Baseline
    cat_path = os.path.join(Config.model_dir, "tinyvit+densenet_test_cat.npy")
    if os.path.exists(cat_path):
        cat_probs = np.load(cat_path)
        baseline_pauc = isic2024_pauc(y_test, cat_probs)
        print(f"\nBaseline (concatenation + CatBoost): {baseline_pauc:.4f}")

    pd.DataFrame(results).T.to_csv(os.path.join(Config.output_dir, "fusion_results.csv"))


if __name__ == "__main__":
    main()