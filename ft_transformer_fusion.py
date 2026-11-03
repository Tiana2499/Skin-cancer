# ============================================================================
# FT-TRANSFORMER FUSION
# Replace CatBoost with FT-Transformer for the final fusion step.
# Same 40 features: TinyViT OOF + DenseNet OOF + tabular features
# Patient-wise 5-fold CV
# ============================================================================

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import os
import copy
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklearn_metrics


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    model_dir = r"./models/sequential_ablation_all3"
    feature_dir = r"./features/tabular"
    output_dir = r"./analysis/ft_transformer"

    n_splits = 5
    seed = 42

    min_tpr = 0.80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------------
    # FT-Transformer hyperparameters
    # ------------------------------------------------------------------------
    d_token = 32
    n_heads = 4
    n_blocks = 2
    dropout = 0.1

    lr = 1e-3
    weight_decay = 1e-2

    # Increased from 10 so early stopping can actually operate
    epochs = 10
    patience = 5

    batch_size = 32


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# ISIC 2024 pAUC
# ============================================================================

def isic2024_pauc(y, p, min_tpr=0.80):
    """
    ISIC-style partial AUC over FPR <= (1 - min_tpr).

    For min_tpr=0.80:
        max_fpr = 0.20
    """

    y = np.asarray(y)
    p = np.asarray(p)

    if len(np.unique(y)) != 2:
        return 0.0

    yf = np.abs(y - 1)
    pf = np.abs(p - 1)

    max_fpr = 1 - min_tpr

    fpr, tpr, _ = sklearn_metrics.roc_curve(yf, pf)

    stop = np.searchsorted(fpr, max_fpr, side="right")

    if stop == 0:
        fpr_i = np.array([0.0, max_fpr])
        tpr_i = np.array([0.0, tpr[0]])
    elif stop >= len(fpr):
        fpr_i = fpr
        tpr_i = tpr
    else:
        fpr_i = np.append(
            fpr[:stop],
            max_fpr
        )

        tpr_i = np.append(
            tpr[:stop],
            np.interp(
                max_fpr,
                [fpr[stop - 1], fpr[stop]],
                [tpr[stop - 1], tpr[stop]]
            )
        )

    return float(sklearn_metrics.auc(fpr_i, tpr_i))


# ============================================================================
# FEATURE STANDARDIZATION
# ============================================================================

class FoldStandardScaler:
    """
    Standardize features using ONLY the training portion of each fold.

    This prevents validation/test information from leaking into the
    preprocessing step.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Prevent division by zero for constant features
        self.std_[self.std_ < 1e-8] = 1.0

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ============================================================================
# FEATURE TOKENIZER
# ============================================================================

class FeatureTokenizer(nn.Module):
    """
    Numerical feature tokenizer.

    Each scalar feature receives its own learnable weight vector and bias
    vector, producing one token per input feature.

    Input:
        (B, n_features)

    Output:
        (B, n_features, d_token)
    """

    def __init__(self, n_features, d_token):
        super().__init__()

        self.n_features = n_features
        self.d_token = d_token

        self.weight = nn.Parameter(
            torch.randn(n_features, d_token) * 0.02
        )

        self.bias = nn.Parameter(
            torch.zeros(n_features, d_token)
        )

    def forward(self, x):
        return (
            x.unsqueeze(-1) * self.weight.unsqueeze(0)
            + self.bias.unsqueeze(0)
        )


# ============================================================================
# FT-TRANSFORMER
# ============================================================================

class FTTransformer(nn.Module):

    def __init__(
        self,
        n_features,
        d_token=32,
        n_heads=4,
        n_blocks=2,
        dropout=0.1
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(
            n_features=n_features,
            d_token=d_token
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_token) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_blocks,
            enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(d_token)

        self.head = nn.Linear(
            d_token,
            1
        )

    def forward(self, x):

        # Feature tokens
        tokens = self.tokenizer(x)

        batch_size = x.shape[0]

        # Expand CLS token for the batch
        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        # [CLS] + feature tokens
        tokens = torch.cat(
            [cls, tokens],
            dim=1
        )

        # Transformer
        out = self.transformer(tokens)

        # Normalize
        out = self.norm(out)

        # CLS representation
        cls_out = out[:, 0, :]

        # Binary classification logit
        return self.head(cls_out).squeeze(-1)


# ============================================================================
# TRAIN ONE FOLD
# ============================================================================

def train_ft_fold(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_test,
    n_features,
    fold
):

    device = Config.device

    # ------------------------------------------------------------------------
    # IMPORTANT:
    # Fit scaler ONLY on training fold.
    # ------------------------------------------------------------------------

    scaler = FoldStandardScaler()

    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    model = FTTransformer(
        n_features=n_features,
        d_token=Config.d_token,
        n_heads=Config.n_heads,
        n_blocks=Config.n_blocks,
        dropout=Config.dropout
    ).to(device)

    # ------------------------------------------------------------------------
    # Class-weighted BCE
    # ------------------------------------------------------------------------

    n_pos = float(np.sum(y_tr))
    n_neg = float(len(y_tr) - n_pos)

    pos_weight_value = n_neg / max(n_pos, 1.0)

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # ------------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.epochs
    )

    # ------------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------------

    X_tr_t = torch.tensor(
        X_tr_scaled,
        dtype=torch.float32
    )

    y_tr_t = torch.tensor(
        y_tr,
        dtype=torch.float32
    )

    train_ds = TensorDataset(
        X_tr_t,
        y_tr_t
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    X_va_t = torch.tensor(
        X_va_scaled,
        dtype=torch.float32,
        device=device
    )

    # ------------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------------

    best_pauc = -np.inf
    best_state = None
    patience_counter = 0
    best_epoch = 0

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    for epoch in range(Config.epochs):

        model.train()

        running_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:

            xb = xb.to(
                device,
                non_blocking=True
            )

            yb = yb.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # --------------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(X_va_t)

            val_probs = torch.sigmoid(
                val_logits
            ).cpu().numpy()

        val_pauc = isic2024_pauc(
            y_va,
            val_probs,
            min_tpr=Config.min_tpr
        )

        avg_loss = running_loss / max(
            n_batches,
            1
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"    Fold {fold} | "
            f"Epoch {epoch + 1:02d}/{Config.epochs} | "
            f"Loss {avg_loss:.5f} | "
            f"Val pAUC {val_pauc:.4f} | "
            f"Best {max(best_pauc, val_pauc):.4f} | "
            f"LR {current_lr:.2e}",
            flush=True
        )

        # --------------------------------------------------------------------
        # Save best model according to validation pAUC
        # --------------------------------------------------------------------

        if val_pauc > best_pauc:

            best_pauc = val_pauc
            best_epoch = epoch + 1

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= Config.patience:

            print(
                f"    Early stopping at epoch {epoch + 1} "
                f"(best epoch {best_epoch})",
                flush=True
            )

            break

    # ------------------------------------------------------------------------
    # Restore best model
    # ------------------------------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            f"Fold {fold}: no best model state was saved."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    # ------------------------------------------------------------------------
    # Test prediction
    # ------------------------------------------------------------------------

    X_test_t = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        test_logits = model(
            X_test_t
        )

        test_probs = torch.sigmoid(
            test_logits
        ).cpu().numpy()

    return (
        test_probs,
        best_pauc,
        best_epoch
    )


# ============================================================================
# LOAD / BUILD FEATURES
# ============================================================================

def load_or_build_features():

    X_path = os.path.join(
        Config.output_dir,
        "X_trainval.npy"
    )

    if os.path.exists(X_path):

        X = np.load(X_path)

        X_test = np.load(
            os.path.join(
                Config.output_dir,
                "X_test.npy"
            )
        )

        y = np.load(
            os.path.join(
                Config.output_dir,
                "y_trainval.npy"
            )
        )

        y_test = np.load(
            os.path.join(
                Config.output_dir,
                "y_test.npy"
            )
        )

        groups = np.load(
            os.path.join(
                Config.output_dir,
                "groups.npy"
            ),
            allow_pickle=True
        )

        print(
            f"Loaded cached features: {X.shape}"
        )

        return (
            X,
            X_test,
            y,
            y_test,
            groups
        )

    # ------------------------------------------------------------------------
    # Build from scratch
    # ------------------------------------------------------------------------

    print(
        "Building feature matrix..."
    )

    tab_full = pd.read_parquet(
        os.path.join(
            Config.feature_dir,
            "trainval_tabular.parquet"
        )
    )

    tab_test = pd.read_parquet(
        os.path.join(
            Config.feature_dir,
            "test_tabular.parquet"
        )
    )

    # ------------------------------------------------------------------------
    # Remove problematic duplicated melanoma-thickness features
    # ------------------------------------------------------------------------

    drop = [
        c
        for c in [
            "mel_thick_mm_dev",
            "mel_thick_mm"
        ]
        if c in tab_full.columns
    ]

    tab_full = tab_full.drop(
        columns=drop
    )

    tab_test = tab_test.drop(
        columns=drop,
        errors="ignore"
    )

    # ------------------------------------------------------------------------
    # Load OOF predictions
    # ------------------------------------------------------------------------

    tiny_oof = np.load(
        r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\oof_probs.npy"
    )

    tiny_ids = np.load(
        r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\oof_ids.npy",
        allow_pickle=True
    )

    dense_oof = np.load(
        r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\oof_probs.npy"
    )

    dense_ids = np.load(
        r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\oof_ids.npy",
        allow_pickle=True
    )

    # ------------------------------------------------------------------------
    # Sort and align OOF predictions
    # ------------------------------------------------------------------------

    tiny_order = np.argsort(tiny_ids)

    tiny_oof = tiny_oof[tiny_order]
    tiny_ids = tiny_ids[tiny_order]

    dense_order = np.argsort(dense_ids)

    dense_oof = dense_oof[dense_order]
    dense_ids = dense_ids[dense_order]

    # Safety check
    if not np.array_equal(
        tiny_ids,
        dense_ids
    ):
        raise ValueError(
            "TinyViT and DenseNet OOF IDs do not match."
        )

    # ------------------------------------------------------------------------
    # Align tabular data to OOF IDs
    # ------------------------------------------------------------------------

    tab_full = (
        tab_full
        .set_index("isic_id")
        .loc[tiny_ids]
        .reset_index()
    )

    META = [
        "isic_id",
        "patient_id",
        "target"
    ]

    tab_cols = [
        c
        for c in tab_full.columns
        if c not in META
    ]

    # ------------------------------------------------------------------------
    # Build train/validation feature matrix
    # ------------------------------------------------------------------------

    X = np.concatenate(
        [
            tiny_oof[:, None],
            dense_oof[:, None],
            tab_full[tab_cols]
                .values
                .astype(np.float32)
        ],
        axis=1
    )

    y = (
        tab_full["target"]
        .values
        .astype(int)
    )

    groups = (
        tab_full["patient_id"]
        .values
    )

    # ------------------------------------------------------------------------
    # Test data
    # ------------------------------------------------------------------------

    test_manifest = pd.read_csv(
        r"F:\thesis 23\tinyvit experiment\cohort_manifests\test_manifest.csv"
    )

    tab_test = (
        tab_test
        .set_index("isic_id")
        .loc[test_manifest["isic_id"].values]
        .reset_index()
    )

    tiny_test = np.load(
        r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\test_probs.npy"
    )

    dense_test = np.load(
        r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\test_probs.npy"
    )

    # Safety checks
    if len(tiny_test) != len(tab_test):
        raise ValueError(
            f"TinyViT test predictions ({len(tiny_test)}) "
            f"do not match test rows ({len(tab_test)})."
        )

    if len(dense_test) != len(tab_test):
        raise ValueError(
            f"DenseNet test predictions ({len(dense_test)}) "
            f"do not match test rows ({len(tab_test)})."
        )

    X_test = np.concatenate(
        [
            tiny_test[:, None],
            dense_test[:, None],
            tab_test[tab_cols]
                .values
                .astype(np.float32)
        ],
        axis=1
    )

    y_test = (
        tab_test["target"]
        .values
        .astype(int)
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    os.makedirs(
        Config.output_dir,
        exist_ok=True
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "X_trainval.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "X_test.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "y_trainval.npy"
        ),
        y
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "y_test.npy"
        ),
        y_test
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "groups.npy"
        ),
        groups
    )

    return (
        X,
        X_test,
        y,
        y_test,
        groups
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    set_seed(
        Config.seed
    )

    os.makedirs(
        Config.output_dir,
        exist_ok=True
    )

    print("=" * 70)
    print("FT-TRANSFORMER FUSION")
    print("=" * 70)

    print(
        f"Device: {Config.device}"
    )

    print(
        f"n_splits: {Config.n_splits}"
    )

    print(
        f"d_token: {Config.d_token}"
    )

    print(
        f"heads: {Config.n_heads}"
    )

    print(
        f"blocks: {Config.n_blocks}"
    )

    print(
        f"epochs: {Config.epochs}"
    )

    print(
        f"patience: {Config.patience}"
    )

    print(
        f"pAUC min TPR: {Config.min_tpr}"
    )

    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------------

    (
        X,
        X_test,
        y,
        y_test,
        groups
    ) = load_or_build_features()

    print(
        f"X: {X.shape} | Positives: {y.sum()}"
    )

    print(
        f"X_test: {X_test.shape} | Positives: {y_test.sum()}"
    )

    print(
        f"Number of features: {X.shape[1]}"
    )

    # ------------------------------------------------------------------------
    # Patient-wise stratified CV
    # ------------------------------------------------------------------------

    sgkf = StratifiedGroupKFold(
        n_splits=Config.n_splits,
        shuffle=True,
        random_state=Config.seed
    )

    splits = list(
        sgkf.split(
            X,
            y,
            groups
        )
    )

    # ------------------------------------------------------------------------
    # Train folds
    # ------------------------------------------------------------------------

    test_preds_per_fold = []
    fold_results = []
    best_epochs = []

    for fold, (tr, va) in enumerate(
        splits,
        1
    ):

        print("\n" + "=" * 70)
        print(
            f"FOLD {fold}/{Config.n_splits}"
        )
        print("=" * 70)

        print(
            f"Train samples: {len(tr)} | "
            f"positives: {y[tr].sum()}"
        )

        print(
            f"Val samples: {len(va)} | "
            f"positives: {y[va].sum()}"
        )

        test_probs, best_val_pauc, best_epoch = train_ft_fold(
            X[tr],
            y[tr],
            X[va],
            y[va],
            X_test,
            X.shape[1],
            fold
        )

        test_preds_per_fold.append(
            test_probs
        )

        fold_results.append(
            best_val_pauc
        )

        best_epochs.append(
            best_epoch
        )

        print(
            f"Best val pAUC: {best_val_pauc:.4f}"
        )

        print(
            f"Best epoch: {best_epoch}"
        )

    # ------------------------------------------------------------------------
    # Average five test predictions
    # ------------------------------------------------------------------------

    test_preds = np.mean(
        test_preds_per_fold,
        axis=0
    )

    # ------------------------------------------------------------------------
    # Final test metrics
    # ------------------------------------------------------------------------

    ft_pauc = isic2024_pauc(
        y_test,
        test_preds,
        min_tpr=Config.min_tpr
    )

    ft_auroc = roc_auc_score(
        y_test,
        test_preds
    )

    mean_val_pauc = np.mean(
        fold_results
    )

    std_val_pauc = np.std(
        fold_results
    )

    # ------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FT-TRANSFORMER FUSION RESULTS")
    print("=" * 70)

    print(
        f"Test pAUC:          {ft_pauc:.4f}"
    )

    print(
        f"Test AUROC:         {ft_auroc:.4f}"
    )

    print(
        f"Per-fold val pAUC:  "
        f"{mean_val_pauc:.4f} ± {std_val_pauc:.4f}"
    )

    print(
        f"Best epochs:        {best_epochs}"
    )

    # ------------------------------------------------------------------------
    # Compare with CatBoost
    # ------------------------------------------------------------------------

    cat_path = os.path.join(
        Config.model_dir,
        "tinyvit+densenet_test_cat.npy"
    )

    if os.path.exists(cat_path):

        cat_probs = np.load(
            cat_path
        )

        cat_pauc = isic2024_pauc(
            y_test,
            cat_probs,
            min_tpr=Config.min_tpr
        )

        cat_auroc = roc_auc_score(
            y_test,
            cat_probs
        )

        print("\n" + "=" * 70)
        print("COMPARISON: CatBoost vs FT-Transformer")
        print("=" * 70)

        print(
            f"{'Model':<20}"
            f"{'pAUC':>10}"
            f"{'AUROC':>10}"
        )

        print("-" * 45)

        print(
            f"{'CatBoost':<20}"
            f"{cat_pauc:>10.4f}"
            f"{cat_auroc:>10.4f}"
        )

        print(
            f"{'FT-Transformer':<20}"
            f"{ft_pauc:>10.4f}"
            f"{ft_auroc:>10.4f}"
        )

        print(
            f"{'Δ (FT - Cat)':<20}"
            f"{ft_pauc - cat_pauc:>+10.4f}"
            f"{ft_auroc - cat_auroc:>+10.4f}"
        )

    # ------------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------------

    np.save(
        os.path.join(
            Config.output_dir,
            "ft_test_probs.npy"
        ),
        test_preds
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "ft_fold_val_pauc.npy"
        ),
        np.asarray(
            fold_results
        )
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "ft_best_epochs.npy"
        ),
        np.asarray(
            best_epochs
        )
    )

    print("\nSaved:")
    print(
        os.path.join(
            Config.output_dir,
            "ft_test_probs.npy"
        )
    )

    print(
        os.path.join(
            Config.output_dir,
            "ft_fold_val_pauc.npy"
        )
    )

    print(
        os.path.join(
            Config.output_dir,
            "ft_best_epochs.npy"
        )
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()