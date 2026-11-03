"""
kan_fusion.py

Replace CatBoost with a Kolmogorov-Arnold Network (KAN) for fusion.

Features:
    TinyViT OOF probability
    DenseNet OOF probability
    Tabular metadata

Uses:
    efficient-kan
    https://github.com/Blealtan/efficient-kan

Important:
    - Patient-wise StratifiedGroupKFold
    - Fold-wise feature scaling
    - Validation pAUC for model selection
    - 5-fold test ensemble
    - No test information used during preprocessing
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklearn_metrics

# efficient-kan
from efficient_kan import KAN


# ============================================================================
# CONFIG
# ============================================================================

class Config:

    model_dir = r"./models/sequential_ablation_all3"
    feature_dir = r"./features/tabular"
    output_dir = r"./analysis/kan_fusion"

    n_splits = 5
    seed = 42

    min_tpr = 0.80

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ------------------------------------------------------------------------
    # KAN architecture
    # ------------------------------------------------------------------------

    hidden_dim = 16

    grid_size = 5
    spline_order = 3

    # IMPORTANT:
    # efficient-kan uses [-1, 1] as its default spline grid.
    # We therefore scale the inputs to approximately this range below.
    grid_range = [-1.0, 1.0]

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    lr = 1e-3
    weight_decay = 1e-2

    epochs = 10
    patience = 3

    batch_size = 64


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed):

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# ISIC 2024 pAUC
# ============================================================================

def isic2024_pauc(
    y,
    p,
    min_tpr=0.80
):
    """
    ISIC-style partial AUC for FPR <= (1 - min_tpr).

    For min_tpr = 0.80:
        max_fpr = 0.20
    """

    y = np.asarray(y)
    p = np.asarray(p)

    if len(np.unique(y)) != 2:
        return 0.0

    yf = np.abs(y - 1)
    pf = np.abs(p - 1)

    max_fpr = 1.0 - min_tpr

    fpr, tpr, _ = sklearn_metrics.roc_curve(
        yf,
        pf
    )

    # Find where FPR exceeds the allowed region.
    stop = np.searchsorted(
        fpr,
        max_fpr,
        side="right"
    )

    # ------------------------------------------------------------------------
    # Handle edge cases safely
    # ------------------------------------------------------------------------

    if stop == 0:

        fpr_i = np.array(
            [0.0, max_fpr],
            dtype=np.float64
        )

        tpr_i = np.array(
            [0.0, tpr[0]],
            dtype=np.float64
        )

    elif stop >= len(fpr):

        fpr_i = fpr
        tpr_i = tpr

    else:

        interpolated_tpr = np.interp(
            max_fpr,
            [
                fpr[stop - 1],
                fpr[stop]
            ],
            [
                tpr[stop - 1],
                tpr[stop]
            ]
        )

        fpr_i = np.append(
            fpr[:stop],
            max_fpr
        )

        tpr_i = np.append(
            tpr[:stop],
            interpolated_tpr
        )

    return float(
        sklearn_metrics.auc(
            fpr_i,
            tpr_i
        )
    )


# ============================================================================
# FOLD-WISE SCALER
# ============================================================================

class FoldStandardScaler:
    """
    Standardize features using training-fold statistics only.

    This prevents validation/test information from leaking into
    preprocessing.

    After standardization, values are clipped to the KAN grid range.

    NOTE:
        The scaler itself is fitted ONLY on the training fold.
    """

    def __init__(
        self,
        clip_min=-1.0,
        clip_max=1.0
    ):

        self.mean_ = None
        self.std_ = None

        self.clip_min = clip_min
        self.clip_max = clip_max

    def fit(self, X):

        X = np.asarray(
            X,
            dtype=np.float32
        )

        self.mean_ = np.mean(
            X,
            axis=0
        )

        self.std_ = np.std(
            X,
            axis=0
        )

        # Avoid division by zero for constant features.
        self.std_[self.std_ < 1e-8] = 1.0

        return self

    def transform(self, X):

        X = np.asarray(
            X,
            dtype=np.float32
        )

        X_scaled = (
            X - self.mean_
        ) / self.std_

        # KAN grid is [-1, 1].
        X_scaled = np.clip(
            X_scaled,
            self.clip_min,
            self.clip_max
        )

        return X_scaled.astype(
            np.float32
        )

    def fit_transform(self, X):

        return self.fit(X).transform(X)


# ============================================================================
# TRAIN ONE KAN FOLD
# ============================================================================

def train_kan_fold(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_test,
    n_features,
    fold
):

    device = Config.device

    # ========================================================================
    # IMPORTANT:
    # Fit scaler ONLY on training data.
    # ========================================================================

    scaler = FoldStandardScaler(
        clip_min=Config.grid_range[0],
        clip_max=Config.grid_range[1]
    )

    X_tr_scaled = scaler.fit_transform(
        X_tr
    )

    X_va_scaled = scaler.transform(
        X_va
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ========================================================================
    # KAN MODEL
    # ========================================================================

    model = KAN(
        layers_hidden=[
            n_features,
            Config.hidden_dim,
            1
        ],
        grid_size=Config.grid_size,
        spline_order=Config.spline_order,
        grid_range=Config.grid_range
    ).to(device)

    # ========================================================================
    # CLASS-WEIGHTED BCE
    # ========================================================================

    n_pos = float(
        np.sum(y_tr)
    )

    n_neg = float(
        len(y_tr) - n_pos
    )

    pos_weight_value = (
        n_neg / max(n_pos, 1.0)
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # ========================================================================
    # OPTIMIZER
    # ========================================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.epochs
    )

    # ========================================================================
    # DATA
    # ========================================================================

    X_tr_t = torch.tensor(
        X_tr_scaled,
        dtype=torch.float32
    )

    y_tr_t = torch.tensor(
        y_tr,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        X_tr_t,
        y_tr_t
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )

    X_va_t = torch.tensor(
        X_va_scaled,
        dtype=torch.float32,
        device=device
    )

    # ========================================================================
    # EARLY STOPPING
    # ========================================================================

    best_pauc = -np.inf

    best_state = None

    best_epoch = 0

    patience_counter = 0

    # ========================================================================
    # TRAINING
    # ========================================================================

    for epoch in range(
        Config.epochs
    ):

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

            logits = model(
                xb
            ).squeeze(-1)

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

        # ====================================================================
        # VALIDATION
        # ====================================================================

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_va_t
            ).squeeze(-1)

            val_probs = torch.sigmoid(
                val_logits
            ).cpu().numpy()

        val_pauc = isic2024_pauc(
            y_va,
            val_probs,
            min_tpr=Config.min_tpr
        )

        avg_loss = (
            running_loss /
            max(n_batches, 1)
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"    Fold {fold} | "
            f"Epoch {epoch + 1:03d}/{Config.epochs} | "
            f"Loss {avg_loss:.6f} | "
            f"Val pAUC {val_pauc:.4f} | "
            f"Best {max(best_pauc, val_pauc):.4f} | "
            f"LR {current_lr:.2e}",
            flush=True
        )

        # ====================================================================
        # BEST MODEL
        # ====================================================================

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

        # ====================================================================
        # EARLY STOP
        # ====================================================================

        if patience_counter >= Config.patience:

            print(
                f"    Early stopping at epoch "
                f"{epoch + 1}; "
                f"best epoch = {best_epoch}",
                flush=True
            )

            break

    # ========================================================================
    # RESTORE BEST MODEL
    # ========================================================================

    if best_state is None:

        raise RuntimeError(
            f"Fold {fold}: no best model state was saved."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    # ========================================================================
    # TEST PREDICTIONS
    # ========================================================================

    X_test_t = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        test_logits = model(
            X_test_t
        ).squeeze(-1)

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

    # ========================================================================
    # LOAD CACHED
    # ========================================================================

    if os.path.exists(X_path):

        X = np.load(
            X_path
        )

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

    # ========================================================================
    # BUILD
    # ========================================================================

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

    # ========================================================================
    # Remove duplicated melanoma thickness variables
    # ========================================================================

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

    # ========================================================================
    # TinyViT OOF
    # ========================================================================

    tiny_oof = np.load(
        r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\oof_probs.npy"
    )

    tiny_ids = np.load(
        r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\oof_ids.npy",
        allow_pickle=True
    )

    # ========================================================================
    # DenseNet OOF
    # ========================================================================

    dense_oof = np.load(
        r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\oof_probs.npy"
    )

    dense_ids = np.load(
        r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\oof_ids.npy",
        allow_pickle=True
    )

    # ========================================================================
    # SORT
    # ========================================================================

    tiny_order = np.argsort(
        tiny_ids
    )

    tiny_oof = tiny_oof[
        tiny_order
    ]

    tiny_ids = tiny_ids[
        tiny_order
    ]

    dense_order = np.argsort(
        dense_ids
    )

    dense_oof = dense_oof[
        dense_order
    ]

    dense_ids = dense_ids[
        dense_order
    ]

    # ========================================================================
    # CRITICAL ID CHECK
    # ========================================================================

    if not np.array_equal(
        tiny_ids,
        dense_ids
    ):

        raise ValueError(
            "TinyViT and DenseNet OOF IDs do not match."
        )

    # ========================================================================
    # ALIGN TABULAR DATA
    # ========================================================================

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

    # ========================================================================
    # TRAIN/VAL FEATURE MATRIX
    # ========================================================================

    X = np.concatenate(
        [
            tiny_oof[:, None],
            dense_oof[:, None],
            tab_full[
                tab_cols
            ].values.astype(
                np.float32
            )
        ],
        axis=1
    )

    y = (
        tab_full[
            "target"
        ]
        .values
        .astype(int)
    )

    groups = (
        tab_full[
            "patient_id"
        ]
        .values
    )

    # ========================================================================
    # TEST
    # ========================================================================

    test_manifest = pd.read_csv(
        r"F:\thesis 23\tinyvit experiment\cohort_manifests\test_manifest.csv"
    )

    tab_test = (
        tab_test
        .set_index("isic_id")
        .loc[
            test_manifest[
                "isic_id"
            ].values
        ]
        .reset_index()
    )

    tiny_test = np.load(
        r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\test_probs.npy"
    )

    dense_test = np.load(
        r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\test_probs.npy"
    )

    # ========================================================================
    # TEST SIZE CHECKS
    # ========================================================================

    if len(tiny_test) != len(tab_test):

        raise ValueError(
            f"TinyViT test predictions "
            f"({len(tiny_test)}) do not match "
            f"test rows ({len(tab_test)})."
        )

    if len(dense_test) != len(tab_test):

        raise ValueError(
            f"DenseNet test predictions "
            f"({len(dense_test)}) do not match "
            f"test rows ({len(tab_test)})."
        )

    # ========================================================================
    # TEST FEATURE MATRIX
    # ========================================================================

    X_test = np.concatenate(
        [
            tiny_test[:, None],
            dense_test[:, None],
            tab_test[
                tab_cols
            ].values.astype(
                np.float32
            )
        ],
        axis=1
    )

    y_test = (
        tab_test[
            "target"
        ]
        .values
        .astype(int)
    )

    # ========================================================================
    # SAVE
    # ========================================================================

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
    print("KAN FUSION")
    print("=" * 70)

    print(
        f"Device: {Config.device}"
    )

    print(
        f"Features: TinyViT OOF + DenseNet OOF + tabular"
    )

    print(
        f"KAN hidden dimension: {Config.hidden_dim}"
    )

    print(
        f"Grid size: {Config.grid_size}"
    )

    print(
        f"Spline order: {Config.spline_order}"
    )

    print(
        f"Epochs: {Config.epochs}"
    )

    print(
        f"Patience: {Config.patience}"
    )

    print(
        f"pAUC min TPR: {Config.min_tpr}"
    )

    print("=" * 70)

    # ========================================================================
    # LOAD FEATURES
    # ========================================================================

    (
        X,
        X_test,
        y,
        y_test,
        groups
    ) = load_or_build_features()

    print(
        f"X: {X.shape} | "
        f"Positives: {y.sum()}"
    )

    print(
        f"X_test: {X_test.shape} | "
        f"Positives: {y_test.sum()}"
    )

    print(
        f"Number of features: {X.shape[1]}"
    )

    # ========================================================================
    # PATIENT-WISE STRATIFIED CV
    # ========================================================================

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

    # ========================================================================
    # TRAIN FOLDS
    # ========================================================================

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
            f"Train: {len(tr)} samples | "
            f"positives: {y[tr].sum()}"
        )

        print(
            f"Val:   {len(va)} samples | "
            f"positives: {y[va].sum()}"
        )

        (
            test_probs,
            best_val_pauc,
            best_epoch
        ) = train_kan_fold(
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
            f"Best val pAUC: "
            f"{best_val_pauc:.4f}"
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

    # ========================================================================
    # FIVE-FOLD TEST ENSEMBLE
    # ========================================================================

    test_preds = np.mean(
        test_preds_per_fold,
        axis=0
    )

    # ========================================================================
    # TEST METRICS
    # ========================================================================

    kan_pauc = isic2024_pauc(
        y_test,
        test_preds,
        min_tpr=Config.min_tpr
    )

    kan_auroc = roc_auc_score(
        y_test,
        test_preds
    )

    mean_val_pauc = np.mean(
        fold_results
    )

    std_val_pauc = np.std(
        fold_results
    )

    # ========================================================================
    # RESULTS
    # ========================================================================

    print("\n" + "=" * 70)

    print(
        "KAN FUSION RESULTS"
    )

    print("=" * 70)

    print(
        f"Test pAUC:          "
        f"{kan_pauc:.4f}"
    )

    print(
        f"Test AUROC:         "
        f"{kan_auroc:.4f}"
    )

    print(
        f"Per-fold val pAUC:  "
        f"{mean_val_pauc:.4f} ± "
        f"{std_val_pauc:.4f}"
    )

    print(
        f"Best epochs:        "
        f"{best_epochs}"
    )

    # ========================================================================
    # CATBOOST COMPARISON
    # ========================================================================

    cat_path = os.path.join(
        Config.model_dir,
        "tinyvit+densenet_test_cat.npy"
    )

    if os.path.exists(
        cat_path
    ):

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

        print(
            "COMPARISON: CatBoost vs KAN"
        )

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
            f"{'KAN':<20}"
            f"{kan_pauc:>10.4f}"
            f"{kan_auroc:>10.4f}"
        )

        print(
            f"{'Δ (KAN - Cat)':<20}"
            f"{kan_pauc - cat_pauc:>+10.4f}"
            f"{kan_auroc - cat_auroc:>+10.4f}"
        )

    # ========================================================================
    # SAVE
    # ========================================================================

    np.save(
        os.path.join(
            Config.output_dir,
            "kan_test_probs.npy"
        ),
        test_preds
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "kan_fold_val_pauc.npy"
        ),
        np.asarray(
            fold_results,
            dtype=np.float32
        )
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "kan_best_epochs.npy"
        ),
        np.asarray(
            best_epochs,
            dtype=np.int32
        )
    )

    print("\nSaved:")

    print(
        os.path.join(
            Config.output_dir,
            "kan_test_probs.npy"
        )
    )

    print(
        os.path.join(
            Config.output_dir,
            "kan_fold_val_pauc.npy"
        )
    )

    print(
        os.path.join(
            Config.output_dir,
            "kan_best_epochs.npy"
        )
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()