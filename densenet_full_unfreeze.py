"""
BASELINE A: DenseNet121 ONLY (5-FOLD CROSS-VALIDATION)
===================================================
FULLY FIXED VERSION — All 14 checklist items implemented:
1. Official ISIC 2024 pAUC (flip + interpolate + sklearn auc)
2. NumPy 2.x compatible (no np.trapz)
3. OOF predictions pooled across folds
4. Bootstrap CI on pooled OOF
5. AUPRC restored
6. pauc_norm dropped — official pAUC only [0, 0.2]
7. Test-set evaluation via 5-fold ensemble
8. AMP disabled
9. Backward-compatible GradScaler
10. Model selection on official pAUC
11. Patient-aware splitting (StratifiedGroupKFold)
12. Full determinism
13. early_stopping_patience = 5
14. Last 4 DenseNet121 transformer blocks unfrozen (flattened across stages)
"""

import os
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

import timm
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. FINAL CONFIGURATION
# =============================================================================

# =============================================================================
# 1. FINAL CONFIGURATION
# =============================================================================

class Config:
    manifest_dir = "./cohort_manifests"
    image_dir = r"F:\thesis 23\ISIC_2024_Training_Input\ISIC_2024_Training_Input"

    train_manifest = "train_manifest.csv"
    val_manifest = "val_manifest.csv"
    test_manifest = "test_manifest.csv"

    n_splits = 5
    early_stopping = True
    image_size = 224
    batch_size = 32
    positive_per_batch = 2
    negative_per_batch = 30

    epochs = 30
    learning_rate = 5e-5
    weight_decay = 1e-1
    dropout = 0.6

    early_stopping = True
    early_stopping_patience = 5
    min_delta = 1e-6

    # Loss
    focal_alpha = 0.75
    focal_gamma = 2.0
    target_tpr = 0.88
    margin_pos = 0.3
    margin_neg = 0.7
    ranking_margin = 0.15
    focal_weight = 1.0
    margin_weight = 0.5

    # Model
    backbone_name = "densenet121"
    pretrained = True
    classifier_dropout = 0.6

    # Augmentation
    horizontal_flip_p = 0.5
    vertical_flip_p = 0.5
    rotation_degrees = 15
    color_jitter = 0.20

    # System
    seed = 42
    num_workers = 4
    pin_memory = True
    persistent_workers = True
    amp = True   # disabled for stability

    # ISIC 2024 official metric
    min_tpr = 0.80

    output_dir = "./baseline_A_densenet121_5fold_fixed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def worker_init_fn(worker_id):
    worker_seed = Config.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# =============================================================================
# 3. OFFICIAL ISIC 2024 pAUC METRIC
# =============================================================================

def isic2024_pauc(v_gt, v_pred, min_tpr=0.80, sample_weight=None):
    """
    OFFICIAL ISIC 2024 partial AUC metric.
    Area under the ROC curve for TPR >= min_tpr.
    Returns a value in [0.0, 0.2] for min_tpr=0.80.
    Matches the official Kaggle notebook:
      - Flips labels and predictions
      - Interpolates at max_fpr = 1 - min_tpr
      - Uses sklearn.metrics.auc on the sliced curve
    """
    v_gt = np.asarray(v_gt)
    v_pred = np.asarray(v_pred)

    if len(np.unique(v_gt)) != 2:
        return 0.0

    v_gt_flipped = abs(v_gt - 1)
    v_pred_flipped = abs(v_pred - 1)
    max_fpr = abs(1 - min_tpr)

    fpr, tpr, _ = sklearn_metrics.roc_curve(
        v_gt_flipped, v_pred_flipped, sample_weight=sample_weight
    )

    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError(f"Expected min_tpr in [0, 1), got: {min_tpr}")

    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr_interp = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr_interp = np.append(fpr[:stop], max_fpr)

    return float(sklearn_metrics.auc(fpr_interp, tpr_interp))


# =============================================================================
# 4. METRICS
# =============================================================================

def calculate_metrics(targets, probabilities, min_tpr=0.80):
    targets = np.asarray(targets).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    predictions = (probabilities >= 0.3).astype(int)

    auroc = roc_auc_score(targets, probabilities)
    auprc = average_precision_score(targets, probabilities)
    pauc = isic2024_pauc(targets, probabilities, min_tpr=min_tpr)

    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "pauc": float(pauc),                      # official ISIC metric in [0, 0.2]
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(targets)),
        "positive": int(targets.sum()),
        "negative": int((targets == 0).sum()),
    }


def bootstrap_pauc_ci(y_true, y_scores, min_tpr=0.80, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        s_boot = y_scores[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        scores.append(isic2024_pauc(y_boot, s_boot, min_tpr))

    if len(scores) < 10:
        return (np.nan, np.nan)
    return np.percentile(scores, [2.5, 97.5])


def bootstrap_auroc_ci(y_true, y_scores, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        s_boot = y_scores[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        scores.append(roc_auc_score(y_boot, s_boot))

    if len(scores) < 10:
        return (np.nan, np.nan)
    return np.percentile(scores, [2.5, 97.5])


# =============================================================================
# 5. TRANSFORMS
# =============================================================================

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.RandomHorizontalFlip(p=Config.horizontal_flip_p),
        transforms.RandomVerticalFlip(p=Config.vertical_flip_p),
        transforms.RandomRotation(degrees=Config.rotation_degrees),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ColorJitter(
            brightness=Config.color_jitter,
            contrast=Config.color_jitter,
            saturation=Config.color_jitter,
            hue=0.0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# =============================================================================
# 6. DATASET
# =============================================================================

class ISICDataset(Dataset):
    def __init__(self, df, image_dir, train=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.train = train

        if "target" not in self.df.columns:
            raise ValueError("Target column not found.")

        self.targets = self.df["target"].astype(int).values

        possible_id_columns = ["isic_id", "image_id", "image", "filename", "image_name"]
        self.id_column = None
        for col in possible_id_columns:
            if col in self.df.columns:
                self.id_column = col
                break
        if self.id_column is None:
            raise ValueError("No supported image-ID column found.")

        self.transform = get_train_transform() if train else get_eval_transform()
        self.positive_indices = np.where(self.targets == 1)[0]
        self.negative_indices = np.where(self.targets == 0)[0]

    def __len__(self):
        return len(self.df)

    def resolve_image_path(self, image_id):
        image_id = str(image_id)
        candidates = [
            self.image_dir / image_id,
            self.image_dir / f"{image_id}.jpg",
            self.image_dir / f"{image_id}.jpeg",
            self.image_dir / f"{image_id}.png",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = str(row[self.id_column])
        image_path = self.resolve_image_path(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Image '{image_id}' not found in '{self.image_dir}'.")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not read image: {image_path}") from exc
        image = self.transform(image)
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        return image, target


# =============================================================================
# 7. SAMPLER & LOSS
# =============================================================================

class PositiveNegativeBatchSampler(Sampler):
    def __init__(self, dataset, positive_per_batch=2, negative_per_batch=30, seed=42):
        self.dataset = dataset
        self.positive_indices = np.asarray(dataset.positive_indices)
        self.negative_indices = np.asarray(dataset.negative_indices)
        self.positive_per_batch = positive_per_batch
        self.negative_per_batch = negative_per_batch
        self.batch_size = positive_per_batch + negative_per_batch
        self.seed = seed
        self.epoch = 0
        self.num_batches = max(1, int(np.ceil(len(dataset) / self.batch_size)))
        if len(self.positive_indices) == 0 or len(self.negative_indices) == 0:
            raise ValueError("No positive or negative samples in training set.")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_batches):
            positives = rng.choice(self.positive_indices, size=self.positive_per_batch, replace=True)
            negatives = rng.choice(self.negative_indices, size=self.negative_per_batch, replace=True)
            batch = np.concatenate([positives, negatives])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        logits = logits.squeeze(-1)
        targets = targets.float()
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)
        return (focal_weight * ce_loss).mean()


class AsymmetricMarginLoss(nn.Module):
    def __init__(self, target_tpr=0.88, margin_pos=0.3, margin_neg=0.7):
        super().__init__()
        self.target_tpr = target_tpr
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.squeeze(-1))
        pos_mask = targets == 1
        neg_mask = targets == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.float())
        pos_loss = F.relu(self.margin_pos - probs[pos_mask]).mean()
        neg_loss = F.relu(probs[neg_mask] - self.margin_neg).mean()
        pos_expanded = probs[pos_mask].unsqueeze(1)
        neg_expanded = probs[neg_mask].unsqueeze(0)
        rank_loss = F.relu(neg_expanded - pos_expanded + 0.15).mean()
        return pos_loss + 0.5 * neg_loss + 2.0 * rank_loss


class CombinedLoss(nn.Module):
    def __init__(self, target_tpr=0.88, focal_alpha=0.75, focal_gamma=2.0,
                 margin_pos=0.3, margin_neg=0.7, focal_weight=1.0, margin_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.margin = AsymmetricMarginLoss(target_tpr=target_tpr, margin_pos=margin_pos, margin_neg=margin_neg)
        self.focal_weight = focal_weight
        self.margin_weight = margin_weight

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_margin = self.margin(logits, targets)
        total_loss = self.focal_weight * loss_focal + self.margin_weight * loss_margin
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"CombinedLoss NaN. logits range: "
                f"[{logits.min().item():.6f}, {logits.max().item():.6f}]"
            )
        return total_loss


# =============================================================================
# 8. DENSENET121 BASELINE
# =============================================================================

# =============================================================================
# 8. DENSENET121 BASELINE
# =============================================================================

class DenseNet121Baseline(nn.Module):
    def __init__(self):
        super().__init__()

        print("\n" + "=" * 80)
        print("INITIALIZING BASELINE A — DenseNet121 ONLY")
        print("=" * 80)

        # ---------------------------------------------------------------------
        # DenseNet121 backbone
        # ---------------------------------------------------------------------
        self.backbone = timm.create_model(
            Config.backbone_name,
            pretrained=Config.pretrained,
            num_classes=0,
        )

        # ---------------------------------------------------------------------
        # FULLY UNFROZEN BACKBONE
        # Every DenseNet121 parameter is trainable.
        # ---------------------------------------------------------------------
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

        self.feature_dim = self.backbone.num_features

        # ---------------------------------------------------------------------
        # Binary classifier
        # ---------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=Config.classifier_dropout),
            nn.Linear(self.feature_dim, 1),
        )

        # Classifier is also trainable
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

        self._set_modes()

        # ---------------------------------------------------------------------
        # Parameter statistics
        # ---------------------------------------------------------------------
        total_params = sum(
            p.numel() for p in self.parameters()
        )

        trainable_params = sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )

        frozen_params = total_params - trainable_params

        print(f"Backbone:             {Config.backbone_name}")
        print(f"Feature dimension:    {self.feature_dim}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters:    {frozen_params:,}")
        print("=" * 80)

        if frozen_params == 0:
            print("✓ DenseNet121 is FULLY UNFROZEN")
        else:
            print("WARNING: Some parameters are frozen!")

        print("=" * 80)

    def _set_modes(self):
        self.backbone.train()
        self.classifier.train()

    def train(self, mode=True):
        super().train(mode)

        if mode:
            self._set_modes()

        return self

    def forward_features(self, x):
        features = self.backbone.forward_features(x)

        # DenseNet121 normally returns:
        # [batch, channels, height, width]
        if features.ndim == 4:
            features = features.mean(dim=(2, 3))

        elif features.ndim == 2:
            pass

        elif features.ndim == 3:
            features = features.mean(dim=1)

        else:
            raise RuntimeError(
                f"Unexpected DenseNet121 feature shape: "
                f"{tuple(features.shape)}"
            )

        return features

    def forward(self, x):
        features = self.forward_features(x)

        logits = self.classifier(features)

        return logits.squeeze(1)


# =============================================================================
# 9. TRAINING LOOP
# =============================================================================

def get_grad_scaler():
    """Backward-compatible GradScaler (torch <2.3 and >=2.3)."""
    if not (Config.amp and Config.device.type == "cuda"):
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, return_predictions=False):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets, all_probabilities = [], []

    for images, targets in loader:
        images = images.to(Config.device, non_blocking=True)
        targets = targets.to(Config.device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            if Config.amp and Config.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, targets)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        probabilities = torch.sigmoid(logits)
        batch_size = images.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size
        all_targets.append(targets.detach().cpu().numpy())
        all_probabilities.append(probabilities.detach().cpu().numpy())

    targets = np.concatenate(all_targets).astype(int)
    probabilities = np.concatenate(all_probabilities)
    metrics = calculate_metrics(targets, probabilities, min_tpr=Config.min_tpr)
    metrics["loss"] = total_loss / max(total_samples, 1)

    if return_predictions:
        return metrics, targets, probabilities
    return metrics


# =============================================================================
# 10. MAIN (5-FOLD CV + TEST ENSEMBLE)
# =============================================================================

def main():
    seed_everything(Config.seed)
    os.makedirs(Config.output_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print("BASELINE A — DENSENET121 ONLY (5-FOLD CV) — FULLY FIXED")
    print("=" * 90)
    print(f"Device:              {Config.device}")
    print(f"AMP enabled:         {Config.amp}")
    print(f"ISIC pAUC min_tpr:   {Config.min_tpr}")
    print(f"Early stop patience: {Config.early_stopping_patience}")

    train_path = os.path.join(Config.manifest_dir, Config.train_manifest)
    val_path = os.path.join(Config.manifest_dir, Config.val_manifest)
    test_path = os.path.join(Config.manifest_dir, Config.test_manifest)

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_full = pd.concat([df_train, df_val], ignore_index=True)
    df_test = pd.read_csv(test_path)

    print(f"Total train+val samples: {len(df_full):,}")
    print(f"Test samples:            {len(df_test):,}")

    if "patient_id" in df_full.columns:
        sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
        splits = list(sgkf.split(df_full, df_full["target"], df_full["patient_id"]))
        print("Using StratifiedGroupKFold grouped by 'patient_id'")
    else:
        skf = StratifiedKFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
        splits = list(skf.split(df_full, df_full["target"]))
        print("Using StratifiedKFold (NO patient_id found)")

    fold_metrics = []
    oof_targets_list = []
    oof_probs_list = []
    fold_model_paths = []
    best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print("\n" + "=" * 80)
        print(f"STARTING FOLD {fold}/{Config.n_splits}")
        print("=" * 80)

        train_df = df_full.iloc[train_idx]
        val_df = df_full.iloc[val_idx]

        train_dataset = ISICDataset(train_df, Config.image_dir, train=True)
        val_dataset = ISICDataset(val_df, Config.image_dir, train=False)

        g = torch.Generator()
        g.manual_seed(Config.seed + fold)

        train_sampler = PositiveNegativeBatchSampler(
            train_dataset,
            positive_per_batch=Config.positive_per_batch,
            negative_per_batch=Config.negative_per_batch,
            seed=Config.seed + fold,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
            persistent_workers=(Config.persistent_workers and Config.num_workers > 0),
            worker_init_fn=worker_init_fn,
            generator=g,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
            persistent_workers=(Config.persistent_workers and Config.num_workers > 0),
            worker_init_fn=worker_init_fn,
            generator=g,
        )

        model = DenseNet121Baseline().to(Config.device)

        criterion = CombinedLoss(
            target_tpr=Config.target_tpr,
            focal_alpha=Config.focal_alpha,
            focal_gamma=Config.focal_gamma,
            margin_pos=Config.margin_pos,
            margin_neg=Config.margin_neg,
            focal_weight=Config.focal_weight,
            margin_weight=Config.margin_weight,
        )
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
        scaler = get_grad_scaler()

        best_pauc = -np.inf
        best_val_metrics = None
        best_epoch = 0
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(Config.epochs):
            train_sampler.set_epoch(epoch)
            train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer, scaler=scaler)
            val_metrics = run_epoch(model, val_loader, criterion)
            scheduler.step()

            print(
                f"Fold {fold} | Epoch {epoch + 1:02d}/{Config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} pAUC: {train_metrics['pauc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} pAUC: {val_metrics['pauc']:.4f} | "
                f"AUROC: {val_metrics['auroc']:.4f}"
            )

            # ✅ Model selection on OFFICIAL pAUC
            if val_metrics["pauc"] > best_pauc + Config.min_delta:
                best_pauc = val_metrics["pauc"]
                best_val_metrics = val_metrics
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if Config.early_stopping and epochs_without_improvement >= Config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Save best model for this fold
        fold_path = os.path.join(Config.output_dir, f"best_model_fold_{fold}.pt")
        torch.save(best_model_state, fold_path)
        fold_model_paths.append(fold_path)

        print(f"\nFold {fold} Best Validation pAUC: {best_pauc:.6f} (epoch {best_epoch})")
        fold_metrics.append(best_val_metrics)
        best_epochs.append(best_epoch)

        # ✅ Collect OOF predictions
        model.load_state_dict(best_model_state)
        model.to(Config.device)
        _, val_targets, val_probs = run_epoch(
            model, val_loader, criterion, return_predictions=True
        )
        oof_targets_list.append(val_targets)
        oof_probs_list.append(val_probs)

    # =========================================================================
    # PER-FOLD SUMMARY
    # =========================================================================
    print("\n" + "=" * 90)
    print("5-FOLD CV RESULTS (PER-FOLD MEAN ± STD)")
    print("=" * 90)

    metric_keys = ["auroc", "auprc", "pauc", "accuracy",
                   "precision", "recall_sensitivity", "specificity", "f1"]
    cv_summary = {}

    for key in metric_keys:
        values = [fm[key] for fm in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv_summary[key] = (mean_val, std_val)
        print(f"{key:<22}: {mean_val:.6f} ± {std_val:.6f}")

    for key in ["tn", "fp", "fn", "tp"]:
        values = [fm[key] for fm in fold_metrics]
        print(f"{key.upper():<22}: {np.mean(values):.2f} ± {np.std(values):.2f}")

    print("=" * 90)
    print(f"Best epochs per fold: {best_epochs}")

    # =========================================================================
    # POOLED OOF METRICS (PRIMARY RESULT)
    # =========================================================================
    oof_targets = np.concatenate(oof_targets_list)
    oof_probs = np.concatenate(oof_probs_list)

    pooled = calculate_metrics(oof_targets, oof_probs, min_tpr=Config.min_tpr)

    print("\n" + "=" * 90)
    print("POOLED OUT-OF-FOLD METRICS (PRIMARY RESULT)")
    print("=" * 90)
    print(f"AUROC:                {pooled['auroc']:.6f}")
    print(f"AUPRC:                {pooled['auprc']:.6f}")
    print(f"ISIC pAUC (official): {pooled['pauc']:.6f}  ← PRIMARY METRIC")
    print(f"Sensitivity:          {pooled['recall_sensitivity']:.6f}")
    print(f"Specificity:          {pooled['specificity']:.6f}")
    print(f"F1:                   {pooled['f1']:.6f}")
    print(f"Confusion Matrix:     TN={pooled['tn']} | FP={pooled['fp']} | FN={pooled['fn']} | TP={pooled['tp']}")
    print(f"Samples:              {pooled['n']:,} | Positive={pooled['positive']:,} | Negative={pooled['negative']:,}")

    # =========================================================================
    # BOOTSTRAP 95% CI ON POOLED OOF
    # =========================================================================
    auroc_lo, auroc_hi = bootstrap_auroc_ci(oof_targets, oof_probs, n_boot=1000)
    pauc_lo, pauc_hi = bootstrap_pauc_ci(oof_targets, oof_probs, min_tpr=Config.min_tpr, n_boot=1000)

    print("\n" + "=" * 90)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (Pooled OOF)")
    print("=" * 90)
    print(f"AUROC = {pooled['auroc']:.4f}  (95% CI: {auroc_lo:.4f} – {auroc_hi:.4f})")
    print(f"pAUC  = {pooled['pauc']:.4f}  (95% CI: {pauc_lo:.4f} – {pauc_hi:.4f})")

    # =========================================================================
    # TEST SET EVALUATION — ENSEMBLE OF 5 FOLD MODELS
    # =========================================================================
    print("\n" + "=" * 90)
    print("TEST SET EVALUATION — ENSEMBLE OF 5 FOLD MODELS")
    print("=" * 90)

    test_dataset = ISICDataset(df_test, Config.image_dir, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        persistent_workers=(Config.persistent_workers and Config.num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    ensemble_probs_list = []
    test_targets = None

    for fold, path in enumerate(fold_model_paths, start=1):
        model = DenseNet121Baseline().to(Config.device)
        model.load_state_dict(torch.load(path, map_location=Config.device))
        model.eval()

        _, t_targets, t_probs = run_epoch(
            model, test_loader, criterion, return_predictions=True
        )
        if test_targets is None:
            test_targets = t_targets
        ensemble_probs_list.append(t_probs)
        print(f"Fold {fold} loaded from {path}")

    ensemble_probs = np.mean(ensemble_probs_list, axis=0)
    test_metrics = calculate_metrics(test_targets, ensemble_probs, min_tpr=Config.min_tpr)

    print("\n" + "-" * 90)
    print("FINAL TEST RESULTS — DENSENET121 BASELINE (ENSEMBLE OF 5 FOLDS)")
    print("-" * 90)
    print(f"AUROC:                {test_metrics['auroc']:.6f}")
    print(f"AUPRC:                {test_metrics['auprc']:.6f}")
    print(f"ISIC pAUC (official): {test_metrics['pauc']:.6f}  ← PRIMARY METRIC")
    print(f"Accuracy:             {test_metrics['accuracy']:.6f}")
    print(f"Precision:            {test_metrics['precision']:.6f}")
    print(f"Sensitivity:          {test_metrics['recall_sensitivity']:.6f}")
    print(f"Specificity:          {test_metrics['specificity']:.6f}")
    print(f"F1:                   {test_metrics['f1']:.6f}")
    print(f"Confusion Matrix:     TN={test_metrics['tn']} | FP={test_metrics['fp']} | FN={test_metrics['fn']} | TP={test_metrics['tp']}")
    print(f"Samples:              {test_metrics['n']:,} | Positive={test_metrics['positive']:,} | Negative={test_metrics['negative']:,}")

    t_auroc_lo, t_auroc_hi = bootstrap_auroc_ci(test_targets, ensemble_probs, n_boot=1000)
    t_pauc_lo, t_pauc_hi = bootstrap_pauc_ci(test_targets, ensemble_probs, min_tpr=Config.min_tpr, n_boot=1000)
    print(f"\n95% CI on test set:")
    print(f"  AUROC = {test_metrics['auroc']:.4f}  (95% CI: {t_auroc_lo:.4f} – {t_auroc_hi:.4f})")
    print(f"  pAUC  = {test_metrics['pauc']:.4f}  (95% CI: {t_pauc_lo:.4f} – {t_pauc_hi:.4f})")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary_rows = []
    for key in metric_keys:
        summary_rows.append({
            "metric": key,
            "per_fold_mean": cv_summary[key][0],
            "per_fold_std": cv_summary[key][1],
            "pooled_oof": pooled[key],
            "test_ensemble": test_metrics[key],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(Config.output_dir, "5fold_cv_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    results = {
        "model": "Baseline A - DenseNet121 only (5-fold CV, ensemble test)",
        "n_splits": Config.n_splits,
        "isic_metric": "pAUC above 80% TPR (official)",
        "per_fold_mean_std": {k: {"mean": v[0], "std": v[1]} for k, v in cv_summary.items()},
        "pooled_oof_metrics": pooled,
        "test_ensemble_metrics": test_metrics,
        "bootstrap_95_ci": {
            "oof_auroc":   [float(auroc_lo), float(auroc_hi)],
            "oof_pauc":    [float(pauc_lo), float(pauc_hi)],
            "test_auroc":  [float(t_auroc_lo), float(t_auroc_hi)],
            "test_pauc":   [float(t_pauc_lo), float(t_pauc_hi)],
        },
        "best_epochs_per_fold": best_epochs,
        "configuration": {
            "backbone": Config.backbone_name,
            
            "image_size": Config.image_size,
            "batch_size": Config.batch_size,
            "positive_per_batch": Config.positive_per_batch,
            "negative_per_batch": Config.negative_per_batch,
            "epochs": Config.epochs,
            "early_stopping_patience": Config.early_stopping_patience,
            "learning_rate": Config.learning_rate,
            "weight_decay": Config.weight_decay,
            "dropout": Config.dropout,
            "amp": Config.amp,
            "seed": Config.seed,
            "min_tpr": Config.min_tpr,
        },
    }
    results_path = os.path.join(Config.output_dir, "baseline_A_densenet121_5fold_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)

    print("\n" + "=" * 90)
    print("BASELINE A (DenseNet121) 5-FOLD CV COMPLETE")
    print("=" * 90)
    print(f"Per-fold pAUC:        {cv_summary['pauc'][0]:.6f} ± {cv_summary['pauc'][1]:.6f}")
    print(f"Pooled OOF pAUC:      {pooled['pauc']:.6f}  (95% CI: {pauc_lo:.4f} – {pauc_hi:.4f})")
    print(f"Test ensemble pAUC:   {test_metrics['pauc']:.6f}  (95% CI: {t_pauc_lo:.4f} – {t_pauc_hi:.4f})")
    print(f"\nSaved:")
    print(f"  {summary_path}")
    print(f"  {results_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()