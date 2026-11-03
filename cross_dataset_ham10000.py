"""
cross_dataset_final_backbones.py
Zero-shot evaluation of TinyViT-5M and DenseNet121 (from final model)
on HAM10000. Image-only (no tabular features).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import sklearn.metrics as sklearn_metrics


class Config:
    ham_dir = r"F:\thesis 23\HAM"
    ham_metadata = r"F:\thesis 23\HAM\HAM10000_metadata"
    output_dir = r"./analysis/ham10000_final_backbones"

    tinyvit_ckpt_dir = r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN"
    densenet_ckpt_dir = r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold"

    image_size = 224
    batch_size = 64
    num_workers = 4
    n_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}


def isic2024_pauc(v_gt, v_pred, min_tpr=0.80):
    v_gt, v_pred = np.asarray(v_gt), np.asarray(v_pred)
    if len(np.unique(v_gt)) != 2:
        return 0.0
    yf, pf = abs(v_gt - 1), abs(v_pred - 1)
    max_fpr = 1 - min_tpr
    fpr, tpr, _ = sklearn_metrics.roc_curve(yf, pf)
    stop = np.searchsorted(fpr, max_fpr, "right")
    tpr_i = np.append(tpr[:stop], np.interp(max_fpr, [fpr[stop-1], fpr[stop]], [tpr[stop-1], tpr[stop]]))
    fpr_i = np.append(fpr[:stop], max_fpr)
    return float(sklearn_metrics.auc(fpr_i, tpr_i))


# ============================================================================
# MODELS — match the final model's backbones
# ============================================================================
class TinyViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tiny_vit_5m_224", pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features  # 320
        self.classifier = nn.Sequential(
            nn.Dropout(0.6), nn.Linear(self.feature_dim, 1))

    def forward(self, x):
        fm = self.backbone.forward_features(x)
        if fm.ndim == 4:
            fm = fm.mean(dim=(2, 3))
        elif fm.ndim == 3:
            fm = fm.mean(dim=1)
        return self.classifier(fm).squeeze(1)


class DenseNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("densenet121", pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features  # 1024
        self.classifier = nn.Sequential(
            nn.Dropout(0.6), nn.Linear(self.feature_dim, 1))

    def forward(self, x):
        fm = self.backbone.forward_features(x)
        if fm.ndim == 4:
            fm = fm.mean(dim=(2, 3))
        return self.classifier(fm).squeeze(1)


class HAM10000Dataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve(self, image_id):
        for ext in [".jpg", ".jpeg", ".png"]:
            matches = list(self.image_dir.rglob(f"{image_id}{ext}"))
            if matches:
                return matches[0]
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = self._resolve(row["image_id"])
        if p is None:
            raise FileNotFoundError(f"Missing image: {row['image_id']}")
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        target = torch.tensor(int(row["binary_target"]), dtype=torch.float32)
        return img, target


def get_transform():
    return transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_targets = [], []
    for images, targets in loader:
        images = images.to(Config.device, non_blocking=True)
        logits = model(images)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(targets.numpy())
    return np.concatenate(all_targets).astype(int), np.concatenate(all_probs)


def evaluate_backbone(model_class, ckpt_dir, loader, name):
    """Evaluate one backbone across 5 folds."""
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    all_fold_probs = []
    y_true = None

    for fold in range(1, Config.n_folds + 1):
        ckpt_path = os.path.join(ckpt_dir, f"best_model_fold_{fold}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Missing: {ckpt_path}")
            continue

        model = model_class().to(Config.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.device, weights_only=False))
        model.eval()

        targets, probs = evaluate(model, loader)
        if y_true is None:
            y_true = targets
        all_fold_probs.append(probs)
        print(f"  Fold {fold} evaluated")
        del model
        torch.cuda.empty_cache()

    probs = np.mean(all_fold_probs, axis=0)

    auroc = sklearn_metrics.roc_auc_score(y_true, probs)
    pauc = isic2024_pauc(y_true, probs)
    auprc = sklearn_metrics.average_precision_score(y_true, probs)

    print(f"\n{name} on HAM10000:")
    print(f"  n_samples:  {len(y_true)}")
    print(f"  n_malignant: {y_true.sum()}")
    print(f"  AUROC:      {auroc:.4f}")
    print(f"  AUPRC:      {auprc:.4f}")
    print(f"  pAUC@80%:   {pauc:.4f}")

    return probs, y_true, auroc, pauc, auprc


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    df = pd.read_csv(Config.ham_metadata)
    print(f"HAM10000 metadata: {len(df)} images")
    df["binary_target"] = df["dx"].apply(lambda x: 1 if x in MALIGNANT_CLASSES else 0)
    print(f"Malignant: {(df['binary_target'] == 1).sum()}")
    print(f"Benign:    {(df['binary_target'] == 0).sum()}")

    transform = get_transform()
    dataset = HAM10000Dataset(df, Config.ham_dir, transform)
    loader = DataLoader(
        dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=Config.num_workers)

    results = {}

    # TinyViT
    tiny_probs, y_true, tiny_auroc, tiny_pauc, tiny_auprc = evaluate_backbone(
        TinyViTModel, Config.tinyvit_ckpt_dir, loader, "TinyViT-5M")

    # DenseNet
    dense_probs, _, dense_auroc, dense_pauc, dense_auprc = evaluate_backbone(
        DenseNetModel, Config.densenet_ckpt_dir, loader, "DenseNet121")

    # Simple average ensemble
    ensemble_probs = (tiny_probs + dense_probs) / 2
    ens_auroc = sklearn_metrics.roc_auc_score(y_true, ensemble_probs)
    ens_pauc = isic2024_pauc(y_true, ensemble_probs)
    ens_auprc = sklearn_metrics.average_precision_score(y_true, ensemble_probs)

    print(f"\n{'=' * 70}")
    print("ENSEMBLE (average of TinyViT + DenseNet)")
    print(f"{'=' * 70}")
    print(f"  AUROC:    {ens_auroc:.4f}")
    print(f"  AUPRC:    {ens_auprc:.4f}")
    print(f"  pAUC@80%: {ens_pauc:.4f}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("CROSS-DATASET GENERALIZATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<30} {'AUROC':>10} {'AUPRC':>10} {'pAUC@80':>10}")
    print("-" * 70)
    print(f"{'TinyViT-5M (ISIC 2024)':<30} {tiny_auroc:>10.4f} {tiny_auprc:>10.4f} {tiny_pauc:>10.4f}")
    print(f"{'DenseNet121 (ISIC 2024)':<30} {dense_auroc:>10.4f} {dense_auprc:>10.4f} {dense_pauc:>10.4f}")
    print(f"{'Ensemble (avg)':<30} {ens_auroc:>10.4f} {ens_auprc:>10.4f} {ens_pauc:>10.4f}")

    print(f"\n{'=' * 70}")
    print("COMPARISON TO ISIC 2024 TEST")
    print(f"{'=' * 70}")
    print(f"{'Model':<30} {'ISIC 2024 pAUC':>15} {'HAM10000 pAUC':>15} {'Drop':>10}")
    print("-" * 70)
    print(f"{'TinyViT-5M image-only':<30} {'0.1427':>15} {tiny_pauc:>15.4f} "
          f"{(0.1427 - tiny_pauc):>+10.4f}")
    print(f"{'DenseNet121 image-only':<30} {'0.1570':>15} {dense_pauc:>15.4f} "
          f"{(0.1570 - dense_pauc):>+10.4f}")
    print(f"\nNote: ISIC 2024 uses 3D-TBP clinical images.")
    print(f"      HAM10000 uses dermoscopic images.")
    print(f"      Domain shift is expected.")

    # Save
    np.save(os.path.join(Config.output_dir, "tinyvit_probs.npy"), tiny_probs)
    np.save(os.path.join(Config.output_dir, "densenet_probs.npy"), dense_probs)
    np.save(os.path.join(Config.output_dir, "y_true.npy"), y_true)

    pd.DataFrame([
        {"model": "TinyViT-5M", "auroc": tiny_auroc, "auprc": tiny_auprc, "pauc": tiny_pauc},
        {"model": "DenseNet121", "auroc": dense_auroc, "auprc": dense_auprc, "pauc": dense_pauc},
        {"model": "Ensemble", "auroc": ens_auroc, "auprc": ens_auprc, "pauc": ens_pauc},
    ]).to_csv(os.path.join(Config.output_dir, "results.csv"), index=False)

    print(f"\n✓ Saved to {Config.output_dir}")


if __name__ == "__main__":
    main()