"""
perturbation_test_all_models.py
Border vs lesion vs random perturbation test on 30 malignant test images.
Runs for TinyViT-5M and DenseNet121.

For each image:
  1. p_original
  2. p_border_masked (outer 20% replaced with mean interior color)
  3. p_lesion_masked (center 60% replaced with mean border color)
  4. p_random_masked (same-size square as lesion, at random offset)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm


class Config:
    manifest_dir = r"F:\thesis 23\tinyvit experiment\cohort_manifests"
    image_dir = r"F:\thesis 23\ISIC_2024_Training_Input\ISIC_2024_Training_Input"
    output_dir = r"./analysis/perturbation_test"

    tinyvit_ckpt = r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\best_model_fold_1.pt"
    densenet_ckpt = r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\best_model_fold_1.pt"

    image_size = 224
    border_frac = 0.20
    lesion_frac = 0.60
    device = torch.device("cpu")
    seed = 42


# ============================================================================
# MODELS
# ============================================================================
class TinyViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tiny_vit_5m_224", pretrained=False, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(self.feature_dim, 1))

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
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(self.feature_dim, 1))

    def forward(self, x):
        fm = self.backbone.forward_features(x)
        if fm.ndim == 4:
            fm = fm.mean(dim=(2, 3))
        return self.classifier(fm).squeeze(1)


# ============================================================================
# TRANSFORMS AND HELPERS
# ============================================================================
def get_transform():
    return transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_image(isic_id):
    for ext in ["", ".jpg", ".jpeg", ".png"]:
        p = os.path.join(Config.image_dir, f"{isic_id}{ext}")
        if os.path.exists(p):
            return Image.open(p).convert("RGB").resize((Config.image_size, Config.image_size))
    return None


def mask_border(pil_img, frac):
    """Replace outer frac of image with mean interior color."""
    img = np.array(pil_img).copy()
    h, w = img.shape[:2]
    mh, mw = int(frac * h), int(frac * w)

    interior = img[mh:h-mh, mw:w-mw]
    mean_color = interior.reshape(-1, 3).mean(axis=0).astype(np.uint8) if interior.size > 0 else np.array([128, 128, 128], dtype=np.uint8)

    img[:mh, :] = mean_color
    img[-mh:, :] = mean_color
    img[:, :mw] = mean_color
    img[:, -mw:] = mean_color
    return Image.fromarray(img)


def mask_lesion(pil_img, frac):
    """Replace center frac of image (lesion region) with mean border color."""
    img = np.array(pil_img).copy()
    h, w = img.shape[:2]
    mh = int((1 - frac) / 2 * h)
    mw = int((1 - frac) / 2 * w)

    # Border region for mean color
    border_pixels = np.concatenate([
        img[:mh, :].reshape(-1, 3),
        img[-mh:, :].reshape(-1, 3),
        img[:, :mw].reshape(-1, 3),
        img[:, -mw:].reshape(-1, 3),
    ]) if mh > 0 and mw > 0 else img.reshape(-1, 3)

    mean_color = border_pixels.mean(axis=0).astype(np.uint8)

    # Mask the center
    img[mh:h-mh, mw:w-mw] = mean_color
    return Image.fromarray(img)


def mask_random(pil_img, frac, rng):
    """Replace a random square of size (frac×frac) with mean color of the region outside it."""
    img = np.array(pil_img).copy()
    h, w = img.shape[:2]
    sh = int(frac * h)
    sw = int(frac * w)

    # Random top-left corner, avoid going out of bounds
    top = rng.integers(0, h - sh + 1)
    left = rng.integers(0, w - sw + 1)

    # Compute mean color from the rest of the image
    mask = np.ones((h, w), dtype=bool)
    mask[top:top+sh, left:left+sw] = False
    rest_pixels = img[mask]
    mean_color = rest_pixels.mean(axis=0).astype(np.uint8)

    img[top:top+sh, left:left+sw] = mean_color
    return Image.fromarray(img)


@torch.no_grad()
def predict(model, pil_img, transform):
    x = transform(pil_img).unsqueeze(0).to(Config.device)
    return torch.sigmoid(model(x)).item()


# ============================================================================
# PERTURBATION TEST
# ============================================================================
def run_perturbation_test(model, malignant_ids, label):
    """Compute Δ_border, Δ_lesion, Δ_random for each image."""
    transform = get_transform()
    rng = np.random.default_rng(Config.seed)

    rows = []

    for isic_id in malignant_ids:
        img = load_image(isic_id)
        if img is None:
            continue

        p_orig = predict(model, img, transform)
        p_border = predict(model, mask_border(img, Config.border_frac), transform)
        p_lesion = predict(model, mask_lesion(img, Config.lesion_frac), transform)
        p_random = predict(model, mask_random(img, Config.lesion_frac, rng), transform)

        delta_border = abs(p_orig - p_border)
        delta_lesion = abs(p_orig - p_lesion)
        delta_random = abs(p_orig - p_random)

        rows.append({
            "isic_id": isic_id,
            "p_orig": p_orig,
            "p_border": p_border,
            "p_lesion": p_lesion,
            "p_random": p_random,
            "delta_border": delta_border,
            "delta_lesion": delta_lesion,
            "delta_random": delta_random,
        })

    df = pd.DataFrame(rows)

    print(f"\n{'=' * 70}")
    print(f"PERTURBATION TEST: {label}")
    print(f"{'=' * 70}")
    print(f"  n = {len(df)}")
    print(f"\n  Mean Δ_border: {df['delta_border'].mean():.4f}  (median: {df['delta_border'].median():.4f})")
    print(f"  Mean Δ_lesion: {df['delta_lesion'].mean():.4f}  (median: {df['delta_lesion'].median():.4f})")
    print(f"  Mean Δ_random: {df['delta_random'].mean():.4f}  (median: {df['delta_random'].median():.4f})")

    # Key comparisons
    ratio_lesion_border = df['delta_lesion'].mean() / max(df['delta_border'].mean(), 1e-8)
    ratio_border_random = df['delta_border'].mean() / max(df['delta_random'].mean(), 1e-8)

    print(f"\n  Δ_lesion / Δ_border = {ratio_lesion_border:.2f}")
    print(f"  Δ_border / Δ_random = {ratio_border_random:.2f}")

    if ratio_lesion_border > 1.5:
        print(f"  ✓ Model depends MORE on lesion than border")
    elif ratio_lesion_border < 0.67:
        print(f"  ✓ Model depends MORE on border than lesion")
    else:
        print(f"  ~ Model depends roughly equally on lesion and border")

    if ratio_border_random < 1.5:
        print(f"  ✓ Border masking ≈ random masking → model does NOT use borders")
    else:
        print(f"  ✓ Border masking > random masking → model partially uses borders")

    return df


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    df_test = pd.read_csv(os.path.join(Config.manifest_dir, "test_manifest.csv"))
    y_test = df_test["target"].values.astype(int)
    malignant_ids = df_test[y_test == 1]["isic_id"].values
    print(f"Malignant test lesions: {len(malignant_ids)}")

    # TinyViT
    print(f"\n{'=' * 70}\nLoading TinyViT-5M\n{'=' * 70}")
    tiny_model = TinyViTModel().to(Config.device)
    tiny_model.load_state_dict(torch.load(Config.tinyvit_ckpt, map_location=Config.device, weights_only=False))
    tiny_model.eval()
    df_tiny = run_perturbation_test(tiny_model, malignant_ids, "TinyViT-5M")

    # DenseNet
    print(f"\n{'=' * 70}\nLoading DenseNet121\n{'=' * 70}")
    dense_model = DenseNetModel().to(Config.device)
    dense_model.load_state_dict(torch.load(Config.densenet_ckpt, map_location=Config.device, weights_only=False))
    dense_model.eval()
    df_dense = run_perturbation_test(dense_model, malignant_ids, "DenseNet121")

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY: PERTURBATION SENSITIVITY")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Δ_border':>12} {'Δ_lesion':>12} {'Δ_random':>12}")
    print("-" * 70)
    print(f"{'TinyViT-5M':<20} {df_tiny['delta_border'].mean():>12.4f} "
          f"{df_tiny['delta_lesion'].mean():>12.4f} {df_tiny['delta_random'].mean():>12.4f}")
    print(f"{'DenseNet121':<20} {df_dense['delta_border'].mean():>12.4f} "
          f"{df_dense['delta_lesion'].mean():>12.4f} {df_dense['delta_random'].mean():>12.4f}")

    # Save
    df_tiny.to_csv(os.path.join(Config.output_dir, "perturbation_tinyvit.csv"), index=False)
    df_dense.to_csv(os.path.join(Config.output_dir, "perturbation_densenet.csv"), index=False)
    print(f"\n✓ Saved to {Config.output_dir}")


if __name__ == "__main__":
    main()