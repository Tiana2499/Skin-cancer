"""
alt_attribution_occlusion.py
Occlusion sensitivity analysis for TinyViT-5M and DenseNet121.
Measures center vs border functional dependence.
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
    output_dir = r"./analysis/occlusion_final"

    tinyvit_ckpt = r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\best_model_fold_1.pt"
    densenet_ckpt = r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\best_model_fold_1.pt"

    image_size = 224
    patch_size = 32
    stride = 16
    border_frac = 0.20  # outer 20% considered "border"
    device = torch.device("cpu")


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
            return Image.open(p).convert("RGB")
    return None


@torch.no_grad()
def occlusion_sensitivity(model, x, patch_size=32, stride=16):
    """Compute occlusion sensitivity heatmap."""
    B, C, H, W = x.shape
    baseline = torch.sigmoid(model(x)).item()

    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            x_occ = x.clone()
            # Mask with mean color (equivalent to zeroing after normalization)
            x_occ[:, :, i:i + patch_size, j:j + patch_size] = 0
            p_occ = torch.sigmoid(model(x_occ)).item()
            delta = baseline - p_occ
            heatmap[i:i + patch_size, j:j + patch_size] += delta
            counts[i:i + patch_size, j:j + patch_size] += 1

    heatmap = heatmap / np.maximum(counts, 1)
    return heatmap, baseline


def run_occlusion(model, malignant_ids, label):
    transform = get_transform()
    center_sens = []
    border_sens = []

    for idx, isic_id in enumerate(malignant_ids):
        img = load_image(isic_id)
        if img is None:
            continue
        try:
            x = transform(img).unsqueeze(0).to(Config.device)
            heatmap, _ = occlusion_sensitivity(
                model, x, patch_size=Config.patch_size, stride=Config.stride)

            margin = int(Config.border_frac * Config.image_size)
            center = heatmap[margin:-margin, margin:-margin]
            border_mask = np.ones_like(heatmap, dtype=bool)
            border_mask[margin:-margin, margin:-margin] = False
            border = heatmap[border_mask]

            center_sens.append(center.mean())
            border_sens.append(border.mean())
        except Exception as e:
            print(f"  ⚠️ {isic_id}: {type(e).__name__}: {e}")
            continue

        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(malignant_ids)}")

    center_sens = np.array(center_sens)
    border_sens = np.array(border_sens)

    print(f"\n[{label}] n={len(center_sens)}")
    print(f"  Mean center sensitivity: {center_sens.mean():.4f}")
    print(f"  Mean border sensitivity: {border_sens.mean():.4f}")
    print(f"  Ratio (center/border):   {center_sens.mean() / max(border_sens.mean(), 1e-8):.3f}")

    if center_sens.mean() > border_sens.mean():
        print(f"  ✓ Occlusion: CENTER > BORDER → model uses lesion content")
    else:
        print(f"  ✗ Occlusion: BORDER > CENTER → model uses borders")

    return center_sens, border_sens


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    df_test = pd.read_csv(os.path.join(Config.manifest_dir, "test_manifest.csv"))
    y_test = df_test["target"].values.astype(int)
    malignant_ids = df_test[y_test == 1]["isic_id"].values
    print(f"Malignant test lesions: {len(malignant_ids)}")

    print(f"\n{'=' * 70}\nTinyViT-5M\n{'=' * 70}")
    tiny_model = TinyViTModel().to(Config.device)
    tiny_model.load_state_dict(torch.load(Config.tinyvit_ckpt, map_location=Config.device, weights_only=False))
    tiny_model.eval()
    center_tiny, border_tiny = run_occlusion(tiny_model, malignant_ids, "TinyViT-5M")

    print(f"\n{'=' * 70}\nDenseNet121\n{'=' * 70}")
    dense_model = DenseNetModel().to(Config.device)
    dense_model.load_state_dict(torch.load(Config.densenet_ckpt, map_location=Config.device, weights_only=False))
    dense_model.eval()
    center_dense, border_dense = run_occlusion(dense_model, malignant_ids, "DenseNet121")

    print(f"\n{'=' * 70}")
    print("OCCLUSION SENSITIVITY SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Center':>12} {'Border':>12} {'Center/Border':>15}")
    print("-" * 70)
    print(f"{'TinyViT-5M':<20} {center_tiny.mean():>12.4f} "
          f"{border_tiny.mean():>12.4f} {center_tiny.mean() / max(border_tiny.mean(), 1e-8):>15.3f}")
    print(f"{'DenseNet121':<20} {center_dense.mean():>12.4f} "
          f"{border_dense.mean():>12.4f} {center_dense.mean() / max(border_dense.mean(), 1e-8):>15.3f}")

    np.save(os.path.join(Config.output_dir, "center_tiny.npy"), center_tiny)
    np.save(os.path.join(Config.output_dir, "border_tiny.npy"), border_tiny)
    np.save(os.path.join(Config.output_dir, "center_dense.npy"), center_dense)
    np.save(os.path.join(Config.output_dir, "border_dense.npy"), border_dense)
    print(f"\n✓ Saved to {Config.output_dir}")


if __name__ == "__main__":
    main()