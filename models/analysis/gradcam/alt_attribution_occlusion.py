"""
alt_attribution_occlusion.py
Use occlusion sensitivity instead of Grad-CAM.
Mask 32×32 patches one at a time and measure prediction change.
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
    manifest_dir = r"./cohort_manifests"
    image_dir = r"F:\thesis 23\ISIC_2024_Training_Input\ISIC_2024_Training_Input"
    checkpoint_dir = r"./baseline_A_efficientnetv2s_5fold_fixed"
    output_dir = r"./analysis/occlusion"
    image_size = 224
    patch_size = 32
    stride = 16
    device = torch.device("cpu")


class EfficientNetV2SBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s",
                                          pretrained=False, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for block in list(self.backbone.blocks)[-4:]:
            for p in block.parameters():
                p.requires_grad = True
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.6), nn.Linear(self.feature_dim, 1))

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


@torch.no_grad()
def occlusion_sensitivity(model, x, patch_size=32, stride=16):
    """Compute occlusion sensitivity map."""
    B, C, H, W = x.shape
    baseline = torch.sigmoid(model(x)).item()

    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    # Iterate over patch positions
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            x_occluded = x.clone()
            x_occluded[:, :, i:i + patch_size, j:j + patch_size] = 0
            p_occluded = torch.sigmoid(model(x_occluded)).item()

            # Drop in prediction when patch is occluded
            delta = baseline - p_occluded
            heatmap[i:i + patch_size, j:j + patch_size] += delta
            counts[i:i + patch_size, j:j + patch_size] += 1

    heatmap = heatmap / np.maximum(counts, 1)
    return heatmap, baseline


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    df_test = pd.read_csv(os.path.join(Config.manifest_dir, "test_manifest.csv"))
    y_test = df_test["target"].values.astype(int)
    malignant_df = df_test[y_test == 1]

    transform = get_transform()
    model = EfficientNetV2SBaseline().to(Config.device)
    ckpt = torch.load(
        os.path.join(Config.checkpoint_dir, "best_model_fold_1.pt"),
        map_location=Config.device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()

    print("=" * 70)
    print("OCCLUSION SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Analyzing {len(malignant_df)} malignant lesions")
    print()

    center_sens = []
    border_sens = []

    for _, row in malignant_df.iterrows():
        isic_id = row["isic_id"]
        img_path = None
        for ext in ["", ".jpg", ".jpeg", ".png"]:
            p = os.path.join(Config.image_dir, f"{isic_id}{ext}")
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue

        pil = Image.open(img_path).convert("RGB")
        x = transform(pil).unsqueeze(0).to(Config.device)

        heatmap, baseline = occlusion_sensitivity(
            model, x, patch_size=Config.patch_size, stride=Config.stride
        )

        # Center vs. border sensitivity
        margin = int(0.20 * Config.image_size)
        center = heatmap[margin:-margin, margin:-margin].mean()
        border = (heatmap.sum() - heatmap[margin:-margin, margin:-margin].sum()) / \
                 (heatmap.size - heatmap[margin:-margin, margin:-margin].size)

        center_sens.append(center)
        border_sens.append(border)

        print(f"{isic_id}: center_sens={center:.4f}, border_sens={border:.4f}")

    center_sens = np.array(center_sens)
    border_sens = np.array(border_sens)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Mean center sensitivity: {center_sens.mean():.4f}")
    print(f"Mean border sensitivity: {border_sens.mean():.4f}")
    print(f"Ratio (center / border): {center_sens.mean() / max(border_sens.mean(), 1e-8):.3f}")

    if center_sens.mean() > border_sens.mean():
        print(f"\n✓ Occlusion shows CENTER sensitivity > border sensitivity")
        print(f"  The model functionally depends more on lesion content than borders.")
    else:
        print(f"\n✗ Occlusion shows border sensitivity > center sensitivity")
        print(f"  The model functionally depends more on borders than lesion content.")

    np.save(os.path.join(Config.output_dir, "center_sens.npy"), center_sens)
    np.save(os.path.join(Config.output_dir, "border_sens.npy"), border_sens)


if __name__ == "__main__":
    main()