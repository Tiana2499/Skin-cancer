"""
exp3_border_perturbation.py
Causal test: replace border pixels with mean skin color.
Measure the change in malignancy probability.
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
    output_dir = r"./analysis/exp3_border_perturbation"
    image_size = 224
    border_frac = 0.15  # perturb outer 15% of each dimension
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


def mask_border(pil_image, frac=0.15):
    """Replace outer border pixels with the mean interior color."""
    img = np.array(pil_image).copy()
    h, w = img.shape[:2]
    margin_h = int(frac * h)
    margin_w = int(frac * w)

    # Compute interior mean color
    interior = img[margin_h:h - margin_h, margin_w:w - margin_w]
    mean_color = interior.reshape(-1, 3).mean(axis=0).astype(np.uint8)

    # Replace border pixels
    img[:margin_h, :] = mean_color
    img[-margin_h:, :] = mean_color
    img[:, :margin_w] = mean_color
    img[:, -margin_w:] = mean_color

    return Image.fromarray(img)


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

    results = []

    print("=" * 70)
    print("EXPERIMENT 3: BORDER PERTURBATION CAUSAL TEST")
    print("=" * 70)
    print(f"Analyzing {len(malignant_df)} malignant test lesions")
    print(f"Border fraction masked: {Config.border_frac}")
    print()

    with torch.no_grad():
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

            pil_orig = Image.open(img_path).convert("RGB")
            pil_masked = mask_border(pil_orig, frac=Config.border_frac)

            x_orig = transform(pil_orig).unsqueeze(0).to(Config.device)
            x_masked = transform(pil_masked).unsqueeze(0).to(Config.device)

            p_orig = torch.sigmoid(model(x_orig)).item()
            p_masked = torch.sigmoid(model(x_masked)).item()
            delta = p_orig - p_masked

            results.append({
                "isic_id": isic_id,
                "p_original": p_orig,
                "p_border_masked": p_masked,
                "delta": delta,
            })

    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format="%.4f"))

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Mean p (original):       {df['p_original'].mean():.4f}")
    print(f"Mean p (border-masked):  {df['p_border_masked'].mean():.4f}")
    print(f"Mean delta:              {df['delta'].mean():+.4f}")
    print(f"Median delta:            {df['delta'].median():+.4f}")
    print(f"Fraction with delta > 0: {(df['delta'] > 0).mean():.4f}")
    print(f"Fraction with delta > 0.1: {(df['delta'] > 0.1).mean():.4f}")

    if df["delta"].mean() > 0.05:
        print(f"\n✓ CAUSAL EVIDENCE: Border pixels contribute to malignant predictions.")
    else:
        print(f"\n✗ No strong evidence that borders drive predictions.")

    df.to_csv(os.path.join(Config.output_dir, "border_perturbation.csv"), index=False)
    print(f"\nSaved to {Config.output_dir}")


if __name__ == "__main__":
    main()