"""
exp4_lesion_centered_crop.py
Crop to central 60% of image (preserving lesion) and measure prediction change.
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
    output_dir = r"./analysis/exp4_lesion_crop"
    image_size = 224
    center_frac = 0.60  # keep central 60%
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


def center_crop(pil_image, frac=0.60):
    """Crop to central (frac × frac) of the image."""
    w, h = pil_image.size
    new_w = int(frac * w)
    new_h = int(frac * h)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return pil_image.crop((left, top, left + new_w, top + new_h))


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    df_test = pd.read_csv(os.path.join(Config.manifest_dir, "test_manifest.csv"))
    y_test = df_test["target"].values.astype(int)

    # Analyze all 30 malignant + 200 random benign
    malignant_df = df_test[y_test == 1]
    benign_df = df_test[y_test == 0].sample(200, random_state=42)
    analysis_df = pd.concat([malignant_df, benign_df], ignore_index=True)

    transform = get_transform()
    model = EfficientNetV2SBaseline().to(Config.device)
    ckpt = torch.load(
        os.path.join(Config.checkpoint_dir, "best_model_fold_1.pt"),
        map_location=Config.device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()

    results = []

    print("=" * 70)
    print("EXPERIMENT 4: LESION-PRESERVING CROP")
    print("=" * 70)
    print(f"Analyzing {len(malignant_df)} malignant + {len(benign_df)} benign")
    print(f"Center crop fraction: {Config.center_frac}")
    print()

    with torch.no_grad():
        for _, row in analysis_df.iterrows():
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
            pil_cropped = center_crop(pil_orig, frac=Config.center_frac)

            x_orig = transform(pil_orig).unsqueeze(0).to(Config.device)
            x_cropped = transform(pil_cropped).unsqueeze(0).to(Config.device)

            p_orig = torch.sigmoid(model(x_orig)).item()
            p_cropped = torch.sigmoid(model(x_cropped)).item()
            delta = p_orig - p_cropped

            results.append({
                "isic_id": isic_id,
                "target": int(row["target"]),
                "p_original": p_orig,
                "p_cropped": p_cropped,
                "delta": delta,
            })

    df = pd.DataFrame(results)

    # Split by malignant / benign
    mal = df[df["target"] == 1]
    ben = df[df["target"] == 0]

    print(f"\n{'=' * 70}")
    print("MALIGNANT LESIONS (n={})".format(len(mal)))
    print(f"{'=' * 70}")
    print(f"Mean p (original):  {mal['p_original'].mean():.4f}")
    print(f"Mean p (cropped):   {mal['p_cropped'].mean():.4f}")
    print(f"Mean delta:         {mal['delta'].mean():+.4f}")

    print(f"\n{'=' * 70}")
    print("BENIGN LESIONS (n={})".format(len(ben)))
    print(f"{'=' * 70}")
    print(f"Mean p (original):  {ben['p_original'].mean():.4f}")
    print(f"Mean p (cropped):   {ben['p_cropped'].mean():.4f}")
    print(f"Mean delta:         {ben['delta'].mean():+.4f}")

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    if mal["delta"].mean() > 0.05:
        print(f"✓ Malignant predictions decrease when borders removed")
        print(f"  This confirms the model relies on peripheral information.")
    else:
        print(f"✗ Malignant predictions are robust to border removal")

    df.to_csv(os.path.join(Config.output_dir, "lesion_crop_results.csv"), index=False)
    print(f"\nSaved to {Config.output_dir}")


if __name__ == "__main__":
    main()