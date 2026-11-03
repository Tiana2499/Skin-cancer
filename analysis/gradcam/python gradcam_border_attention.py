"""
gradcam_border_attention.py
Quantify border vs. lesion attention across all malignant test lesions.
No OpenCV needed — uses PIL and NumPy only.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm


class Config:
    manifest_dir = r"./cohort_manifests"
    image_dir = r"F:\thesis 23\ISIC_2024_Training_Input\ISIC_2024_Training_Input"
    checkpoint_dir = r"./baseline_A_efficientnetv2s_5fold_fixed"
    image_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EfficientNetV2SBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for block in list(self.backbone.blocks)[-4:]:
            for p in block.parameters():
                p.requires_grad = True
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(self.feature_dim, 1))

    def forward(self, x):
        fm = self.backbone.forward_features(x)
        if fm.ndim == 4:
            fm = fm.mean(dim=(2, 3))
        return self.classifier(fm).squeeze(1)


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        layer = model.backbone.blocks[-1]
        layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0]))

    def generate(self, x):
        self.model.eval()
        self.model.zero_grad()
        x = x.requires_grad_(True)
        logit = self.model(x)
        logit.sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() + 1e-8)


def get_transform():
    return transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def main():
    df_test = pd.read_csv(os.path.join(Config.manifest_dir, "test_manifest.csv"))
    y_test = df_test["target"].values.astype(int)
    malignant_idx = np.where(y_test == 1)[0]

    print(f"Malignant test lesions: {len(malignant_idx)}")

    transform = get_transform()
    model = EfficientNetV2SBaseline().to(Config.device)
    ckpt = torch.load(os.path.join(Config.checkpoint_dir, "best_model_fold_1.pt"),
                      map_location=Config.device, weights_only=False)
    model.load_state_dict(ckpt)

    border_fractions = []
    processed = 0

    for idx in malignant_idx:
        row = df_test.iloc[idx]
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

        gcam = GradCAM(model)
        cam = gcam.generate(x)

        # Border = outer 20% of each dimension
        margin = int(0.20 * Config.image_size)
        border_mask = np.zeros_like(cam, dtype=bool)
        border_mask[:margin, :] = True
        border_mask[-margin:, :] = True
        border_mask[:, :margin] = True
        border_mask[:, -margin:] = True

        total_attention = cam.sum()
        border_attention = cam[border_mask].sum()
        border_fraction = border_attention / (total_attention + 1e-8)
        border_fractions.append(border_fraction)
        processed += 1

    border_fractions = np.array(border_fractions)

    print(f"\n{'=' * 70}")
    print("BORDER ATTENTION ANALYSIS (Malignant Test Lesions)")
    print(f"{'=' * 70}")
    print(f"Processed: {processed} lesions")
    print(f"\nBorder attention fraction:")
    print(f"  Mean:   {border_fractions.mean():.4f}")
    print(f"  Median: {np.median(border_fractions):.4f}")
    print(f"  Std:    {border_fractions.std():.4f}")
    print(f"  Min:    {border_fractions.min():.4f}")
    print(f"  Max:    {border_fractions.max():.4f}")

    print(f"\nReference values:")
    print(f"  Uniform attention (36% border area): 0.36")
    print(f"  Lesion-focused (<10%):                <0.10")
    print(f"  Border-focused (>50%):                >0.50")
    print(f"\nObserved: {border_fractions.mean():.4f}")

    if border_fractions.mean() > 0.50:
        print(f"\n✓ CONFIRMED: Model is border-focused")
        print(f"  The encoder attends primarily to image borders, not lesion content.")
    elif border_fractions.mean() > 0.36:
        print(f"\n⚠️ Border attention above uniform baseline")
        print(f"  The encoder is biased toward borders.")
    else:
        print(f"\n✗ Model attends more to center than borders")

    # Save results
    np.save("./models/tree_ensemble_v3/border_attention.npy", border_fractions)
    print(f"\nSaved: ./models/tree_ensemble_v3/border_attention.npy")


if __name__ == "__main__":
    main()