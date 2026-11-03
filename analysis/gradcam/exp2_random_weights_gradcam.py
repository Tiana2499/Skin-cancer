"""
exp2_random_weights_gradcam.py
Sanity check: does the border pattern persist with random weights?
If yes → Grad-CAM pipeline is suspect.
If no → pattern is learned by the trained model.
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
    output_dir = r"./analysis/exp2_random_weights"
    image_size = 224
    device = torch.device("cpu")  # CPU to avoid CUDA warnings


class EfficientNetV2SBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s",
                                          pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.6), nn.Linear(self.feature_dim, 1))

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


def compute_border_fraction(cam, margin_frac=0.20):
    h, w = cam.shape
    margin = int(margin_frac * min(h, w))
    border_mask = np.zeros_like(cam, dtype=bool)
    border_mask[:margin, :] = True
    border_mask[-margin:, :] = True
    border_mask[:, :margin] = True
    border_mask[:, -margin:] = True
    return cam[border_mask].sum() / (cam.sum() + 1e-8)


def run_gradcam_analysis(model, isic_ids, transform, label):
    border_fractions = []
    for isic_id in isic_ids:
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
        border_fractions.append(compute_border_fraction(cam))

    bf = np.array(border_fractions)
    print(f"\n[{label}]")
    print(f"  n = {len(bf)}")
    print(f"  Mean border attention:   {bf.mean():.4f}")
    print(f"  Median border attention: {np.median(bf):.4f}")
    print(f"  Std:                     {bf.std():.4f}")
    return bf


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    df_test = pd.read_csv(os.path.join(Config.manifest_dir, "test_manifest.csv"))
    y_test = df_test["target"].values.astype(int)
    malignant_ids = df_test[y_test == 1]["isic_id"].values

    transform = get_transform()

    print("=" * 70)
    print("EXPERIMENT 2: RANDOM-WEIGHT SANITY CHECK")
    print("=" * 70)
    print(f"Analyzing {len(malignant_ids)} malignant test lesions")

    # --- Trained model ---
    torch.manual_seed(42)
    model_trained = EfficientNetV2SBaseline().to(Config.device)
    ckpt = torch.load(
        os.path.join(Config.checkpoint_dir, "best_model_fold_1.pt"),
        map_location=Config.device, weights_only=False)
    model_trained.load_state_dict(ckpt)

    bf_trained = run_gradcam_analysis(
        model_trained, malignant_ids, transform, "TRAINED MODEL")

    # --- Random-weight model ---
    torch.manual_seed(0)
    model_random = EfficientNetV2SBaseline().to(Config.device)
    # Reinitialize with random weights
    for m in model_random.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    bf_random = run_gradcam_analysis(
        model_random, malignant_ids, transform, "RANDOM WEIGHTS")

    # --- Comparison ---
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    print(f"Trained model border attention:  {bf_trained.mean():.4f}")
    print(f"Random weights border attention: {bf_random.mean():.4f}")
    print(f"Difference:                      {bf_trained.mean() - bf_random.mean():+.4f}")

    if bf_random.mean() > 0.70:
        print("\n⚠️ Random weights also show high border attention.")
        print("   The pattern may be partially an artifact of the Grad-CAM method.")
    else:
        print("\n✓ Border attention is specific to the trained model.")
        print("   The pattern is learned, not a visualization artifact.")

    np.save(os.path.join(Config.output_dir, "bf_trained.npy"), bf_trained)
    np.save(os.path.join(Config.output_dir, "bf_random.npy"), bf_random)
    print(f"\nSaved to {Config.output_dir}")


if __name__ == "__main__":
    main()