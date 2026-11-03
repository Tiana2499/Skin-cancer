"""
compress_final_backbones.py
INT8 dynamic quantization of TinyViT-5M and DenseNet121 fold checkpoints.
Measures pAUC change, model size reduction, and inference speedup.
"""

import os
import time
import io
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import timm
import sklearn.metrics as sklearn_metrics


class Config:
    manifest_dir = r"F:\thesis 23\tinyvit experiment\cohort_manifests"
    image_dir = r"F:\thesis 23\ISIC_2024_Training_Input\ISIC_2024_Training_Input"
    output_dir = r"./analysis/compression"

    tinyvit_ckpt_dir = r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN"
    densenet_ckpt_dir = r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold"

    image_size = 224
    batch_size = 32
    num_workers = 4
    n_folds = 5
    min_tpr = 0.80
    device = torch.device("cpu")  # INT8 dynamic quantization requires CPU


class TinyViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tiny_vit_5m_224", pretrained=False, num_classes=0)
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
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(self.feature_dim, 1))

    def forward(self, x):
        fm = self.backbone.forward_features(x)
        if fm.ndim == 4:
            fm = fm.mean(dim=(2, 3))
        return self.classifier(fm).squeeze(1)


class TestDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        for ext in ["", ".jpg", ".jpeg", ".png"]:
            p = os.path.join(self.image_dir, f"{row['isic_id']}{ext}")
            if os.path.exists(p):
                break
        img = Image.open(p).convert("RGB")
        return self.transform(img), row["target"]


def get_transform():
    return transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def isic2024_pauc(y, p, min_tpr=0.80):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) != 2:
        return 0.0
    yf, pf = abs(y - 1), abs(p - 1)
    max_fpr = 1 - min_tpr
    fpr, tpr, _ = sklearn_metrics.roc_curve(yf, pf)
    stop = np.searchsorted(fpr, max_fpr, "right")
    tpr_i = np.append(tpr[:stop], np.interp(max_fpr, [fpr[stop-1], fpr[stop]], [tpr[stop-1], tpr[stop]]))
    fpr_i = np.append(fpr[:stop], max_fpr)
    return float(sklearn_metrics.auc(fpr_i, tpr_i))


def model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, targets = [], []
    t0 = time.time()
    for x, y in loader:
        p = torch.sigmoid(model(x)).numpy()
        probs.append(p)
        targets.append(y.numpy())
    elapsed = time.time() - t0
    return np.concatenate(targets).astype(int), np.concatenate(probs), elapsed


def quantize_and_evaluate(ModelClass, ckpt_dir, loader, name):
    """Full pipeline: FP32 eval, INT8 quantize, INT8 eval, comparison."""
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    # ---- FP32 ----
    print("Loading FP32 model...")
    model_fp32 = ModelClass().to(Config.device)
    ckpt = os.path.join(ckpt_dir, "best_model_fold_1.pt")
    model_fp32.load_state_dict(torch.load(ckpt, map_location=Config.device, weights_only=False))
    model_fp32.eval()

    size_fp32 = model_size_mb(model_fp32)
    y, p_fp32, time_fp32 = evaluate(model_fp32, loader)
    pauc_fp32 = isic2024_pauc(y, p_fp32)
    auc_fp32 = sklearn_metrics.roc_auc_score(y, p_fp32)

    print(f"  FP32: size={size_fp32:.2f}MB, pAUC={pauc_fp32:.4f}, "
          f"AUROC={auc_fp32:.4f}, time={time_fp32:.1f}s")

    # ---- INT8 dynamic quantization ----
    print("Quantizing to INT8...")
    try:
        model_int8 = torch.quantization.quantize_dynamic(
            model_fp32, qconfig_spec={nn.Linear}, dtype=torch.qint8)
        model_int8.eval()

        size_int8 = model_size_mb(model_int8)
        _, p_int8, time_int8 = evaluate(model_int8, loader)
        pauc_int8 = isic2024_pauc(y, p_int8)
        auc_int8 = sklearn_metrics.roc_auc_score(y, p_int8)

        print(f"  INT8: size={size_int8:.2f}MB, pAUC={pauc_int8:.4f}, "
              f"AUROC={auc_int8:.4f}, time={time_int8:.1f}s")
        print(f"  ---")
        print(f"  Size reduction:  {size_fp32 / size_int8:.2f}×")
        print(f"  Speedup:         {time_fp32 / time_int8:.2f}×")
        print(f"  Δ pAUC:          {pauc_int8 - pauc_fp32:+.4f}")
        print(f"  Δ AUROC:         {auc_int8 - auc_fp32:+.4f}")

        return {
            "model": name,
            "size_fp32_mb": size_fp32,
            "size_int8_mb": size_int8,
            "size_reduction": size_fp32 / size_int8,
            "pauc_fp32": pauc_fp32,
            "pauc_int8": pauc_int8,
            "delta_pauc": pauc_int8 - pauc_fp32,
            "auroc_fp32": auc_fp32,
            "auroc_int8": auc_int8,
            "delta_auroc": auc_int8 - auc_fp32,
            "time_fp32_s": time_fp32,
            "time_int8_s": time_int8,
            "speedup": time_fp32 / time_int8 if time_int8 > 0 else 0,
        }
    except Exception as e:
        print(f"  ⚠️ Quantization failed: {e}")
        return None


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    df_test = pd.read_csv(os.path.join(Config.manifest_dir, "test_manifest.csv"))
    print(f"Test set: {len(df_test)} lesions, {df_test['target'].sum()} malignant")

    transform = get_transform()
    ds = TestDataset(df_test, Config.image_dir, transform)
    loader = DataLoader(ds, batch_size=Config.batch_size, shuffle=False,
                        num_workers=Config.num_workers)

    results = []

    # TinyViT
    r = quantize_and_evaluate(TinyViTModel, Config.tinyvit_ckpt_dir, loader, "TinyViT-5M")
    if r:
        results.append(r)

    # DenseNet
    r = quantize_and_evaluate(DenseNetModel, Config.densenet_ckpt_dir, loader, "DenseNet121")
    if r:
        results.append(r)

    # =====================================================================
    # Summary
    # =====================================================================
    if results:
        print(f"\n{'=' * 70}")
        print("COMPRESSION SUMMARY")
        print(f"{'=' * 70}")
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format="%.4f"))

        # Combined pipeline estimate
        total_fp32 = df["size_fp32_mb"].sum()
        total_int8 = df["size_int8_mb"].sum()
        print(f"\nPipeline (both backbones):")
        print(f"  FP32 total: {total_fp32:.2f} MB")
        print(f"  INT8 total: {total_int8:.2f} MB")
        print(f"  Reduction:  {total_fp32 / total_int8:.2f}×")

        df.to_csv(os.path.join(Config.output_dir, "compression_results.csv"), index=False)
        print(f"\n✓ Saved to {Config.output_dir}")


if __name__ == "__main__":
    main()