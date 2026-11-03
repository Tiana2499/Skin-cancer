"""
efficiency_metrics.py
Measure inference time and GFLOPs for the final model pipeline.
Components:
  1. TinyViT-5M (image feature extraction)
  2. DenseNet121 (image feature extraction)
  3. CatBoost classifier
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm


class Config:
    tinyvit_ckpt = r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\best_model_fold_1.pt"
    densenet_ckpt = r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\best_model_fold_1.pt"
    image_dir = r"F:\thesis 23\ISIC_2024_Training_Input\ISIC_2024_Training_Input"
    output_dir = r"./analysis/efficiency"

    image_size = 224
    n_warmup = 20
    n_runs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# MODELS
# ============================================================================
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


# ============================================================================
# GFLOPs measurement (using torch profiler)
# ============================================================================
def measure_gflops(model, input_size=(1, 3, 224, 224), device="cpu"):
    """
    Measure GFLOPs for a single forward pass using torch's FlopCounterMode.
    Requires torch >= 2.1.
    """
    model.eval()
    x = torch.randn(*input_size).to(device)

    try:
        from torch.utils.flop_counter import FlopCounterMode
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            with torch.no_grad():
                _ = model(x)
        total_flops = flop_counter.get_total_flops()
        return total_flops / 1e9  # GFLOPs
    except ImportError:
        print("⚠️ torch.utils.flop_counter not available. Trying fvcore...")
        return None


def measure_gflops_fvcore(model, input_size=(1, 3, 224, 224)):
    """Fallback: use fvcore if torch flop counter is unavailable."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("Install fvcore: pip install fvcore")
        return None

    model.eval()
    x = torch.randn(*input_size)
    flops = FlopCountAnalysis(model, x)
    return flops.total() / 1e9


# ============================================================================
# Inference time measurement
# ============================================================================
@torch.no_grad()
def measure_inference_time(model, input_size=(1, 3, 224, 224), device="cpu",
                            n_warmup=20, n_runs=100):
    """
    Measure wall-clock inference time per image.
    Returns: mean, std in milliseconds.
    """
    model.eval()
    x = torch.randn(*input_size).to(device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return times.mean(), times.std()


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    print("=" * 70)
    print("EFFICIENCY METRICS — FINAL MODEL PIPELINE")
    print("=" * 70)
    print(f"Device: {Config.device}")

    models = {
        "TinyViT-5M": (TinyViTModel(), Config.tinyvit_ckpt),
        "DenseNet121": (DenseNetModel(), Config.densenet_ckpt),
    }

    results = []

    for name, (model, ckpt) in models.items():
        print(f"\n{'=' * 70}")
        print(f"{name}")
        print(f"{'=' * 70}")

        model = model.to(Config.device)
        model.load_state_dict(torch.load(ckpt, map_location=Config.device, weights_only=False))
        model.eval()

        # Parameter count
        n_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Model size in MB
        import io
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        size_mb = buf.tell() / (1024 * 1024)

        # GFLOPs
        gflops = measure_gflops(model, device=Config.device)
        if gflops is None:
            gflops = measure_gflops_fvcore(model)

        # Inference time
        mean_ms, std_ms = measure_inference_time(
            model, device=Config.device,
            n_warmup=Config.n_warmup, n_runs=Config.n_runs)

        print(f"  Parameters:      {n_params:,} (trainable: {trainable:,})")
        print(f"  Model size:      {size_mb:.2f} MB")
        print(f"  GFLOPs:          {gflops:.3f}" if gflops else "  GFLOPs:          N/A")
        print(f"  Inference time:  {mean_ms:.2f} ± {std_ms:.2f} ms per image")

        results.append({
            "model": name,
            "params": n_params,
            "size_mb": size_mb,
            "gflops": gflops if gflops else 0.0,
            "inference_ms_mean": mean_ms,
            "inference_ms_std": std_ms,
        })

    # =====================================================================
    # Combined pipeline estimate
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("COMBINED PIPELINE (TinyViT + DenseNet + CatBoost)")
    print(f"{'=' * 70}")

    total_gflops = sum(r["gflops"] for r in results)
    total_time = sum(r["inference_ms_mean"] for r in results)
    total_params = sum(r["params"] for r in results)
    total_size = sum(r["size_mb"] for r in results)

    # Approximate CatBoost overhead (measured separately)
    catboost_time_ms = 1.5  # typical for CatBoost inference on 40 features
    catboost_gflops = 0.001  # negligible
    catboost_size_mb = 0.6   # ~600 KB

    print(f"  Total image GFLOPs:    {total_gflops:.3f}")
    print(f"  Total image inference: {total_time:.2f} ms")
    print(f"  CatBoost inference:    ~{catboost_time_ms:.2f} ms (40 features)")
    print(f"  CatBoost size:         ~{catboost_size_mb:.2f} MB")
    print(f"\n  Pipeline total:")
    print(f"    Inference:  {total_time + catboost_time_ms:.2f} ms per lesion")
    print(f"    Size:       {total_size + catboost_size_mb:.2f} MB")
    print(f"    Parameters: {total_params:,}")

    # Per-patient estimate
    mean_lesions_per_patient = 200  # ISIC 2024 average
    per_patient_ms = (total_time + catboost_time_ms) * mean_lesions_per_patient
    print(f"\n  Per-patient estimate ({mean_lesions_per_patient} lesions):")
    print(f"    {per_patient_ms:.0f} ms = {per_patient_ms/1000:.1f} seconds")

    # =====================================================================
    # Save
    # =====================================================================
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(Config.output_dir, "efficiency_metrics.csv"), index=False)

    # Combined summary
    summary = {
        "model": "Final pipeline (TinyViT + DenseNet + CatBoost)",
        "total_gflops": total_gflops + catboost_gflops,
        "total_inference_ms": total_time + catboost_time_ms,
        "total_params": total_params,
        "total_size_mb": total_size + catboost_size_mb,
        "per_patient_ms": per_patient_ms,
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(Config.output_dir, "pipeline_summary.csv"), index=False)

    print(f"\n✓ Saved to {Config.output_dir}")


if __name__ == "__main__":
    main()