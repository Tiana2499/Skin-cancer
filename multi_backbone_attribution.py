# ============================================================================
# multi_backbone_attribution.py
#
# Corrected Grad-CAM border attention analysis for:
#   1. TinyViT-5M
#   2. DenseNet121
#
# IMPORTANT:
# - Model architectures are kept the same.
# - Checkpoints are kept the same.
# - Preprocessing is kept the same.
# - Test cohort is kept the same.
# - Border fraction is kept at 0.20.
# - Trained vs random comparison is kept.
#
# CORRECTION:
# Grad-CAM now explicitly converts feature maps to [B,C,H,W]
# before calculating channel weights and spatial CAMs.
#
# It supports:
#   [B,C,H,W]
#   [B,H,W,C]
#   [B,N,C] transformer tokens
# ============================================================================

import os
import gc
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
import timm


# ============================================================================
# CONFIG
# ============================================================================

class Config:

    manifest_dir = r"F:\thesis 23\tinyvit experiment\cohort_manifests"

    image_dir = (
        r"F:\thesis 23\ISIC_2024_Training_Input"
        r"\ISIC_2024_Training_Input"
    )

    output_dir = r"./analysis/multi_backbone_attribution"

    image_size = 224

    # Keep exactly the same as your original script
    border_frac = 0.20

    # Keep CPU exactly as original
    device = torch.device("cpu")

    tinyvit_ckpt = (
        r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN"
        r"\best_model_fold_1.pt"
    )

    densenet_ckpt = (
        r"F:\thesis 23\5 fold experiments"
        r"\baseline_B_densenet121_5fold"
        r"\best_model_fold_1.pt"
    )


# ============================================================================
# TINYVIT MODEL
# ============================================================================

class TinyViTModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "tiny_vit_5m_224",
            pretrained=False,
            num_classes=0
        )

        # Keep fully unfrozen
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(self.feature_dim, 1)
        )

    def forward(self, x):

        fm = self.backbone.forward_features(x)

        if fm.ndim == 4:

            # Keep original model behavior
            fm = fm.mean(dim=(2, 3))

        elif fm.ndim == 3:

            fm = fm.mean(dim=1)

        else:

            raise RuntimeError(
                f"Unexpected TinyViT forward_features shape: {fm.shape}"
            )

        return self.classifier(fm).squeeze(1)


# ============================================================================
# DENSENET MODEL
# ============================================================================

class DenseNetModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "densenet121",
            pretrained=False,
            num_classes=0
        )

        for p in self.backbone.parameters():
            p.requires_grad = True

        self.feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(self.feature_dim, 1)
        )

    def forward(self, x):

        fm = self.backbone.forward_features(x)

        if fm.ndim == 4:

            fm = fm.mean(dim=(2, 3))

        else:

            raise RuntimeError(
                f"Unexpected DenseNet forward_features shape: {fm.shape}"
            )

        return self.classifier(fm).squeeze(1)


# ============================================================================
# FEATURE-MAP STANDARDIZATION
# ============================================================================

def convert_to_nchw(
    activations,
    gradients,
    verbose=False,
    layer_name=""
):
    """
    Convert feature representations into:

        [B, C, H, W]

    Supported input formats:

        [B, C, H, W]
        [B, H, W, C]
        [B, N, C]

    This is particularly important for transformer-style
    feature representations.
    """

    if activations.ndim != gradients.ndim:

        raise RuntimeError(
            f"Activation/gradient dimensionality mismatch: "
            f"{activations.shape} vs {gradients.shape}"
        )

    # ------------------------------------------------------------------------
    # CASE 1: Standard CNN feature map
    # [B, C, H, W]
    # ------------------------------------------------------------------------

    if activations.ndim == 4:

        b, d1, d2, d3 = activations.shape

        # Heuristic:
        #
        # CNN convention:
        #   [B,C,H,W]
        #
        # NHWC convention:
        #   [B,H,W,C]
        #
        # Spatial dimensions are normally substantially smaller than C.

        # If first feature dimension is channel-like and the last two
        # dimensions are equal spatial dimensions -> NCHW.
        if d2 == d3 and d1 != d2:

            if verbose:
                print(
                    f"[{layer_name}] Detected NCHW: "
                    f"{activations.shape}"
                )

            return activations, gradients

        # If second and third dimensions are equal and the final dimension
        # looks like the channel dimension -> NHWC.
        if d1 == d2 and d3 != d1:

            if verbose:
                print(
                    f"[{layer_name}] Detected NHWC: "
                    f"{activations.shape} -> "
                    f"[B,C,H,W]"
                )

            activations = activations.permute(
                0, 3, 1, 2
            ).contiguous()

            gradients = gradients.permute(
                0, 3, 1, 2
            ).contiguous()

            return activations, gradients

        # --------------------------------------------------------------------
        # Fallback: infer channel dimension by assuming the largest dimension
        # is channel-like.
        # --------------------------------------------------------------------

        dims = [d1, d2, d3]

        channel_axis = int(np.argmax(dims)) + 1

        if channel_axis == 1:

            return activations, gradients

        elif channel_axis == 2:

            activations = activations.permute(
                0, 2, 1, 3
            ).contiguous()

            gradients = gradients.permute(
                0, 2, 1, 3
            ).contiguous()

        elif channel_axis == 3:

            activations = activations.permute(
                0, 3, 1, 2
            ).contiguous()

            gradients = gradients.permute(
                0, 3, 1, 2
            ).contiguous()

        if verbose:
            print(
                f"[{layer_name}] 4D fallback conversion: "
                f"{activations.shape}"
            )

        return activations, gradients

    # ------------------------------------------------------------------------
    # CASE 2: Transformer tokens
    #
    # [B, N, C]
    #
    # Need to recover:
    #
    # [B,C,H,W]
    # ------------------------------------------------------------------------

    elif activations.ndim == 3:

        B, N, C = activations.shape

        if verbose:
            print(
                f"[{layer_name}] Token representation detected: "
                f"{activations.shape}"
            )

        # Check whether N is already a perfect square
        side = int(np.sqrt(N))

        if side * side == N:

            # No CLS token
            spatial_tokens = activations
            spatial_grads = gradients

        else:

            # Try removing one special token
            if N > 1:

                N_without_cls = N - 1
                side = int(np.sqrt(N_without_cls))

                if side * side == N_without_cls:

                    if verbose:
                        print(
                            f"[{layer_name}] Removing 1 special token "
                            f"before spatial reshape."
                        )

                    spatial_tokens = activations[:, 1:, :]
                    spatial_grads = gradients[:, 1:, :]

                else:

                    raise RuntimeError(
                        f"Cannot infer square spatial grid from "
                        f"token count N={N}. "
                        f"Activation shape={activations.shape}"
                    )

            else:

                raise RuntimeError(
                    f"Too few transformer tokens: {activations.shape}"
                )

        B, N_spatial, C = spatial_tokens.shape

        side = int(np.sqrt(N_spatial))

        if side * side != N_spatial:

            raise RuntimeError(
                f"Spatial token count {N_spatial} is not square."
            )

        # [B,N,C] -> [B,C,H,W]
        activations = spatial_tokens.transpose(1, 2).contiguous()

        gradients = spatial_grads.transpose(1, 2).contiguous()

        activations = activations.view(
            B, C, side, side
        )

        gradients = gradients.view(
            B, C, side, side
        )

        if verbose:
            print(
                f"[{layer_name}] Converted tokens to NCHW: "
                f"{activations.shape}"
            )

        return activations, gradients

    else:

        raise RuntimeError(
            f"Unsupported activation shape: {activations.shape}"
        )


# ============================================================================
# GRAD-CAM
# ============================================================================

class GradCAM:

    """
    Grad-CAM implementation with explicit feature-layout handling.

    The target layer is hooked directly.

    Output:
        normalized CAM with shape [H,W]
    """

    def __init__(
        self,
        model,
        target_layer,
        layer_name="target_layer"
    ):

        self.model = model

        self.gradients = None
        self.activations = None

        self.layer_name = layer_name

        self._fwd_handle = target_layer.register_forward_hook(
            self._save_activation
        )

        self._bwd_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    # ------------------------------------------------------------------------

    def _save_activation(
        self,
        module,
        input,
        output
    ):

        # Some modules can return tuples.
        if isinstance(output, (tuple, list)):

            output = output[0]

        self.activations = output

    # ------------------------------------------------------------------------

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        gradient = grad_output[0]

        if isinstance(gradient, (tuple, list)):

            gradient = gradient[0]

        self.gradients = gradient

    # ------------------------------------------------------------------------

    def generate(
        self,
        x,
        verbose=False
    ):

        self.model.eval()

        self.model.zero_grad(set_to_none=True)

        # Fresh graph
        x = (
            x.detach()
            .clone()
            .requires_grad_(True)
        )

        # ------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------

        logit = self.model(x)

        # Single-image batch
        score = logit.sum()

        # ------------------------------------------------------------
        # Backward
        # ------------------------------------------------------------

        score.backward(
            retain_graph=False
        )

        # ------------------------------------------------------------
        # Check hooks
        # ------------------------------------------------------------

        if self.activations is None:

            raise RuntimeError(
                "Forward hook did not capture activations."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Backward hook did not capture gradients."
            )

        activations = self.activations.detach()

        gradients = self.gradients.detach()

        if verbose:

            print(
                f"\n[{self.layer_name}]"
            )

            print(
                f"Raw activation shape: "
                f"{tuple(activations.shape)}"
            )

            print(
                f"Raw gradient shape:   "
                f"{tuple(gradients.shape)}"
            )

        # ------------------------------------------------------------
        # Convert to [B,C,H,W]
        # ------------------------------------------------------------

        activations, gradients = convert_to_nchw(
            activations,
            gradients,
            verbose=verbose,
            layer_name=self.layer_name
        )

        # ------------------------------------------------------------
        # Standard Grad-CAM
        # ------------------------------------------------------------

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps
        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # ReLU
        cam = F.relu(cam)

        # ------------------------------------------------------------
        # Resize to input resolution
        # ------------------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(
                x.shape[2],
                x.shape[3]
            ),
            mode="bilinear",
            align_corners=False
        )

        # Remove batch/channel dimensions
        cam = (
            cam[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        # ------------------------------------------------------------
        # Normalize
        # ------------------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min < 1e-12:

            cam = np.zeros_like(cam)

        else:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        # ------------------------------------------------------------
        # Clear cached tensors
        # ------------------------------------------------------------

        self.activations = None
        self.gradients = None

        return cam

    # ------------------------------------------------------------------------

    def remove_hooks(self):

        self._fwd_handle.remove()
        self._bwd_handle.remove()

        self.activations = None
        self.gradients = None


# ============================================================================
# TRANSFORM
# ============================================================================

def get_transform():

    return transforms.Compose([

        transforms.Resize(
            (
                Config.image_size,
                Config.image_size
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )

    ])


# ============================================================================
# BORDER ATTENTION
# ============================================================================

def compute_border_fraction(
    cam,
    margin_frac
):

    h, w = cam.shape

    m = int(
        margin_frac * min(h, w)
    )

    mask = np.zeros_like(
        cam,
        dtype=bool
    )

    mask[:m, :] = True
    mask[-m:, :] = True
    mask[:, :m] = True
    mask[:, -m:] = True

    border_sum = cam[mask].sum()

    total_sum = cam.sum()

    return border_sum / (
        total_sum + 1e-8
    )


# ============================================================================
# IMAGE LOADING
# ============================================================================

def load_image(isic_id):

    for ext in [
        "",
        ".jpg",
        ".jpeg",
        ".png"
    ]:

        p = os.path.join(
            Config.image_dir,
            f"{isic_id}{ext}"
        )

        if os.path.exists(p):

            return Image.open(
                p
            ).convert("RGB")

    return None


# ============================================================================
# TARGET LAYER CHECK
# ============================================================================

def inspect_target_layer(
    model,
    target_layer,
    x,
    name
):

    print("\n" + "=" * 70)
    print(f"TARGET LAYER CHECK: {name}")
    print("=" * 70)

    captured = {}

    def hook(module, inp, output):

        if isinstance(output, (tuple, list)):
            output = output[0]

        captured["output"] = output

    handle = target_layer.register_forward_hook(
        hook
    )

    with torch.no_grad():

        _ = model(x)

    handle.remove()

    if "output" not in captured:

        raise RuntimeError(
            f"Could not capture output from {name}"
        )

    output = captured["output"]

    print(
        f"Raw target activation shape: "
        f"{tuple(output.shape)}"
    )

    print(
        f"Activation dtype: "
        f"{output.dtype}"
    )

    print(
        f"Activation min: "
        f"{output.detach().min().item():.6f}"
    )

    print(
        f"Activation max: "
        f"{output.detach().max().item():.6f}"
    )

    print("=" * 70)

    return output.shape


# ============================================================================
# TRAINED MODEL ANALYSIS
# ============================================================================

def run_gradcam_reload(
    ckpt_path,
    model_class,
    target_layer_fn,
    ids,
    label
):

    """
    Reload model for each image.

    Kept from original script for isolation and reproducibility.
    """

    transform = get_transform()

    bf = []

    print(
        f"\nStarting: {label}"
    )

    for i, isic_id in enumerate(ids):

        img = load_image(isic_id)

        if img is None:

            print(
                f"  ⚠️ Missing image: {isic_id}"
            )

            continue

        model = None
        gcam = None

        try:

            # --------------------------------------------------------
            # Load model
            # --------------------------------------------------------

            model = model_class().to(
                Config.device
            )

            state = torch.load(
                ckpt_path,
                map_location=Config.device,
                weights_only=False
            )

            model.load_state_dict(
                state
            )

            model.eval()

            # --------------------------------------------------------
            # Target layer
            # --------------------------------------------------------

            target_layer = target_layer_fn(
                model
            )

            # --------------------------------------------------------
            # Grad-CAM
            # --------------------------------------------------------

            gcam = GradCAM(
                model,
                target_layer,
                layer_name=label
            )

            x = transform(
                img
            ).unsqueeze(0).to(
                Config.device
            )

            cam = gcam.generate(
                x,
                verbose=(i == 0)
            )

            border_fraction = (
                compute_border_fraction(
                    cam,
                    Config.border_frac
                )
            )

            bf.append(
                border_fraction
            )

            if i == 0:

                print(
                    f"\nFirst CAM shape: "
                    f"{cam.shape}"
                )

                print(
                    f"First border attention: "
                    f"{border_fraction:.4f}"
                )

        except Exception as e:

            print(
                f"  ⚠️ {isic_id}: "
                f"{type(e).__name__}: {e}"
            )

        finally:

            if gcam is not None:

                try:
                    gcam.remove_hooks()
                except Exception:
                    pass

            del gcam
            del model

            gc.collect()

        if (i + 1) % 5 == 0:

            print(
                f"  Processed "
                f"{i + 1}/{len(ids)}"
            )

    bf = np.array(
        bf,
        dtype=np.float32
    )

    print(
        f"\n[{label}] n={len(bf)}"
    )

    if len(bf) > 0:

        print(
            f"  Mean border attention:   "
            f"{bf.mean():.4f}"
        )

        print(
            f"  Median border attention: "
            f"{np.median(bf):.4f}"
        )

        print(
            f"  Std border attention:     "
            f"{bf.std():.4f}"
        )

    return bf


# ============================================================================
# RANDOM MODEL ANALYSIS
# ============================================================================

def run_gradcam_random_reload(
    model_class,
    target_layer_fn,
    ids,
    label
):

    """
    Random-weight control.

    Same procedure as trained analysis,
    but without loading a checkpoint.
    """

    transform = get_transform()

    bf = []

    print(
        f"\nStarting: {label}"
    )

    for i, isic_id in enumerate(ids):

        img = load_image(isic_id)

        if img is None:

            print(
                f"  ⚠️ Missing image: {isic_id}"
            )

            continue

        model = None
        gcam = None

        try:

            # --------------------------------------------------------
            # Deterministic random initialization
            # --------------------------------------------------------

            torch.manual_seed(i)

            model = model_class().to(
                Config.device
            )

            # Explicitly reset learnable layers
            for m in model.modules():

                if isinstance(
                    m,
                    (
                        nn.Conv2d,
                        nn.Linear,
                        nn.BatchNorm2d
                    )
                ):

                    if hasattr(
                        m,
                        "reset_parameters"
                    ):

                        m.reset_parameters()

            model.eval()

            # --------------------------------------------------------
            # Target layer
            # --------------------------------------------------------

            target_layer = target_layer_fn(
                model
            )

            # --------------------------------------------------------
            # Grad-CAM
            # --------------------------------------------------------

            gcam = GradCAM(
                model,
                target_layer,
                layer_name=label
            )

            x = transform(
                img
            ).unsqueeze(0).to(
                Config.device
            )

            cam = gcam.generate(
                x,
                verbose=(i == 0)
            )

            border_fraction = (
                compute_border_fraction(
                    cam,
                    Config.border_frac
                )
            )

            bf.append(
                border_fraction
            )

        except Exception as e:

            print(
                f"  ⚠️ {isic_id}: "
                f"{type(e).__name__}: {e}"
            )

        finally:

            if gcam is not None:

                try:
                    gcam.remove_hooks()
                except Exception:
                    pass

            del gcam
            del model

            gc.collect()

        if (i + 1) % 5 == 0:

            print(
                f"  Processed "
                f"{i + 1}/{len(ids)}"
            )

    bf = np.array(
        bf,
        dtype=np.float32
    )

    print(
        f"\n[{label}] n={len(bf)}"
    )

    if len(bf) > 0:

        print(
            f"  Mean border attention:   "
            f"{bf.mean():.4f}"
        )

        print(
            f"  Median border attention: "
            f"{np.median(bf):.4f}"
        )

        print(
            f"  Std border attention:     "
            f"{bf.std():.4f}"
        )

    return bf


# ============================================================================
# MAIN
# ============================================================================

def main():

    os.makedirs(
        Config.output_dir,
        exist_ok=True
    )

    print("=" * 70)
    print("MULTI-BACKBONE GRAD-CAM BORDER ATTRIBUTION")
    print("=" * 70)

    print(
        f"Device: {Config.device}"
    )

    print(
        f"Image size: {Config.image_size}"
    )

    print(
        f"Border fraction: {Config.border_frac}"
    )

    # ========================================================================
    # LOAD TEST MANIFEST
    # ========================================================================

    manifest_path = os.path.join(
        Config.manifest_dir,
        "test_manifest.csv"
    )

    if not os.path.exists(
        manifest_path
    ):

        raise FileNotFoundError(
            f"Test manifest not found:\n"
            f"{manifest_path}"
        )

    df_test = pd.read_csv(
        manifest_path
    )

    y_test = (
        df_test["target"]
        .values
        .astype(int)
    )

    malignant_ids = (
        df_test[
            y_test == 1
        ]["isic_id"]
        .values
    )

    print(
        f"\nTotal test images: "
        f"{len(df_test)}"
    )

    print(
        f"Malignant test lesions: "
        f"{len(malignant_ids)}"
    )

    # ========================================================================
    # TINYVIT
    # ========================================================================

    print(
        "\n" +
        "=" * 70
    )

    print(
        "TINYVIT-5M"
    )

    print(
        "=" * 70
    )

    bf_tiny = run_gradcam_reload(

        Config.tinyvit_ckpt,

        TinyViTModel,

        lambda m:
            m.backbone.stages[-1]
            .blocks[-1],

        malignant_ids,

        "TinyViT trained"

    )

    bf_tiny_rnd = run_gradcam_random_reload(

        TinyViTModel,

        lambda m:
            m.backbone.stages[-1]
            .blocks[-1],

        malignant_ids,

        "TinyViT random"

    )

    # ========================================================================
    # DENSENET
    # ========================================================================

    print(
        "\n" +
        "=" * 70
    )

    print(
        "DENSENET121"
    )

    print(
        "=" * 70
    )

    bf_dense = run_gradcam_reload(

        Config.densenet_ckpt,

        DenseNetModel,

        lambda m:
            m.backbone.features.denseblock4,

        malignant_ids,

        "DenseNet trained"

    )

    bf_dense_rnd = run_gradcam_random_reload(

        DenseNetModel,

        lambda m:
            m.backbone.features.denseblock4,

        malignant_ids,

        "DenseNet random"

    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print(
        "\n" +
        "=" * 70
    )

    print(
        "ATTENTION COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        f"{'Model':<30}"
        f"{'Mean':>12}"
        f"{'Median':>12}"
        f"{'Std':>12}"
        f"{'N':>8}"
    )

    print(
        "-" * 70
    )

    def print_result(
        name,
        arr
    ):

        if len(arr) == 0:

            return

        print(
            f"{name:<30}"
            f"{arr.mean():>12.4f}"
            f"{np.median(arr):>12.4f}"
            f"{arr.std():>12.4f}"
            f"{len(arr):>8}"
        )

    print_result(
        "TinyViT trained",
        bf_tiny
    )

    print_result(
        "TinyViT random",
        bf_tiny_rnd
    )

    print_result(
        "DenseNet trained",
        bf_dense
    )

    print_result(
        "DenseNet random",
        bf_dense_rnd
    )

    # ========================================================================
    # TRAINED - RANDOM DIFFERENCE
    # ========================================================================

    print(
        "\n" +
        "=" * 70
    )

    print(
        "TRAINED VS RANDOM DIFFERENCE"
    )

    print(
        "=" * 70
    )

    if len(bf_tiny) > 0 and len(bf_tiny_rnd) > 0:

        print(
            "TinyViT:"
        )

        print(
            f"  Trained mean: "
            f"{bf_tiny.mean():.4f}"
        )

        print(
            f"  Random mean:  "
            f"{bf_tiny_rnd.mean():.4f}"
        )

        print(
            f"  Difference:    "
            f"{bf_tiny.mean() - bf_tiny_rnd.mean():.4f}"
        )

    if len(bf_dense) > 0 and len(bf_dense_rnd) > 0:

        print(
            "\nDenseNet:"
        )

        print(
            f"  Trained mean: "
            f"{bf_dense.mean():.4f}"
        )

        print(
            f"  Random mean:  "
            f"{bf_dense_rnd.mean():.4f}"
        )

        print(
            f"  Difference:    "
            f"{bf_dense.mean() - bf_dense_rnd.mean():.4f}"
        )

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    np.save(
        os.path.join(
            Config.output_dir,
            "bf_tiny_trained.npy"
        ),
        bf_tiny
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "bf_tiny_random.npy"
        ),
        bf_tiny_rnd
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "bf_dense_trained.npy"
        ),
        bf_dense
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "bf_dense_random.npy"
        ),
        bf_dense_rnd
    )

    # Also save a simple CSV summary
    summary_rows = []

    for name, arr in [
        ("TinyViT trained", bf_tiny),
        ("TinyViT random", bf_tiny_rnd),
        ("DenseNet trained", bf_dense),
        ("DenseNet random", bf_dense_rnd),
    ]:

        if len(arr) > 0:

            summary_rows.append({
                "model": name,
                "n": len(arr),
                "mean_border_attention": float(
                    arr.mean()
                ),
                "median_border_attention": float(
                    np.median(arr)
                ),
                "std_border_attention": float(
                    arr.std()
                )
            })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        os.path.join(
            Config.output_dir,
            "border_attention_summary.csv"
        ),
        index=False
    )

    print(
        "\nSaved results to:"
    )

    print(
        Config.output_dir
    )

    print(
        "\nDONE."
    )


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":

    main()