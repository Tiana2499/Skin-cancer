"""
set_transformer_aggregator.py
Learned patient-level aggregation using a small Set Transformer.
Requires: per-lesion 1280-D features (not just probabilities).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetTransformerAggregator(nn.Module):
    """
    Takes N lesions per patient, outputs a single patient-level logit.

    Architecture:
        1. Linear projection: 1280 → 256
        2. Multi-head self-attention (across lesions)
        3. Mean pooling → patient embedding (256-D)
        4. MLP classifier: 256 → 64 → 1
    """

    def __init__(self, input_dim=1280, hidden_dim=256, n_heads=4, dropout=0.3):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, lesion_features, mask=None):
        """
        lesion_features: (batch, max_lesions, 1280)
        mask: (batch, max_lesions) — True for real lesions, False for padding
        """
        # Project
        x = self.projection(lesion_features)  # (B, N, H)

        # Self-attention across lesions
        # key_padding_mask: True for positions to ignore
        attn_out, _ = self.attention(
            x, x, x,
            key_padding_mask=~mask if mask is not None else None,
        )
        x = self.norm(x + attn_out)

        # Masked mean pooling
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # (B, N, 1)
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)

        # Classify
        logits = self.classifier(x).squeeze(-1)  # (B,)
        return logits