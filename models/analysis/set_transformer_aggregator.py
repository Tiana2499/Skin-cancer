"""
set_transformer_aggregator.py
Learned patient-level aggregation via Set Transformer.
Compares against lesion-level baseline and naive aggregations.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklearn_metrics


class Config:
    model_dir = r"./models/sequential_ablation_all3"
    feature_dir = r"./features/tabular"
    tinyvit_dir = r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN"
    densenet_dir = r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold"
    output_dir = r"./analysis/set_transformer"
    max_lesions = 100
    hidden_dim = 128
    n_heads = 4
    batch_size = 8
    epochs = 30
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class SetTransformerAggregator(nn.Module):
    """
    Aggregates N lesions into one patient-level prediction.

    Input: (B, N, F) lesion features
    Output: (B,) patient-level logit
    """
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, lesion_features, mask):
        """
        lesion_features: (B, N, F)
        mask: (B, N) boolean — True for real lesions, False for padding
        """
        x = self.projection(lesion_features)
        attn_out, _ = self.attention(
            x, x, x, key_padding_mask=~mask)
        x = self.norm(x + attn_out)

        # Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return self.classifier(x).squeeze(-1)


class PatientDataset(Dataset):
    def __init__(self, features, patient_ids, labels, max_lesions):
        self.max_lesions = max_lesions
        self.samples = []

        df = pd.DataFrame({
            "idx": np.arange(len(features)),
            "patient_id": patient_ids,
            "label": labels,
        })

        for pid, group in df.groupby("patient_id"):
            idxs = group["idx"].values[:max_lesions]
            self.samples.append({
                "indices": idxs,
                "label": int(group["label"].max()),
            })

        self.features = features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feats = self.features[s["indices"]]
        n = len(s["indices"])

        padded = np.zeros((self.max_lesions, feats.shape[1]), dtype=np.float32)
        padded[:n] = feats
        mask = np.zeros(self.max_lesions, dtype=bool)
        mask[:n] = True

        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
            torch.tensor(s["label"], dtype=torch.float32),
        )


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    # Load features
    tab = pd.read_parquet(os.path.join(Config.feature_dir, "test_tabular.parquet"))
    tab = tab.drop(columns=[c for c in ["mel_thick_mm_dev", "mel_thick_mm"] if c in tab.columns])
    META = ["isic_id", "patient_id", "target"]
    tab_cols = [c for c in tab.columns if c not in META]

    # Load OOF predictions (here we use OOF for training, test for evaluation)
    # NOTE: For simplicity, this uses test features for both. A proper setup
    # would train on train+val OOF and test on test predictions.

    # Actually, since Set Transformer is trained, we need train+val data:
    tab_train = pd.read_parquet(os.path.join(Config.feature_dir, "trainval_tabular.parquet"))
    tab_train = tab_train.drop(columns=[c for c in ["mel_thick_mm_dev", "mel_thick_mm"] if c in tab_train.columns])
    tab_test = tab

    tiny_train = np.load(os.path.join(Config.tinyvit_dir, "oof_probs.npy"))
    tiny_train_ids = np.load(os.path.join(Config.tinyvit_dir, "oof_ids.npy"), allow_pickle=True)
    dense_train = np.load(os.path.join(Config.densenet_dir, "oof_probs.npy"))
    dense_train_ids = np.load(os.path.join(Config.densenet_dir, "oof_ids.npy"), allow_pickle=True)

    # Sort and align
    order = np.argsort(tiny_train_ids)
    tiny_train = tiny_train[order]
    tiny_train_ids = tiny_train_ids[order]
    dense_order = np.argsort(dense_train_ids)
    dense_train = dense_train[dense_order]

    tab_train = tab_train.set_index("isic_id").loc[tiny_train_ids].reset_index()

    X_train = np.concatenate([
        tiny_train[:, None], dense_train[:, None],
        tab_train[tab_cols].values.astype(np.float32)], axis=1)
    y_train = tab_train["target"].values.astype(int)
    pid_train = tab_train["patient_id"].values

    # Test features
    tiny_test = np.load(os.path.join(Config.tinyvit_dir, "test_probs.npy"))
    dense_test = np.load(os.path.join(Config.densenet_dir, "test_probs.npy"))
    # Assume row order matches test manifest
    X_test = np.concatenate([
        tiny_test[:, None], dense_test[:, None],
        tab_test[tab_cols].values.astype(np.float32)], axis=1)
    y_test = tab_test["target"].values.astype(int)
    pid_test = tab_test["patient_id"].values

    print(f"Train+Val: {len(y_train):,}  Positives: {y_train.sum()}")
    print(f"Test:      {len(y_test):,}  Positives: {y_test.sum()}")

    # Lesion-level baseline
    lesion_pauc = isic2024_pauc(y_test, X_test[:, 0] * 0.5 + X_test[:, 1] * 0.5)
    print(f"\nLesion-level baseline (avg of image probs): pAUC = {lesion_pauc:.4f}")

    # Patient-level naive aggregations (from earlier analysis, showing here for reference)
    df_test = pd.DataFrame({"pid": pid_test, "y": y_test,
                            "p_avg": (X_test[:, 0] + X_test[:, 1]) / 2})
    agg_max = df_test.groupby("pid").agg(p=("p_avg", "max"), y=("y", "max"))
    patient_max_pauc = isic2024_pauc(agg_max["y"].values, agg_max["p"].values)
    print(f"Patient-max (naive): pAUC = {patient_max_pauc:.4f}")

    # ====================================================================
    # Set Transformer
    # ====================================================================
    print(f"\nTraining Set Transformer...")

    train_ds = PatientDataset(X_train, pid_train, y_train, Config.max_lesions)
    test_ds = PatientDataset(X_test, pid_test, y_test, Config.max_lesions)

    # Split train/val by patient
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(
        range(len(train_ds.samples)),
        groups=[s["label"] for s in train_ds.samples]))

    train_subset = torch.utils.data.Subset(train_ds, train_idx)
    val_subset = torch.utils.data.Subset(train_ds, val_idx)

    train_loader = DataLoader(train_subset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False)

    model = SetTransformerAggregator(
        input_dim=X_train.shape[1],
        hidden_dim=Config.hidden_dim,
        n_heads=Config.n_heads,
    ).to(Config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
        for feats, mask, label in train_loader:
            feats = feats.to(Config.device)
            mask = mask.to(Config.device)
            label = label.to(Config.device)

            optimizer.zero_grad()
            logit = model(feats, mask)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{Config.epochs}  loss={train_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for feats, mask, label in test_loader:
            feats = feats.to(Config.device)
            mask = mask.to(Config.device)
            logits = model(feats, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(label.numpy())

    patient_probs = np.concatenate(all_probs)
    patient_labels = np.concatenate(all_labels)

    set_transformer_pauc = isic2024_pauc(patient_labels, patient_probs)
    set_transformer_auroc = roc_auc_score(patient_labels, patient_probs)

    print(f"\nSet Transformer patient-level:")
    print(f"  AUROC: {set_transformer_auroc:.4f}")
    print(f"  pAUC:  {set_transformer_pauc:.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("PATIENT-LEVEL AGGREGATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Method':<30} {'AUROC':>10} {'pAUC':>10}")
    print("-" * 70)
    print(f"{'Lesion-level':<30} {roc_auc_score(y_test, (X_test[:, 0] + X_test[:, 1]) / 2):>10.4f} {lesion_pauc:>10.4f}")
    print(f"{'Patient-max (naive)':<30} {roc_auc_score(agg_max['y'], agg_max['p']):>10.4f} {patient_max_pauc:>10.4f}")
    print(f"{'Set Transformer':<30} {set_transformer_auroc:>10.4f} {set_transformer_pauc:>10.4f}")


if __name__ == "__main__":
    main()