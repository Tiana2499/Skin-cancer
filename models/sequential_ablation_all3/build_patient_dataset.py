"""
build_patient_dataset.py
Build per-patient lesion sets for Set Transformer training.
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PatientLesionDataset(Dataset):
    """
    Each __getitem__ returns:
        lesion_features: (N_lesions, 1280)
        mask: (N_lesions,) boolean
        label: scalar (1 if any lesion malignant)
        patient_id: string
    """

    def __init__(self, features, patient_ids, labels, max_lesions=50):
        self.max_lesions = max_lesions
        self.samples = []

        # Group by patient
        df = pd.DataFrame({
            "feature_idx": np.arange(len(features)),
            "patient_id": patient_ids,
            "label": labels,
        })

        for pid, group in df.groupby("patient_id"):
            idxs = group["feature_idx"].values[:max_lesions]
            self.samples.append({
                "indices": idxs,
                "label": int(group["label"].max()),
                "patient_id": pid,
            })

        self.features = features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        idxs = sample["indices"]

        feats = self.features[idxs]  # (N, 1280)
        n = len(idxs)

        # Pad to max_lesions
        padded = np.zeros((self.max_lesions, feats.shape[1]), dtype=np.float32)
        padded[:n] = feats

        mask = np.zeros(self.max_lesions, dtype=bool)
        mask[:n] = True

        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
            torch.tensor(sample["label"], dtype=torch.float32),
            sample["patient_id"],
        )