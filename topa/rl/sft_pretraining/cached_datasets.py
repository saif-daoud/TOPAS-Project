from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class CachedFeatures:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    labels: np.ndarray
    extra: Dict[str, np.ndarray]

class CachedTensorDataset(Dataset):
    """Dataset that serves already-tokenized tensors from cached pickle."""

    def __init__(self, features: CachedFeatures):
        self.features = features
        # Propagate optional metadata (e.g., conv-state label maps) from cached features.
        for _attr in ("dim_names", "dim_to_label2id", "dim_num_labels", "spec_dim_names"):
            if hasattr(features, _attr):
                setattr(self, _attr, getattr(features, _attr))

    def __len__(self) -> int:
        return int(self.features.input_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {
            "input_ids": torch.tensor(self.features.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.features.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.features.labels[idx]),
        }
        # dtype fixups
        if out["labels"].dtype not in (torch.long, torch.int64, torch.float32):
            out["labels"] = out["labels"].to(torch.long)

        for k, v in self.features.extra.items():
            t = torch.tensor(v[idx])
            # Heuristics: ids -> long, vectors -> float
            if t.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8):
                out[k] = t.to(torch.long)
            else:
                out[k] = t.to(torch.float32)
        return out

def stack_collate(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Simple collator for cached fixed-length tensors."""
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out