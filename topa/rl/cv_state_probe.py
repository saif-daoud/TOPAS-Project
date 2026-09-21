"""Frozen conversation-state probe utilities.

The classes below are intentionally kept as lightweight PyTorch modules so they
can be reused by activation-space SFT and offline RL without depending on the
original probe training script.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenProbeFeatureExtractor(nn.Module):
    """
    Best-use frozen conversation-state probe for RL.

    Keeps the full class probability distribution from every conversation-state head:

        probs_all = concat(softmax(head_1), ..., softmax(head_62))

    Optionally appends the frozen probe hidden representation z_probe.

    Output:
        concat(probs_all, z_probe) if include_probe_hidden=True
        probs_all otherwise
    """
    def __init__(self, checkpoint_path: Path, include_probe_hidden: bool = True):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.include_probe_hidden = include_probe_hidden

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format: {self.checkpoint_path}")

        clean_sd = {}
        for k, v in state_dict.items():
            kk = k.replace("module.", "")
            if isinstance(v, torch.Tensor):
                clean_sd[kk] = v.detach().float().cpu()

        head_prefix = None
        for cand in ["heads", "classifiers", "output_heads"]:
            if any(re.match(rf"^{cand}\.\d+\.weight$", k) for k in clean_sd):
                head_prefix = cand
                break

        if head_prefix is None:
            raise ValueError("Could not find probe heads. Expected heads.0.weight or classifiers.0.weight.")

        head_ids = sorted({
            int(re.match(rf"^{head_prefix}\.(\d+)\.weight$", k).group(1))
            for k in clean_sd
            if re.match(rf"^{head_prefix}\.\d+\.weight$", k)
        })

        self.head_weights = nn.ParameterList()
        self.head_biases = nn.ParameterList()
        for i in head_ids:
            w = clean_sd[f"{head_prefix}.{i}.weight"]
            b = clean_sd.get(f"{head_prefix}.{i}.bias", torch.zeros(w.shape[0]))
            self.head_weights.append(nn.Parameter(w, requires_grad=False))
            self.head_biases.append(nn.Parameter(b, requires_grad=False))

        self.num_classes_per_dim = [int(w.shape[0]) for w in self.head_weights]
        self.total_prob_dim = int(sum(self.num_classes_per_dim))
        self.head_input_dim = int(self.head_weights[0].shape[1])

        self.backbone_prefix = None
        for cand in ["encoder", "encoder.net", "shared", "net", "backbone", "feature_extractor", "projection", "hidden", "mlp"]:
            pattern = rf"^{re.escape(cand)}\.\d+\.weight$"
            if any(re.match(pattern, k) for k in clean_sd):
                self.backbone_prefix = cand
                break

        self.backbone_layers = nn.ModuleList()
        if self.backbone_prefix is not None:
            pattern = rf"^{re.escape(self.backbone_prefix)}\.(\d+)\.weight$"
            layer_ids = sorted({
                int(re.match(pattern, k).group(1))
                for k in clean_sd
                if re.match(pattern, k)
            })
            for idx in layer_ids:
                w = clean_sd[f"{self.backbone_prefix}.{idx}.weight"]
                b = clean_sd.get(f"{self.backbone_prefix}.{idx}.bias", torch.zeros(w.shape[0]))
                lin = nn.Linear(w.shape[1], w.shape[0])
                lin.weight.data.copy_(w)
                lin.bias.data.copy_(b)
                for p in lin.parameters():
                    p.requires_grad = False
                self.backbone_layers.append(lin)

        for p in self.parameters():
            p.requires_grad = False

        self.hidden_dim = self.head_input_dim
        self.output_dim = self.total_prob_dim + (self.hidden_dim if self.include_probe_hidden else 0)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.backbone_layers) == 0:
            return x
        h = x
        for i, layer in enumerate(self.backbone_layers):
            h = layer(h)
            if i < len(self.backbone_layers) - 1:
                h = F.relu(h)
        return h

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_probe = self._forward_backbone(x)

        if z_probe.shape[-1] != self.head_input_dim:
            raise RuntimeError(
                f"Probe mismatch: backbone output dim {z_probe.shape[-1]} "
                f"but head input dim {self.head_input_dim}. "
                "Your checkpoint architecture may need a custom loader."
            )

        probs_list = []
        for w, b in zip(self.head_weights, self.head_biases):
            logits = F.linear(z_probe, w, b)
            probs_list.append(F.softmax(logits, dim=-1))

        probs_all = torch.cat(probs_list, dim=-1)

        if self.include_probe_hidden:
            return torch.cat([probs_all, z_probe], dim=-1)

        return probs_all


class BestCVStateEncoder(nn.Module):
    """
    Supports two variants:

    baseline_full_rl_512:
        h_t -> trainable RL projector -> 512 dims

    rl_plus_full_frozen_cv:
        h_t -> trainable RL projector -> rl_dim
        h_t -> frozen CV probe:
                full head probabilities + optional probe hidden
             -> trainable CV adapter -> cv_adapter_dim
        z_t = concat(rl_z, cv_z)
    """
    def __init__(
        self,
        input_dim: int,
        variant: str,
        rl_dim: int,
        cv_adapter_dim: int,
        hidden_dim: int,
        dropout: float,
        probe_checkpoint: Optional[Path] = None,
        include_probe_hidden: bool = True,
    ):
        super().__init__()

        self.variant = variant
        self.rl_dim = int(rl_dim)
        self.cv_adapter_dim = int(cv_adapter_dim)

        self.rl_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rl_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if variant == "baseline_full_rl_512":
            self.cv_probe = None
            self.cv_adapter = None
            self.output_dim = rl_dim

        elif variant == "rl_plus_full_frozen_cv":
            if probe_checkpoint is None:
                raise ValueError("probe_checkpoint is required for rl_plus_full_frozen_cv.")

            self.cv_probe = FrozenProbeFeatureExtractor(
                checkpoint_path=probe_checkpoint,
                include_probe_hidden=include_probe_hidden,
            )
            for p in self.cv_probe.parameters():
                p.requires_grad = False

            self.cv_adapter = nn.Sequential(
                nn.Linear(self.cv_probe.output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, cv_adapter_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(cv_adapter_dim),
            )

            self.output_dim = rl_dim + cv_adapter_dim

        else:
            raise ValueError(f"Unknown variant={variant}")

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        rl_z = self.rl_projector(h)

        if self.variant == "baseline_full_rl_512":
            return rl_z, None

        with torch.no_grad():
            cv_raw = self.cv_probe(h)

        cv_z = self.cv_adapter(cv_raw)
        z = torch.cat([rl_z, cv_z], dim=-1)

        return z, cv_z


class MLPDownprojectProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes_per_dim: List[int], projection_dim=512, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, n_cls) for n_cls in num_classes_per_dim])

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return [head(z) for head in self.heads], z


def frozen_probe_vectors_from_numpy(
    X: np.ndarray,
    checkpoint_path: str | Path,
    *,
    batch_size: int = 1024,
    device: Optional[str] = None,
    include_probe_hidden: bool = False,
) -> np.ndarray:
    """Return frozen probe features for a numpy activation matrix.

    By default this returns only the concatenated probability vector, matching
    the requested `include_probe_hidden=False` setting.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D activation matrix, got shape={X.shape}")
    if X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    probe = FrozenProbeFeatureExtractor(
        checkpoint_path=Path(checkpoint_path),
        include_probe_hidden=bool(include_probe_hidden),
    ).to(device).eval()

    outs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], int(batch_size)):
            end = start + int(batch_size)
            xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            outs.append(probe(xb).detach().float().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def append_frozen_probe_features(
    X: np.ndarray,
    checkpoint_path: str | Path,
    *,
    batch_size: int = 1024,
    include_probe_hidden: bool = False,
) -> tuple[np.ndarray, int]:
    """Append frozen probe probabilities to activation features."""
    probe_X = frozen_probe_vectors_from_numpy(
        X,
        checkpoint_path=checkpoint_path,
        batch_size=int(batch_size),
        include_probe_hidden=bool(include_probe_hidden),
    )
    return np.concatenate([np.asarray(X, dtype=np.float32), probe_X], axis=1).astype(np.float32), int(probe_X.shape[1])


def _load_json(path: Path) -> dict:
    import json
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class _SFTActivationConvStateProbe(nn.Module):
    """Loader for the existing train_conv_state_activations.py MultiHeadMLP."""

    def __init__(self, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        clean_sd = {k.replace("module.", ""): v.detach().float().cpu() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        head_ids = sorted({int(re.match(r"^heads\.(\d+)\.weight$", k).group(1)) for k in clean_sd if re.match(r"^heads\.\d+\.weight$", k)})
        if not head_ids:
            raise ValueError("Could not find SFT conv-state heads in actor.pt")

        if "trunk.0.weight" in clean_sd:
            w = clean_sd["trunk.0.weight"]
            b = clean_sd.get("trunk.0.bias", torch.zeros(w.shape[0]))
            self.trunk = nn.Sequential(nn.Linear(w.shape[1], w.shape[0]), nn.ReLU())
            self.trunk[0].weight.data.copy_(w)
            self.trunk[0].bias.data.copy_(b)
            head_input_dim = int(w.shape[0])
        else:
            first_head_w = clean_sd[f"heads.{head_ids[0]}.weight"]
            self.trunk = nn.Identity()
            head_input_dim = int(first_head_w.shape[1])

        self.heads = nn.ModuleList()
        for i in head_ids:
            w = clean_sd[f"heads.{i}.weight"]
            b = clean_sd.get(f"heads.{i}.bias", torch.zeros(w.shape[0]))
            if int(w.shape[1]) != head_input_dim:
                raise ValueError("Inconsistent SFT conv-state head dimensions")
            head = nn.Linear(w.shape[1], w.shape[0])
            head.weight.data.copy_(w)
            head.bias.data.copy_(b)
            self.heads.append(head)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.trunk(x)
        return [head(h) for head in self.heads]


def sft_conv_state_vectors_from_numpy(
    X: np.ndarray,
    model_dir: str | Path,
    *,
    encoding: str = "one_hot",
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> tuple[np.ndarray, List[str]]:
    """Predict conv-state vectors with an existing activation-SFT conv-state model."""
    X = np.asarray(X, dtype=np.float32)
    model_dir = Path(model_dir)
    label_map = _load_json(model_dir / "label_map.json")
    dim_names = list(label_map["dim_names"])
    dim_to_label2id = label_map["dim_to_label2id"]
    dim_num_labels = [len(dim_to_label2id[d]) for d in dim_names]

    offsets = {}
    feature_names: List[str] = []
    off = 0
    for dim in dim_names:
        offsets[dim] = off
        id2label = {int(v): str(k) for k, v in dim_to_label2id[dim].items()}
        for i in range(len(dim_to_label2id[dim])):
            feature_names.append(f"{dim}={id2label[i]}")
        off += len(dim_to_label2id[dim])

    state = torch.load(model_dir / "actor.pt", map_location="cpu", weights_only=False)
    model = _SFTActivationConvStateProbe(state)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    out_dim = off if str(encoding).strip().lower() == "one_hot" else len(dim_names)
    out = np.zeros((X.shape[0], out_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, X.shape[0], int(batch_size)):
            end = min(start + int(batch_size), X.shape[0])
            logits_list = model(torch.tensor(X[start:end], dtype=torch.float32, device=device))
            rows = np.arange(start, end)
            for j, dim in enumerate(dim_names):
                pred = torch.argmax(logits_list[j], dim=-1).detach().cpu().numpy()
                if str(encoding).strip().lower() == "one_hot":
                    out[rows, int(offsets[dim]) + pred] = 1.0
                else:
                    out[rows, j] = pred.astype(np.float32)
    return out.astype(np.float32), (feature_names if str(encoding).strip().lower() == "one_hot" else dim_names)


def append_sft_conv_state_features(
    X: np.ndarray,
    model_dir: str | Path,
    *,
    encoding: str = "one_hot",
    batch_size: int = 1024,
) -> tuple[np.ndarray, int, List[str]]:
    conv_X, names = sft_conv_state_vectors_from_numpy(
        X,
        model_dir=model_dir,
        encoding=str(encoding),
        batch_size=int(batch_size),
    )
    return np.concatenate([np.asarray(X, dtype=np.float32), conv_X], axis=1).astype(np.float32), int(conv_X.shape[1]), names
