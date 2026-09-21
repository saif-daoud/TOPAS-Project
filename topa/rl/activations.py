"""Utilities for loading cached LLM internal representations ("activations").

This module centralizes activation loading for:
  - offline RL rollouts in activation space
  - SFT pretraining on activation vectors

Activation cache format (expected):
  <activation_root>/user_<user_idx>/session_<session_idx>.pt

The .pt file should contain a dict with key "activations" of shape:
  [n_layers, seq_len, hidden_dim]

Notes
-----
We keep these helpers small and dependency-light. They are intentionally
domain-agnostic: they only deal with file paths and tensors.
"""



from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import os
import json
import re
import torch

from .sft_pretraining.utils import safe_name


@dataclass
class ActivationSpec:
    """Specifies how to locate and read activation tensors."""

    root: Path
    key: str = "activations"

    @staticmethod
    def resolve(root_dir: Path, activation_path: str) -> "ActivationSpec":
        """Resolve activation root.

        Parameters
        ----------
        root_dir:
            A DataRoot root. Used only when `activation_path` is relative.
        activation_path:
            Either an absolute path, or a path relative to `root_dir`.

        Returns
        -------
        ActivationSpec
        """

        # Empty string -> try common defaults relative to the repo root.
        if not activation_path or not str(activation_path).strip():
            cand1 = (root_dir / "extract_internal_representation" / "activations").resolve()
            if cand1.exists():
                return ActivationSpec(root=cand1)
            cand2 = (root_dir / "activations").resolve()
            return ActivationSpec(root=cand2)

        p = Path(activation_path).expanduser()
        if not p.is_absolute():
            p = (root_dir / p).resolve()

        # Backward-compat: some scripts pass a directory containing "activations_*".
        # If the user points at a parent directory and it contains exactly one
        # matching activations folder, pick it.
        if p.exists() and p.is_dir():
            candidates = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("activations")]
            if (p.name.startswith("activations") and _has_user_dir(p)) or _has_user_dir(p):
                return ActivationSpec(root=p)
            if len(candidates) == 1 and _has_user_dir(candidates[0]):
                return ActivationSpec(root=candidates[0])
        return ActivationSpec(root=p)


_ID_DIR_RE = re.compile(r"^([A-Za-z][A-Za-z0-9._-]*)_(\d+)$")


def short_model_name(model_id: str) -> str:
    name = str(model_id).replace("\\", "/").split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or "model"


def role_labels_for_root(root: Path) -> dict:
    path = root / "role_labels.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"system": "system", "user": "user"}


def resolve_activation_path(root: Path, activation_out_dir: str, model_id: Optional[str]) -> Path:
    path_text = str(activation_out_dir).strip()
    if path_text and "path/to/activations" in path_text:
        path_text = ""
    if "{model}" in path_text:
        if not model_id:
            pattern = path_text.replace("{model}", "*")
            path = Path(pattern)
            if not path.is_absolute():
                path = (root / path).resolve()
            candidates = [d for d in path.parent.glob(path.name) if d.is_dir()]
            assert len(candidates) == 1
            return candidates[0]
        path_text = path_text.replace("{model}", short_model_name(model_id))

    default = root / "extract_internal_representation" / f"activations_{short_model_name(model_id)}" if model_id else root / "extract_internal_representation" / "activations"
    path = Path(path_text).expanduser() if path_text else default
    return path if path.is_absolute() else (root / path).resolve()


def _find_id_dirs(act_root: Path) -> list[Path]:
    if not act_root.exists():
        return []
    return [p for p in act_root.iterdir() if p.is_dir() and _ID_DIR_RE.match(p.name)]


def _has_user_dir(act_root: Path) -> bool:
    """Return True if the root contains any user/session directories."""
    return len(_find_id_dirs(act_root)) > 0


def activation_cache_has_files(act_root: Path) -> bool:
    return any(p.is_dir() and list(p.glob("session_*.pt")) for p in _find_id_dirs(act_root))


def dataset_json_for_root(root: Path, dataset_path: Optional[str] = None) -> Optional[Path]:
    if dataset_path:
        path = Path(dataset_path).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        if path.exists():
            return path
    candidates = sorted((root / "data").glob("*.json"))
    return candidates[0] if candidates else None


def activation_prefixes(root: Path, act_root: Path) -> list[str]:
    labels = role_labels_for_root(root)
    prefixes = [safe_name(labels["user"]), "user"]
    prefixes += [m.group(1) for p in _find_id_dirs(act_root) if (m := _ID_DIR_RE.match(p.name))]
    return list(dict.fromkeys(prefixes))


def activation_cache_complete(root: Path, act_root: Path, dataset_path: Optional[str] = None) -> bool:
    data_path = dataset_json_for_root(root, dataset_path)
    if data_path is None:
        return activation_cache_has_files(act_root)

    data = json.loads(data_path.read_text(encoding="utf-8"))
    for prefix in activation_prefixes(root, act_root):
        expected = [
            act_root / f"{prefix}_{user_idx}" / f"session_{session_idx}.pt"
            for user_idx, user in enumerate(data)
            for session_idx, _ in enumerate(user["sessions"])
        ]
        if expected and all(path.exists() for path in expected):
            return True
    return False


def _activation_prefix(act_root: Path) -> str:
    # Prefer a "user_*" prefix when present, otherwise take the first id dir prefix.
    for p in _find_id_dirs(act_root):
        if p.name.startswith("user_"):
            return "user"
    for p in sorted(_find_id_dirs(act_root), key=lambda x: x.name):
        m = _ID_DIR_RE.match(p.name)
        if m:
            return m.group(1)
    return "user"


def activation_file(act_root: Path, user_idx: int, session_idx: int) -> Path:
    prefix = _activation_prefix(act_root)
    p = act_root / f"{prefix}_{int(user_idx)}" / f"session_{int(session_idx)}.pt"
    if p.exists():
        return p
    # Fallback: try any directory that matches "*_{user_idx}"
    for cand in act_root.glob(f"*_{int(user_idx)}"):
        if cand.is_dir():
            alt = cand / f"session_{int(session_idx)}.pt"
            if alt.exists():
                return alt
    return p


def load_session_activations(
    act_root: Path,
    user_idx: int,
    session_idx: int,
    *,
    key: str = "activations",
    map_location: str = "cpu",
) -> torch.Tensor:
    """Load activations tensor for one (user, session)."""

    p = activation_file(act_root, user_idx, session_idx)
    if not p.exists():
        raise FileNotFoundError(f"Activation file not found: {p}")
    obj = torch.load(str(p), map_location=map_location)
    if key not in obj:
        raise KeyError(f"Activation file {p} missing key '{key}'. Keys: {list(obj.keys())}")
    return obj[key]


def load_layer_matrix(
    act_root: Path,
    user_idx: int,
    session_idx: int,
    layer_idx: int,
    *,
    key: str = "activations",
    cache: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
) -> torch.Tensor:
    """Load the [seq_len, hidden_dim] matrix for a single layer.

    If `cache` is provided, we keep the full [n_layers, seq_len, hidden_dim]
    tensor in memory per (user, session) to avoid repeated disk reads.
    """

    k = (int(user_idx), int(session_idx))
    if cache is not None and k in cache:
        acts = cache[k]
    else:
        acts = load_session_activations(act_root, user_idx, session_idx, key=key)
        if cache is not None:
            cache[k] = acts

    if layer_idx < 0:
        layer_idx = acts.shape[0] + int(layer_idx)
    if layer_idx < 0 or layer_idx >= acts.shape[0]:
        raise IndexError(f"layer_idx={layer_idx} out of range for activations with n_layers={acts.shape[0]}")
    return acts[int(layer_idx)]


def activation_vector(
    act_root: Path,
    user_idx: int,
    session_idx: int,
    token_idx: int,
    layer_idx: int,
    *,
    key: str = "activations",
    cache: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
) -> torch.Tensor:
    """Return a single activation vector at (layer_idx, token_idx)."""

    mat = load_layer_matrix(act_root, user_idx, session_idx, layer_idx, key=key, cache=cache)
    token_idx = int(token_idx)
    if token_idx < 0:
        token_idx = mat.shape[0] + token_idx
    if token_idx < 0 or token_idx >= mat.shape[0]:
        raise IndexError(
            f"token_idx={token_idx} out of range for layer matrix with seq_len={mat.shape[0]} (user={user_idx}, session={session_idx})"
        )
    return mat[token_idx]


def default_activation_layer() -> int:
    """Default activation layer, override with TOPA_ACTIVATION_LAYER."""

    v = (os.environ["TOPA_ACTIVATION_LAYER"] if "TOPA_ACTIVATION_LAYER" in os.environ else "-1").strip()
    return int(v)


def infer_num_layers(act_root: Path, key: str = "activations") -> int:
    """Infer number of layers from the first activation file found."""
    if not act_root.exists():
        raise FileNotFoundError(f"Activation root not found: {act_root}")

    for user_dir in _find_id_dirs(act_root):
        for sf in sorted(user_dir.glob("session_*.pt")):
            obj = torch.load(str(sf), map_location="cpu")
            acts = obj[key]
            if isinstance(acts, (list, tuple)):
                return int(len(acts))
            if hasattr(acts, "shape") and len(acts.shape) >= 1:
                return int(acts.shape[0])
    raise FileNotFoundError(f"No activation files with key '{key}' found under {act_root}")
