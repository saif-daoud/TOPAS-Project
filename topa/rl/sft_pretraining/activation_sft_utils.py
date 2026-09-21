"""SFT utilities for training simple MLP policies on activation vectors.

These models are meant to be compatible with the activation-space offline RL
heads, so SFT and offline RL use the same small MLP policy shape.
"""



from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import math
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from ..activations import ActivationSpec, load_layer_matrix
from ..simple_models import MLPPolicy

from .data_loading import (
    DataRoot,
    build_conv_state_label_maps,
    load_conversation_state_spec,
    norm_str,
)
from .utils import make_torch_generator, seed_worker, set_seed, topk_accuracy


class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        return {
            "input_embeds": torch.tensor(self.X[i], dtype=torch.float32),
            "labels": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


@dataclass
class TrainResult:
    out_dir: Path
    best_val_acc: float
    best_val_acc_at_2: float
    best_val_acc_at_3: float
    best_val_macro_f1: float
    selection_metric: str
    best_selection_score: float
    num_train: int
    num_val: int


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return float((pred == y).float().mean().item())


def _acc_at_k(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    return float(topk_accuracy(logits.detach().cpu().numpy(), y.detach().cpu().numpy(), k=int(k)))


def balanced_class_weights(y: np.ndarray, num_labels: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=int(num_labels)).astype(np.float32)
    return (float(len(y)) / (float(num_labels) * np.maximum(counts, 1.0))).astype(np.float32)


def _macro_f1(logits: torch.Tensor, y: torch.Tensor, num_labels: int) -> float:
    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    true = y.detach().cpu().numpy()
    scores = []
    for cls in range(int(num_labels)):
        tp = float(np.sum((pred == cls) & (true == cls)))
        fp = float(np.sum((pred == cls) & (true != cls)))
        fn = float(np.sum((pred != cls) & (true == cls)))
        denom = 2.0 * tp + fp + fn
        scores.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(scores))


def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    out_dir: Path,
    num_labels: int,
    hidden_dim: int = 256,
    projection_dim: int = 0,
    conv_state_embedding_dim: int = 0,
    dropout: float = 0.1,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    epochs: int = 10,
    seed: int = 48,
    device: Optional[str] = None,
    tb_dir: Optional[Path] = None,
    conv_state_dim: int = 0,
    class_weights: Optional[np.ndarray] = None,
    selection_metric: str = "accuracy",
) -> TrainResult:
    """Train an `MLPPolicy` with cross-entropy loss and save the best checkpoint."""

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    rng = np.random.default_rng(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = EmbeddingDataset(X_train, y_train)
    ds_val = EmbeddingDataset(X_val, y_val)

    generator = make_torch_generator(int(seed))
    dl_train = DataLoader(ds_train, batch_size=int(batch_size), shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=generator)
    dl_val = DataLoader(ds_val, batch_size=int(batch_size), shuffle=False, drop_last=False, worker_init_fn=seed_worker)

    model = MLPPolicy(
        in_dim=int(X_train.shape[1]),
        num_actions=int(num_labels),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        conv_state_dim=int(conv_state_dim),
        projection_dim=int(projection_dim),
        conv_state_embedding_dim=int(conv_state_embedding_dim),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    if class_weights is not None:
        weight_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight_t)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
    best_acc2 = -1.0
    best_acc3 = -1.0
    best_macro_f1 = -1.0
    best_score = -1.0
    best_path = out_dir / "actor.pt"
    history: List[Dict] = []

    writer = SummaryWriter(log_dir=str(tb_dir)) if tb_dir is not None else None

    for epoch in range(int(epochs)):
        model.train()
        train_losses = []
        train_accs = []
        train_acc2s = []
        train_acc3s = []
        for batch in dl_train:
            x = batch["input_embeds"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
            train_accs.append(_accuracy(logits.detach(), y.detach()))
            train_acc2s.append(_acc_at_k(logits.detach(), y.detach(), 2))
            train_acc3s.append(_acc_at_k(logits.detach(), y.detach(), 3))

        model.eval()
        with torch.no_grad():
            val_accs = []
            val_acc2s = []
            val_acc3s = []
            val_macro_f1s = []
            val_losses = []
            for batch in dl_val:
                x = batch["input_embeds"].to(device)
                y = batch["labels"].to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                val_losses.append(float(loss.detach().cpu()))
                val_accs.append(_accuracy(logits.detach(), y.detach()))
                val_acc2s.append(_acc_at_k(logits.detach(), y.detach(), 2))
                val_acc3s.append(_acc_at_k(logits.detach(), y.detach(), 3))
                val_macro_f1s.append(_macro_f1(logits.detach(), y.detach(), int(num_labels)))

        tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        tr_acc = float(np.mean(train_accs)) if train_accs else float("nan")
        tr_acc2 = float(np.mean(train_acc2s)) if train_acc2s else float("nan")
        tr_acc3 = float(np.mean(train_acc3s)) if train_acc3s else float("nan")
        va_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        va_acc = float(np.mean(val_accs)) if val_accs else float("nan")
        va_acc2 = float(np.mean(val_acc2s)) if val_acc2s else float("nan")
        va_acc3 = float(np.mean(val_acc3s)) if val_acc3s else float("nan")
        va_macro_f1 = float(np.mean(val_macro_f1s)) if val_macro_f1s else float("nan")
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "train_acc_at_2": tr_acc2,
                "train_acc_at_3": tr_acc3,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "val_acc_at_2": va_acc2,
                "val_acc_at_3": va_acc3,
                "val_macro_f1": va_macro_f1,
            }
        )

        if writer is not None:
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("loss/val", va_loss, epoch)
            writer.add_scalar("acc/train", tr_acc, epoch)
            writer.add_scalar("acc/val", va_acc, epoch)
            writer.add_scalar("acc_at_2/train", tr_acc2, epoch)
            writer.add_scalar("acc_at_2/val", va_acc2, epoch)
            writer.add_scalar("acc_at_3/train", tr_acc3, epoch)
            writer.add_scalar("acc_at_3/val", va_acc3, epoch)
            writer.add_scalar("macro_f1/val", va_macro_f1, epoch)

        score = va_macro_f1 if str(selection_metric) == "macro_f1" else va_acc
        if score > best_score:
            best_score = score
            best_acc = va_acc
            best_acc2 = va_acc2
            best_acc3 = va_acc3
            best_macro_f1 = va_macro_f1
            torch.save(model.state_dict(), best_path)

        print(f"[epoch {epoch}] train_acc={tr_acc:.4f} val_acc={va_acc:.4f} val_macro_f1={va_macro_f1:.4f} val_acc@2={va_acc2:.4f} val_acc@3={va_acc3:.4f}")

    # Save final + history
    with (out_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    effective_projection_dim = (int(projection_dim) if int(projection_dim) > 0 else (int(hidden_dim) if int(hidden_dim) > 0 else 0)) if int(conv_state_dim) > 0 else 0
    effective_conv_embedding_dim = int(conv_state_embedding_dim) if int(conv_state_dim) > 0 and int(conv_state_embedding_dim) > 0 else 0
    if int(conv_state_dim) <= 0:
        fusion = "plain"
    elif int(effective_projection_dim) > 0 and int(effective_conv_embedding_dim) > 0:
        fusion = "latent_projection_conv_embedding_concat"
    elif int(effective_projection_dim) > 0:
        fusion = "latent_projection_concat"
    elif int(effective_conv_embedding_dim) > 0:
        fusion = "latent_conv_embedding_concat"
    else:
        fusion = "latent_concat"
    model_meta = {
        "input_dim": int(X_train.shape[1]),
        "latent_dim": int(X_train.shape[1]) - int(conv_state_dim),
        "conv_state_dim": int(conv_state_dim),
        "projection_dim": int(effective_projection_dim),
        "conv_state_embedding_dim": int(effective_conv_embedding_dim),
        "fusion": fusion,
        "selection_metric": str(selection_metric),
        "best_selection_score": float(best_score),
    }
    with (out_dir / "model_meta.json").open("w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    if writer is not None:
        writer.flush()
        writer.close()

    return TrainResult(
        out_dir=out_dir,
        best_val_acc=float(best_acc),
        best_val_acc_at_2=float(best_acc2),
        best_val_acc_at_3=float(best_acc3),
        best_val_macro_f1=float(best_macro_f1),
        selection_metric=str(selection_metric),
        best_selection_score=float(best_score),
        num_train=len(ds_train),
        num_val=len(ds_val),
    )


def build_conv_state_vectors(root: DataRoot, df: pd.DataFrame, encoding: str = "one_hot") -> Tuple[np.ndarray, List[str]]:
    """Create conversation-state features per row.

    Uses `components/conversation_states.json` (domain-agnostic) to determine
    which columns represent conversation-state dimensions and how to map
    labels to IDs.
    """
    encoding = str(encoding).strip().lower()
    spec = load_conversation_state_spec(root)
    dim_names, dim_to_label2id = build_conv_state_label_maps(spec)

    if encoding == "label":
        X = np.zeros((len(df), len(dim_names)), dtype=np.float32)
        for j, dim in enumerate(dim_names):
            label2id = dim_to_label2id[dim]
            none_id = int(label2id["none"])
            if dim not in df.columns:
                X[:, j] = float(none_id)
                continue
            for i, raw in enumerate(df[dim].tolist()):
                lab = norm_str(raw).lower()
                X[i, j] = float(label2id[lab]) if lab in label2id else float(none_id)
        return X, dim_names

    offsets: Dict[str, int] = {}
    feature_names: List[str] = []
    off = 0
    for dim in dim_names:
        offsets[dim] = off
        label2id = dim_to_label2id[dim]
        id2label = {int(v): str(k) for k, v in label2id.items()}
        for i in range(len(label2id)):
            feature_names.append(f"{dim}={id2label[i]}")
        off += len(label2id)

    X = np.zeros((len(df), off), dtype=np.float32)

    for dim in dim_names:
        if dim not in df.columns:
            label2id = dim_to_label2id[dim]
            X[:, int(offsets[dim]) + int(label2id["none"])] = 1.0
            continue
        label2id = dim_to_label2id[dim]
        none_id = int(label2id["none"])
        offset = int(offsets[dim])
        for i, raw in enumerate(df[dim].tolist()):
            lab = norm_str(raw).lower()
            label_id = int(label2id[lab]) if lab in label2id else none_id
            X[i, offset + label_id] = 1.0

    return X, feature_names


def build_activation_features(
    *,
    root: DataRoot,
    df: pd.DataFrame,
    activation_path: str,
    activation_layer: int,
    activation_key: str,
    index_col: str,
    concat_extra: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load activation vectors for each row and return X plus a mask of valid rows.

    Args:
        index_col: column holding the integer token/utterance index used to
            slice the activations matrix.
        concat_extra: optional extra features [N, D_extra] to concatenate.

    Returns:
        X: [M, D] activation features (plus concat_extra if provided)
        keep_mask: boolean mask [N] indicating which rows were kept
    """
    act_root = ActivationSpec.resolve(root.root, activation_path).root

    sess_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    feats: List[np.ndarray] = []
    keep = np.zeros(len(df), dtype=bool)

    for i, row in df.iterrows():
        u = int(row["user_idx"]) if "user_idx" in row else None
        s = int(row["session_idx"]) if "session_idx" in row else None
        if u is None or s is None:
            continue
        idx = int(row[index_col])
        if (u, s) not in sess_cache:
            try:
                sess_cache[(u, s)] = load_layer_matrix(
                    act_root,
                    u,
                    s,
                    layer_idx=int(activation_layer),
                    key=str(activation_key),
                    cache=None,
                )
            except FileNotFoundError:
                continue

        mat = sess_cache[(u, s)]
        if idx < 0 or idx >= int(mat.shape[0]):
            continue

        v = mat[idx].detach().to(dtype=torch.float32).cpu().numpy()
        if concat_extra is not None:
            v = np.concatenate([v, concat_extra[i].astype(np.float32)], axis=0)
        feats.append(v.astype(np.float32))
        keep[i] = True

    if len(feats) == 0:
        return np.zeros((0, 0), dtype=np.float32), keep

    X = np.stack(feats, axis=0)
    return X, keep
