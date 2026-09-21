"""Conversation-state SFT on latent vectors (LLM activations).

Trains a simple multi-head MLP on activation vectors to predict conversation
state labels for each dimension.
"""



import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from .data_loading import (
    DataRoot,
    load_conv_state_annotations,
    load_conversation_state_spec,
    build_conv_state_label_maps,
)
from .utils import add_preferred_turn_index, load_json, make_torch_generator, save_json, seed_worker, set_seed, split_df_user, split_df_stratify, norm_str, role_labels
from .activation_sft_utils import build_activation_features


class ConvStateActivationDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        return {
            "input_embeds": torch.tensor(self.X[i], dtype=torch.float32),
            "labels": torch.tensor(self.y[i], dtype=torch.long),
        }


class MultiHeadMLP(nn.Module):
    def __init__(self, input_dim: int, dim_num_labels: List[int], hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        if int(hidden_dim) > 0:
            self.trunk = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )
            head_dim = int(hidden_dim)
        else:
            self.trunk = nn.Identity()
            head_dim = int(input_dim)
        self.heads = nn.ModuleList([nn.Linear(int(head_dim), int(n)) for n in dim_num_labels])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.trunk(x)
        return [head(h) for head in self.heads]


def _resolve_index_col(df: pd.DataFrame) -> str:
    for c in ["activation_turn_idx", "from_idx", "utterance_idx", "to_idx"]:
        if c in df.columns:
            return c
    raise ValueError("Conv-state annotations must include a turn id or idx column to align activations.")


def _resolve_user_split_file(path: str, root: DataRoot) -> Path:
    p = Path(path).expanduser()
    if p.exists():
        return p
    data_dir = root.data_dir
    _, user_label = role_labels()
    user_label = user_label.strip().lower().replace(" ", "_")
    candidates = [data_dir / f"{user_label}_split.json", data_dir / "user_split.json"]
    for c in candidates:
        if c.exists():
            return c
    return p


def _parse_idx_cols(df: pd.DataFrame) -> pd.DataFrame:
    return add_preferred_turn_index(
        df,
        "activation_turn_idx",
        ["from_id", "utterance_id", "turn_id", "to_id"],
        ["from_idx", "utterance_idx", "to_idx"],
    )


def _build_label_matrix(df: pd.DataFrame, dim_names: List[str], dim_to_label2id: Dict[str, Dict[str, int]]) -> np.ndarray:
    y = np.zeros((len(df), len(dim_names)), dtype=np.int64)
    for j, dim in enumerate(dim_names):
        label2id = dim_to_label2id[dim]
        none_id = label2id["none"] if "none" in label2id else max(label2id.values())
        if dim not in df.columns:
            y[:, j] = int(none_id)
            continue
        for i, raw in enumerate(df[dim].tolist()):
            lab = norm_str(raw).lower()
            if not lab:
                y[i, j] = int(none_id)
            elif lab in label2id:
                y[i, j] = int(label2id[lab])
            else:
                y[i, j] = int(none_id)
    return y


def _accuracy_per_dim(logits_list: List[torch.Tensor], labels: torch.Tensor) -> List[float]:
    accs = []
    for j, logits in enumerate(logits_list):
        preds = torch.argmax(logits, dim=-1)
        acc = float((preds == labels[:, j]).float().mean().item())
        accs.append(acc)
    return accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--output_dir", type=str, default="outputs/conv_state_activations")

    ap.add_argument("--activation_path", type=str, default="")
    ap.add_argument("--activation_layer", type=int, default=-1)
    ap.add_argument("--activation_key", type=str, default="activations")

    ap.add_argument("--split_mode", type=str, default="user", choices=["user", "stratify"])
    ap.add_argument(
        "--user_split_file",
        type=str,
        default=str(Path(__file__).resolve().parent / "splits_test/user_split.json"),
        help="Used only when split_mode=user",
    )
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=48)

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    set_seed(int(args.seed))

    root = DataRoot(Path(args.root).resolve())
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_conv_state_annotations(root)
    df = _parse_idx_cols(df)
    df = df.dropna(subset=["user_idx", "session_idx", "activation_turn_idx"]).reset_index(drop=True)
    df["activation_turn_idx"] = df["activation_turn_idx"].astype(int)

    spec = load_conversation_state_spec(root)
    dim_names, dim_to_label2id = build_conv_state_label_maps(spec)
    y = _build_label_matrix(df, dim_names, dim_to_label2id)

    index_col = _resolve_index_col(df)
    X, keep = build_activation_features(
        root=root,
        df=df,
        activation_path=str(args.activation_path),
        activation_layer=int(args.activation_layer),
        activation_key=str(args.activation_key),
        index_col=index_col,
        concat_extra=None,
    )
    if X.size == 0:
        raise RuntimeError("No activation features were loaded. Check --activation_path / --activation_key and your CSV indices.")

    df_keep = df.loc[keep].reset_index(drop=True)
    y = y[keep]

    if args.split_mode == "user":
        user_split = load_json(_resolve_user_split_file(args.user_split_file, root))
        train_df, val_df = split_df_user(df_keep, user_split)
        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()
    else:
        train_df, val_df, _ = split_df_stratify(df_keep, label_col=dim_names[0], val_frac=float(args.val_frac), seed=int(args.seed))
        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    ds_train = ConvStateActivationDataset(X_train, y_train)
    ds_val = ConvStateActivationDataset(X_val, y_val)

    generator = make_torch_generator(int(args.seed))
    dl_train = DataLoader(ds_train, batch_size=int(args.batch_size), shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=generator)
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False, drop_last=False, worker_init_fn=seed_worker)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim_num_labels = [len(dim_to_label2id[d]) for d in dim_names]
    model = MultiHeadMLP(X.shape[1], dim_num_labels, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_acc = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(int(args.epochs)):
        model.train()
        train_losses = []
        train_accs = []
        for batch in dl_train:
            x = batch["input_embeds"].to(device)
            yb = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            logits_list = model(x)
            loss = 0.0
            for j, logits in enumerate(logits_list):
                loss = loss + loss_fn(logits, yb[:, j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
            train_accs.append(float(np.mean(_accuracy_per_dim(logits_list, yb))))

        model.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            for batch in dl_val:
                x = batch["input_embeds"].to(device)
                yb = batch["labels"].to(device)
                logits_list = model(x)
                loss = 0.0
                for j, logits in enumerate(logits_list):
                    loss = loss + loss_fn(logits, yb[:, j])
                val_losses.append(float(loss.detach().cpu()))
                val_accs.append(float(np.mean(_accuracy_per_dim(logits_list, yb))))

        tr_loss = float(np.mean(train_losses)) if train_losses else 0.0
        va_loss = float(np.mean(val_losses)) if val_losses else 0.0
        tr_acc = float(np.mean(train_accs)) if train_accs else 0.0
        va_acc = float(np.mean(val_accs)) if val_accs else 0.0
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "train_mean_acc": tr_acc,
                "val_mean_acc": va_acc,
            }
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("acc/train_mean", tr_acc, epoch)
        writer.add_scalar("acc/val_mean", va_acc, epoch)

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), out_dir / "actor.pt")

        print(f"[epoch {epoch}] train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

    writer.flush()
    writer.close()

    # Save metadata + metrics
    id2label = {d: {int(v): k for k, v in dim_to_label2id[d].items()} for d in dim_names}
    save_json({"dim_names": dim_names, "dim_to_label2id": dim_to_label2id, "dim_id2label": id2label}, out_dir / "label_map.json")
    save_json({"best_val_mean_acc": float(best_acc), "num_train": len(ds_train), "num_val": len(ds_val)}, out_dir / "metrics.json")
    save_json(history, out_dir / "train_history.json")

    print(f"Saved conv_state activation SFT to: {out_dir}")


if __name__ == "__main__":
    main()
