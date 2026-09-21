"""Termination-function SFT on latent vectors (LLM activations).

This mirrors the label construction used in `train_termination.py`, but trains
on activation vectors instead of tokenized text.

Labels are derived from the *micro-action* annotation stream:
  terminate=1 at a step where the next step's macro action changes (or the
  sequence ends), else terminate=0.

Saves `actor.pt` for the activation-space termination head.
"""



import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from .data_loading import DataRoot, build_macro_action_map, load_action_space, load_micro_action_annotations
from .utils import add_preferred_turn_index, load_json, make_torch_generator, save_json, seed_worker, set_seed, split_df_user, split_df_stratify, role_labels, termination_labels_from_options
from .activation_sft_utils import build_activation_features
from .train_conv_state_activations import MultiHeadMLP
from ..simple_models import MLPPolicy
from ..cv_state_probe import append_frozen_probe_features, append_sft_conv_state_features


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


def _add_termination_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute terminate label per micro step.

    Assumes micro CSV is already ordered by time within each (user, session).
    """
    df = df.copy()
    df = df.sort_values(["user_idx", "session_idx", "activation_turn_idx"]).reset_index(drop=True)

    df["terminate"] = 0
    for _, idx in df.groupby(["user_idx", "session_idx"], sort=False).groups.items():
        options = [str(x) for x in df.loc[idx, "selected_macro_action"].tolist()]
        df.loc[idx, "terminate"] = termination_labels_from_options(options)
    return df


def _predict_termination(model: MLPPolicy, X: np.ndarray, actions: np.ndarray, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X.shape[0], int(batch_size)):
            end = start + int(batch_size)
            x = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            a = torch.tensor(actions[start:end], dtype=torch.long, device=device)
            logits = model(x).gather(1, a.view(-1, 1)).squeeze(1)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    p = np.concatenate(probs, axis=0).astype(np.float32)
    return p, (p >= 0.5).astype(np.int64)


def _eval_termination(model: MLPPolicy, X: np.ndarray, actions: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> dict:
    probs, pred = _predict_termination(model, X, actions, batch_size, device)
    rep = classification_report(
        y,
        pred,
        labels=[0, 1],
        target_names=["continue", "terminate"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    rep["prob_mean"] = float(probs.mean()) if len(probs) else 0.0
    return rep


def _train_termination_mlp(
    X_train: np.ndarray,
    action_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    action_val: np.ndarray,
    y_val: np.ndarray,
    *,
    out_dir: Path,
    num_options: int,
    conv_state_dim: int,
    args,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(action_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(action_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.float32),
    )
    generator = make_torch_generator(int(args.seed))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False, worker_init_fn=seed_worker)

    model = MLPPolicy(
        in_dim=int(X_train.shape[1]),
        num_actions=int(num_options),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        conv_state_dim=int(conv_state_dim),
        projection_dim=int(args.projection_dim),
        conv_state_embedding_dim=int(args.conv_state_embedding_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    pos = float(np.sum(y_train))
    neg = float(len(y_train)) - pos
    pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=device)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    best_f1 = -1.0
    best = {}
    history = []
    for epoch in range(int(args.epochs)):
        model.train()
        train_losses = []
        for x, action, term in train_loader:
            x = x.to(device)
            action = action.to(device)
            term = term.to(device)
            logits = model(x).gather(1, action.view(-1, 1)).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, term, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, action, term in val_loader:
                x = x.to(device)
                action = action.to(device)
                term = term.to(device)
                logits = model(x).gather(1, action.view(-1, 1)).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, term, pos_weight=pos_weight)
                val_losses.append(float(loss.detach().cpu()))
        report = _eval_termination(model, X_val, action_val, y_val, int(args.batch_size), device)
        macro_f1 = float(report["macro avg"]["f1-score"])
        term_f1 = float(report["terminate"]["f1-score"])
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "val_macro_f1": macro_f1,
            "val_terminate_f1": term_f1,
        }
        history.append(row)
        writer.add_scalar("loss/train", row["train_loss"], epoch)
        writer.add_scalar("loss/val", row["val_loss"], epoch)
        writer.add_scalar("macro_f1/val", macro_f1, epoch)
        writer.add_scalar("terminate_f1/val", term_f1, epoch)
        writer.add_scalar("termination/pos_weight", float(pos_weight.detach().cpu()), epoch)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best = dict(row)
            torch.save(model.state_dict(), out_dir / "actor.pt")
        print(f"[epoch {epoch}] val_macro_f1={macro_f1:.4f} val_terminate_f1={term_f1:.4f}")

    writer.flush()
    writer.close()
    save_json(history, out_dir / "train_history.json")
    return {
        "best": best,
        "num_train": int(len(train_ds)),
        "num_val": int(len(val_ds)),
        "pos_weight": float(pos_weight.detach().cpu()),
    }


def _save_validation_report(out_dir: Path, X_val: np.ndarray, action_val: np.ndarray, y_val: np.ndarray, args, conv_state_dim: int, num_options: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPPolicy(
        in_dim=int(X_val.shape[1]),
        num_actions=int(num_options),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        conv_state_dim=int(conv_state_dim),
        projection_dim=int(args.projection_dim),
        conv_state_embedding_dim=int(args.conv_state_embedding_dim),
    ).to(device)
    model.load_state_dict(torch.load(out_dir / "actor.pt", map_location=device))
    _, y_pred = _predict_termination(model, X_val, action_val, int(args.batch_size), device)
    report_txt = classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["continue", "terminate"],
        digits=4,
        zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    report_json = classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["continue", "terminate"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    save_json(report_json, out_dir / "classification_report.json")

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=150)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Validation)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["continue", "terminate"], rotation=30, ha="right")
    ax.set_yticklabels(["continue", "terminate"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)


def _predicted_conv_state_vectors(X: np.ndarray, conv_state_model_dir: Path, args) -> np.ndarray:
    label_map = load_json(conv_state_model_dir / "label_map.json")
    dim_names = list(label_map["dim_names"])
    dim_to_label2id = label_map["dim_to_label2id"]
    dim_num_labels = [len(dim_to_label2id[d]) for d in dim_names]
    encoding = str(args.conv_state_encoding).strip().lower()
    offsets = {}
    off = 0
    for dim in dim_names:
        offsets[dim] = off
        off += len(dim_to_label2id[dim])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiHeadMLP(
        input_dim=int(X.shape[1]),
        dim_num_labels=dim_num_labels,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    model.load_state_dict(torch.load(conv_state_model_dir / "actor.pt", map_location=device))
    model.eval()

    out_dim = off if encoding == "one_hot" else len(dim_names)
    out = np.zeros((X.shape[0], out_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, X.shape[0], int(args.batch_size)):
            end = start + int(args.batch_size)
            logits_list = model(torch.tensor(X[start:end], dtype=torch.float32, device=device))
            for j, dim in enumerate(dim_names):
                pred = torch.argmax(logits_list[j], dim=-1).detach().cpu().numpy()
                rows = np.arange(start, min(end, X.shape[0]))
                if encoding == "one_hot":
                    out[rows, int(offsets[dim]) + pred] = 1.0
                else:
                    out[rows, j] = pred.astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--output_dir", type=str, default="outputs/termination_activations")

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
    ap.add_argument("--projection_dim", type=int, default=0)
    ap.add_argument("--conv_state_embedding_dim", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--use_conv_state", action="store_true")
    ap.add_argument("--conv_state_model_dir", type=str, default="")
    ap.add_argument("--conv_state_encoding", type=str, default="one_hot", choices=["one_hot", "label"])
    ap.add_argument("--conv_state_source", type=str, default="sft", choices=["sft", "probe"], help="sft: existing activation-SFT conv-state model; probe: frozen probability probe.")
    ap.add_argument("--conv_state_sft_dir", type=str, default="", help="Alias for --conv_state_model_dir when conv_state_source=sft.")
    ap.add_argument("--probe_checkpoint", type=str, default="/home/local/QCRI/saif.sedaoud/clean-env/new_env/cv_state_preds/probe_results_conversation_states/checkpoints/probe_layer_22.pt")
    ap.add_argument("--probe_batch_size", type=int, default=1024)

    args = ap.parse_args()
    set_seed(int(args.seed))
    split_mode = str(args.split_mode or "").strip().lower()

    root = DataRoot(Path(args.root).resolve())
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    action_space = load_action_space(root)
    macro2id = build_macro_action_map(action_space)
    id2macro = {int(v): str(k) for k, v in macro2id.items()}

    df = load_micro_action_annotations(root)
    df = add_preferred_turn_index(df, "activation_turn_idx", ["utterance_id", "turn_id", "from_id", "to_id"], ["utterance_idx"])
    df = df.dropna(subset=["user_idx", "session_idx", "activation_turn_idx", "selected_macro_action"]).reset_index(drop=True)
    df["activation_turn_idx"] = df["activation_turn_idx"].astype(int)
    df = _add_termination_labels(df)

    X, keep = build_activation_features(
        root=root,
        df=df,
        activation_path=str(args.activation_path),
        activation_layer=int(args.activation_layer),
        activation_key=str(args.activation_key),
        index_col="activation_turn_idx",
        concat_extra=None,
    )
    if X.size == 0:
        raise RuntimeError("No activation features were loaded. Check --activation_path / --activation_key and your CSV indices.")

    df_keep = df.loc[keep].reset_index(drop=True)
    y = df_keep["terminate"].to_numpy(dtype=np.int64)
    option_ids = df_keep["selected_macro_action"].astype(str).map(macro2id).to_numpy(dtype=np.int64)

    if split_mode == "user":
        user_split = load_json(_resolve_user_split_file(args.user_split_file, root))
        train_df, val_df = split_df_user(df_keep, user_split)
        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()
    else:
        train_df, val_df, _ = split_df_stratify(df_keep, label_col="terminate", val_frac=float(args.val_frac), seed=int(args.seed))
        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()

    X_train, y_train, action_train = X[train_idx], y[train_idx], option_ids[train_idx]
    X_val, y_val, action_val = X[val_idx], y[val_idx], option_ids[val_idx]

    conv_state_dim = 0
    conv_source = str(args.conv_state_source).strip().lower()
    if bool(args.use_conv_state):
        if conv_source == "probe":
            X_train, conv_state_dim = append_frozen_probe_features(
                X_train,
                checkpoint_path=str(args.probe_checkpoint),
                batch_size=int(args.probe_batch_size),
                include_probe_hidden=False,
            )
            X_val, val_conv_state_dim = append_frozen_probe_features(
                X_val,
                checkpoint_path=str(args.probe_checkpoint),
                batch_size=int(args.probe_batch_size),
                include_probe_hidden=False,
            )
            assert int(val_conv_state_dim) == int(conv_state_dim)
        else:
            conv_dir = str(args.conv_state_sft_dir).strip() or str(args.conv_state_model_dir).strip()
            X_train, conv_state_dim, _ = append_sft_conv_state_features(
                X_train,
                model_dir=conv_dir,
                encoding=str(args.conv_state_encoding),
                batch_size=int(args.probe_batch_size),
            )
            X_val, val_conv_state_dim, _ = append_sft_conv_state_features(
                X_val,
                model_dir=conv_dir,
                encoding=str(args.conv_state_encoding),
                batch_size=int(args.probe_batch_size),
            )
            assert int(val_conv_state_dim) == int(conv_state_dim)

    res = _train_termination_mlp(
        X_train,
        action_train,
        y_train,
        X_val,
        action_val,
        y_val,
        out_dir=out_dir,
        num_options=int(len(macro2id)),
        conv_state_dim=int(conv_state_dim),
        args=args,
    )

    effective_projection_dim = (int(args.projection_dim) if int(args.projection_dim) > 0 else (int(args.hidden_dim) if int(args.hidden_dim) > 0 else 0)) if int(conv_state_dim) > 0 else 0
    effective_conv_embedding_dim = int(args.conv_state_embedding_dim) if int(conv_state_dim) > 0 and int(args.conv_state_embedding_dim) > 0 else 0
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
    save_json(
        {
            "id2label": {0: "continue", 1: "terminate"},
            "macro2id": macro2id,
            "id2macro": id2macro,
            "num_options": int(len(macro2id)),
            "output_type": "option_conditioned_termination",
            "use_conv_state": bool(args.use_conv_state),
            "conv_state_dim": int(conv_state_dim),
            "conv_state_encoding": str(args.conv_state_encoding),
            "projection_dim": int(effective_projection_dim),
            "conv_state_embedding_dim": int(effective_conv_embedding_dim),
            "fusion": fusion,
        },
        out_dir / "label_map.json",
    )
    save_json(
        {
            "best": res["best"],
            "selection_metric": "macro_f1",
            "best_selection_score": float(res["best"]["val_macro_f1"]),
            "num_train": int(res["num_train"]),
            "num_val": int(res["num_val"]),
            "pos_weight": float(res["pos_weight"]),
            "use_conv_state": bool(args.use_conv_state),
            "conv_state_source": str(conv_source),
            "conv_state_sft_dir": str(args.conv_state_sft_dir or args.conv_state_model_dir),
            "probe_checkpoint": str(args.probe_checkpoint),
            "probe_include_hidden": 0,
            "conv_state_dim": int(conv_state_dim),
            "conv_state_encoding": str(args.conv_state_encoding),
            "projection_dim": int(effective_projection_dim),
            "conv_state_embedding_dim": int(effective_conv_embedding_dim),
            "num_options": int(len(macro2id)),
        },
        out_dir / "metrics.json",
    )
    _save_validation_report(out_dir, X_val, action_val, y_val, args, conv_state_dim=int(conv_state_dim), num_options=int(len(macro2id)))

    print(f"Saved termination activation SFT to: {out_dir}")


if __name__ == "__main__":
    main()
