"""Intra-option and flat-RL SFT training on latent vectors (LLM activations).

Trains simple MLP heads on activation vectors for either per-option
intra-option policies or one flat policy over all option/action pairs.

Two common workflows:
  1) Per-option intra-option policies:
     python -m topa.rl.sft_pretraining.train_micro_action_activations \
        --root ... --activation_path ... --macro_action "..."

  2) Train all per-option policies in one go:
     python -m topa.rl.sft_pretraining.train_micro_action_activations \
        --root ... --activation_path ... --train_all_macros
"""



import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loading import (
    DataRoot,
    build_flat_micro_label_map,
    load_action_space,
    load_micro_action_annotations,
    build_micro_action_maps,
)
from .utils import add_preferred_turn_index, load_json, save_json, split_df_user, split_df_stratify, safe_name, role_labels
from .activation_sft_utils import build_activation_features, train_mlp_classifier
from ..cv_state_probe import append_frozen_probe_features, append_sft_conv_state_features




def _append_conv_state_if_requested(X: np.ndarray, *, use_conv_state: bool, conv_state_source: str, conv_state_sft_dir: str, conv_state_encoding: str, probe_checkpoint: str, probe_batch_size: int) -> tuple[np.ndarray, int]:
    if not bool(use_conv_state):
        return X, 0
    source = str(conv_state_source).strip().lower()
    if source == "probe":
        return append_frozen_probe_features(
            X,
            checkpoint_path=str(probe_checkpoint),
            batch_size=int(probe_batch_size),
            include_probe_hidden=False,
        )
    if source == "sft":
        X_new, conv_dim, _ = append_sft_conv_state_features(
            X,
            model_dir=str(conv_state_sft_dir),
            encoding=str(conv_state_encoding),
            batch_size=int(probe_batch_size),
        )
        return X_new, conv_dim
    raise ValueError(f"Unsupported conv_state_source={conv_state_source!r}; expected 'sft' or 'probe'.")


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


def _train_one(
    *,
    root: DataRoot,
    df_all: pd.DataFrame,
    action_space: dict,
    macro_action: str,
    activation_path: str,
    activation_layer: int,
    activation_key: str,
    out_dir: Path,
    split_mode: str,
    user_split_file: Path,
    val_frac: float,
    seed: int,
    hidden_dim: int,
    projection_dim: int,
    conv_state_embedding_dim: int,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    use_conv_state: bool = False,
    conv_state_source: str = "sft",
    conv_state_sft_dir: str = "",
    conv_state_encoding: str = "one_hot",
    probe_checkpoint: str = "",
    probe_batch_size: int = 1024,
) -> None:
    macro2id, macro_id_to_micro_label2id = build_micro_action_maps(action_space)
    macro_id = macro2id[str(macro_action)]
    micro2id = macro_id_to_micro_label2id[macro_id]

    df = df_all[df_all["selected_macro_action"].astype(str) == str(macro_action)].copy()
    df = add_preferred_turn_index(df, "activation_turn_idx", ["utterance_id", "turn_id", "from_id", "to_id"], ["utterance_idx"])
    df["micro_id"] = df["selected_micro_action"].astype(str).map(micro2id)
    df = df.dropna(subset=["micro_id", "activation_turn_idx", "user_idx", "session_idx"]).reset_index(drop=True)
    df["micro_id"] = df["micro_id"].astype(int)
    df["activation_turn_idx"] = df["activation_turn_idx"].astype(int)
    if len(df) == 0:
        print(f"[WARN] No rows for macro_action={macro_action!r} after filtering; skipping.")
        return

    X, keep = build_activation_features(
        root=root,
        df=df,
        activation_path=str(activation_path),
        activation_layer=int(activation_layer),
        activation_key=str(activation_key),
        index_col="activation_turn_idx",
        concat_extra=None,
    )
    if X.size == 0:
        raise RuntimeError(
            f"No activation features were loaded for macro_action={macro_action!r}. "
            "Check --activation_path / --activation_key and your CSV indices."
        )

    X, conv_state_dim = _append_conv_state_if_requested(
        X,
        use_conv_state=bool(use_conv_state),
        conv_state_source=str(conv_state_source),
        conv_state_sft_dir=str(conv_state_sft_dir),
        conv_state_encoding=str(conv_state_encoding),
        probe_checkpoint=str(probe_checkpoint),
        probe_batch_size=int(probe_batch_size),
    )

    df_keep = df.loc[keep].reset_index(drop=True)
    if len(df_keep) == 0:
        print(f"[WARN] No activation features kept for macro_action={macro_action!r}; skipping.")
        return
    y = df_keep["micro_id"].to_numpy(dtype=np.int64)

    if split_mode == "user":
        user_split = load_json(_resolve_user_split_file(str(user_split_file), root))
        train_df, val_df = split_df_user(df_keep, user_split)
        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()
    else:
        train_df, val_df, _ = split_df_stratify(df_keep, label_col="micro_id", val_frac=float(val_frac), seed=int(seed))
        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        print(
            f"[WARN] Empty train/val split for macro_action={macro_action!r} "
            f"(train={X_train.shape[0]}, val={X_val.shape[0]}); skipping."
        )
        return

    res = train_mlp_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        out_dir=out_dir,
        num_labels=int(max(micro2id.values()) + 1),
        hidden_dim=int(hidden_dim),
        projection_dim=int(projection_dim),
        conv_state_embedding_dim=int(conv_state_embedding_dim),
        dropout=float(dropout),
        lr=float(lr),
        batch_size=int(batch_size),
        epochs=int(epochs),
        seed=int(seed),
        tb_dir=out_dir / "tb",
        conv_state_dim=int(conv_state_dim),
    )

    id2micro = {int(v): str(k) for k, v in micro2id.items()}
    save_json({
        "macro_action": str(macro_action),
        "micro2id": micro2id,
        "id2micro": id2micro,
        "use_conv_state": bool(use_conv_state),
        "conv_state_source": str(conv_state_source),
        "conv_state_sft_dir": str(conv_state_sft_dir),
        "probe_checkpoint": str(probe_checkpoint),
        "probe_include_hidden": 0,
        "conv_state_dim": int(conv_state_dim),
        "conv_state_encoding": str(conv_state_encoding),
        "projection_dim": int(projection_dim) if int(conv_state_dim) > 0 else 0,
        "conv_state_embedding_dim": int(conv_state_embedding_dim) if int(conv_state_dim) > 0 else 0,
    }, out_dir / "label_map.json")
    save_json(
        {
            "best_val_acc": res.best_val_acc,
            "best_val_acc_at_2": res.best_val_acc_at_2,
            "best_val_acc_at_3": res.best_val_acc_at_3,
            "num_train": res.num_train,
            "num_val": res.num_val,
        },
        out_dir / "metrics.json",
    )


def _train_flat(
    *,
    root: DataRoot,
    df_all: pd.DataFrame,
    action_space: dict,
    activation_path: str,
    activation_layer: int,
    activation_key: str,
    out_dir: Path,
    split_mode: str,
    user_split_file: Path,
    val_frac: float,
    seed: int,
    hidden_dim: int,
    projection_dim: int,
    conv_state_embedding_dim: int,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    use_conv_state: bool = False,
    conv_state_source: str = "sft",
    conv_state_sft_dir: str = "",
    conv_state_encoding: str = "one_hot",
    probe_checkpoint: str = "",
    probe_batch_size: int = 1024,
) -> None:
    class_names, label2id = build_flat_micro_label_map(action_space)
    df = df_all.copy()
    df = add_preferred_turn_index(df, "activation_turn_idx", ["utterance_id", "turn_id", "from_id", "to_id"], ["utterance_idx"])
    df["flat_label"] = df["selected_macro_action"].astype(str) + "::" + df["selected_micro_action"].astype(str)
    df["flat_id"] = df["flat_label"].astype(str).map(label2id)
    df = df.dropna(subset=["flat_id", "activation_turn_idx", "user_idx", "session_idx"]).reset_index(drop=True)
    df["flat_id"] = df["flat_id"].astype(int)
    df["activation_turn_idx"] = df["activation_turn_idx"].astype(int)

    X, keep = build_activation_features(
        root=root,
        df=df,
        activation_path=str(activation_path),
        activation_layer=int(activation_layer),
        activation_key=str(activation_key),
        index_col="activation_turn_idx",
        concat_extra=None,
    )
    if X.size == 0:
        raise RuntimeError("No activation features were loaded for flat RL SFT.")

    X, conv_state_dim = _append_conv_state_if_requested(
        X,
        use_conv_state=bool(use_conv_state),
        conv_state_source=str(conv_state_source),
        conv_state_sft_dir=str(conv_state_sft_dir),
        conv_state_encoding=str(conv_state_encoding),
        probe_checkpoint=str(probe_checkpoint),
        probe_batch_size=int(probe_batch_size),
    )

    df_keep = df.loc[keep].reset_index(drop=True)
    y = df_keep["flat_id"].to_numpy(dtype=np.int64)

    if split_mode == "user":
        user_split = load_json(_resolve_user_split_file(str(user_split_file), root))
        train_df, val_df = split_df_user(df_keep, user_split)
        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()
    else:
        train_df, val_df, _ = split_df_stratify(df_keep, label_col="flat_id", val_frac=float(val_frac), seed=int(seed))
        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    res = train_mlp_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        out_dir=out_dir,
        num_labels=int(len(class_names)),
        hidden_dim=int(hidden_dim),
        projection_dim=int(projection_dim),
        conv_state_embedding_dim=int(conv_state_embedding_dim),
        dropout=float(dropout),
        lr=float(lr),
        batch_size=int(batch_size),
        epochs=int(epochs),
        seed=int(seed),
        tb_dir=out_dir / "tb",
        conv_state_dim=int(conv_state_dim),
    )

    id2label = {int(v): str(k) for k, v in label2id.items()}
    save_json({
        "label2id": label2id,
        "id2label": id2label,
        "classes": class_names,
        "use_conv_state": bool(use_conv_state),
        "conv_state_source": str(conv_state_source),
        "conv_state_sft_dir": str(conv_state_sft_dir),
        "probe_checkpoint": str(probe_checkpoint),
        "probe_include_hidden": 0,
        "conv_state_dim": int(conv_state_dim),
        "conv_state_encoding": str(conv_state_encoding),
        "projection_dim": int(projection_dim) if int(conv_state_dim) > 0 else 0,
        "conv_state_embedding_dim": int(conv_state_embedding_dim) if int(conv_state_dim) > 0 else 0,
    }, out_dir / "label_map.json")
    save_json(
        {
            "best_val_acc": res.best_val_acc,
            "best_val_acc_at_2": res.best_val_acc_at_2,
            "best_val_acc_at_3": res.best_val_acc_at_3,
            "num_train": res.num_train,
            "num_val": res.num_val,
        },
        out_dir / "metrics.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--output_dir", type=str, default="outputs/micro_action_activations")

    ap.add_argument("--activation_path", type=str, default="")
    ap.add_argument("--activation_layer", type=int, default=-1)
    ap.add_argument("--activation_key", type=str, default="activations")

    ap.add_argument("--macro_action", type=str, default="", help="If set, train only for this macro action.")
    ap.add_argument("--train_all_macros", action="store_true", help="Train one intra-option policy per option.")
    ap.add_argument("--train_flat", action="store_true", help="Train one flat RL policy over all option/action pairs.")
    ap.add_argument("--use_conv_state", action="store_true")
    ap.add_argument("--conv_state_source", type=str, default="sft", choices=["sft", "probe"])
    ap.add_argument("--conv_state_sft_dir", type=str, default="")
    ap.add_argument("--conv_state_encoding", type=str, default="one_hot", choices=["one_hot", "label"])
    ap.add_argument("--probe_checkpoint", type=str, default="/home/local/QCRI/saif.sedaoud/clean-env/new_env/cv_state_preds/probe_results_conversation_states/checkpoints/probe_layer_22.pt")
    ap.add_argument("--probe_batch_size", type=int, default=1024)

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

    args = ap.parse_args()
    split_mode = str(args.split_mode or "").strip().lower()

    root = DataRoot(Path(args.root).resolve())
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    action_space = load_action_space(root)
    df_all = load_micro_action_annotations(root)

    macro2id, _ = build_micro_action_maps(action_space)
    macros = list(macro2id.keys())
    if bool(args.train_flat):
        _train_flat(
            root=root,
            df_all=df_all,
            action_space=action_space,
            activation_path=str(args.activation_path),
            activation_layer=int(args.activation_layer),
            activation_key=str(args.activation_key),
            out_dir=out_root,
            split_mode=split_mode,
            user_split_file=Path(args.user_split_file),
            val_frac=float(args.val_frac),
            seed=int(args.seed),
            hidden_dim=int(args.hidden_dim),
            projection_dim=int(args.projection_dim),
            conv_state_embedding_dim=int(args.conv_state_embedding_dim),
            dropout=float(args.dropout),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            use_conv_state=bool(args.use_conv_state),
            conv_state_source=str(args.conv_state_source),
            conv_state_sft_dir=str(args.conv_state_sft_dir),
            conv_state_encoding=str(args.conv_state_encoding),
            probe_checkpoint=str(args.probe_checkpoint),
            probe_batch_size=int(args.probe_batch_size),
        )
        print(f"Saved flat RL activation SFT to: {out_root}")
        return

    if bool(args.train_all_macros):
        targets = macros
    elif str(args.macro_action).strip():
        targets = [str(args.macro_action)]
    else:
        raise ValueError("Provide --macro_action, --train_all_macros, or --train_flat")

    for macro in targets:
        out_dir = out_root / f"intra_option_{safe_name(str(macro))}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Training intra-option policy for option={macro!r} -> {out_dir} ===")
        _train_one(
            root=root,
            df_all=df_all,
            action_space=action_space,
            macro_action=macro,
            activation_path=str(args.activation_path),
            activation_layer=int(args.activation_layer),
            activation_key=str(args.activation_key),
            out_dir=out_dir,
            split_mode=split_mode,
            user_split_file=Path(args.user_split_file),
            val_frac=float(args.val_frac),
            seed=int(args.seed),
            hidden_dim=int(args.hidden_dim),
            projection_dim=int(args.projection_dim),
            conv_state_embedding_dim=int(args.conv_state_embedding_dim),
            dropout=float(args.dropout),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            use_conv_state=bool(args.use_conv_state),
            conv_state_source=str(args.conv_state_source),
            conv_state_sft_dir=str(args.conv_state_sft_dir),
            conv_state_encoding=str(args.conv_state_encoding),
            probe_checkpoint=str(args.probe_checkpoint),
            probe_batch_size=int(args.probe_batch_size),
        )

    print(f"Saved intra-option activation SFT to: {out_root}")


if __name__ == "__main__":
    main()
