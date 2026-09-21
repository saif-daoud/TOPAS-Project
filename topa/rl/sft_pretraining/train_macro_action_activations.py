"""Policy-over-options SFT training on latent vectors (LLM activations).

This trains a simple MLP policy-over-options head on precomputed activation
vectors.

Expected files under --root:
  - components/action_space.json (or components/micro_actions.json)
  - annotations/annotations_macro_actions.csv

Activations are expected at:
  <activation_root>/user_<user_idx>/session_<session_idx>.pt

The .pt should contain a dict with key `--activation_key` (default: activations)
that is either:
  - a tensor [layers, seq_len, hidden], or
  - a list/tuple of length L with tensors [seq_len, hidden]
"""



import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loading import (
    DataRoot,
    load_action_space,
    load_macro_action_annotations,
    load_conv_state_annotations,
    build_macro_action_map,
)
from .utils import add_preferred_turn_index, load_json, save_json, split_df_user, split_df_stratify, safe_name, role_labels
from .activation_sft_utils import build_activation_features, build_conv_state_vectors, train_mlp_classifier
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]), help="Root containing data/, annotations/, components/")
    ap.add_argument("--output_dir", type=str, default="outputs/macro_action_activations")

    ap.add_argument("--activation_path", type=str, default="")
    ap.add_argument("--activation_layer", type=int, default=-1)
    ap.add_argument("--activation_key", type=str, default="activations")

    ap.add_argument("--use_conv_state", action="store_true", help="Fuse conversation-state features with the activation vector.")
    ap.add_argument("--conv_state_encoding", type=str, default="one_hot", choices=["one_hot", "label"])
    ap.add_argument("--conv_state_source", type=str, default="gold", choices=["gold", "sft", "probe"], help="gold: annotation labels; sft: train_conv_state_activations.py model; probe: frozen probability probe.")
    ap.add_argument("--conv_state_sft_dir", type=str, default="")
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
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    conv_source = str(args.conv_state_source).strip().lower()

    # label mapping
    action_space = load_action_space(root)
    macro2id = build_macro_action_map(action_space)
    id2macro = {int(v): str(k) for k, v in macro2id.items()}

    df = load_macro_action_annotations(root)
    df = df.copy()
    df = add_preferred_turn_index(df, "activation_turn_idx", ["from_id"], ["from_idx"])
    df["macro_id"] = df["selected_macro_action"].astype(str).map(macro2id)
    df = df.dropna(subset=["macro_id", "activation_turn_idx", "user_idx", "session_idx"]).reset_index(drop=True)
    df["macro_id"] = df["macro_id"].astype(int)
    df["activation_turn_idx"] = df["activation_turn_idx"].astype(int)

    extra = None
    extra_dim_names = []
    conv_state_dim = 0
    if bool(args.use_conv_state) and conv_source == "gold":
        # Prefer merging the conversation-state annotation CSV if present.
        # We merge on the strongest available key intersection.
        try:
            df_cs = load_conv_state_annotations(root)
            candidate_keys = [
                "user_idx",
                "session_idx",
                "from_id",
                "to_id",
                "from_idx",
                "to_idx",
                "selected_macro_action",
            ]
            merge_keys = [k for k in candidate_keys if k in df.columns and k in df_cs.columns]
            if merge_keys:
                df = df.merge(df_cs, on=merge_keys, how="left", suffixes=("", "_cs"))
        except FileNotFoundError:
            df_cs = None

        try:
            extra, extra_dim_names = build_conv_state_vectors(root, df, encoding=str(args.conv_state_encoding))
            conv_state_dim = int(extra.shape[1])
        except FileNotFoundError:
            print("[warn] conversation_states.json not found; proceeding without conv-state features")
            extra = None

    X, keep = build_activation_features(
        root=root,
        df=df,
        activation_path=str(args.activation_path),
        activation_layer=int(args.activation_layer),
        activation_key=str(args.activation_key),
        index_col="activation_turn_idx",
        concat_extra=extra,
    )
    if X.size == 0:
        raise RuntimeError("No activation features were loaded. Check --activation_path / --activation_key and your CSV indices.")

    if bool(args.use_conv_state) and conv_source == "probe":
        X, conv_state_dim = append_frozen_probe_features(
            X,
            checkpoint_path=str(args.probe_checkpoint),
            batch_size=int(args.probe_batch_size),
            include_probe_hidden=False,
        )
        extra_dim_names = [f"probe_prob_{i}" for i in range(int(conv_state_dim))]
    elif bool(args.use_conv_state) and conv_source == "sft":
        X, conv_state_dim, extra_dim_names = append_sft_conv_state_features(
            X,
            model_dir=str(args.conv_state_sft_dir),
            encoding=str(args.conv_state_encoding),
            batch_size=int(args.probe_batch_size),
        )

    df_keep = df.loc[keep].reset_index(drop=True)
    y = df_keep["macro_id"].to_numpy(dtype=np.int64)

    if split_mode == "user":
        user_split = load_json(_resolve_user_split_file(args.user_split_file, root))
        train_df, val_df = split_df_user(df_keep, user_split)
        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()
    else:
        train_df, val_df, _ = split_df_stratify(df_keep, label_col="macro_id", val_frac=float(args.val_frac), seed=int(args.seed))
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
        num_labels=int(max(macro2id.values()) + 1),
        hidden_dim=int(args.hidden_dim),
        projection_dim=int(args.projection_dim),
        conv_state_embedding_dim=int(args.conv_state_embedding_dim),
        dropout=float(args.dropout),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        seed=int(args.seed),
        tb_dir=out_dir / "tb",
        conv_state_dim=int(conv_state_dim),
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
            "macro2id": macro2id,
            "id2macro": id2macro,
            "use_conv_state": bool(args.use_conv_state),
            "conv_state_dims": extra_dim_names,
            "conv_state_dim": int(conv_state_dim),
            "conv_state_encoding": str(args.conv_state_encoding),
            "conv_state_source": str(conv_source),
            "conv_state_sft_dir": str(args.conv_state_sft_dir),
            "probe_checkpoint": str(args.probe_checkpoint),
            "probe_include_hidden": 0,
            "projection_dim": int(effective_projection_dim),
            "conv_state_embedding_dim": int(effective_conv_embedding_dim),
            "fusion": fusion,
        },
        out_dir / "label_map.json",
    )
    save_json(
        {
            "best_val_acc": res.best_val_acc,
            "best_val_acc_at_2": res.best_val_acc_at_2,
            "best_val_acc_at_3": res.best_val_acc_at_3,
            "best_val_macro_f1": res.best_val_macro_f1,
            "selection_metric": res.selection_metric,
            "best_selection_score": res.best_selection_score,
            "num_train": res.num_train,
            "num_val": res.num_val,
        },
        out_dir / "metrics.json",
    )

    print(f"Saved macro activation SFT to: {out_dir}")


if __name__ == "__main__":
    main()
