"""Train a termination function (SFT) for macro actions.

Goal: given dialogue context (and optionally current macro), predict whether the current macro
should terminate at this step.

Labels:
- terminate = 1 if macro action changes at the next micro-step within the same (user, session),
  OR if this is the last micro-step of the session.
- else 0

This is intentionally trained with SFT (supervised) per your request.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments

from .data_loading import DataRoot, load_dataset_json, load_micro_action_annotations, build_micro_action_text, get_default_prefix
from .utils import set_seed, load_json, save_json, remove_logging, split_df_user, split_df_stratify, resolve_mixed_precision, role_labels
from .feature_cache import CacheKey, cache_or_build, fingerprint_dict
from .cached_datasets import CachedFeatures, CachedTensorDataset, stack_collate
from .conv_state_prediction import predict_conv_state_vectors_encoder
from .models import RobertaForMacroActionFusion

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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

def build_termination_dataframe(df_micro: pd.DataFrame) -> pd.DataFrame:
    df = df_micro.copy()
    df["utterance_idx"] = df["utterance_idx"].astype(int)
    df = df.sort_values(["user_idx", "session_idx", "utterance_idx"]).reset_index(drop=True)

    # termination if macro changes at next step or session ends
    nxt_macro = df.groupby(["user_idx","session_idx"])["selected_macro_action"].shift(-1)
    df["terminate"] = (nxt_macro.notna() & (nxt_macro != df["selected_macro_action"])).astype(int)

    # last step in session -> terminate
    is_last = df.groupby(["user_idx","session_idx"]).cumcount(ascending=False) == 0
    df.loc[is_last, "terminate"] = 1
    return df

def build_cached_features(
    cache_dir: Path,
    root: DataRoot,
    df_term: pd.DataFrame,
    tokenizer,
    max_length: int,
    context_turns: int,
    include_macro_in_text: bool,
    conv_state_vec: np.ndarray | None = None,
    conv_state_source: str = "",
    force_rebuild: bool = False,
) -> CachedFeatures:
    data_json = load_dataset_json(root)

    meta = {
        "kind": "termination",
        "n": len(df_term),
        "max_length": max_length,
        "context_turns": context_turns,
        "include_macro_in_text": int(include_macro_in_text),
        "conv_state_source": str(conv_state_source),
        "tokenizer": getattr(tokenizer, "name_or_path", "tok"),
    }
    key = CacheKey("termination", fingerprint_dict(meta))

    def _build() -> CachedFeatures:
        texts = []
        labels = []
        for _, row in df_term.iterrows():
            rowd = row.to_dict()
            txt = build_micro_action_text(data_json, rowd, prefix=get_default_prefix(), context_turns=context_turns)
            if include_macro_in_text:
                txt = txt + f"\n\nCURRENT_MACRO_ACTION: {rowd['selected_macro_action']}"
            texts.append(txt)
            labels.append(int(rowd["terminate"]))
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        extra = {}
        if conv_state_vec is not None:
            extra["conv_state_vec"] = np.asarray(conv_state_vec, dtype=np.float32)
        return CachedFeatures(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=np.asarray(labels, dtype=np.int64),
            extra=extra,
        )

    return cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--model_name", type=str, default="roberta-large")
    ap.add_argument("--output_dir", type=str, default="runs/termination_sft")
    ap.add_argument("--cache_dir", type=str, default="cache/features")
    ap.add_argument("--force_rebuild_cache", action="store_true")

    ap.add_argument("--context_turns", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--include_macro_in_text", action="store_true")
    ap.add_argument("--use_conv_state", action="store_true")
    ap.add_argument("--conv_state_adapter_dir", type=str, default="")
    ap.add_argument("--conv_state_meta_path", type=str, default="")
    ap.add_argument("--conv_state_max_turns", type=int, default=None)
    ap.add_argument("--conv_state_encoding", type=str, default="one_hot", choices=["one_hot", "label"])

    ap.add_argument(
        "--split_mode",
        type=str,
        default="user",
        choices=["user", "stratify"],
        help="user: use user_split.json (user-level split). stratify: row-level stratified split.",
    )
    ap.add_argument(
        "--user_split_file",
        type=str,
        default=str(Path(__file__).resolve().parent / "splits_test/user_split.json"),
        help="Used only when split_mode=user",
    )
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=48)
    ap.add_argument("--amp", type=str, default=os.environ.get("AMP", "auto"), choices=["none", "fp16", "bf16", "auto"], help="Mixed precision: auto chooses bf16 if supported, else fp16.")

    # Logging
    ap.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "none"],
        help="HF Trainer reporting backend. Use 'tensorboard' for TensorBoard logging.",
    )
    ap.add_argument("--tb_dir", type=str, default="", help="TensorBoard log directory. Default: <output_dir>/tb")

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    args = ap.parse_args()
    split_mode = str(args.split_mode or "").strip().lower()
    set_seed(args.seed)

    root = DataRoot(Path(args.root).resolve())
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_micro = load_micro_action_annotations(root)
    df_term = build_termination_dataframe(df_micro)

    # split row-level (termination is local)
    if split_mode == "user":
        split = load_json(_resolve_user_split_file(args.user_split_file, root))
        train_df, val_df = split_df_user(df_term, split)
    else:
        train_df, val_df, _ = split_df_stratify(df_term, label_col="terminate", val_frac=args.val_frac, seed=args.seed)

    is_main = (os.environ.get("RANK", "0") == "0")
    if is_main:
        print(f"Total rows: {len(df_term)}")
        print(f"Train rows: {len(train_df)}")
        print(f"Validation rows: {len(val_df)}")

        print("Train dist:")
        print(train_df["terminate"].value_counts())
        print("Validation dist:")
        print(val_df["terminate"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    train_conv_vec = None
    val_conv_vec = None
    conv_state_dim = 0
    if bool(args.use_conv_state):
        train_conv_vec, _ = predict_conv_state_vectors_encoder(
            root=root,
            df=train_df,
            model_name=str(args.model_name),
            adapter_dir=str(args.conv_state_adapter_dir),
            meta_path=str(args.conv_state_meta_path),
            max_length=int(args.max_length),
            max_turns=args.conv_state_max_turns,
            batch_size=int(args.batch_size),
            encoding=str(args.conv_state_encoding),
        )
        val_conv_vec, _ = predict_conv_state_vectors_encoder(
            root=root,
            df=val_df,
            model_name=str(args.model_name),
            adapter_dir=str(args.conv_state_adapter_dir),
            meta_path=str(args.conv_state_meta_path),
            max_length=int(args.max_length),
            max_turns=args.conv_state_max_turns,
            batch_size=int(args.batch_size),
            encoding=str(args.conv_state_encoding),
        )
        conv_state_dim = int(train_conv_vec.shape[1])

    train_feat = build_cached_features(Path(args.cache_dir), root, train_df, tokenizer, args.max_length, args.context_turns,
                                       include_macro_in_text=bool(args.include_macro_in_text),
                                       conv_state_vec=train_conv_vec,
                                       conv_state_source=str(args.conv_state_adapter_dir),
                                       force_rebuild=bool(args.force_rebuild_cache))
    val_feat = build_cached_features(Path(args.cache_dir), root, val_df, tokenizer, args.max_length, args.context_turns,
                                     include_macro_in_text=bool(args.include_macro_in_text),
                                     conv_state_vec=val_conv_vec,
                                     conv_state_source=str(args.conv_state_adapter_dir),
                                     force_rebuild=bool(args.force_rebuild_cache))

    train_ds = CachedTensorDataset(train_feat)
    val_ds = CachedTensorDataset(val_feat)

    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)
    if bool(args.use_conv_state):
        base_model = RobertaForMacroActionFusion.from_pretrained(
            args.model_name,
            config=config,
            conv_state_dim=conv_state_dim,
        )
        modules_to_save = ["classifier", "conv_proj"]
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
        modules_to_save = ["classifier"]

    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["query", "value"],
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(base_model, peft_cfg)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        return {"accuracy": acc}

    tb_dir = Path(args.tb_dir).expanduser().resolve() if args.tb_dir.strip() else (out_dir / "tb")
    tb_dir.mkdir(parents=True, exist_ok=True)
    report_to = "none" if args.report_to == "none" else ["tensorboard"]
    n_workers = int(args.num_workers)
    fp16, bf16, amp_mode = resolve_mixed_precision(args.amp)
    if os.environ.get("RANK", "0") == "0":
        print(f"[SFT] Mixed precision amp={args.amp} -> {amp_mode}")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        fp16=fp16,
        bf16=bf16,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        dataloader_num_workers=n_workers,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=1,
        report_to=report_to,
        logging_dir=str(tb_dir),
        run_name=str(out_dir.name),
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=stack_collate,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    remove_logging(trainer)

    trainer.train()
    metrics = trainer.evaluate()
    (out_dir / "eval_metrics.json").write_text(str(metrics), encoding="utf-8")

    # Save adapter + label info
    model.save_pretrained(str(out_dir / "adapter"))
    save_json(
        {
            "labels": {"0": "continue", "1": "terminate"},
            "use_conv_state": bool(args.use_conv_state),
            "conv_state_source": str(args.conv_state_adapter_dir),
            "conv_state_dim": int(conv_state_dim),
            "conv_state_encoding": str(args.conv_state_encoding),
        },
        out_dir / "label_map.json",
    )

    # --- Save classification report + confusion matrix on validation set ---
    pred_out = trainer.predict(val_ds)
    logits = pred_out.predictions
    y_true = pred_out.label_ids
    y_pred = np.argmax(logits, axis=-1)

    # Classification report
    report_txt = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["continue", "terminate"],
        digits=4,
        zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    # Also save a structured version
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["continue", "terminate"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    save_json(report_dict, out_dir / "classification_report.json")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
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

    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
