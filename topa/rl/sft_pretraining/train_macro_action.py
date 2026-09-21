import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .data_loading import (
    DataRoot,
    load_dataset_json,
    load_macro_action_annotations,
    load_action_space,
    load_conv_state_annotations,
    IGNORE_CONV_STATE_COLS,
)
from .models import RobertaForMacroActionFusion
from .cache_builders import build_cached_macro
from .cached_datasets import CachedTensorDataset, stack_collate
from .utils import load_json, save_json, set_seed, topk_accuracy, remove_logging, resolve_mixed_precision, role_labels

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

def save_macro_reports(out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]):
    """Save accuracy + per-class accuracy + confusion matrix + classification report."""
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    overall_acc = float(accuracy_score(y_true, y_pred))

    # Confusion matrix (rows=true, cols=pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Per-class accuracy = correct/support (a.k.a. recall per class)
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    per_class_acc = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support != 0)

    per_class_df = pd.DataFrame(
        {
            "macro_action": class_names,
            "support": support.astype(int),
            "correct": correct.astype(int),
            "accuracy": per_class_acc.astype(float),
        }
    )
    per_class_df.to_csv(reports_dir / "accuracy_per_macro_action.csv", index=False)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(reports_dir / "confusion_matrix.csv", index=True)

    # Classification report
    rep_txt = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, digits=4, zero_division=0
    )
    (reports_dir / "classification_report.txt").write_text(rep_txt, encoding="utf-8")

    rep_json = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0
    )
    save_json(rep_json, reports_dir / "classification_report.json")

    # Summary JSON
    save_json(
        {
            "overall_accuracy": overall_acc,
            "num_examples": int(len(y_true)),
        },
        reports_dir / "metrics.json",
    )

    # Confusion matrix plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Macro Actions)")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]), help="encoders_RL root")

    ap.add_argument(
        "--split_mode",
        type=str,
        default="user",
        choices=["user", "stratify"],
        help="user: use user_split.json (user-level split). stratify: row-level split stratified by label_col.",
    )
    ap.add_argument(
        "--split_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "splits_test/user_split.json"),
        help="Used only when split_mode=user",
    )
    ap.add_argument("--val_frac", type=float, default=0.1, help="Validation fraction when split_mode=stratify")
    ap.add_argument("--label_col", type=str, default="selected_macro_action", help="Label column for stratified split")

    ap.add_argument("--model_name", type=str, default="roberta-large")
    ap.add_argument("--output_dir", type=str, default="outputs/macro_action")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_turns", type=int, default=None)
    ap.add_argument(
        "--use_conv_state",
        type=int,
        default=0,
        choices=[0, 1],
        help="1: include CONV_STATE (gold labels) in input; 0: dialogue only",
    )
    ap.add_argument("--conv_state_encoding", type=str, default="one_hot", choices=["one_hot", "label"])

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
        help="HF Trainer reporting backend. Use 'tensorboard' to enable TensorBoard logging.",
    )
    ap.add_argument(
        "--tb_dir",
        type=str,
        default="",
        help="TensorBoard logging dir (default: <output_dir>/tb)",
    )

    # Feature caching (data preparation)
    ap.add_argument("--cache_dir", type=str, default="cache/features", help="Where to store tokenized feature caches")
    ap.add_argument("--force_rebuild_cache", action="store_true", help="Rebuild caches even if they exist")

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    args = ap.parse_args()

    set_seed(args.seed)
    root = DataRoot(Path(args.root))

    df = load_macro_action_annotations(root)

    print(f"Total rows: {len(df)}")

    # When using conv states, we must merge the conversation-state annotations (per phase)
    # into the macro-action dataframe so MacroActionDataset can construct a non-empty
    # conv_state_vec. Without this merge, conv_state_dim becomes 0 and fusion model init fails.
    if bool(args.use_conv_state):
        df_cs = load_conv_state_annotations(root)
        # Drop metadata columns from conv-state annotations; keep only the categorical dimensions.
        cs_cols = [c for c in df_cs.columns if c not in IGNORE_CONV_STATE_COLS]
        # De-duplicate just in case.
        key_cols = ["user_idx", "session_idx", "phase_idx", "from_idx", "to_idx", "selected_macro_action"]
        df_cs = df_cs.drop_duplicates(subset=key_cols, keep="first")
        df = df.merge(
            df_cs[key_cols + cs_cols],
            on=key_cols,
            how="left",
        )
        print(f"Total rows after merge: {len(df)}")

    split_mode = str(args.split_mode or "").strip().lower()

    if split_mode == "user":
        split = load_json(_resolve_user_split_file(args.split_path, root))
        train_users = set(split["train_user_idxs"])
        val_users = set(split["val_user_idxs"])
        df_train = df[df.user_idx.isin(train_users)].reset_index(drop=True)
        df_val = df[df.user_idx.isin(val_users)].reset_index(drop=True)
    else:
        # Row-level stratified split by label_col
        le = LabelEncoder()
        y = le.fit_transform(df[args.label_col].astype(str).to_numpy())
        idx = np.arange(len(df))
        train_idx, val_idx = train_test_split(
            idx, test_size=args.val_frac, random_state=args.seed, stratify=y
        )
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

    is_main = (os.environ.get("RANK", "0") == "0")
    if is_main:
        print(f"Total rows: {len(df)}")
        print(f"Train rows: {len(df_train)}")
        print(f"Validation rows: {len(df_val)}")

    action_space = load_action_space(root)
    num_labels = len(action_space)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.truncation_side = "left"  # keep the most recent utterances
    tokenizer.padding_side = "right"

    # ---- Cache features first (data preparation) ----
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_feat = build_cached_macro(
        cache_dir=cache_dir,
        root=root,
        df=df_train,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_turns=args.max_turns,
        use_conv_states=bool(args.use_conv_state),
        conv_state_encoding=str(args.conv_state_encoding),
        force_rebuild=bool(args.force_rebuild_cache),
    )
    val_feat = build_cached_macro(
        cache_dir=cache_dir,
        root=root,
        df=df_val,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_turns=args.max_turns,
        use_conv_states=bool(args.use_conv_state),
        conv_state_encoding=str(args.conv_state_encoding),
        force_rebuild=bool(args.force_rebuild_cache),
    )

    train_ds = CachedTensorDataset(train_feat)
    val_ds = CachedTensorDataset(val_feat)

    # Model:
    # - if use_conv_state=1: dialogue-only RoBERTa features + late-fused conv_state_vec
    # - else: plain RoBERTa sequence classifier on dialogue only
    if bool(args.use_conv_state):
        if "conv_state_vec" not in train_feat.extra:
            raise RuntimeError('use_conv_state=1 but cached features did not include conv_state_vec')
        conv_state_vector_dim = int(train_feat.extra["conv_state_vec"].shape[1])
        config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels)
        model = RobertaForMacroActionFusion.from_pretrained(
            args.model_name,
            config=config,
            conv_state_dim=conv_state_vector_dim,
        )
        modules_to_save = ["classifier", "conv_proj"]
        fusion = "late_fusion_concat"
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
        modules_to_save = ["classifier"]
        fusion = "none"
        conv_state_vector_dim = 0

    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["query", "value"],
        modules_to_save=modules_to_save,  # save classifier (+ fusion layers if exists) inside the adapter
    )
    model = get_peft_model(model, peft_cfg)

    collator = stack_collate

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "macro_actions": [a["name"] for a in action_space],
            "split_mode": args.split_mode,
            "use_conv_state": int(args.use_conv_state),
            "conv_state_fusion": fusion,
            "conv_state_vector_dim": conv_state_vector_dim,
            "conv_state_encoding": str(args.conv_state_encoding),
            "label_col": args.label_col,
            "val_frac": args.val_frac,
        },
        out_dir / "macro_action_meta.json",
    )

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
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=report_to,
        logging_dir=str(tb_dir),
        run_name=str(out_dir.name),
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        acc2 = float(topk_accuracy(logits, labels, k=2))
        acc3 = float(topk_accuracy(logits, labels, k=3))
        return {"accuracy": acc, "acc_at_2": acc2, "acc_at_3": acc3}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    remove_logging(trainer)

    trainer.train()

    if trainer.is_world_process_zero():
        # Final evaluation reports on validation split
        pred_out = trainer.predict(val_ds)
        y_true = pred_out.label_ids
        logits = pred_out.predictions
        y_pred = np.argmax(logits, axis=-1)

        # Class names in label-id order
        class_names = [a["name"] for a in action_space]
        save_macro_reports(out_dir, y_true, y_pred, class_names)
        print("Saved macro-action reports to:", out_dir / "reports")

        model.save_pretrained(str(out_dir / "adapter"))
        tokenizer.save_pretrained(str(out_dir / "adapter"))
        print("Saved adapter to:", out_dir / "adapter")

if __name__ == "__main__":
    main()
