"""Micro-action SFT training.

Two modes:
- independent: train one micro-policy per macro action (N LoRA adapters, trained independently)
- flat: train a single policy over ALL micro actions (1 LoRA adapter)
"""

import os
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .data_loading import (DataRoot, load_dataset_json, load_micro_action_annotations, load_action_space, 
                           build_flat_micro_label_map, build_micro_action_maps)
from .cache_builders import build_cached_micro_independent, build_cached_micro_flat
from .cached_datasets import CachedTensorDataset, stack_collate
from .utils import save_json, load_json, set_seed, safe_name, topk_accuracy, remove_logging, split_df_user, split_df_stratify, resolve_mixed_precision, role_labels

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

def save_reports(out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> None:
    """Save overall accuracy, per-class accuracy, confusion matrix, classification report (+ plots)."""
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    overall_acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    support = cm.sum(axis=1)
    correct = np.diag(cm)
    per_class_acc = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support != 0)

    per_class_df = pd.DataFrame(
        {
            "micro_action": class_names,
            "support": support.astype(int),
            "correct": correct.astype(int),
            "accuracy": per_class_acc,
        }
    )
    per_class_df.to_csv(reports_dir / "accuracy_per_micro_action.csv", index=False)

    # Confusion matrix CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(reports_dir / "confusion_matrix.csv")

    # Classification report
    rep_txt = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, digits=4, zero_division=0
    )
    (reports_dir / "classification_report.txt").write_text(rep_txt, encoding="utf-8")

    rep_json = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0
    )
    save_json(rep_json, reports_dir / "classification_report.json")

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
    im = ax.imshow(cm, aspect="auto")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)

    # If too many classes, skip tick labels to avoid unreadable plot.
    if len(class_names) <= 60:
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=6)

    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

def save_training_curves(trainer: Trainer, out_dir: Path) -> None:
    """Save training curves from Trainer.log_history."""
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    hist = trainer.state.log_history
    if not hist:
        return

    df = pd.DataFrame(hist)
    df.to_csv(reports_dir / "trainer_log_history.csv", index=False)

    # Accuracy curve
    if "eval_accuracy" in df.columns and "epoch" in df.columns:
        d = df.dropna(subset=["eval_accuracy", "epoch"])
        if len(d) > 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(d["epoch"].to_numpy(), d["eval_accuracy"].to_numpy(), marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Eval accuracy")
            ax.set_title("Eval Accuracy")
            fig.tight_layout()
            fig.savefig(reports_dir / "eval_accuracy_curve.png", dpi=200)
            plt.close(fig)

    # Loss curve (train and eval)
    have_epoch = "epoch" in df.columns
    if have_epoch and ("loss" in df.columns or "eval_loss" in df.columns):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if "loss" in df.columns:
            d = df.dropna(subset=["loss", "epoch"])
            if len(d) > 0:
                ax.plot(d["epoch"].to_numpy(), d["loss"].to_numpy(), marker="o", label="train_loss")
        if "eval_loss" in df.columns:
            d = df.dropna(subset=["eval_loss", "epoch"])
            if len(d) > 0:
                ax.plot(d["epoch"].to_numpy(), d["eval_loss"].to_numpy(), marker="o", label="eval_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports_dir / "loss_curve.png", dpi=200)
        plt.close(fig)

def train_policy(policy_out: Path, model_name: str, tokenizer, train_dataset, val_dataset, num_labels: int, 
                 class_names: List[str], args) -> None:
    policy_out.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # LoRA
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base_model, peft_cfg)
    collator = stack_collate

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        acc2 = float(topk_accuracy(logits, labels, k=2))
        acc3 = float(topk_accuracy(logits, labels, k=3))
        return {"accuracy": acc, "acc_at_2": acc2, "acc_at_3": acc3}

    tb_dir = Path(args.tb_dir).expanduser().resolve() if getattr(args, "tb_dir", "").strip() else (policy_out / "tb")
    tb_dir.mkdir(parents=True, exist_ok=True)
    report_to = "none" if getattr(args, "report_to", "tensorboard") == "none" else ["tensorboard"]
    n_workers = int(args.num_workers)

    training_args = TrainingArguments(
        output_dir=str(policy_out),
        fp16=args.fp16,
        bf16=args.bf16,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        dataloader_num_workers=n_workers,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=1,
        save_total_limit=2,
        report_to=report_to,
        logging_dir=str(tb_dir),
        run_name=str(policy_out.name),
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    remove_logging(trainer)

    trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(str(policy_out / "adapter"))
    tokenizer.save_pretrained(str(policy_out / "adapter"))

    # Final evaluation reports on validation split
    pred_out = trainer.predict(val_dataset)
    y_true = pred_out.label_ids.astype(int)
    y_pred = np.argmax(pred_out.predictions, axis=-1).astype(int)

    save_reports(policy_out, y_true, y_pred, class_names)
    save_training_curves(trainer, policy_out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]), help="Project root (contains data/, annotations/, components/)")
    ap.add_argument("--output_dir", type=str, default="outputs/micro_action", help="Output directory")
    ap.add_argument("--model_name", type=str, default="roberta-large")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--context_turns", type=int, default=None)

    ap.add_argument("--policy_mode", type=str, default="flat", choices=["independent", "flat"],
                    help="independent: N micro policies (one per macro action). flat: single policy over all micro actions.")

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

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

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
    ap.add_argument("--tb_dir", type=str, default="", help="TensorBoard log dir (default: <output_dir>/.../<policy>/tb)")
    ap.add_argument("--cache_dir", type=str, default="cache/features", help="Where to store tokenized feature caches")
    ap.add_argument("--force_rebuild_cache", action="store_true", help="Rebuild caches even if they exist")

    # Optional: restrict independent training to a subset of macro actions
    ap.add_argument("--macro_actions", type=str, default="", help="Comma-separated macro actions to train (independent mode only). Default: all.")

    args = ap.parse_args()
    split_mode = str(args.split_mode or "").strip().lower()
    set_seed(args.seed)

    args.fp16, args.bf16, args.amp_mode = resolve_mixed_precision(args.amp)
    if os.environ.get('RANK', '0') == '0':
        print(f"[SFT] Mixed precision amp={args.amp} -> {args.amp_mode}")

    root = DataRoot(Path(args.root).resolve())
    out_dir = Path(args.output_dir).resolve()

    data_json = load_dataset_json(root)
    df = load_micro_action_annotations(root)
    action_space = load_action_space(root)

    macro2id, macro_id_to_micro_label2id = build_micro_action_maps(action_space)

    # Tokenizer config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.truncation_side = "left" # keep the most recent utterances
    tokenizer.padding_side = "right"

    if args.policy_mode == "flat":
        class_names_all, _ = build_flat_micro_label_map(action_space)

        df_work = df.copy()
        df_work["flat_label"] = df_work["selected_macro_action"].astype(str) + "::" + df_work["selected_micro_action"].astype(str)

        # Keep only labels that actually appear in the dataframe
        present = set(df_work["flat_label"].astype(str).unique())
        class_names = [c for c in class_names_all if c in present]
        # Rebuild label2id for the filtered label set
        label2id = {c: i for i, c in enumerate(class_names)}

        # Drop rows with labels not in map
        mask = df_work["flat_label"].isin(label2id)
        if mask.mean() != 1.0:
            unexpected_actions = df_work[~df_work["flat_label"].isin(label2id)]["flat_label"].unique()
            print(f"There are some actions that are not included in the action space: {list(unexpected_actions)} !")
            raise Exception()
        df_work = df_work[df_work["flat_label"].isin(label2id)].reset_index(drop=True)

        if split_mode == "user":
            split = load_json(_resolve_user_split_file(args.user_split_file, root))
            train_df, val_df = split_df_user(df_work, split)
        else:
            train_df, val_df, _ = split_df_stratify(df_work, label_col="flat_label", val_frac=args.val_frac, seed=args.seed)

        is_main = (os.environ.get("RANK", "0") == "0")
        if is_main:
            print(f"Total rows: {len(df)}")
            print(f"Train rows: {len(train_df)}")
            print(f"Validation rows: {len(val_df)}")
        # ---- Cache features first (data preparation) ----
        cache_dir = Path(args.cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_feat = build_cached_micro_flat(
            cache_dir=cache_dir,
            df=train_df,
            data_json=data_json,
            tokenizer=tokenizer,
            max_length=args.max_length,
            label2id=label2id,
            context_turns=args.context_turns,
            force_rebuild=bool(args.force_rebuild_cache),
        )
        val_feat = build_cached_micro_flat(
            cache_dir=cache_dir,
            df=val_df,
            data_json=data_json,
            tokenizer=tokenizer,
            max_length=args.max_length,
            label2id=label2id,
            context_turns=args.context_turns,
            force_rebuild=bool(args.force_rebuild_cache),
        )
        train_ds = CachedTensorDataset(train_feat)
        val_ds = CachedTensorDataset(val_feat)

        policy_out = out_dir / "flat_policy"
        save_json(
            {
                "policy_mode": "flat",
                "num_labels": len(class_names),
                "classes": class_names,
            },
            policy_out / "micro_action_meta.json",
        )

        train_policy(policy_out=policy_out, model_name=args.model_name, tokenizer=tokenizer, train_dataset=train_ds, 
                     val_dataset=val_ds, num_labels=len(class_names), class_names=class_names, args=args)
        print("Saved flat micro policy to:", policy_out)

        return

    # independent mode
    macros = []
    for m in action_space:
        macro = m["name"]
        if macro is not None:
            macros.append(str(macro))

    macros = sorted(set(macros))

    if args.macro_actions.strip():
        wanted = {x.strip() for x in args.macro_actions.split(",") if x.strip()}
        macros = [m for m in macros if m in wanted]

    policies_root = out_dir / "policies"
    policies_root.mkdir(parents=True, exist_ok=True)

    print(f"Total rows: {len(df)}")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if split_mode == "user":
        split = load_json(_resolve_user_split_file(args.user_split_file, root))

    for macro_name in macros:
        sub = df[df["selected_macro_action"].astype(str) == str(macro_name)].copy()
        if len(sub) == 0:
            continue

        if split_mode == "user":
            train_df, val_df = split_df_user(sub, split)
        else:
            train_df, val_df, _ = split_df_stratify(sub, label_col="selected_micro_action", val_frac=args.val_frac, seed=args.seed)

        print(f"Training  micro policy on {macro_name}")
        print(f"Subset rows: {len(sub)}")
        print(f"Train rows: {len(train_df)}")
        print(f"Validation rows: {len(val_df)}")
        print(sub["selected_micro_action"].value_counts())

        # ---- Cache features first (data preparation) ----
        train_feat = build_cached_micro_independent(
            cache_dir=cache_dir,
            root=root,
            df=train_df,
            tokenizer=tokenizer,
            max_length=args.max_length,
            context_turns=args.context_turns,
            force_rebuild=bool(args.force_rebuild_cache),
        )
        val_feat = build_cached_micro_independent(
            cache_dir=cache_dir,
            root=root,
            df=val_df,
            tokenizer=tokenizer,
            max_length=args.max_length,
            context_turns=args.context_turns,
            force_rebuild=bool(args.force_rebuild_cache),
        )
        train_ds = CachedTensorDataset(train_feat)
        val_ds = CachedTensorDataset(val_feat)

        # Class names for this macro in ID order
        macro_id = macro2id[str(macro_name)]
        micro2id = macro_id_to_micro_label2id[macro_id]
        inv = [None] * len(micro2id)
        for k, v in micro2id.items():
            inv[int(v)] = k
        class_names = [str(x) for x in inv]

        policy_out = policies_root / safe_name(macro_name)
        save_json(
            {
                "policy_mode": "independent",
                "macro_action": str(macro_name),
                "num_labels": len(class_names),
                "classes": class_names,
            },
            policy_out / "micro_action_meta.json",
        )

        train_policy(
            policy_out=policy_out,
            model_name=args.model_name,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            num_labels=len(class_names),
            class_names=class_names,
            args=args,
        )
        print("Saved micro policy for", macro_name, "to:", policy_out)

if __name__ == "__main__":
    main()
