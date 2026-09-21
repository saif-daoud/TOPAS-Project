"""Conversation-state SFT training.

- Multi-head classification: one head per conversation-state dimension.
- Saves adapter + tokenizer.
- Saves evaluation reports (overall + per-dimension + per-label) and plots to output_dir/reports/.
- Supports being run/imported as a top-level script or as a package module.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments

from .data_loading import DataRoot, load_conv_state_annotations
from .cache_builders import build_cached_conv_state
from .cached_datasets import CachedTensorDataset, stack_collate
from .models import RobertaForConvStateMultiHead
from .utils import load_json, save_json, set_seed, safe_name, topk_accuracy, remove_logging, resolve_mixed_precision, role_labels

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

def save_training_curves(trainer: Trainer, out_dir: Path) -> None:
    """Save training curves + log history CSV from Trainer.log_history."""
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    hist = trainer.state.log_history
    if not hist:
        return

    df = pd.DataFrame(hist)
    df.to_csv(reports_dir / "trainer_log_history.csv", index=False)

    # Eval accuracy curve
    if "eval_accuracy" in df.columns and "epoch" in df.columns:
        d = df.dropna(subset=["eval_accuracy", "epoch"])
        if len(d) > 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(d["epoch"].to_numpy(), d["eval_accuracy"].to_numpy(), marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Eval mean accuracy")
            ax.set_title("Eval Mean Accuracy (Conv State)")
            fig.tight_layout()
            fig.savefig(reports_dir / "eval_accuracy_curve.png", dpi=200)
            plt.close(fig)

    # Loss curve (train and eval)
    if "epoch" in df.columns and ("loss" in df.columns or "eval_loss" in df.columns):
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
        ax.set_title("Loss (Conv State)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports_dir / "loss_curve.png", dpi=200)
        plt.close(fig)

def save_conv_state_reports(
    out_dir: Path,
    dim_names: list[str],
    dim_id2label: dict[str, list[str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Save overall + per-dimension reports: accuracy, per-label accuracy, confusion matrices, classification reports."""
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    assert y_true.shape == y_pred.shape, (y_true.shape, y_pred.shape)
    n, d = y_true.shape

    per_dim_rows = []
    per_dim_accs = []

    for i, dim in enumerate(dim_names):
        safe_dim = safe_name(dim)
        class_names = dim_id2label[dim]

        yt = y_true[:, i]
        yp = y_pred[:, i]

        acc = float(accuracy_score(yt, yp))
        per_dim_accs.append(acc)
        per_dim_rows.append({"dimension": dim, "accuracy": acc, "num_examples": int(len(yt))})

        # Confusion matrix (rows=true, cols=pred)
        cm = confusion_matrix(yt, yp, labels=list(range(len(class_names))))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(reports_dir / f"confusion_matrix__{safe_dim}.csv", index=True)

        # Per-label accuracy (= recall per label)
        support = cm.sum(axis=1)
        correct = np.diag(cm)
        per_label_acc = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support != 0)

        per_label_df = pd.DataFrame(
            {
                "label": class_names,
                "support": support.astype(int),
                "correct": correct.astype(int),
                "accuracy": per_label_acc.astype(float),
            }
        )
        per_label_df.to_csv(reports_dir / f"accuracy_per_label__{safe_dim}.csv", index=False)

        # Classification report
        rep_txt = classification_report(
            yt,
            yp,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        (reports_dir / f"classification_report__{safe_dim}.txt").write_text(rep_txt, encoding="utf-8")

        rep_json = classification_report(
            yt,
            yp,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        save_json(rep_json, reports_dir / f"classification_report__{safe_dim}.json")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(f"Confusion Matrix ({dim})")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        fig.tight_layout()
        fig.savefig(reports_dir / f"confusion_matrix__{safe_dim}.png", dpi=200)
        plt.close(fig)

    per_dim_df = pd.DataFrame(per_dim_rows)
    per_dim_df.to_csv(reports_dir / "accuracy_per_dimension.csv", index=False)

    mean_acc = float(np.mean(per_dim_accs)) if per_dim_accs else 0.0
    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1))) if n > 0 else 0.0

    save_json(
        {
            "overall_mean_accuracy": mean_acc,
            "exact_match_accuracy": exact_match,
            "num_examples": int(n),
            "num_dimensions": int(d),
        },
        reports_dir / "metrics.json",
    )

    # Helpful meta for later freezing/usage
    save_json(
        {
            "dim_names": dim_names,
            "dim_id2label": dim_id2label,
        },
        reports_dir / "conv_state_meta.json",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]), help="encoders_RL root")
    ap.add_argument("--split_path", type=str, default=str(Path(__file__).resolve().parent / "splits_test/user_split.json"))
    ap.add_argument("--model_name", type=str, default="roberta-large")
    ap.add_argument("--output_dir", type=str, default="outputs/conv_state")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_turns", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", type=str, default=os.environ.get("AMP", "auto"), choices=["none", "fp16", "bf16", "auto"], help="Mixed precision: auto chooses bf16 if supported, else fp16.")
    ap.add_argument("--seed", type=int, default=48)

    # Logging
    ap.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "none"],
        help="HF Trainer reporting backend. Use 'tensorboard' for TensorBoard logging.",
    )
    ap.add_argument("--tb_dir", type=str, default="", help="TensorBoard log dir. Default: <output_dir>/tb")
    ap.add_argument("--cache_dir", type=str, default="cache/features", help="Where to store tokenized feature caches")
    ap.add_argument("--force_rebuild_cache", action="store_true", help="Rebuild caches even if they exist")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    set_seed(args.seed)
    root = DataRoot(Path(args.root))

    split = load_json(_resolve_user_split_file(args.split_path, root))
    train_users = set(split["train_user_idxs"])
    val_users = set(split["val_user_idxs"])

    df = load_conv_state_annotations(root)
    df_train = df[df.user_idx.isin(train_users)].reset_index(drop=True)
    df_val = df[df.user_idx.isin(val_users)].reset_index(drop=True)

    is_main = (os.environ.get("RANK", "0") == "0")
    if is_main:
        print(f"Total rows: {len(df)}")
        print(f"Train rows: {len(df_train)}")
        print(f"Validation rows: {len(df_val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # ---- Cache features first (data preparation) ----
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_feat = build_cached_conv_state(
        cache_dir=cache_dir, root=root, df=df_train, tokenizer=tokenizer, max_length=args.max_length, max_turns=args.max_turns,
        force_rebuild=bool(args.force_rebuild_cache),
    )
    val_feat = build_cached_conv_state(
        cache_dir=cache_dir, root=root, df=df_val, tokenizer=tokenizer, max_length=args.max_length, max_turns=args.max_turns,
        force_rebuild=bool(args.force_rebuild_cache),
    )
    train_ds = CachedTensorDataset(train_feat)
    val_ds = CachedTensorDataset(val_feat)

    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(args.model_name)
    base_model = RobertaForConvStateMultiHead.from_pretrained(
        args.model_name,
        config=config,
        dim_num_labels=train_ds.dim_num_labels,
    )

    # PEFT cannot wrap ModuleDict with modules_to_save; we save each head module separately.
    head_modules_to_save = [f"heads.{i}" for i in range(len(train_ds.dim_names))]

    peft_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["query", "value"],
        modules_to_save=head_modules_to_save, # save task heads inside the adapter
    )
    model = get_peft_model(base_model, peft_cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collator = stack_collate
    dim_names = list(train_ds.dim_names)

    # Build id2label lists for reporting
    dim_id2label = {}
    for dim in dim_names:
        label2id = train_ds.dim_to_label2id[dim]
        id2label = [None] * len(label2id)
        for lab, idx in label2id.items():
            id2label[int(idx)] = lab
        dim_id2label[dim] = id2label

    def compute_metrics(eval_pred):
        # eval_pred can be EvalPrediction or (preds, labels)
        preds = getattr(eval_pred, "predictions", eval_pred[0])
        labels = getattr(eval_pred, "label_ids", eval_pred[1])  # (N, D)

        # preds is a tuple/list of arrays, one per dim: (N, C_d)
        preds_list = list(preds)
        y_pred = np.stack([p.argmax(axis=-1) for p in preds_list], axis=1)

        labels = np.asarray(labels)
        per_dim = (y_pred == labels).mean(axis=0)
        mean_acc = float(per_dim.mean()) if len(per_dim) else 0.0
        exact = float(np.mean(np.all(y_pred == labels, axis=1))) if len(labels) else 0.0

        # Top-2 accuracy (mean over dimensions)
        per_dim_top2 = []
        for d, logits_d in enumerate(preds_list):
            try:
                per_dim_top2.append(topk_accuracy(np.asarray(logits_d), labels[:, d], k=2))
            except Exception:
                pass

        # Top-3 accuracy (mean over dimensions)
        per_dim_top3 = []
        for d, logits_d in enumerate(preds_list):
            try:
                per_dim_top3.append(topk_accuracy(np.asarray(logits_d), labels[:, d], k=3))
            except Exception:
                pass
        out = {"accuracy": mean_acc, "acc_at_2": float(np.mean(per_dim_top2)) if len(per_dim_top2) else 0.0, "acc_at_3": float(np.mean(per_dim_top3)) if len(per_dim_top3) else 0.0, "exact_match": exact}
        for i, dim in enumerate(dim_names):
            out[f"acc_{safe_name(dim)}"] = float(per_dim[i])
        return out

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
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
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
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    remove_logging(trainer, show_eval=False)

    trainer.train()

    # Save log-history plots
    save_training_curves(trainer, out_dir)

    # Final evaluation reports on validation split
    pred_out = trainer.predict(val_ds)
    preds = pred_out.predictions
    labels = np.asarray(pred_out.label_ids)

    preds_list = list(preds)
    y_pred = np.stack([p.argmax(axis=-1) for p in preds_list], axis=1)
    y_true = labels

    save_conv_state_reports(out_dir, dim_names, dim_id2label, y_true=y_true, y_pred=y_pred)

    # Save adapter + tokenizer
    model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    print("Saved adapter to:", out_dir / "adapter")
    print("Saved reports to:", out_dir / "reports")

if __name__ == "__main__":
    main()
