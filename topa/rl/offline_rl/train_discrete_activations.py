import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..simple_models import MLPPolicy, MLPValue
from ..sft_pretraining.data_loading import DataRoot, build_micro_action_maps, load_action_space
from ..sft_pretraining.utils import make_torch_generator, safe_name, seed_worker, set_seed
from .activation_rollouts import (
    build_cached_flat_rl_rollout_activations,
    build_cached_intra_option_rollout_activations,
    build_cached_option_utterance_rollout_activations,
)
from .encoder_rollouts import (
    build_cached_flat_rl_rollout_encoder,
    build_cached_intra_option_rollout_encoder,
    build_cached_option_utterance_rollout_encoder,
)
from .reward_loading import ExtrinsicConfig
from .utils import make_writer, tensor_stats


def _next_embeddings(x: np.ndarray, dones: np.ndarray) -> np.ndarray:
    next_x = np.roll(x, shift=-1, axis=0)
    done_mask = np.asarray(dones).astype(bool)
    next_x[done_mask] = x[done_mask]
    return next_x.astype(np.float32)


def _dataset_from_rollout(feat, with_term: bool) -> TensorDataset:
    x = np.asarray(feat.extra["input_embeds"], dtype=np.float32)
    next_x = _next_embeddings(x, feat.extra["dones"])
    actions = np.asarray(feat.extra["actions"], dtype=np.int64)
    rewards = np.asarray(feat.extra["rewards"], dtype=np.float32)
    dones = np.asarray(feat.extra["dones"], dtype=np.float32)
    if with_term:
        term = np.asarray(feat.extra["term_labels"], dtype=np.float32)
    else:
        term = np.zeros_like(dones, dtype=np.float32)
    return TensorDataset(
        torch.tensor(x),
        torch.tensor(next_x),
        torch.tensor(actions),
        torch.tensor(rewards),
        torch.tensor(dones),
        torch.tensor(term),
    )

def _make_iql_sampler(feat, n_actions: int, seed: int):
    """
    Oversample rare and high-return transitions.

    Env controls:
      TOPA_IQL_SAMPLER=0                  disable
      TOPA_IQL_SAMPLER=1                  enable
      TOPA_IQL_SAMPLE_ACTION_ALPHA=0.5    action rarity strength
      TOPA_IQL_SAMPLE_RETURN_ALPHA=1.0    high-return strength
      TOPA_IQL_SAMPLE_MAX_WEIGHT=20.0     clamp max sample weight
    """
    import os
    import numpy as np
    import torch

    enabled = int(os.environ.get("TOPA_IQL_SAMPLER", "1"))
    if enabled == 0:
        return None
    
    print("Enabling TOPA_IQL Sampler")
    action_alpha = float(os.environ.get("TOPA_IQL_SAMPLE_ACTION_ALPHA", "0.5"))
    return_alpha = float(os.environ.get("TOPA_IQL_SAMPLE_RETURN_ALPHA", "0.5"))
    max_w = float(os.environ.get("TOPA_IQL_SAMPLE_MAX_WEIGHT", "10.0"))

    actions = np.asarray(feat.extra["actions"], dtype=np.int64)

    # Prefer returns, because your reward is terminal/session-level.
    # returns propagate terminal reward backward to earlier utterances.
    if "returns" in feat.extra:
        returns = np.asarray(feat.extra["returns"], dtype=np.float32)
    else:
        returns = np.asarray(feat.extra["rewards"], dtype=np.float32)

    counts = np.bincount(actions, minlength=int(n_actions)).astype(np.float32)

    # action rarity weight
    action_w = np.ones_like(counts, dtype=np.float32)
    observed = counts > 0
    median_count = np.median(counts[observed])
    action_w[observed] = (median_count / np.maximum(counts[observed], 1.0)) ** action_alpha

    # return weight: high-return traces sampled more often
    r = returns.copy()
    r = r - np.nanmin(r)
    r = r / (np.nanmax(r) + 1e-8)
    return_w = 1.0 + return_alpha * r

    sample_w = action_w[actions] * return_w

    # avoid one weird transition dominating
    sample_w = np.clip(sample_w, 1.0 / max_w, max_w)

    print("[IQL sampler]")
    print("  action_alpha:", action_alpha)
    print("  return_alpha:", return_alpha)
    print("  max_w:", max_w)
    print("  action counts:", counts.tolist())
    print("  action weights:", action_w.tolist())
    print("  sample weight min/mean/max:",
          float(sample_w.min()), float(sample_w.mean()), float(sample_w.max()))

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
        generator=generator,
    )

def _num_actions(feat) -> int:
    if "num_labels" in feat.extra:
        return int(feat.extra["num_labels"])
    return int(np.max(feat.extra["actions"])) + 1

def _actor_class_weights_from_actions(actions_np, num_actions: int, device: str):
    """
    Env controls:
      TOPA_ACTOR_CLASS_WEIGHT_ALPHA=0.0  -> no class weighting
      TOPA_ACTOR_CLASS_WEIGHT_ALPHA=0.5  -> inverse sqrt frequency
      TOPA_ACTOR_CLASS_WEIGHT_ALPHA=1.0  -> inverse frequency
      TOPA_ACTOR_CLASS_WEIGHT_MAX=5.0    -> max weight clamp
    """
    import os

    alpha = float(os.environ.get("TOPA_ACTOR_CLASS_WEIGHT_ALPHA", "0.0"))
    max_w = float(os.environ.get("TOPA_ACTOR_CLASS_WEIGHT_MAX", "5.0"))

    actions = torch.as_tensor(actions_np, dtype=torch.long)
    counts = torch.bincount(actions, minlength=num_actions).float()

    weights = torch.ones(num_actions, dtype=torch.float32)

    if alpha > 0:
        observed = counts > 0

        # inverse frequency^alpha
        weights[observed] = counts[observed].pow(-alpha)

        # normalize observed weights around mean 1
        weights[observed] = weights[observed] / weights[observed].mean().clamp(min=1e-8)

        # clamp to avoid exploding rare actions
        weights = torch.clamp(weights, min=1.0 / max_w, max=max_w)

    print("[IQL] Actor class weighting")
    print("  alpha:", alpha)
    print("  max_w:", max_w)
    print("  counts:", counts.tolist())
    print("  weights:", weights.tolist())

    return weights.to(device)


def _q_action(q_net: MLPPolicy, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return q_net(x).gather(1, actions.view(-1, 1)).squeeze(1)


def _logp_action(actor: MLPPolicy, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(actor(x), dim=-1).gather(1, actions.view(-1, 1)).squeeze(1)


def _entropy(actor: MLPPolicy, x: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(actor(x), dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1).mean()


def _task_prefix(task: str) -> str:
    if task == "policy_over_options":
        return "policy_over_options"
    if task == "intra_option_all":
        return "intra_option"
    return "flat_rl"


def _log_scalars(writer, vals: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for k, v in vals.items():
        writer.add_scalar(k, float(v), int(step))


def _termination_report(beta: MLPPolicy, loader: DataLoader, device: str) -> dict:
    y_true = []
    y_pred = []
    probs = []
    beta.eval()
    with torch.no_grad():
        for _, next_x, actions, _, _, term in loader:
            next_x = next_x.to(device)
            actions = actions.to(device)
            logits = beta(next_x).gather(1, actions.view(-1, 1)).squeeze(1)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.extend(p.tolist())
            y_pred.extend((p >= 0.5).astype(np.int64).tolist())
            y_true.extend(term.numpy().astype(np.int64).tolist())
    beta.train()

    rep = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["continue", "terminate"],
        output_dict=True,
        zero_division=0,
    )
    rep["text"] = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["continue", "terminate"],
        zero_division=0,
    )
    p_arr = np.asarray(probs, dtype=np.float32)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    rep["summary"] = {
        "accuracy": float(rep["accuracy"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "terminate_precision": float(rep["terminate"]["precision"]),
        "terminate_recall": float(rep["terminate"]["recall"]),
        "terminate_f1": float(rep["terminate"]["f1-score"]),
        "true_terminate_rate": float(y_true_arr.mean()) if len(y_true_arr) else 0.0,
        "pred_terminate_rate": float(y_pred_arr.mean()) if len(y_pred_arr) else 0.0,
        "prob_mean": float(p_arr.mean()) if len(p_arr) else 0.0,
        "prob_min": float(p_arr.min()) if len(p_arr) else 0.0,
        "prob_max": float(p_arr.max()) if len(p_arr) else 0.0,
    }
    return rep


def _log_termination_report(writer, report: dict, step: int, prefix: str = "termination") -> None:
    summary = report["summary"]
    vals = {
        f"{prefix}/accuracy": summary["accuracy"],
        f"{prefix}/macro_f1": summary["macro_f1"],
        f"{prefix}/weighted_f1": summary["weighted_f1"],
        f"{prefix}/terminate_precision": summary["terminate_precision"],
        f"{prefix}/terminate_recall": summary["terminate_recall"],
        f"{prefix}/terminate_f1": summary["terminate_f1"],
        f"{prefix}/true_terminate_rate": summary["true_terminate_rate"],
        f"{prefix}/pred_terminate_rate": summary["pred_terminate_rate"],
        f"{prefix}/prob_mean": summary["prob_mean"],
        f"{prefix}/prob_min": summary["prob_min"],
        f"{prefix}/prob_max": summary["prob_max"],
    }
    _log_scalars(writer, vals, step)


def _termination_signal_report(
    signal: np.ndarray,
    beta_prob: np.ndarray,
    term_labels: np.ndarray,
    out_dir: Path,
    writer,
    epoch: int,
    title: str,
) -> dict:
    signal = np.asarray(signal, dtype=np.float32)
    beta_prob = np.asarray(beta_prob, dtype=np.float32)
    term_labels = np.asarray(term_labels, dtype=np.int64)
    y_signal = (signal < 0.0).astype(np.int64)
    y_pred = (beta_prob >= 0.5).astype(np.int64)

    rep = classification_report(
        y_signal,
        y_pred,
        labels=[0, 1],
        target_names=["continue_signal", "switch_signal"],
        output_dict=True,
        zero_division=0,
    )
    rep["text"] = classification_report(
        y_signal,
        y_pred,
        labels=[0, 1],
        target_names=["continue_signal", "switch_signal"],
        zero_division=0,
    )
    label_alignment = classification_report(
        term_labels,
        y_signal,
        labels=[0, 1],
        target_names=["continue_label", "terminate_label"],
        output_dict=True,
        zero_division=0,
    )
    rep["label_alignment"] = {
        "accuracy": float(label_alignment["accuracy"]),
        "macro_f1": float(label_alignment["macro avg"]["f1-score"]),
        "terminate_recall": float(label_alignment["terminate_label"]["recall"]),
        "text": classification_report(
            term_labels,
            y_signal,
            labels=[0, 1],
            target_names=["continue_label", "terminate_label"],
            zero_division=0,
        ),
    }
    rep["summary"] = {
        "accuracy": float(rep["accuracy"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "switch_precision": float(rep["switch_signal"]["precision"]),
        "switch_recall": float(rep["switch_signal"]["recall"]),
        "switch_f1": float(rep["switch_signal"]["f1-score"]),
        "continue_precision": float(rep["continue_signal"]["precision"]),
        "continue_recall": float(rep["continue_signal"]["recall"]),
        "continue_f1": float(rep["continue_signal"]["f1-score"]),
        "true_switch_rate": float(y_signal.mean()) if len(y_signal) else 0.0,
        "pred_switch_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
        "annotation_terminate_rate": float(term_labels.mean()) if len(term_labels) else 0.0,
        "signal_mean": float(signal.mean()) if len(signal) else 0.0,
        "signal_min": float(signal.min()) if len(signal) else 0.0,
        "signal_max": float(signal.max()) if len(signal) else 0.0,
        "prob_mean": float(beta_prob.mean()) if len(beta_prob) else 0.0,
        "prob_min": float(beta_prob.min()) if len(beta_prob) else 0.0,
        "prob_max": float(beta_prob.max()) if len(beta_prob) else 0.0,
    }

    plot_dir = Path(out_dir) / "termination_signal"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].hist(signal[y_signal == 0], bins=40, alpha=0.65, label="continue signal")
    axes[0].hist(signal[y_signal == 1], bins=40, alpha=0.65, label="switch signal")
    axes[0].axvline(0.0, color="black", linewidth=1.0)
    axes[0].set_title("Q continue - V switch")
    axes[0].set_xlabel("signal")
    axes[0].set_ylabel("count")
    axes[0].legend()

    bins = np.linspace(0.0, 1.0, 31)
    axes[1].hist(beta_prob[y_signal == 0], bins=bins, alpha=0.65, label="continue signal")
    axes[1].hist(beta_prob[y_signal == 1], bins=bins, alpha=0.65, label="switch signal")
    axes[1].axvline(0.5, color="black", linewidth=1.0)
    axes[1].set_title("Learned termination probability")
    axes[1].set_xlabel("beta probability")
    axes[1].set_ylabel("count")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_dir / f"epoch_{epoch}.png", dpi=160)
    if writer is not None:
        writer.add_figure("termination_signal/q_minus_v_vs_beta", fig, int(epoch))
    plt.close(fig)

    with (plot_dir / f"epoch_{epoch}.csv").open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["signal", "switch_signal", "beta_prob", "beta_pred", "termination_label"])
        writer_csv.writeheader()
        for i in range(len(signal)):
            writer_csv.writerow(
                {
                    "signal": float(signal[i]),
                    "switch_signal": int(y_signal[i]),
                    "beta_prob": float(beta_prob[i]),
                    "beta_pred": int(y_pred[i]),
                    "termination_label": int(term_labels[i]),
                }
            )
    return rep


def _log_termination_signal_report(writer, report: dict, step: int, prefix: str = "termination_signal") -> None:
    summary = report["summary"]
    vals = {
        f"{prefix}/accuracy": summary["accuracy"],
        f"{prefix}/macro_f1": summary["macro_f1"],
        f"{prefix}/weighted_f1": summary["weighted_f1"],
        f"{prefix}/switch_precision": summary["switch_precision"],
        f"{prefix}/switch_recall": summary["switch_recall"],
        f"{prefix}/switch_f1": summary["switch_f1"],
        f"{prefix}/continue_precision": summary["continue_precision"],
        f"{prefix}/continue_recall": summary["continue_recall"],
        f"{prefix}/continue_f1": summary["continue_f1"],
        f"{prefix}/true_switch_rate": summary["true_switch_rate"],
        f"{prefix}/pred_switch_rate": summary["pred_switch_rate"],
        f"{prefix}/annotation_terminate_rate": summary["annotation_terminate_rate"],
        f"{prefix}/signal_mean": summary["signal_mean"],
        f"{prefix}/signal_min": summary["signal_min"],
        f"{prefix}/signal_max": summary["signal_max"],
        f"{prefix}/prob_mean": summary["prob_mean"],
        f"{prefix}/prob_min": summary["prob_min"],
        f"{prefix}/prob_max": summary["prob_max"],
        f"{prefix}/label_alignment_accuracy": report["label_alignment"]["accuracy"],
        f"{prefix}/label_alignment_macro_f1": report["label_alignment"]["macro_f1"],
        f"{prefix}/label_alignment_terminate_recall": report["label_alignment"]["terminate_recall"],
    }
    _log_scalars(writer, vals, step)


def _termination_signal_report_iql(
    beta: MLPPolicy,
    q_net: MLPPolicy,
    v_net: MLPValue,
    loader: DataLoader,
    device: str,
    out_dir: Path,
    writer,
    epoch: int,
    title: str,
) -> dict:
    signals = []
    probs = []
    labels = []
    beta.eval()
    q_net.eval()
    v_net.eval()
    with torch.no_grad():
        for _, next_x, actions, _, _, term in loader:
            next_x = next_x.to(device)
            actions = actions.to(device)
            signal = _q_action(q_net, next_x, actions) - v_net(next_x)
            logits = beta(next_x).gather(1, actions.view(-1, 1)).squeeze(1)
            signals.extend(signal.detach().cpu().numpy().tolist())
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            labels.extend(term.numpy().astype(np.int64).tolist())
    beta.train()
    q_net.train()
    v_net.train()
    return _termination_signal_report(np.asarray(signals), np.asarray(probs), np.asarray(labels), out_dir, writer, epoch, title)


def _termination_signal_report_cql(
    beta: MLPPolicy,
    actor: MLPPolicy,
    q1: MLPPolicy,
    q2: MLPPolicy,
    loader: DataLoader,
    device: str,
    entropy_alpha: float,
    out_dir: Path,
    writer,
    epoch: int,
    title: str,
) -> dict:
    signals = []
    probs = []
    labels = []
    beta.eval()
    actor.eval()
    q1.eval()
    q2.eval()
    with torch.no_grad():
        for _, next_x, actions, _, _, term in loader:
            next_x = next_x.to(device)
            actions = actions.to(device)
            q_next = torch.minimum(q1(next_x), q2(next_x))
            logp_next = F.log_softmax(actor(next_x), dim=-1)
            p_next = logp_next.exp()
            v_switch = (p_next * (q_next - float(entropy_alpha) * logp_next)).sum(dim=-1)
            q_cont = q_next.gather(1, actions.view(-1, 1)).squeeze(1)
            signal = q_cont - v_switch
            logits = beta(next_x).gather(1, actions.view(-1, 1)).squeeze(1)
            signals.extend(signal.detach().cpu().numpy().tolist())
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            labels.extend(term.numpy().astype(np.int64).tolist())
    beta.train()
    actor.train()
    q1.train()
    q2.train()
    return _termination_signal_report(np.asarray(signals), np.asarray(probs), np.asarray(labels), out_dir, writer, epoch, title)


def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff >= 0, torch.full_like(diff, float(expectile)), torch.full_like(diff, 1.0 - float(expectile)))
    return (weight * diff.pow(2)).mean()


def _id2name(feat, n_actions: int) -> list[str]:
    id2name = [str(i) for i in range(int(n_actions))]
    if "label2id" in feat.extra:
        for name, idx in feat.extra["label2id"].items():
            id2name[int(idx)] = str(name)
    return id2name


def _action_counts(actor: MLPPolicy, loader: DataLoader, n_actions: int, device: str, max_batches: int) -> np.ndarray:
    counts = np.zeros(int(n_actions), dtype=np.int64)
    actor.eval()
    with torch.no_grad():
        for batch_idx, (x, *_rest) in enumerate(loader):
            if batch_idx >= int(max_batches):
                break
            pred = torch.argmax(actor(x.to(device)), dim=-1).detach().cpu().numpy()
            counts += np.bincount(pred, minlength=int(n_actions)).astype(np.int64)
    actor.train()
    return counts


def _gt_action_counts(loader: DataLoader, n_actions: int, max_batches: int) -> np.ndarray:
    counts = np.zeros(int(n_actions), dtype=np.int64)
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(max_batches):
            break
        actions = batch[2].detach().cpu().numpy()
        counts += np.bincount(actions, minlength=int(n_actions)).astype(np.int64)
    return counts


def _plot_action_dist(actor: MLPPolicy, loader: DataLoader, feat, out_dir: Path, writer, epoch: int, device: str, args, title: str) -> None:
    if not bool(args.plot_action_dist):
        return
    n_actions = _num_actions(feat)
    pred_counts = _action_counts(actor, loader, n_actions, device, int(args.action_dist_max_batches))
    gt_counts = _gt_action_counts(loader, n_actions, int(args.action_dist_max_batches))
    id2name = _id2name(feat, n_actions)
    order = np.argsort(-(pred_counts + gt_counts))
    k = int(min(int(args.action_dist_top_k), len(order)))
    top = order[:k]

    fig = plt.figure(figsize=(max(8, 0.35 * k), 4.5))
    ax = fig.add_subplot(111)
    x = np.arange(k)
    width = 0.42
    ax.bar(x - width / 2.0, gt_counts[top], width=width, label="ground truth")
    ax.bar(x + width / 2.0, pred_counts[top], width=width, label="actor argmax")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([id2name[int(i)] for i in top], rotation=90, fontsize=7)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    plot_dir = Path(out_dir) / "action_dist"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"epoch_{epoch}.png", dpi=160)
    with (plot_dir / f"epoch_{epoch}.csv").open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["action_id", "action_name", "ground_truth", "actor_argmax"])
        writer_csv.writeheader()
        for action_id in top:
            writer_csv.writerow(
                {
                    "action_id": int(action_id),
                    "action_name": id2name[int(action_id)],
                    "ground_truth": int(gt_counts[int(action_id)]),
                    "actor_argmax": int(pred_counts[int(action_id)]),
                }
            )
    if writer is not None:
        writer.add_figure("action_dist/gt_vs_argmax_topk", fig, int(epoch))
    plt.close(fig)


def _mlp_kwargs(feat, args) -> dict[str, int]:
    conv_state_dim = int(feat.extra["conv_state_dim"]) if "conv_state_dim" in feat.extra else 0
    projection_dim = (int(args.projection_dim) if int(args.projection_dim) > 0 else (int(args.hidden_dim) if int(args.hidden_dim) > 0 else 0)) if conv_state_dim > 0 else 0
    conv_state_embedding_dim = int(args.conv_state_embedding_dim) if conv_state_dim > 0 and int(args.conv_state_embedding_dim) > 0 else 0
    return {
        "conv_state_dim": int(conv_state_dim),
        "projection_dim": int(projection_dim),
        "conv_state_embedding_dim": int(conv_state_embedding_dim),
    }


def _fusion_name(conv_state_dim: int, projection_dim: int, conv_state_embedding_dim: int) -> str:
    if int(conv_state_dim) <= 0:
        return "plain"
    if int(projection_dim) > 0 and int(conv_state_embedding_dim) > 0:
        return "latent_projection_conv_embedding_concat"
    if int(projection_dim) > 0:
        return "latent_projection_concat"
    if int(conv_state_embedding_dim) > 0:
        return "latent_conv_embedding_concat"
    return "latent_concat"


def _load_sft_model(model: MLPPolicy, path: str, device: str, writer, tag: str) -> None:
    p = Path(str(path))
    if not str(path).strip() or not p.exists():
        if writer is not None:
            writer.add_scalar(f"{tag}/loaded_sft_checkpoint", 0.0, 0)
        raise FileNotFoundError("offline_rl.init_from_sft=true but an SFT checkpoint was not found. Run the SFT stage first or set init_from_sft=false.")
    state = torch.load(p, map_location=device)
    model_state = model.state_dict()
    compatible = set(state.keys()) == set(model_state.keys())
    if compatible:
        compatible = all(tuple(state[k].shape) == tuple(model_state[k].shape) for k in state.keys())
    if not compatible:
        if writer is not None:
            writer.add_scalar(f"{tag}/loaded_sft_checkpoint", 0.0, 0)
        raise RuntimeError("offline_rl.init_from_sft=true but an SFT checkpoint is not compatible with this run. Re-run SFT with the same state representation, layer, conv-state setting, and conv_state_encoding.")
    model.load_state_dict(state)
    if writer is not None:
        writer.add_scalar(f"{tag}/loaded_sft_checkpoint", 1.0, 0)
    print(f"[offline_rl] loaded SFT {tag} init from: {p}")


def _load_sft_fusion_layers(model, path: str, device: str, writer, tag: str) -> None:
    rep_prefixes = ("latent_proj.", "conv_state_proj.")
    rep_keys = [k for k in model.state_dict().keys() if k.startswith(rep_prefixes)]
    if not rep_keys:
        if writer is not None:
            writer.add_scalar(f"{tag}/loaded_sft_fusion_layers", 0.0, 0)
        return

    p = Path(str(path))
    if not str(path).strip() or not p.exists():
        if writer is not None:
            writer.add_scalar(f"{tag}/loaded_sft_fusion_layers", 0.0, 0)
        raise FileNotFoundError("offline_rl.init_critic_fusion_from_sft=true but an SFT actor checkpoint was not found. Run the SFT stage first or set init_critic_fusion_from_sft=false.")

    state = torch.load(p, map_location=device)
    model_state = model.state_dict()
    compatible = all(k in state and tuple(state[k].shape) == tuple(model_state[k].shape) for k in rep_keys)
    if not compatible:
        if writer is not None:
            writer.add_scalar(f"{tag}/loaded_sft_fusion_layers", 0.0, 0)
        raise RuntimeError("offline_rl.init_critic_fusion_from_sft=true but the SFT actor fusion layers are not compatible with this run. Re-run SFT with the same layer, conv-state setting, projection dim, and conv-state embedding dim.")

    for k in rep_keys:
        model_state[k] = state[k]
    model.load_state_dict(model_state)
    if writer is not None:
        writer.add_scalar(f"{tag}/loaded_sft_fusion_layers", 1.0, 0)
    print(f"[offline_rl] loaded SFT fusion layers for {tag} from: {p}")


def train_iql(feat, out_dir: Path, args, with_term: bool, log_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = make_writer(log_dir)
    task_prefix = _task_prefix(str(args.task))
    dataset = _dataset_from_rollout(feat, with_term=with_term)
    num_workers = int(args.num_workers)

    in_dim = int(feat.extra["input_embeds"].shape[1])
    n_actions = _num_actions(feat)

    sampler = _make_iql_sampler(feat, n_actions, int(args.seed)) #######

    loader = DataLoader(
    dataset,
    batch_size=int(args.batch_size),
    shuffle=(sampler is None),
    sampler=sampler,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=bool(device == "cuda"),
    persistent_workers=bool(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=make_torch_generator(int(args.seed)),
    )


    # loader = DataLoader(
    #     dataset,
    #     batch_size=int(args.batch_size),
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=num_workers,
    #     pin_memory=bool(device == "cuda"),
    #     persistent_workers=bool(num_workers > 0),
    #     prefetch_factor=4 if num_workers > 0 else None,
    #     worker_init_fn=seed_worker,
    #     generator=make_torch_generator(int(args.seed)),
    # )

    model_kwargs = _mlp_kwargs(feat, args)

    ### minority classes ### - weighting
    actor_class_weights = _actor_class_weights_from_actions(
    feat.extra["actions"],
    n_actions,
    device,)

    actor = MLPPolicy(in_dim, n_actions, args.hidden_dim, args.dropout, **model_kwargs).to(device)
    q_net = MLPPolicy(in_dim, n_actions, args.hidden_dim, args.dropout, **model_kwargs).to(device)
    v_net = MLPValue(in_dim, args.hidden_dim, args.dropout, **model_kwargs).to(device)
    beta = MLPPolicy(in_dim, n_actions, args.hidden_dim, args.dropout, **model_kwargs).to(device) if with_term else None

    if bool(args.init_from_sft):
        _load_sft_model(actor, str(args.actor_init_path), device, writer, "policy")
        if beta is not None:
            _load_sft_model(beta, str(args.termination_init_path), device, writer, "termination")
    elif writer is not None:
        writer.add_scalar("policy/loaded_sft_checkpoint", 0.0, 0)
        if beta is not None:
            writer.add_scalar("termination/loaded_sft_checkpoint", 0.0, 0)

    if bool(args.init_critic_fusion_from_sft):
        _load_sft_fusion_layers(q_net, str(args.actor_init_path), device, writer, "critic_q")
        _load_sft_fusion_layers(v_net, str(args.actor_init_path), device, writer, "critic_v")
    elif writer is not None:
        writer.add_scalar("critic_q/loaded_sft_fusion_layers", 0.0, 0)
        writer.add_scalar("critic_v/loaded_sft_fusion_layers", 0.0, 0)

    opt_actor = torch.optim.AdamW(actor.parameters(), lr=float(args.lr_actor))
    opt_q = torch.optim.AdamW(q_net.parameters(), lr=float(args.lr_critic))
    opt_v = torch.optim.AdamW(v_net.parameters(), lr=float(args.lr_critic))
    opt_beta = torch.optim.AdamW(beta.parameters(), lr=float(args.lr_beta)) if beta is not None else None
    global_step = 0
    metrics = []
    step_metrics = []
    termination_reports = []
    termination_signal_reports = []
    if beta is not None:
        init_report = _termination_report(beta, loader, device)
        init_report["epoch"] = -1
        init_report["stage"] = "sft_init" if bool(args.init_from_sft) else "random_init"
        termination_reports.append(init_report)
        _log_termination_report(writer, init_report, 0, prefix="termination/init")

    for epoch in range(int(args.epochs)):
        target_q = deepcopy(q_net).eval()
        target_v = deepcopy(v_net).eval()
        epoch_metrics = {"q": [], "v": [], "actor": [], "beta": [], "entropy": []}
        for x, next_x, actions, rewards, dones, _term in tqdm(loader, desc=f"IQL epoch {epoch+1}", leave=True):
            x = x.to(device)
            next_x = next_x.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)
            _term = _term.to(device)

            with torch.no_grad():
                next_v = target_v(next_x)
                if beta is not None:
                    p_term_target = torch.sigmoid(beta(next_x).gather(1, actions.view(-1, 1)).squeeze(1))
                    q_cont = _q_action(target_q, next_x, actions)
                    u_next = (1.0 - p_term_target) * q_cont + p_term_target * next_v
                else:
                    u_next = next_v
                y = rewards + float(args.gamma) * (1.0 - dones) * u_next
                q_old = _q_action(target_q, x, actions)

            q_pred = _q_action(q_net, x, actions)
            q_loss = F.mse_loss(q_pred, y)
            opt_q.zero_grad(set_to_none=True)
            q_loss.backward()
            opt_q.step()

            v_pred = v_net(x)
            v_loss = _expectile_loss(q_old - v_pred, float(args.expectile))
            opt_v.zero_grad(set_to_none=True)
            v_loss.backward()
            opt_v.step()

            with torch.no_grad():
                adv = q_old - v_net(x)
                weights = torch.exp(adv / float(args.temperature)).clamp(max=100.0)
            logp_a = _logp_action(actor, x, actions)
            # actor_loss = -(weights * logp_a).mean()
            ## tahayol try
            class_w = actor_class_weights[actions]
            actor_loss = -(class_w * weights * logp_a).mean()

            opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            opt_actor.step()

            beta_loss = torch.tensor(0.0, device=device)
            if beta is not None:
                with torch.no_grad():
                    g_term = _q_action(target_q, next_x, actions) - target_v(next_x)
                p_term = torch.sigmoid(beta(next_x).gather(1, actions.view(-1, 1)).squeeze(1))
                beta_loss = (p_term * g_term.detach()).mean()
                opt_beta.zero_grad(set_to_none=True)
                beta_loss.backward()
                opt_beta.step()

            ent = _entropy(actor, x)
            vals = {
                "loss/q": float(q_loss.detach().cpu()),
                "loss/value": float(v_loss.detach().cpu()),
                "loss/actor": float(actor_loss.detach().cpu()),
                "loss/beta": float(beta_loss.detach().cpu()),
                "critic/q.loss": float(q_loss.detach().cpu()),
                "critic/v.loss": float(v_loss.detach().cpu()),
                "actor/loss": float(actor_loss.detach().cpu()),
                "policy/entropy": float(ent.detach().cpu()),
                f"{task_prefix}/policy_entropy": float(ent.detach().cpu()),
                f"{task_prefix}/actor_loss": float(actor_loss.detach().cpu()),
                "stats/reward_mean": float(rewards.detach().cpu().mean()),
                "stats/q_mean": float(q_pred.detach().cpu().mean()),
                "stats/v_mean": float(v_pred.detach().cpu().mean()),
            }
            vals.update(tensor_stats("critic/q", q_pred))
            vals.update(tensor_stats("critic/v", v_pred))
            vals.update(tensor_stats("critic/target_q", q_old))
            vals.update(tensor_stats("critic/td_target", y))
            vals.update(tensor_stats("critic/target_u", u_next))
            vals.update(tensor_stats("actor/advantages", adv))
            vals.update(tensor_stats("actor/factor", weights))
            vals.update(tensor_stats("actor/log_prob", logp_a))
            if beta is not None:
                vals.update(
                    {
                        "termination/loss": float(beta_loss.detach().cpu()),
                        "termination/label_rate_batch": float(_term.detach().cpu().mean()),
                    }
                )
                vals.update(tensor_stats("termination/prob", p_term))
                vals.update(tensor_stats("termination/advantage", g_term))
            _log_scalars(writer, vals, global_step)
            row = {"epoch": int(epoch), "step": int(global_step)}
            row.update(vals)
            step_metrics.append(row)
            epoch_metrics["q"].append(vals["loss/q"])
            epoch_metrics["v"].append(vals["loss/value"])
            epoch_metrics["actor"].append(vals["loss/actor"])
            epoch_metrics["beta"].append(vals["loss/beta"])
            epoch_metrics["entropy"].append(vals["policy/entropy"])
            global_step += 1

        summary = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
        summary["epoch"] = int(epoch)
        metrics.append(summary)
        if beta is not None:
            report = _termination_report(beta, loader, device)
            report["epoch"] = int(epoch)
            termination_reports.append(report)
            _log_termination_report(writer, report, epoch)
            signal_report = _termination_signal_report_iql(
                beta,
                q_net,
                v_net,
                loader,
                device,
                out_dir,
                writer,
                epoch,
                title=f"IQL {args.task} termination signal",
            )
            signal_report["epoch"] = int(epoch)
            termination_signal_reports.append(signal_report)
            _log_termination_signal_report(writer, signal_report, epoch)
        _plot_action_dist(actor, loader, feat, out_dir, writer, epoch, device, args, title=f"IQL {args.task} action distribution")

    _save_models(
        out_dir,
        actor=actor,
        q=q_net,
        v=v_net,
        beta=beta,
        metrics=metrics,
        step_metrics=step_metrics,
        feat=feat,
        args=args,
        termination_reports=termination_reports,
        termination_signal_reports=termination_signal_reports,
    )
    if writer is not None:
        writer.flush()
        writer.close()


def train_cql(feat, out_dir: Path, args, with_term: bool, log_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = make_writer(log_dir)
    task_prefix = _task_prefix(str(args.task))
    dataset = _dataset_from_rollout(feat, with_term=with_term)
    num_workers = int(args.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=bool(device == "cuda"),
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(int(args.seed)),
    )
    in_dim = int(feat.extra["input_embeds"].shape[1])
    n_actions = _num_actions(feat)
    model_kwargs = _mlp_kwargs(feat, args)

    actor = MLPPolicy(in_dim, n_actions, args.hidden_dim, args.dropout, **model_kwargs).to(device)
    q1 = MLPPolicy(in_dim, n_actions, args.hidden_dim, args.dropout, **model_kwargs).to(device)
    q2 = MLPPolicy(in_dim, n_actions, args.hidden_dim, args.dropout, **model_kwargs).to(device)
    beta = MLPPolicy(in_dim, n_actions, args.hidden_dim, args.dropout, **model_kwargs).to(device) if with_term else None

    if bool(args.init_from_sft):
        _load_sft_model(actor, str(args.actor_init_path), device, writer, "policy")
        if beta is not None:
            _load_sft_model(beta, str(args.termination_init_path), device, writer, "termination")
    elif writer is not None:
        writer.add_scalar("policy/loaded_sft_checkpoint", 0.0, 0)
        if beta is not None:
            writer.add_scalar("termination/loaded_sft_checkpoint", 0.0, 0)

    if bool(args.init_critic_fusion_from_sft):
        _load_sft_fusion_layers(q1, str(args.actor_init_path), device, writer, "critic_q1")
        _load_sft_fusion_layers(q2, str(args.actor_init_path), device, writer, "critic_q2")
    elif writer is not None:
        writer.add_scalar("critic_q1/loaded_sft_fusion_layers", 0.0, 0)
        writer.add_scalar("critic_q2/loaded_sft_fusion_layers", 0.0, 0)

    opt_actor = torch.optim.AdamW(actor.parameters(), lr=float(args.lr_actor))
    opt_q1 = torch.optim.AdamW(q1.parameters(), lr=float(args.lr_critic))
    opt_q2 = torch.optim.AdamW(q2.parameters(), lr=float(args.lr_critic))
    opt_beta = torch.optim.AdamW(beta.parameters(), lr=float(args.lr_beta)) if beta is not None else None
    global_step = 0
    metrics = []
    step_metrics = []
    termination_reports = []
    termination_signal_reports = []
    if beta is not None:
        init_report = _termination_report(beta, loader, device)
        init_report["epoch"] = -1
        init_report["stage"] = "sft_init" if bool(args.init_from_sft) else "random_init"
        termination_reports.append(init_report)
        _log_termination_report(writer, init_report, 0, prefix="termination/init")

    for epoch in range(int(args.epochs)):
        tq1 = deepcopy(q1).eval()
        tq2 = deepcopy(q2).eval()
        epoch_metrics = {"q1": [], "q2": [], "cql": [], "actor": [], "beta": [], "entropy": []}
        for x, next_x, actions, rewards, dones, _term in tqdm(loader, desc=f"CQL epoch {epoch+1}", leave=True):
            x = x.to(device)
            next_x = next_x.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)
            _term = _term.to(device)

            with torch.no_grad():
                q_next = torch.minimum(tq1(next_x), tq2(next_x))
                logp_next = F.log_softmax(actor(next_x), dim=-1)
                p_next = logp_next.exp()
                v_switch = (p_next * (q_next - float(args.entropy_alpha) * logp_next)).sum(dim=-1)
                q_cont = q_next.gather(1, actions.view(-1, 1)).squeeze(1)
                if beta is not None:
                    p_term_target = torch.sigmoid(beta(next_x).gather(1, actions.view(-1, 1)).squeeze(1))
                    u_next = (1.0 - p_term_target) * q_cont + p_term_target * v_switch
                else:
                    u_next = v_switch
                y = rewards + float(args.gamma) * (1.0 - dones) * u_next

            q1_all = q1(x)
            q2_all = q2(x)
            q1_a = q1_all.gather(1, actions.view(-1, 1)).squeeze(1)
            q2_a = q2_all.gather(1, actions.view(-1, 1)).squeeze(1)
            cql1 = torch.logsumexp(q1_all, dim=-1).mean() - q1_a.mean()
            cql2 = torch.logsumexp(q2_all, dim=-1).mean() - q2_a.mean()
            q1_td_loss = F.mse_loss(q1_a, y)
            q2_td_loss = F.mse_loss(q2_a, y)
            q1_loss = q1_td_loss + float(args.cql_alpha) * cql1
            q2_loss = q2_td_loss + float(args.cql_alpha) * cql2
            opt_q1.zero_grad(set_to_none=True)
            q1_loss.backward()
            opt_q1.step()
            opt_q2.zero_grad(set_to_none=True)
            q2_loss.backward()
            opt_q2.step()

            logp = F.log_softmax(actor(x), dim=-1)
            p = logp.exp()
            with torch.no_grad():
                q_min = torch.minimum(q1(x), q2(x))
                q_data = q_min.gather(1, actions.view(-1, 1)).squeeze(1)
                policy_q = (p * q_min).sum(dim=-1)
                soft_adv = q_data - policy_q
                cql_factor = torch.exp(soft_adv / float(args.temperature)).clamp(max=100.0)
            rl_loss = (p * (float(args.entropy_alpha) * logp - q_min)).sum(dim=-1).mean()
            bc_loss = F.cross_entropy(actor(x), actions)
            actor_loss = rl_loss + float(args.bc_weight) * bc_loss
            logp_a = logp.gather(1, actions.view(-1, 1)).squeeze(1)
            opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            opt_actor.step()

            beta_loss = torch.tensor(0.0, device=device)
            if beta is not None:
                with torch.no_grad():
                    q_next_now = torch.minimum(tq1(next_x), tq2(next_x))
                    q_cont_now = q_next_now.gather(1, actions.view(-1, 1)).squeeze(1)
                    g_term = q_cont_now - v_switch
                p_term = torch.sigmoid(beta(next_x).gather(1, actions.view(-1, 1)).squeeze(1))
                beta_loss = (p_term * g_term.detach()).mean()
                opt_beta.zero_grad(set_to_none=True)
                beta_loss.backward()
                opt_beta.step()

            ent = _entropy(actor, x)
            vals = {
                "loss/q1": float(q1_loss.detach().cpu()),
                "loss/q2": float(q2_loss.detach().cpu()),
                "loss/cql": float((cql1 + cql2).detach().cpu()),
                "loss/actor": float(actor_loss.detach().cpu()),
                "loss/beta": float(beta_loss.detach().cpu()),
                "critic/q1.loss": float(q1_loss.detach().cpu()),
                "critic/q2.loss": float(q2_loss.detach().cpu()),
                "critic/q1.td_loss": float(q1_td_loss.detach().cpu()),
                "critic/q2.td_loss": float(q2_td_loss.detach().cpu()),
                "critic/cql.penalty": float((cql1 + cql2).detach().cpu()),
                "critic/cql1.penalty": float(cql1.detach().cpu()),
                "critic/cql2.penalty": float(cql2.detach().cpu()),
                "actor/loss": float(actor_loss.detach().cpu()),
                "actor/rl_loss": float(rl_loss.detach().cpu()),
                "actor/bc_loss": float(bc_loss.detach().cpu()),
                "policy/entropy": float(ent.detach().cpu()),
                f"{task_prefix}/policy_entropy": float(ent.detach().cpu()),
                f"{task_prefix}/actor_loss": float(actor_loss.detach().cpu()),
                "stats/reward_mean": float(rewards.detach().cpu().mean()),
                "stats/q_mean": float(q1_a.detach().cpu().mean()),
            }
            vals.update(tensor_stats("critic/q1", q1_a))
            vals.update(tensor_stats("critic/q2", q2_a))
            vals.update(tensor_stats("critic/target_q_cont", q_cont))
            vals.update(tensor_stats("critic/target_v_switch", v_switch))
            vals.update(tensor_stats("critic/td_target", y))
            vals.update(tensor_stats("actor/advantages", soft_adv))
            vals.update(tensor_stats("actor/factor", cql_factor))
            vals.update(tensor_stats("actor/log_prob", logp_a))
            if beta is not None:
                vals.update(
                    {
                        "termination/loss": float(beta_loss.detach().cpu()),
                        "termination/label_rate_batch": float(_term.detach().cpu().mean()),
                    }
                )
                vals.update(tensor_stats("termination/prob", p_term))
                vals.update(tensor_stats("termination/advantage", g_term))
            _log_scalars(writer, vals, global_step)
            row = {"epoch": int(epoch), "step": int(global_step)}
            row.update(vals)
            step_metrics.append(row)
            epoch_metrics["q1"].append(vals["loss/q1"])
            epoch_metrics["q2"].append(vals["loss/q2"])
            epoch_metrics["cql"].append(vals["loss/cql"])
            epoch_metrics["actor"].append(vals["loss/actor"])
            epoch_metrics["beta"].append(vals["loss/beta"])
            epoch_metrics["entropy"].append(vals["policy/entropy"])
            global_step += 1

        summary = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
        summary["epoch"] = int(epoch)
        metrics.append(summary)
        if beta is not None:
            report = _termination_report(beta, loader, device)
            report["epoch"] = int(epoch)
            termination_reports.append(report)
            _log_termination_report(writer, report, epoch)
            signal_report = _termination_signal_report_cql(
                beta,
                actor,
                q1,
                q2,
                loader,
                device,
                float(args.entropy_alpha),
                out_dir,
                writer,
                epoch,
                title=f"CQL {args.task} termination signal",
            )
            signal_report["epoch"] = int(epoch)
            termination_signal_reports.append(signal_report)
            _log_termination_signal_report(writer, signal_report, epoch)
        _plot_action_dist(actor, loader, feat, out_dir, writer, epoch, device, args, title=f"CQL {args.task} action distribution")

    _save_models(
        out_dir,
        actor=actor,
        q1=q1,
        q2=q2,
        beta=beta,
        metrics=metrics,
        step_metrics=step_metrics,
        feat=feat,
        args=args,
        termination_reports=termination_reports,
        termination_signal_reports=termination_signal_reports,
    )
    if writer is not None:
        writer.flush()
        writer.close()


def _save_models(
    out_dir: Path,
    *,
    actor,
    metrics,
    step_metrics,
    feat,
    args,
    q=None,
    v=None,
    q1=None,
    q2=None,
    beta=None,
    termination_reports=None,
    termination_signal_reports=None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), out_dir / "actor.pt")
    if q is not None:
        torch.save(q.state_dict(), out_dir / "q.pt")
    if v is not None:
        torch.save(v.state_dict(), out_dir / "v.pt")
    if q1 is not None:
        torch.save(q1.state_dict(), out_dir / "q1.pt")
    if q2 is not None:
        torch.save(q2.state_dict(), out_dir / "q2.pt")
    if beta is not None:
        torch.save(beta.state_dict(), out_dir / "termination.pt")
    conv_state_dim = int(feat.extra["conv_state_dim"]) if "conv_state_dim" in feat.extra else 0
    model_kwargs = _mlp_kwargs(feat, args)
    projection_dim = model_kwargs["projection_dim"]
    conv_state_embedding_dim = model_kwargs["conv_state_embedding_dim"]
    meta = {
        "algorithm": str(args.algorithm),
        "task": str(args.task),
        "state_repr": str(args.state_repr),
        "num_actions": _num_actions(feat),
        "input_dim": int(feat.extra["input_embeds"].shape[1]),
        "latent_dim": int(feat.extra["latent_dim"]) if "latent_dim" in feat.extra else int(feat.extra["input_embeds"].shape[1]),
        "conv_state_dim": int(conv_state_dim),
        "projection_dim": int(projection_dim),
        "conv_state_embedding_dim": int(conv_state_embedding_dim),
        "fusion": _fusion_name(int(conv_state_dim), int(projection_dim), int(conv_state_embedding_dim)),
        "conv_state_encoding": str(args.conv_state_encoding),
        "conv_state_source": str(args.conv_state_source),
        "conv_state_sft_dir": str(args.conv_state_sft_dir),
        "probe_checkpoint": str(args.probe_checkpoint),
        "probe_include_hidden": 0,
        "init_from_sft": bool(args.init_from_sft),
        "init_critic_fusion_from_sft": bool(args.init_critic_fusion_from_sft),
        "actor_init_path": str(args.actor_init_path),
        "termination_init_path": str(args.termination_init_path),
        "actor_init_loaded": bool(args.init_from_sft),
        "termination_init_loaded": bool(args.init_from_sft and beta is not None),
        "critic_fusion_init_loaded": bool(args.init_critic_fusion_from_sft),
        "num_transitions": int(feat.extra["input_embeds"].shape[0]),
    }
    if str(args.state_repr) == "encoder":
        meta["encoder_name"] = str(args.encoder_name)
    payload = {"epochs": metrics, "meta": meta}
    if termination_reports:
        payload["termination_reports"] = termination_reports
        (out_dir / "termination_classification_report.json").write_text(json.dumps(termination_reports[-1], indent=2), encoding="utf-8")
        final = termination_reports[-1]
        lines = [
            "Final termination report summary",
            json.dumps(final["summary"], indent=2),
            "",
            final["text"],
            "Full JSON report is saved in termination_classification_report.json",
        ]
        (out_dir / "termination_classification_report.txt").write_text("\n".join(lines), encoding="utf-8")
    if termination_signal_reports:
        payload["termination_signal_reports"] = termination_signal_reports
        (out_dir / "termination_signal_classification_report.json").write_text(json.dumps(termination_signal_reports[-1], indent=2), encoding="utf-8")
        final = termination_signal_reports[-1]
        lines = [
            "Final termination-vs-switch-signal report summary",
            json.dumps(final["summary"], indent=2),
            "",
            final["text"],
            "",
            "Annotation labels compared to the learned switch signal",
            json.dumps(final["label_alignment"], indent=2),
            "Full JSON report is saved in termination_signal_classification_report.json",
        ]
        (out_dir / "termination_signal_classification_report.txt").write_text("\n".join(lines), encoding="utf-8")
    if step_metrics:
        keys = sorted({k for row in step_metrics for k in row.keys()})
        with (out_dir / "metrics_steps.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(step_metrics)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _train_one(feat, out_dir: Path, args, with_term: bool, log_dir: Path) -> None:
    if str(args.algorithm).lower() == "iql":
        train_iql(feat, out_dir, args, with_term=with_term, log_dir=log_dir)
    else:
        train_cql(feat, out_dir, args, with_term=with_term, log_dir=log_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--task", type=str, required=True, choices=["policy_over_options", "flat_rl", "intra_option_all"])
    ap.add_argument("--algorithm", type=str, required=True, choices=["iql", "cql"])
    ap.add_argument("--state_repr", type=str, default="activations", choices=["encoder", "activations"])
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--log_dir", type=str, required=True)
    ap.add_argument("--encoder_name", type=str, default="")
    ap.add_argument("--encoder_batch_size", type=int, default=128)
    ap.add_argument("--conv_state_adapter_dir", type=str, default="")
    ap.add_argument("--conv_state_meta_path", type=str, default="")
    ap.add_argument("--conv_state_encoding", type=str, default="one_hot", choices=["one_hot", "label"])
    ap.add_argument("--conv_state_source", type=str, default="gold", choices=["gold", "sft", "probe"], help="gold: use annotation vectors; sft: use activation-SFT conv-state model; probe: use frozen probability probe.")
    ap.add_argument("--conv_state_sft_dir", type=str, default="", help="Directory containing actor.pt and label_map.json from train_conv_state_activations.py when conv_state_source=sft.")
    ap.add_argument("--probe_checkpoint", type=str, default="/home/local/QCRI/saif.sedaoud/clean-env/new_env/cv_state_preds/probe_results_conversation_states/checkpoints/probe_layer_22.pt", help="Frozen probe checkpoint used when conv_state_source=probe.")
    ap.add_argument("--probe_batch_size", type=int, default=1024)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--context_turns", type=int, default=-1)
    ap.add_argument("--activation_path", type=str, default="")
    ap.add_argument("--activation_layer", type=int, default=-1)
    ap.add_argument("--activation_key", type=str, default="activations")
    ap.add_argument("--use_conv_state", action="store_true")
    ap.add_argument("--hidden_dim", type=int, required=True)
    ap.add_argument("--projection_dim", type=int, default=0)
    ap.add_argument("--conv_state_embedding_dim", type=int, default=0)
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--num_workers", type=int, required=True)
    ap.add_argument("--lr_actor", type=float, required=True)
    ap.add_argument("--lr_critic", type=float, required=True)
    ap.add_argument("--lr_beta", type=float, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--expectile", type=float, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--entropy_alpha", type=float, required=True)
    ap.add_argument("--cql_alpha", type=float, required=True)
    ap.add_argument("--bc_weight", type=float, required=True)
    ap.add_argument("--init_from_sft", action="store_true")
    ap.add_argument("--init_critic_fusion_from_sft", action="store_true")
    ap.add_argument("--actor_init_path", type=str, default="")
    ap.add_argument("--termination_init_path", type=str, default="")
    ap.add_argument("--actor_init_dir", type=str, default="")
    ap.add_argument("--plot_action_dist", action="store_true")
    ap.add_argument("--action_dist_top_k", type=int, required=True)
    ap.add_argument("--action_dist_max_batches", type=int, required=True)
    ap.add_argument("--seed", type=int, default=48)
    args = ap.parse_args()

    set_seed(int(args.seed))
    root = DataRoot(Path(args.root).resolve())
    out_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    extr_cfg = ExtrinsicConfig()
    context_turns = None if int(args.context_turns) < 0 else int(args.context_turns)
    tokenizer = None
    if args.state_repr == "encoder":
        assert str(args.encoder_name).strip()
        tokenizer = AutoTokenizer.from_pretrained(str(args.encoder_name), use_fast=True)
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "right"

    if args.task == "policy_over_options":
        if args.state_repr == "encoder":
            feat = build_cached_option_utterance_rollout_encoder(
                cache_dir=cache_dir,
                root=root,
                tokenizer=tokenizer,
                encoder_name=str(args.encoder_name),
                max_length=int(args.max_length),
                context_turns=context_turns,
                gamma=float(args.gamma),
                use_conv_state=bool(args.use_conv_state),
                extr_cfg=extr_cfg,
                encode_batch_size=int(args.encoder_batch_size),
                conv_state_adapter_dir=str(args.conv_state_adapter_dir),
                conv_state_meta_path=str(args.conv_state_meta_path),
                conv_state_encoding=str(args.conv_state_encoding),
            )
        else:
            feat = build_cached_option_utterance_rollout_activations(
                cache_dir=cache_dir,
                root=root,
                activation_path=str(args.activation_path),
                activation_layer=int(args.activation_layer),
                activation_key=str(args.activation_key),
                gamma=float(args.gamma),
                use_conv_state=bool(args.use_conv_state),
                extr_cfg=extr_cfg,
                conv_state_encoding=str(args.conv_state_encoding),
                conv_state_source=str(args.conv_state_source),
                conv_state_sft_dir=str(args.conv_state_sft_dir),
                probe_checkpoint=str(args.probe_checkpoint),
                probe_batch_size=int(args.probe_batch_size),
            )
        _train_one(feat, out_dir, args, with_term=True, log_dir=log_dir)
    elif args.task == "flat_rl":
        if args.state_repr == "encoder":
            feat = build_cached_flat_rl_rollout_encoder(
                cache_dir=cache_dir,
                root=root,
                tokenizer=tokenizer,
                encoder_name=str(args.encoder_name),
                max_length=int(args.max_length),
                context_turns=context_turns,
                gamma=float(args.gamma),
                use_conv_state=bool(args.use_conv_state),
                extr_cfg=extr_cfg,
                encode_batch_size=int(args.encoder_batch_size),
                conv_state_encoding=str(args.conv_state_encoding),
            )
        else:
            feat = build_cached_flat_rl_rollout_activations(
                cache_dir=cache_dir,
                root=root,
                activation_path=str(args.activation_path),
                activation_layer=int(args.activation_layer),
                activation_key=str(args.activation_key),
                gamma=float(args.gamma),
                use_conv_state=bool(args.use_conv_state),
                extr_cfg=extr_cfg,
                conv_state_encoding=str(args.conv_state_encoding),
                conv_state_source=str(args.conv_state_source),
                conv_state_sft_dir=str(args.conv_state_sft_dir),
                probe_checkpoint=str(args.probe_checkpoint),
                probe_batch_size=int(args.probe_batch_size),
            )
        _train_one(feat, out_dir, args, with_term=False, log_dir=log_dir)
    else:
        action_space = load_action_space(root)
        macro2id, _ = build_micro_action_maps(action_space)
        for macro_action in macro2id:
            macro_out = out_dir / f"micro_{safe_name(str(macro_action))}"
            macro_log = log_dir / f"micro_{safe_name(str(macro_action))}"
            if args.state_repr == "encoder":
                feat = build_cached_intra_option_rollout_encoder(
                    cache_dir=cache_dir,
                    root=root,
                    tokenizer=tokenizer,
                    encoder_name=str(args.encoder_name),
                    max_length=int(args.max_length),
                    context_turns=context_turns,
                    macro_action=str(macro_action),
                    gamma=float(args.gamma),
                    use_conv_state=bool(args.use_conv_state),
                    encode_batch_size=int(args.encoder_batch_size),
                    conv_state_encoding=str(args.conv_state_encoding),
                )
            else:
                feat = build_cached_intra_option_rollout_activations(
                    cache_dir=cache_dir,
                    root=root,
                    activation_path=str(args.activation_path),
                    activation_layer=int(args.activation_layer),
                    activation_key=str(args.activation_key),
                    macro_action=str(macro_action),
                    gamma=float(args.gamma),
                    use_conv_state=bool(args.use_conv_state),
                    conv_state_encoding=str(args.conv_state_encoding),
                    conv_state_source=str(args.conv_state_source),
                    conv_state_sft_dir=str(args.conv_state_sft_dir),
                    probe_checkpoint=str(args.probe_checkpoint),
                    probe_batch_size=int(args.probe_batch_size),
                )
            if str(args.actor_init_dir).strip():
                args.actor_init_path = str(Path(args.actor_init_dir) / f"intra_option_{safe_name(str(macro_action))}" / "actor.pt")
            _train_one(feat, macro_out, args, with_term=False, log_dir=macro_log)


if __name__ == "__main__":
    main()
