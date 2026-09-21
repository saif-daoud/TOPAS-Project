from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..activations import ActivationSpec, load_layer_matrix
from ..cv_state_probe import append_frozen_probe_features, append_sft_conv_state_features
from ..sft_pretraining.activation_sft_utils import build_conv_state_vectors
from ..sft_pretraining.data_loading import (
    DataRoot,
    build_flat_micro_label_map,
    build_macro_action_map,
    build_micro_action_maps,
    load_action_space,
)
from ..sft_pretraining.feature_cache import CacheKey, cache_or_build, fingerprint_dict
from ..sft_pretraining.utils import add_preferred_turn_index, termination_labels_from_options
from .reward_loading import ExtrinsicConfig, load_extrinsic_reward_csv, load_intrinsic_reward_csv

import os

def _max_train_utterances():
    v = os.environ.get("TOPA_RL_MAX_UTTERANCES", "").strip()
    if not v:
        return None
    return int(v)

def _filter_first_utterances(df, col="turn_idx"):
    H = _max_train_utterances()
    if H is None:
        return df
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    df[col] = df[col].astype(int)
    return df[df[col] < H].reset_index(drop=True)

def _filter_first_utterance_phases(df, from_col="from_turn_idx", to_col="to_turn_idx"):
    H = _max_train_utterances()
    if H is None:
        return df
    df = df.copy()
    df[from_col] = pd.to_numeric(df[from_col], errors="coerce")
    df[to_col] = pd.to_numeric(df[to_col], errors="coerce")
    df = df.dropna(subset=[from_col, to_col])
    df[from_col] = df[from_col].astype(int)
    df[to_col] = df[to_col].astype(int)

    # keep only phases that start before the truncation point
    df = df[df[from_col] < H].copy()

    # clip phases that cross the truncation boundary
    df[to_col] = df[to_col].clip(upper=H - 1)

    return df.reset_index(drop=True)

@dataclass
class OfflineRolloutFeatures:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    labels: np.ndarray
    extra: Dict[str, object]


def _compute_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        if dones[i] == 1:
            g = 0.0
        g = float(rewards[i]) + float(gamma) * g
        returns[i] = g
    return returns


def _append_generated_conv_state_features(
    input_embeds_np: np.ndarray,
    *,
    use_conv_state: bool,
    conv_source: str,
    conv_state_dim: int,
    conv_state_encoding: str,
    conv_state_sft_dir: str,
    probe_checkpoint: str,
    probe_batch_size: int,
) -> tuple[np.ndarray, int, int]:
    """Append generated conv-state features from either the frozen probe or SFT model.

    Existing gold/annotation features are already concatenated before this helper
    is called; in that case this simply returns the original matrix.
    """
    input_embeds_np = np.asarray(input_embeds_np, dtype=np.float32)
    latent_dim = int(input_embeds_np.shape[1]) - int(conv_state_dim)
    if not bool(use_conv_state):
        return input_embeds_np, latent_dim, int(conv_state_dim)

    conv_source = str(conv_source).strip().lower()
    if conv_source == "probe":
        if not str(probe_checkpoint).strip():
            raise ValueError("conv_state_source=probe requires --probe_checkpoint")
        latent_dim = int(input_embeds_np.shape[1])
        input_embeds_np, conv_state_dim = append_frozen_probe_features(
            input_embeds_np,
            checkpoint_path=str(probe_checkpoint),
            batch_size=int(probe_batch_size),
            include_probe_hidden=False,
        )
    elif conv_source == "sft":
        if not str(conv_state_sft_dir).strip():
            raise ValueError("conv_state_source=sft requires --conv_state_sft_dir")
        latent_dim = int(input_embeds_np.shape[1])
        input_embeds_np, conv_state_dim, _ = append_sft_conv_state_features(
            input_embeds_np,
            model_dir=str(conv_state_sft_dir),
            encoding=str(conv_state_encoding),
            batch_size=int(probe_batch_size),
        )

    return input_embeds_np.astype(np.float32), int(latent_dim), int(conv_state_dim)


def build_cached_option_utterance_rollout_activations(
    cache_dir: Path,
    root: DataRoot,
    activation_path: str,
    activation_layer: int,
    activation_key: str,
    gamma: float,
    use_conv_state: bool,
    extr_cfg: ExtrinsicConfig,
    conv_state_encoding: str = "one_hot",
    conv_state_source: str = "gold",
    conv_state_sft_dir: str = "",
    probe_checkpoint: str = "",
    probe_batch_size: int = 1024,
    force_rebuild: bool = False,
) -> OfflineRolloutFeatures:
    act_root = ActivationSpec.resolve(root.root, activation_path).root
    conv_source = str(conv_state_source).strip().lower()
    use_generated_conv_state = bool(use_conv_state) and conv_source in {"probe", "sft"}
    meta = {
        "kind": "option_utterance_offline_activations",
        "activation_root": str(act_root),
        "activation_layer": int(activation_layer),
        "activation_key": str(activation_key),
        "gamma": float(gamma),
        "use_conv_state": int(bool(use_conv_state)),
        "threshold": float(extr_cfg.threshold),
        "high_threshold": float(extr_cfg.high_threshold),
        "reward_mapping": "continuous_overall_mean_div_6_clipped_0_6",
        # "reward_mapping": "0_below_threshold_1_below_high_2_at_high",
        "extr_reward_key": str(extr_cfg.reward_key or ""),
        "indexing": "preferred_id_turn_index",
        "termination_labeling": "next_option_change_only",
        "conv_state_encoding": str(conv_state_encoding),
        "conv_state_source": str(conv_source),
        "conv_state_sft_dir": str(conv_state_sft_dir),
        "probe_checkpoint": str(probe_checkpoint),
        "probe_include_hidden": 0,
    }
    key = CacheKey("option_utterance_offline_activations", fingerprint_dict(meta))

    def _build() -> OfflineRolloutFeatures:
        df = pd.read_csv(root.annotations_dir / "annotations_micro_actions_refined.csv")
        df["user_idx"] = df["user_idx"].astype(int)
        df["session_idx"] = df["session_idx"].astype(int)
        df = add_preferred_turn_index(df, "turn_idx", ["utterance_id", "turn_id", "from_id", "to_id"], ["utterance_idx"])
        df = df.dropna(subset=["turn_idx"]).reset_index(drop=True)
        df["turn_idx"] = df["turn_idx"].astype(int)
        df = df.sort_values(["user_idx", "session_idx", "turn_idx"]).reset_index(drop=True)

        action_space = load_action_space(root)
        macro2id = build_macro_action_map(action_space)
        num_options = int(max(macro2id.values()) + 1)

        extra_vecs = None
        conv_state_dim = 0
        if use_conv_state and not use_generated_conv_state:
            extra_vecs, _ = build_conv_state_vectors(root, df, encoding=str(conv_state_encoding))
            conv_state_dim = int(extra_vecs.shape[1])

        df_extr = load_extrinsic_reward_csv(root.annotations_dir / "annotations_extrinsic_reward.csv", cfg=extr_cfg)
        extr_map = {
            (int(r["user_idx"]), int(r["session_idx"])): float(r["extrinsic_reward"])
            for r in df_extr.to_dict(orient="records")
        }

        sess_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        input_embeds: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[int] = []
        term_labels: List[int] = []
        episode_ids: List[int] = []

        ep = 0
        for (u, s), sub in tqdm(df.groupby(["user_idx", "session_idx"], sort=False), desc="[cache] build option offline", unit="sess"):
            u = int(u)
            s = int(s)
            if (u, s) not in sess_cache:
                sess_cache[(u, s)] = load_layer_matrix(act_root, u, s, activation_layer, key=activation_key, cache=None)
            layer_mat = sess_cache[(u, s)]

            local_vecs: List[np.ndarray] = []
            local_actions: List[int] = []
            for idx, row in sub.iterrows():
                macro_name = str(row["selected_macro_action"])
                if macro_name not in macro2id:
                    continue
                utt = int(row["turn_idx"])
                if utt < 0 or utt >= int(layer_mat.shape[0]):
                    continue
                vec = layer_mat[utt].detach().to(dtype=torch.float32).cpu().numpy()
                if extra_vecs is not None:
                    vec = np.concatenate([vec, extra_vecs[int(idx)].astype(np.float32)], axis=0)
                local_vecs.append(vec.astype(np.float32))
                local_actions.append(int(macro2id[macro_name]))

            if not local_actions:
                continue

            local_terms = termination_labels_from_options(local_actions)
            for i, action_id in enumerate(local_actions):
                done = int(i == len(local_actions) - 1)
                input_embeds.append(local_vecs[i])
                actions.append(action_id)
                rewards.append(float(extr_map[(u, s)]) if done else 0.0)
                dones.append(done)
                term_labels.append(local_terms[i])
                episode_ids.append(ep)
            ep += 1

        input_embeds_np = np.stack(input_embeds, axis=0).astype(np.float32)
        input_embeds_np, latent_dim, conv_state_dim = _append_generated_conv_state_features(
            input_embeds_np,
            use_conv_state=bool(use_conv_state),
            conv_source=str(conv_source),
            conv_state_dim=int(conv_state_dim),
            conv_state_encoding=str(conv_state_encoding),
            conv_state_sft_dir=str(conv_state_sft_dir),
            probe_checkpoint=str(probe_checkpoint),
            probe_batch_size=int(probe_batch_size),
        )
        actions_np = np.asarray(actions, dtype=np.int64)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.int64)
        extra = {
            "input_embeds": input_embeds_np,
            "latent_dim": int(latent_dim),
            "conv_state_dim": int(conv_state_dim),
            "actions": actions_np,
            "rewards": rewards_np,
            "dones": dones_np,
            "returns": _compute_returns(rewards_np, dones_np, gamma=float(gamma)),
            "term_labels": np.asarray(term_labels, dtype=np.float32),
            "episode_id": np.asarray(episode_ids, dtype=np.int64),
            "label2id": macro2id,
            "num_labels": num_options,
        }

        dummy_input_ids = np.zeros((len(actions_np), 1), dtype=np.int64)
        dummy_attn = np.ones((len(actions_np), 1), dtype=np.int64)
        return OfflineRolloutFeatures(input_ids=dummy_input_ids, attention_mask=dummy_attn, labels=actions_np.copy(), extra=extra)

    feat = cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)
    feat.extra["_cache_key"] = key.filename()
    feat.extra["_cache_dir"] = str(Path(cache_dir).resolve())
    feat.extra["_cache_meta"] = dict(meta)
    return feat


def _load_session_acts_np(act_root: Path, user_idx: int, session_idx: int, layer_idx: int, activation_key: str) -> np.ndarray:
    mat = load_layer_matrix(act_root, user_idx, session_idx, layer_idx, key=activation_key, cache=None)
    return mat.detach().to(dtype=torch.float32).cpu().numpy()


def build_cached_intra_option_rollout_activations(
    cache_dir: Path,
    root: DataRoot,
    activation_path: str,
    activation_layer: int,
    activation_key: str,
    macro_action: str,
    gamma: float,
    use_conv_state: bool = False,
    conv_state_encoding: str = "one_hot",
    conv_state_source: str = "gold",
    conv_state_sft_dir: str = "",
    probe_checkpoint: str = "",
    probe_batch_size: int = 1024,
    intrinsic_center: bool = True,
    force_rebuild: bool = False,
) -> OfflineRolloutFeatures:
    act_root = ActivationSpec.resolve(root.root, activation_path).root
    conv_source = str(conv_state_source).strip().lower()
    use_generated_conv_state = bool(use_conv_state) and conv_source in {"probe", "sft"}
    meta = {
        "kind": "intra_option_offline_activations",
        "macro_action": str(macro_action),
        "activation_root": str(act_root),
        "activation_layer": int(activation_layer),
        "activation_key": str(activation_key),
        "gamma": float(gamma),
        "use_conv_state": int(bool(use_conv_state)),
        "intrinsic_center": int(bool(intrinsic_center)),
        "indexing": "preferred_id_turn_index",
        "conv_state_encoding": str(conv_state_encoding),
        "conv_state_source": str(conv_source),
        "conv_state_sft_dir": str(conv_state_sft_dir),
        "probe_checkpoint": str(probe_checkpoint),
        "probe_include_hidden": 0,
    }
    key = CacheKey("intra_option_offline_activations", fingerprint_dict(meta))

    def _build() -> OfflineRolloutFeatures:
        df_micro = pd.read_csv(root.annotations_dir / "annotations_micro_actions_refined.csv")
        df_micro["user_idx"] = df_micro["user_idx"].astype(int)
        df_micro["session_idx"] = df_micro["session_idx"].astype(int)
        df_micro = add_preferred_turn_index(df_micro, "turn_idx", ["utterance_id", "turn_id", "from_id", "to_id"], ["utterance_idx"])
        df_micro = df_micro.dropna(subset=["turn_idx"]).reset_index(drop=True)
        df_micro["turn_idx"] = df_micro["turn_idx"].astype(int)
        df_micro = df_micro[df_micro["selected_macro_action"].astype(str) == str(macro_action)].reset_index(drop=True)

        extra_vecs = None
        conv_state_dim = 0
        if use_conv_state and not use_generated_conv_state:
            extra_vecs, _ = build_conv_state_vectors(root, df_micro, encoding=str(conv_state_encoding))
            conv_state_dim = int(extra_vecs.shape[1])

        action_space = load_action_space(root)
        macro2id, macro_id_to_micro2id = build_micro_action_maps(action_space)
        mid = macro2id[str(macro_action)]
        micro2id = macro_id_to_micro2id[mid]

        df_intr = load_intrinsic_reward_csv(root.annotations_dir / "annotations_intrinsic_reward.csv")
        df_intr = df_intr[df_intr["selected_macro_action"].astype(str) == str(macro_action)].copy()
        df_intr = add_preferred_turn_index(df_intr, "from_turn_idx", ["from_id"], ["from_idx"])
        df_intr = add_preferred_turn_index(df_intr, "to_turn_idx", ["to_id"], ["to_idx"])
        df_intr = df_intr.dropna(subset=["from_turn_idx", "to_turn_idx"]).reset_index(drop=True)
        df_intr["from_turn_idx"] = df_intr["from_turn_idx"].astype(int)
        df_intr["to_turn_idx"] = df_intr["to_turn_idx"].astype(int)



        df_intr = df_intr.sort_values(["user_idx", "session_idx", "from_turn_idx", "to_turn_idx"]).reset_index(drop=True)

        sess_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        input_embeds: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[int] = []
        episode_ids: List[int] = []

        ep = 0
        for pr in tqdm(df_intr.to_dict(orient="records"), desc=f"[cache] build intra-option offline ({macro_action})", unit="phase"):
            u = int(pr["user_idx"])
            s = int(pr["session_idx"])
            from_i = int(pr["from_turn_idx"])
            to_i = int(pr["to_turn_idx"])
            sub = df_micro[
                (df_micro["user_idx"] == u)
                & (df_micro["session_idx"] == s)
                & (df_micro["turn_idx"] >= from_i)
                & (df_micro["turn_idx"] <= to_i)
            ].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values("turn_idx").reset_index()
            r_phase = float(pr["reward"])
            if intrinsic_center:
                r_phase = r_phase - 1.0

            if (u, s) not in sess_cache:
                sess_cache[(u, s)] = load_layer_matrix(act_root, u, s, activation_layer, key=activation_key, cache=None)
            layer_mat = sess_cache[(u, s)]

            for i, row in sub.iterrows():
                micro_name = str(row["selected_micro_action"])
                if micro_name not in micro2id:
                    continue
                idx = int(row["turn_idx"])
                if idx < 0 or idx >= int(layer_mat.shape[0]):
                    continue
                vec = layer_mat[idx].detach().to(dtype=torch.float32).cpu().numpy()
                if extra_vecs is not None:
                    vec = np.concatenate([vec, extra_vecs[int(row["index"])].astype(np.float32)], axis=0)
                input_embeds.append(vec.astype(np.float32))
                actions.append(int(micro2id[micro_name]))
                is_last = i == len(sub) - 1
                rewards.append(float(r_phase) if is_last else 0.0)
                dones.append(1 if is_last else 0)
                episode_ids.append(ep)
            ep += 1

        input_embeds_np = np.stack(input_embeds, axis=0).astype(np.float32)
        input_embeds_np, latent_dim, conv_state_dim = _append_generated_conv_state_features(
            input_embeds_np,
            use_conv_state=bool(use_conv_state),
            conv_source=str(conv_source),
            conv_state_dim=int(conv_state_dim),
            conv_state_encoding=str(conv_state_encoding),
            conv_state_sft_dir=str(conv_state_sft_dir),
            probe_checkpoint=str(probe_checkpoint),
            probe_batch_size=int(probe_batch_size),
        )
        actions_np = np.asarray(actions, dtype=np.int64)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.int64)
        extra = {
            "input_embeds": input_embeds_np,
            "latent_dim": int(latent_dim),
            "conv_state_dim": int(conv_state_dim),
            "actions": actions_np,
            "rewards": rewards_np,
            "dones": dones_np,
            "returns": _compute_returns(rewards_np, dones_np, gamma=float(gamma)),
            "episode_id": np.asarray(episode_ids, dtype=np.int64),
            "label2id": micro2id,
            "num_labels": int(len(micro2id)),
        }
        dummy_input_ids = np.zeros((len(actions_np), 1), dtype=np.int64)
        dummy_attn = np.ones((len(actions_np), 1), dtype=np.int64)
        return OfflineRolloutFeatures(input_ids=dummy_input_ids, attention_mask=dummy_attn, labels=actions_np.copy(), extra=extra)

    feat = cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)
    feat.extra["_cache_key"] = key.filename()
    feat.extra["_cache_dir"] = str(Path(cache_dir).resolve())
    feat.extra["_cache_meta"] = dict(meta)
    return feat


def build_cached_flat_rl_rollout_activations(
    cache_dir: Path,
    root: DataRoot,
    activation_path: str,
    activation_layer: int,
    activation_key: str,
    gamma: float,
    use_conv_state: bool = False,
    conv_state_encoding: str = "one_hot",
    conv_state_source: str = "gold",
    conv_state_sft_dir: str = "",
    probe_checkpoint: str = "",
    probe_batch_size: int = 1024,
    extr_cfg: ExtrinsicConfig = ExtrinsicConfig(),
    force_rebuild: bool = False,
) -> OfflineRolloutFeatures:
    act_root = ActivationSpec.resolve(root.root, activation_path).root
    conv_source = str(conv_state_source).strip().lower()
    use_generated_conv_state = bool(use_conv_state) and conv_source in {"probe", "sft"}
    meta = {
        "kind": "flat_rl_offline_activations",
        "activation_root": str(act_root),
        "activation_layer": int(activation_layer),
        "activation_key": str(activation_key),
        "gamma": float(gamma),
        "use_conv_state": int(bool(use_conv_state)),
        "extr_threshold": float(extr_cfg.threshold),
        "extr_high_threshold": float(extr_cfg.high_threshold),
        # "reward_mapping": "0_below_threshold_1_below_high_2_at_high",
        "reward_mapping": "continuous_overall_mean_div_6_clipped_0_6",
        "extr_reward_key": str(extr_cfg.reward_key or ""),
        "indexing": "preferred_id_turn_index",
        "conv_state_encoding": str(conv_state_encoding),
        "conv_state_source": str(conv_source),
        "conv_state_sft_dir": str(conv_state_sft_dir),
        "probe_checkpoint": str(probe_checkpoint),
        "probe_include_hidden": 0,
    }
    key = CacheKey("flat_rl_offline_activations", fingerprint_dict(meta))

    def _build() -> OfflineRolloutFeatures:
        df_micro = pd.read_csv(root.annotations_dir / "annotations_micro_actions_refined.csv")
        df_micro["user_idx"] = df_micro["user_idx"].astype(int)
        df_micro["session_idx"] = df_micro["session_idx"].astype(int)
        df_micro = add_preferred_turn_index(df_micro, "turn_idx", ["utterance_id", "turn_id", "from_id", "to_id"], ["utterance_idx"])
        df_micro = df_micro.dropna(subset=["turn_idx"]).reset_index(drop=True)
        df_micro["turn_idx"] = df_micro["turn_idx"].astype(int)


        df_micro = df_micro.sort_values(["user_idx", "session_idx", "turn_idx"]).reset_index(drop=True)

        extra_vecs = None
        conv_state_dim = 0
        if use_conv_state and not use_generated_conv_state:
            extra_vecs, _ = build_conv_state_vectors(root, df_micro, encoding=str(conv_state_encoding))
            conv_state_dim = int(extra_vecs.shape[1])

        action_space = load_action_space(root)
        class_names, label2id = build_flat_micro_label_map(action_space)
        df_extr = load_extrinsic_reward_csv(root.annotations_dir / "annotations_extrinsic_reward.csv", cfg=extr_cfg)
        extr_map: Dict[Tuple[int, int], float] = {
            (int(r["user_idx"]), int(r["session_idx"])): float(r["extrinsic_reward"])
            for r in df_extr.to_dict(orient="records")
        }

        sess_cache: Dict[Tuple[int, int], np.ndarray] = {}
        input_embeds: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[int] = []
        episode_ids: List[int] = []

        ep = 0
        for (u, s), sub in tqdm(df_micro.groupby(["user_idx", "session_idx"], sort=False), desc="[cache] build flat RL offline", unit="sess"):
            u = int(u)
            s = int(s)
            r_sess = float(extr_map[(u, s)])
            if (u, s) not in sess_cache:
                sess_cache[(u, s)] = _load_session_acts_np(act_root, u, s, activation_layer, activation_key)
            sess_vecs = sess_cache[(u, s)]

            local_embeds: List[np.ndarray] = []
            local_actions: List[int] = []
            for idx, row in sub.sort_values("turn_idx").iterrows():
                utt = int(row["turn_idx"])
                if utt < 0 or utt >= len(sess_vecs):
                    continue
                label = f"{str(row['selected_macro_action'])}::{str(row['selected_micro_action'])}"
                if label not in label2id:
                    continue
                vec = sess_vecs[utt].astype(np.float32)
                if extra_vecs is not None:
                    vec = np.concatenate([vec, extra_vecs[int(idx)].astype(np.float32)], axis=0)
                local_embeds.append(vec.astype(np.float32))
                local_actions.append(int(label2id[label]))

            if not local_actions:
                continue
            for i, action_id in enumerate(local_actions):
                is_last = i == len(local_actions) - 1
                input_embeds.append(local_embeds[i])
                actions.append(action_id)
                rewards.append(float(r_sess) if is_last else 0.0)
                dones.append(1 if is_last else 0)
                episode_ids.append(ep)
            ep += 1

        input_embeds_np = np.stack(input_embeds, axis=0).astype(np.float32)
        input_embeds_np, latent_dim, conv_state_dim = _append_generated_conv_state_features(
            input_embeds_np,
            use_conv_state=bool(use_conv_state),
            conv_source=str(conv_source),
            conv_state_dim=int(conv_state_dim),
            conv_state_encoding=str(conv_state_encoding),
            conv_state_sft_dir=str(conv_state_sft_dir),
            probe_checkpoint=str(probe_checkpoint),
            probe_batch_size=int(probe_batch_size),
        )
        actions_np = np.asarray(actions, dtype=np.int64)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.int64)
        extra = {
            "input_embeds": input_embeds_np,
            "latent_dim": int(latent_dim),
            "conv_state_dim": int(conv_state_dim),
            "actions": actions_np,
            "rewards": rewards_np,
            "dones": dones_np,
            "returns": _compute_returns(rewards_np, dones_np, gamma=float(gamma)),
            "episode_id": np.asarray(episode_ids, dtype=np.int64),
            "label2id": label2id,
            "num_labels": int(len(class_names)),
        }
        dummy_input_ids = np.zeros((len(actions_np), 1), dtype=np.int64)
        dummy_attn = np.ones((len(actions_np), 1), dtype=np.int64)
        return OfflineRolloutFeatures(input_ids=dummy_input_ids, attention_mask=dummy_attn, labels=actions_np.copy(), extra=extra)

    feat = cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)
    feat.extra["_cache_key"] = key.filename()
    feat.extra["_cache_dir"] = str(Path(cache_dir).resolve())
    feat.extra["_cache_meta"] = dict(meta)
    return feat
