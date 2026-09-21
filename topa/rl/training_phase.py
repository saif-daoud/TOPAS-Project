
"""TOPA RL phase (domain-agnostic) wiring.

The RL subproject in this repo (topa/rl/...) historically assumed a
CBT/AlexanderStreet layout. This phase makes it *domain agnostic* by:
  - building an RL "data root" under outputs/<domain>/... with the expected
    subfolders (data/ components/ annotations/)
  - copying refined components and annotations produced by earlier TOPA
    phases into that root

By default this phase prepares artifacts (prepare_only=True). When any RL block
declares activation states, it builds/checks the activation cache independently
of SFT/offline-RL training.
"""



import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Config
from .activations import (
    ActivationSpec,
    activation_cache_complete,
    infer_num_layers,
    resolve_activation_path,
    role_labels_for_root,
)
from .sft_pretraining.utils import safe_name



def _copytree_files(src_dir: Path, dst_dir: Path, pattern: str) -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.glob(pattern):
        shutil.copy2(p, dst_dir / p.name)
        n += 1
    return n


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _resolve_path(root: Path, value: Optional[str], default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_activation_layers(layer_spec: Any, activation_path: str, activation_key: str, root: Path) -> list[int]:
    """Parse activation_layer spec into a list of layer indices."""
    if isinstance(layer_spec, list):
        return [int(x) for x in layer_spec]
    if isinstance(layer_spec, str):
        s = layer_spec.strip()
        if s.lower() in {"all", "*"}:
            act_root = ActivationSpec.resolve(root, activation_path).root
            n_layers = infer_num_layers(act_root, key=activation_key)
            return list(range(int(n_layers)))
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if parts:
            return [int(p) for p in parts]
    try:
        return [int(layer_spec)]
    except Exception:
        return [-1]


def _require_keys(block: Dict[str, Any], keys: list[str], label: str) -> None:
    missing = [k for k in keys if k not in block]
    if missing:
        raise KeyError(f"{label} is missing required keys: {missing}")


def _validate_rl_config(params: Dict[str, Any]) -> None:
    _require_keys(params, ["prepare_only", "artifact_dir", "runs_dir", "cache_dir", "sft", "offline_rl"], "rl")

    sft = params["sft"]
    offline_rl = params["offline_rl"]

    _require_keys(
        sft,
        [
            "enabled",
            "state_repr",
            "conv_state_encoding",
            "epochs",
            "batch_size",
            "num_workers",
            "lr",
            "run_macro_no_conv",
            "run_conv_state",
            "run_macro_with_conv",
            "run_micro_flat",
            "run_micro_independent",
            "run_termination",
        ],
        "rl.sft",
    )
    sft_state = str(sft["state_repr"]).strip().lower()
    if sft_state not in {"encoder", "activations"}:
        raise ValueError("rl.sft.state_repr must be 'encoder' or 'activations'")
    if str(sft["conv_state_encoding"]).strip().lower() not in {"one_hot", "label"}:
        raise ValueError("rl.sft.conv_state_encoding must be 'one_hot' or 'label'")
    if sft_state == "activations":
        _require_keys(sft, ["activation_layer", "activation_key", "mlp_hidden_dim", "mlp_projection_dim", "mlp_conv_state_embedding_dim", "mlp_dropout"], "rl.sft")
    else:
        _require_keys(sft, ["encoder_name", "amp"], "rl.sft")

    _require_keys(
        offline_rl,
        [
            "enabled",
            "algorithm",
            "state_repr",
            "conv_state_encoding",
            "init_from_sft",
            "init_critic_fusion_from_sft",
            "mlp_hidden_dim",
            "mlp_projection_dim",
            "mlp_conv_state_embedding_dim",
            "mlp_dropout",
            "epochs",
            "batch_size",
            "num_workers",
            "plot_action_dist",
            "action_dist_top_k",
            "action_dist_max_batches",
            "run_policy_over_options_no_conv",
            "run_policy_over_options_with_conv",
            "run_flat_rl",
            "run_intra_option_policies",
            "use_conv_state_for_intra_options",
            "iql",
            "cql",
        ],
        "rl.offline_rl",
    )
    algo = str(offline_rl["algorithm"]).strip().lower()
    if algo not in {"iql", "cql"}:
        raise ValueError("rl.offline_rl.algorithm must be 'iql' or 'cql'")
    offline_state = str(offline_rl["state_repr"]).strip().lower()
    if offline_state not in {"encoder", "activations"}:
        raise ValueError("rl.offline_rl.state_repr must be 'encoder' or 'activations'")
    if str(offline_rl["conv_state_encoding"]).strip().lower() not in {"one_hot", "label"}:
        raise ValueError("rl.offline_rl.conv_state_encoding must be 'one_hot' or 'label'")
    if offline_state == "activations":
        _require_keys(offline_rl, ["activation_layer", "activation_key"], "rl.offline_rl")
    else:
        _require_keys(offline_rl, ["encoder_batch_size"], "rl.offline_rl")

    _require_keys(offline_rl["iql"], ["lr_actor", "lr_critic", "lr_beta", "gamma", "expectile", "temperature"], "rl.offline_rl.iql")
    _require_keys(offline_rl["cql"], ["lr_actor", "lr_critic", "lr_beta", "gamma", "entropy_alpha", "cql_alpha", "bc_weight"], "rl.offline_rl.cql")




def _resolve_conv_state_runtime(
    block: Dict[str, Any],
    *,
    fallback_block: Optional[Dict[str, Any]] = None,
    default_source: str = "gold",
    label: str = "conv_state",
) -> tuple[str, str, str, int]:
    """Resolve conv-state predictor/fusion settings for config-driven runs.

    The YAML commonly defines the predictor once under ``rl.sft``.  Offline RL
    may omit the same fields; in that case we intentionally inherit them from
    SFT so one config value controls both SFT and offline RL.
    """
    fallback_block = fallback_block or {}

    source = str(block.get("conv_state_source", fallback_block.get("conv_state_source", default_source))).strip().lower()
    conv_state_sft_dir = str(block.get("conv_state_sft_dir", fallback_block.get("conv_state_sft_dir", ""))).strip()
    probe_checkpoint = str(block.get("probe_checkpoint", fallback_block.get("probe_checkpoint", ""))).strip()
    probe_batch_size = int(block.get("probe_batch_size", fallback_block.get("probe_batch_size", 1024)))

    if source not in {"gold", "sft", "probe"}:
        raise ValueError(f"{label}.conv_state_source must be one of: gold, sft, probe")
    if source == "probe" and not probe_checkpoint:
        raise ValueError(f"{label}.probe_checkpoint is required when conv_state_source=probe")

    return source, conv_state_sft_dir, probe_checkpoint, probe_batch_size

def _best_val_acc_from_dir(out_dir: Path) -> Optional[float]:
    mpath = out_dir / "metrics.json"
    if not mpath.exists():
        return None
    try:
        d = _read_json(mpath)
        for k in ["best_val_acc", "best_val_mean_acc", "best_val_accuracy"]:
            if k in d:
                return float(d[k])
    except Exception:
        return None
    return None


def _run_activation_sanity_checks(root: Path, act_root: Path, env: Dict[str, str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "topa.rl.extract_internal_representation.check_indexing",
        "--root",
        str(root),
        "--activations_dir",
        str(act_root),
    ]
    subprocess.run(cmd, check=True, env=env)

def _maybe_build_activations(
    *,
    root: Path,
    activation_key: str,
    cfg: Dict[str, Any],
    env: Dict[str, str],
) -> str:
    model_id = str(cfg["activation_model_id"]).strip()
    model_id = model_id or None

    act_path = resolve_activation_path(root, str(cfg["activation_out_dir"]), model_id)
    act_root = ActivationSpec.resolve(root, str(act_path)).root
    if activation_cache_complete(root, act_root, env["TOPA_DATASET_PATH"]):
        _run_activation_sanity_checks(root, act_root, env)
        return str(act_root)

    if not model_id:
        raise ValueError(
            "Activation cache missing. Set rl.activation_model_id "
            "to auto-extract activations, or provide a valid activation_out_dir."
        )

    transcripts_dir = _resolve_path(root, cfg["activation_transcripts_dir"], root / "extract_internal_representation" / "transcripts")
    out_dir = act_root
    labels = role_labels_for_root(root)
    system_label = str(labels["system"])
    user_label = str(labels["user"])
    system_tag = f"<|{system_label}|>"
    user_tag = f"<|{user_label}|>"
    id_dir_prefix = safe_name(user_label)
    dtype = str(cfg["activation_dtype"])
    device_map = str(cfg["activation_device_map"])
    select_layers = str(cfg["activation_select_layers"]).strip()

    cmd1 = [
        sys.executable,
        "-m",
        "topa.rl.extract_internal_representation.preprocess_sessions",
        "--root",
        str(root),
        "--out_dir",
        str(transcripts_dir),
        "--system_tag",
        system_tag,
        "--user_tag",
        user_tag,
        "--id_dir_prefix",
        id_dir_prefix,
    ]

    cmd2 = [
        sys.executable,
        "-m",
        "topa.rl.extract_internal_representation.extract_internal_activations",
        "--root",
        str(root),
        "--transcripts_dir",
        str(transcripts_dir),
        "--out_dir",
        str(out_dir),
        "--model_id",
        str(model_id),
        "--dtype",
        dtype,
        "--device_map",
        device_map,
        "--system_tag",
        system_tag,
        "--user_tag",
        user_tag,
    ]
    if select_layers:
        cmd2 += ["--select_layers", select_layers]

    subprocess.run(cmd1, check=True, env=env)
    subprocess.run(cmd2, check=True, env=env)
    _run_activation_sanity_checks(root, out_dir, env)

    return str(out_dir)

def _mean_micro_acc(out_root: Path) -> Optional[float]:
    if not out_root.exists():
        return None
    vals: list[float] = []
    for p in list(out_root.glob("intra_option_*")) + list(out_root.glob("micro_*")):
        if not p.is_dir():
            continue
        v = _best_val_acc_from_dir(p)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


@dataclass
class RLArtifacts:
    rl_root: Path
    data_json: Path
    components_dir: Path
    annotations_dir: Path

    def to_json(self) -> Dict[str, str]:
        return {
            "rl_root": str(self.rl_root),
            "data_json": str(self.data_json),
            "components_dir": str(self.components_dir),
            "annotations_dir": str(self.annotations_dir),
        }


class RLTrainingPhase:
    """Prepare RL artifacts and optionally launch training."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.domain = cfg.domain_params["adj"] or cfg.domain

    def prepare(
        self,
        *,
        refined_components: str,
        annotations: str,
    ) -> RLArtifacts:
        """Create a self-contained RL root directory.

        Args:
            refined_components: path returned by RefinementPhase.run (typically .../refined_components)
            annotations: path returned by AnnotationManager.annotate (typically .../annotations)
        """

        cache_dir = Path(self.cfg.output_path) / self.domain
        artifact_dir = str(self.cfg.rl_params["artifact_dir"])
        rl_root = (cache_dir / artifact_dir).resolve()

        data_dir = rl_root / "data"
        components_dir = rl_root / "components"
        annotations_dir = rl_root / "annotations"
        data_dir.mkdir(parents=True, exist_ok=True)
        components_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # 1) Data JSON (copy into rl_root/data/)
        data_src = Path(self.cfg.data_path).expanduser().resolve()
        if data_src.is_dir():
            # If a folder is provided, pick the first json file (or error)
            json_files = sorted(list(data_src.glob("*.json")))
            if not json_files:
                raise FileNotFoundError(f"No *.json dataset found in data directory: {data_src}")
            data_src = json_files[0]
        data_json = data_dir / data_src.name
        _copy_file(data_src, data_json)
        # Copy user split file if present next to the dataset
        user_label = str(self.cfg.domain_params["user"]).strip().lower().replace(" ", "_")
        split_names = [f"{user_label}_split.json", "user_split.json"]
        for name in split_names:
            cand = data_src.parent / name
            if cand.exists():
                _copy_file(cand, data_dir / cand.name)
                break

        # 2) Components (copy refined components)
        comp_src = Path(refined_components).expanduser().resolve()
        if comp_src.is_file():
            comp_src = comp_src.parent
        _copytree_files(comp_src, components_dir, "*.json")

        # Ensure `action_space.json` exists for older RL scripts
        micro_actions = components_dir / "micro_actions.json"
        action_space = components_dir / "action_space.json"
        if not action_space.exists() and micro_actions.exists():
            shutil.copy2(micro_actions, action_space)

        # 3) Annotations (copy csv files)
        ann_src = Path(annotations).expanduser().resolve()
        if ann_src.is_file():
            ann_src = ann_src.parent
        _copytree_files(ann_src, annotations_dir, "*.csv")
        _copytree_files(ann_src, annotations_dir, "*.json")

        # Helpful manifest
        (rl_root / "manifest.txt").write_text(
            "\n".join(
                [
                    f"domain={self.domain}",
                    f"data_src={data_src}",
                    f"refined_components_src={comp_src}",
                    f"annotations_src={ann_src}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        # Role labels for dataset parsing (used by SFT/RL helpers).
        role_labels = {
            "system": str(self.cfg.domain_params["system"]),
            "user": str(self.cfg.domain_params["user"]),
        }
        (rl_root / "role_labels.json").write_text(
            json.dumps(role_labels, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return RLArtifacts(
            rl_root=rl_root,
            data_json=data_json,
            components_dir=components_dir,
            annotations_dir=annotations_dir,
        )

    def run(
        self,
        *,
        extracted_components: Optional[str] = None,
        refined_components: str,
        annotations: str,
    ) -> Dict[str, Any]:
        """Prepare artifacts and optionally execute training scripts.

        Training is controlled by cfg.rl_params. By default, we only prepare.
        """

        artifacts = self.prepare(refined_components=refined_components, annotations=annotations)

        params = self.cfg.rl_params
        _validate_rl_config(params)
        prepare_only = bool(params["prepare_only"])
        seed = int(self.cfg.seed)
        out: Dict[str, Any] = {"artifacts": artifacts.to_json(), "prepare_only": prepare_only}

        env = os.environ.copy()
        # Point RL loaders at the prepared root/dataset.
        env.setdefault("TOPA_DATASET_PATH", str(artifacts.data_json))
        env.setdefault("TOPA_DATA_DIR", str(artifacts.rl_root / "data"))
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env["PYTHONHASHSEED"] = str(seed)
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Role labels are written to <rl_root>/role_labels.json (no env needed).

        sft = params["sft"]
        offline_rl = params["offline_rl"]
        sft_uses_activations = str(sft["state_repr"]).strip().lower() == "activations"
        offline_rl_uses_activations = str(offline_rl["state_repr"]).strip().lower() == "activations"
        sft_enabled = bool(sft["enabled"])
        offline_rl_enabled = bool(offline_rl["enabled"])

        activation_dir: Optional[str] = None
        if sft_uses_activations or offline_rl_uses_activations:
            source_cfg = sft if sft_uses_activations else offline_rl
            activation_dir = _maybe_build_activations(
                root=artifacts.rl_root,
                activation_key=str(source_cfg["activation_key"]),
                cfg=params,
                env=env,
            )
            out["activation_out_dir"] = activation_dir

        if prepare_only or not (sft_enabled or offline_rl_enabled):
            return out

        # Optional training orchestration (SFT then offline RL).
        # We mirror the RL scripts in topa/rl/scripts for consistency.
        runs_dir = _resolve_path(artifacts.rl_root, params["runs_dir"], artifacts.rl_root / "runs")
        cache_dir = _resolve_path(artifacts.rl_root, params["cache_dir"], artifacts.rl_root / ".cache")

        # --------------------
        # SFT (encoder or activations)
        # --------------------
        sft_runs = runs_dir
        sft_cache = cache_dir
        tb_dir = sft_runs / "tb"
        if sft_enabled:
            state_repr = str(sft["state_repr"]).strip().lower()

            epochs = int(sft["epochs"])
            batch_size = int(sft["batch_size"])
            lr = float(sft["lr"])

            run_macro_no_conv = bool(sft["run_macro_no_conv"])
            run_conv_state = bool(sft["run_conv_state"])
            run_macro_with_conv = bool(sft["run_macro_with_conv"])
            run_micro_flat = bool(sft["run_micro_flat"])
            run_micro_independent = bool(sft["run_micro_independent"])
            run_termination = bool(sft["run_termination"])
            conv_state_encoding = str(sft["conv_state_encoding"]).strip().lower()
            sft_use_conv_state = bool(sft.get("use_conv_state", run_conv_state))
            sft_conv_state_source, sft_conv_state_sft_dir, sft_probe_checkpoint, sft_probe_batch_size = _resolve_conv_state_runtime(
                sft,
                default_source=("sft" if run_conv_state else "gold"),
                label="rl.sft",
            )

            cmds = []
            cmd_meta: list[dict] = []
            if state_repr == "activations":
                activation_key = str(sft["activation_key"])
                activation_path = str(activation_dir)
                activation_layer_spec = sft["activation_layer"]
                hidden_dim = int(sft["mlp_hidden_dim"])
                projection_dim = int(sft["mlp_projection_dim"])
                conv_state_embedding_dim = int(sft["mlp_conv_state_embedding_dim"])
                dropout = float(sft["mlp_dropout"])

                activation_layers = _parse_activation_layers(
                    activation_layer_spec,
                    activation_path=activation_path,
                    activation_key=activation_key,
                    root=artifacts.rl_root,
                )

                for layer in activation_layers:
                    layer_tag = f"L{layer}"

                    if run_macro_no_conv:
                        out_dir = sft_runs / f"sft_policy_over_options_no_conv_acts_{layer_tag}"
                        cmd = [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_macro_action_activations",
                            "--root",
                            str(artifacts.rl_root),
                            "--activation_path",
                            activation_path,
                            "--activation_layer",
                            str(layer),
                            "--activation_key",
                            activation_key,
                            "--hidden_dim",
                            str(hidden_dim),
                            "--projection_dim",
                            str(projection_dim),
                            "--conv_state_embedding_dim",
                            str(conv_state_embedding_dim),
                            "--dropout",
                            str(dropout),
                            "--lr",
                            str(lr),
                            "--batch_size",
                            str(batch_size),
                            "--epochs",
                            str(epochs),
                            "--output_dir",
                            str(out_dir),
                        ]
                        cmds.append(cmd)
                        cmd_meta.append({"task": "policy_over_options_no_conv", "layer": int(layer), "out_dir": out_dir})

                    if run_conv_state:
                        out_dir = sft_runs / f"sft_conv_state_acts_{layer_tag}"
                        cmd = [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_conv_state_activations",
                            "--root",
                            str(artifacts.rl_root),
                            "--activation_path",
                            activation_path,
                            "--activation_layer",
                            str(layer),
                            "--activation_key",
                            activation_key,
                            "--hidden_dim",
                            str(hidden_dim),
                            "--dropout",
                            str(dropout),
                            "--lr",
                            str(lr),
                            "--batch_size",
                            str(batch_size),
                            "--epochs",
                            str(epochs),
                            "--output_dir",
                            str(out_dir),
                        ]
                        cmds.append(cmd)
                        cmd_meta.append({"task": "conv_state", "layer": int(layer), "out_dir": out_dir})

                    if run_macro_with_conv:
                        out_dir = sft_runs / f"sft_policy_over_options_with_conv_acts_{layer_tag}"
                        cmd = [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_macro_action_activations",
                            "--root",
                            str(artifacts.rl_root),
                            "--activation_path",
                            activation_path,
                            "--activation_layer",
                            str(layer),
                            "--activation_key",
                            activation_key,
                            "--hidden_dim",
                            str(hidden_dim),
                            "--projection_dim",
                            str(projection_dim),
                            "--conv_state_embedding_dim",
                            str(conv_state_embedding_dim),
                            "--dropout",
                            str(dropout),
                            "--lr",
                            str(lr),
                            "--batch_size",
                            str(batch_size),
                            "--epochs",
                            str(epochs),
                            "--use_conv_state",
                            "--conv_state_encoding",
                            conv_state_encoding,
                            "--conv_state_source",
                            sft_conv_state_source,
                            "--probe_batch_size",
                            str(sft_probe_batch_size),
                            "--output_dir",
                            str(out_dir),
                        ]
                        if sft_conv_state_source == "probe":
                            cmd += ["--probe_checkpoint", sft_probe_checkpoint]
                        elif sft_conv_state_source == "sft":
                            cmd += [
                                "--conv_state_sft_dir",
                                sft_conv_state_sft_dir or str(sft_runs / f"sft_conv_state_acts_{layer_tag}"),
                            ]
                        cmds.append(cmd)
                        cmd_meta.append({"task": "policy_over_options_with_conv", "layer": int(layer), "out_dir": out_dir})

                    if run_micro_flat:
                        out_dir = sft_runs / f"sft_flat_rl_acts_{layer_tag}"
                        cmd = [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_micro_action_activations",
                            "--root",
                            str(artifacts.rl_root),
                            "--activation_path",
                            activation_path,
                            "--activation_layer",
                            str(layer),
                            "--activation_key",
                            activation_key,
                            "--hidden_dim",
                            str(hidden_dim),
                            "--projection_dim",
                            str(projection_dim),
                            "--conv_state_embedding_dim",
                            str(conv_state_embedding_dim),
                            "--dropout",
                            str(dropout),
                            "--lr",
                            str(lr),
                            "--batch_size",
                            str(batch_size),
                            "--epochs",
                            str(epochs),
                            "--train_flat",
                            "--output_dir",
                            str(out_dir),
                        ]
                        cmds.append(cmd)
                        cmd_meta.append({"task": "flat_rl", "layer": int(layer), "out_dir": out_dir})

                    if run_micro_independent:
                        out_dir = sft_runs / f"sft_intra_option_policies_acts_{layer_tag}"
                        cmd = [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_micro_action_activations",
                            "--root",
                            str(artifacts.rl_root),
                            "--activation_path",
                            activation_path,
                            "--activation_layer",
                            str(layer),
                            "--activation_key",
                            activation_key,
                            "--hidden_dim",
                            str(hidden_dim),
                            "--projection_dim",
                            str(projection_dim),
                            "--conv_state_embedding_dim",
                            str(conv_state_embedding_dim),
                            "--dropout",
                            str(dropout),
                            "--lr",
                            str(lr),
                            "--batch_size",
                            str(batch_size),
                            "--epochs",
                            str(epochs),
                            "--train_all_macros",
                            "--output_dir",
                            str(out_dir),
                        ]
                        cmds.append(cmd)
                        cmd_meta.append({"task": "intra_option_policies", "layer": int(layer), "out_dir": out_dir})

                    if run_termination:
                        base_cmd = [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_termination_activations",
                            "--root",
                            str(artifacts.rl_root),
                            "--activation_path",
                            activation_path,
                            "--activation_layer",
                            str(layer),
                            "--activation_key",
                            activation_key,
                            "--hidden_dim",
                            str(hidden_dim),
                            "--projection_dim",
                            str(projection_dim),
                            "--conv_state_embedding_dim",
                            str(conv_state_embedding_dim),
                            "--dropout",
                            str(dropout),
                            "--lr",
                            str(lr),
                            "--batch_size",
                            str(batch_size),
                            "--epochs",
                            str(epochs),
                        ]
                        out_dir = sft_runs / f"sft_termination_no_conv_acts_{layer_tag}"
                        cmd = base_cmd + ["--output_dir", str(out_dir)]
                        cmds.append(cmd)
                        cmd_meta.append({"task": "termination_no_conv", "layer": int(layer), "out_dir": out_dir})
                        if sft_use_conv_state:
                            out_dir = sft_runs / f"sft_termination_with_conv_acts_{layer_tag}"
                            term_conv_source = "sft" if sft_conv_state_source == "gold" else sft_conv_state_source
                            cmd = base_cmd + [
                                "--use_conv_state",
                                "--conv_state_encoding",
                                conv_state_encoding,
                                "--conv_state_source",
                                term_conv_source,
                                "--probe_batch_size",
                                str(sft_probe_batch_size),
                                "--output_dir",
                                str(out_dir),
                            ]
                            if term_conv_source == "probe":
                                cmd += ["--probe_checkpoint", sft_probe_checkpoint]
                            else:
                                cmd += [
                                    "--conv_state_model_dir",
                                    sft_conv_state_sft_dir or str(sft_runs / f"sft_conv_state_acts_{layer_tag}"),
                                ]
                            cmds.append(cmd)
                            cmd_meta.append({"task": "termination_with_conv", "layer": int(layer), "out_dir": out_dir})
            else:
                # encoder-based SFT (mirrors run_sft_encoders.sh)
                encoder_name = str(sft["encoder_name"])
                amp = str(sft["amp"])
                sft_num_workers = int(sft["num_workers"])
                if run_macro_no_conv:
                    cmds.append(
                        [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_macro_action",
                            "--root",
                            str(artifacts.rl_root),
                            "--model_name",
                            encoder_name,
                            "--output_dir",
                            str(sft_runs / "sft_macro_no_conv"),
                            "--cache_dir",
                            str(sft_cache),
                            "--epochs",
                            str(epochs),
                            "--batch_size",
                            str(batch_size),
                            "--num_workers",
                            str(sft_num_workers),
                            "--lr",
                            str(lr),
                            "--amp",
                            amp,
                            "--report_to",
                            "tensorboard",
                            "--tb_dir",
                            str(tb_dir / "macro_no_conv"),
                        ]
                    )
                if run_conv_state:
                    cmds.append(
                        [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_conv_state",
                            "--root",
                            str(artifacts.rl_root),
                            "--model_name",
                            encoder_name,
                            "--output_dir",
                            str(sft_runs / "sft_conv_state"),
                            "--cache_dir",
                            str(sft_cache),
                            "--epochs",
                            str(epochs),
                            "--batch_size",
                            str(batch_size),
                            "--num_workers",
                            str(sft_num_workers),
                            "--lr",
                            str(lr),
                            "--amp",
                            amp,
                            "--report_to",
                            "tensorboard",
                            "--tb_dir",
                            str(tb_dir / "conv_state"),
                        ]
                    )
                if run_macro_with_conv:
                    cmds.append(
                        [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_macro_action",
                            "--root",
                            str(artifacts.rl_root),
                            "--model_name",
                            encoder_name,
                            "--output_dir",
                            str(sft_runs / "sft_macro_with_conv"),
                            "--cache_dir",
                            str(sft_cache),
                            "--use_conv_state",
                            "1",
                            "--conv_state_encoding",
                            conv_state_encoding,
                            "--epochs",
                            str(epochs),
                            "--batch_size",
                            str(batch_size),
                            "--num_workers",
                            str(sft_num_workers),
                            "--lr",
                            str(lr),
                            "--amp",
                            amp,
                            "--report_to",
                            "tensorboard",
                            "--tb_dir",
                            str(tb_dir / "macro_with_conv"),
                        ]
                    )
                if run_micro_flat:
                    cmds.append(
                        [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_micro_action",
                            "--root",
                            str(artifacts.rl_root),
                            "--model_name",
                            encoder_name,
                            "--output_dir",
                            str(sft_runs / "sft_micro_flat"),
                            "--cache_dir",
                            str(sft_cache),
                            "--policy_mode",
                            "flat",
                            "--epochs",
                            str(epochs),
                            "--batch_size",
                            str(batch_size),
                            "--num_workers",
                            str(sft_num_workers),
                            "--lr",
                            str(lr),
                            "--amp",
                            amp,
                            "--report_to",
                            "tensorboard",
                            "--tb_dir",
                            str(tb_dir / "micro_flat"),
                        ]
                    )
                if run_micro_independent:
                    cmds.append(
                        [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_micro_action",
                            "--root",
                            str(artifacts.rl_root),
                            "--model_name",
                            encoder_name,
                            "--output_dir",
                            str(sft_runs / "sft_micro_independent"),
                            "--cache_dir",
                            str(sft_cache),
                            "--policy_mode",
                            "independent",
                            "--epochs",
                            str(epochs),
                            "--batch_size",
                            str(batch_size),
                            "--num_workers",
                            str(sft_num_workers),
                            "--lr",
                            str(lr),
                            "--amp",
                            amp,
                            "--report_to",
                            "tensorboard",
                            "--tb_dir",
                            str(tb_dir / "micro_independent"),
                        ]
                    )
                if run_termination:
                    cmd = [
                            sys.executable,
                            "-m",
                            "topa.rl.sft_pretraining.train_termination",
                            "--root",
                            str(artifacts.rl_root),
                            "--model_name",
                            encoder_name,
                            "--output_dir",
                            str(sft_runs / "sft_termination"),
                            "--cache_dir",
                            str(sft_cache),
                            "--include_macro_in_text",
                            "--epochs",
                            str(epochs),
                            "--batch_size",
                            str(batch_size),
                            "--num_workers",
                            str(sft_num_workers),
                            "--lr",
                            str(lr),
                            "--amp",
                            amp,
                            "--report_to",
                            "tensorboard",
                            "--tb_dir",
                            str(tb_dir / "termination"),
                    ]
                    if run_conv_state:
                        cmd += [
                            "--use_conv_state",
                            "--conv_state_adapter_dir",
                            str(sft_runs / "sft_conv_state" / "adapter"),
                            "--conv_state_meta_path",
                            str(sft_runs / "sft_conv_state" / "reports" / "conv_state_meta.json"),
                            "--conv_state_encoding",
                            conv_state_encoding,
                        ]
                    cmds.append(cmd)

            out["sft_commands"] = cmds
            layer_metrics: Dict[str, Dict[int, float]] = {}
            for i, cmd in enumerate(cmds):
                if "--seed" not in cmd:
                    cmd += ["--seed", str(seed)]
                subprocess.run(cmd, check=True, env=env)
                if cmd_meta:
                    meta = cmd_meta[i]
                    task = meta["task"]
                    layer = int(meta["layer"])
                    out_dir = Path(meta["out_dir"])
                    if task == "intra_option_policies":
                        acc = _mean_micro_acc(out_dir)
                    else:
                        acc = _best_val_acc_from_dir(out_dir)
                    if acc is not None:
                        layer_metrics.setdefault(str(task), {})[int(layer)] = float(acc)

            if layer_metrics:
                out["sft_layer_metrics"] = layer_metrics
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    import matplotlib.pyplot as plt

                    layer_tb = tb_dir / "activation_layers"
                    writer = SummaryWriter(log_dir=str(layer_tb))
                    plots_dir = sft_runs / "layer_plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    for task, lm in layer_metrics.items():
                        layers_sorted = sorted(lm.keys())
                        vals = [lm[k] for k in layers_sorted]
                        for l, v in zip(layers_sorted, vals):
                            writer.add_scalar(f"{task}/best_val_acc", float(v), int(l))

                        if layers_sorted:
                            fig = plt.figure()
                            ax = fig.add_subplot(111)
                            ax.plot(layers_sorted, vals, marker="o")
                            ax.set_xlabel("Layer")
                            ax.set_ylabel("Best val acc")
                            ax.set_title(f"{task}: best val acc by layer")
                            fig.tight_layout()
                            fig_path = plots_dir / f"{task}_best_val_acc.png"
                            fig.savefig(fig_path, dpi=200)
                            writer.add_figure(f"{task}/best_val_acc_by_layer", fig, global_step=0)
                            plt.close(fig)

                    writer.flush()
                    writer.close()
                except Exception:
                    pass

        # --------------------
        # Offline RL (IQL/CQL over encoder or activation states)
        # --------------------
        if offline_rl_enabled:
            state_repr = str(offline_rl["state_repr"]).strip().lower()
            assert state_repr in {"encoder", "activations"}
            algorithm = str(offline_rl["algorithm"]).strip().lower()
            assert algorithm in {"iql", "cql"}

            rl_runs = runs_dir
            rl_cache = cache_dir
            rl_tb = rl_runs / "tb"

            hidden_dim = int(offline_rl["mlp_hidden_dim"])
            projection_dim = int(offline_rl["mlp_projection_dim"])
            conv_state_embedding_dim = int(offline_rl["mlp_conv_state_embedding_dim"])
            dropout = float(offline_rl["mlp_dropout"])
            epochs = int(offline_rl["epochs"])
            batch_size = int(offline_rl["batch_size"])
            num_workers = int(offline_rl["num_workers"])
            conv_state_encoding = str(offline_rl["conv_state_encoding"]).strip().lower()
            offline_conv_state_source, offline_conv_state_sft_dir, offline_probe_checkpoint, offline_probe_batch_size = _resolve_conv_state_runtime(
                offline_rl,
                fallback_block=sft,
                default_source="gold",
                label="rl.offline_rl",
            )
            init_from_sft = bool(offline_rl["init_from_sft"])
            init_critic_fusion_from_sft = bool(offline_rl["init_critic_fusion_from_sft"])
            if init_from_sft and state_repr == "encoder":
                raise ValueError("offline_rl.init_from_sft=true is only supported for activation MLP SFT checkpoints. Encoder SFT checkpoints are PEFT adapters, so set init_from_sft=false for encoder offline RL.")
            if init_critic_fusion_from_sft and state_repr == "encoder":
                raise ValueError("offline_rl.init_critic_fusion_from_sft=true is only supported for activation MLP SFT checkpoints. Encoder SFT checkpoints are PEFT adapters, so set init_critic_fusion_from_sft=false for encoder offline RL.")
            algo_params = offline_rl[algorithm]
            lr_actor = float(algo_params["lr_actor"])
            lr_critic = float(algo_params["lr_critic"])
            lr_beta = float(algo_params["lr_beta"])
            gamma = float(algo_params["gamma"])
            if algorithm == "iql":
                expectile = float(algo_params["expectile"])
                temperature = float(algo_params["temperature"])
                entropy_alpha = 0.0
                cql_alpha = 0.0
                bc_weight = 0.0
            else:
                expectile = 0.0
                temperature = 1.0
                entropy_alpha = float(algo_params["entropy_alpha"])
                cql_alpha = float(algo_params["cql_alpha"])
                bc_weight = float(algo_params["bc_weight"])
            plot_action_dist = bool(offline_rl["plot_action_dist"])
            action_dist_top_k = int(offline_rl["action_dist_top_k"])
            action_dist_max_batches = int(offline_rl["action_dist_max_batches"])

            run_policy_over_options_no_conv = bool(offline_rl["run_policy_over_options_no_conv"])
            run_policy_over_options_with_conv = bool(offline_rl["run_policy_over_options_with_conv"])
            run_flat_rl = bool(offline_rl["run_flat_rl"])
            run_intra_option_policies = bool(offline_rl["run_intra_option_policies"])
            use_conv_state_for_intra_options = bool(offline_rl["use_conv_state_for_intra_options"])

            if state_repr == "activations":
                activation_key = str(offline_rl["activation_key"])
                activation_path = str(activation_dir)
                activation_layer_spec = offline_rl["activation_layer"]
                activation_layers = _parse_activation_layers(
                    activation_layer_spec,
                    activation_path=activation_path,
                    activation_key=activation_key,
                    root=artifacts.rl_root,
                )
                encoder_name = ""
            else:
                activation_key = ""
                activation_path = ""
                activation_layers = [None]
                encoder_name = str(sft["encoder_name"])
                encoder_batch_size = int(offline_rl["encoder_batch_size"])

            def _offline_cmd(task: str, layer: int | None, output_name: str, *, use_conv_state: bool = False) -> list[str]:
                run_tag = f"acts_L{layer}" if state_repr == "activations" else "enc"
                sft_init_args: list[str] = []
                if state_repr == "activations":
                    layer_tag = f"L{int(layer)}"
                    if task == "policy_over_options":
                        sft_name = "sft_policy_over_options_with_conv_acts" if use_conv_state else "sft_policy_over_options_no_conv_acts"
                        term_sft_name = "sft_termination_with_conv_acts" if use_conv_state else "sft_termination_no_conv_acts"
                        sft_init_args = [
                            "--actor_init_path",
                            str(sft_runs / f"{sft_name}_{layer_tag}" / "actor.pt"),
                            "--termination_init_path",
                            str(sft_runs / f"{term_sft_name}_{layer_tag}" / "actor.pt"),
                        ]
                    elif task == "flat_rl" and not use_conv_state:
                        sft_init_args = ["--actor_init_path", str(sft_runs / f"sft_flat_rl_acts_{layer_tag}" / "actor.pt")]
                    elif task == "intra_option_all" and not use_conv_state:
                        sft_init_args = ["--actor_init_dir", str(sft_runs / f"sft_intra_option_policies_acts_{layer_tag}")]
                cmd = [
                    sys.executable,
                    "-m",
                    "topa.rl.offline_rl.train_discrete_activations",
                    "--root",
                    str(artifacts.rl_root),
                    "--task",
                    task,
                    "--algorithm",
                    algorithm,
                    "--state_repr",
                    state_repr,
                    "--hidden_dim",
                    str(hidden_dim),
                    "--projection_dim",
                    str(projection_dim),
                    "--conv_state_embedding_dim",
                    str(conv_state_embedding_dim),
                    "--dropout",
                    str(dropout),
                    "--output_dir",
                    str(rl_runs / f"{algorithm}_{output_name}_{run_tag}"),
                    "--cache_dir",
                    str(rl_cache),
                    "--log_dir",
                    str(rl_tb / f"{algorithm}_{output_name}_{run_tag}"),
                    "--epochs",
                    str(epochs),
                    "--batch_size",
                    str(batch_size),
                    "--num_workers",
                    str(num_workers),
                    "--lr_actor",
                    str(lr_actor),
                    "--lr_critic",
                    str(lr_critic),
                    "--lr_beta",
                    str(lr_beta),
                    "--gamma",
                    str(gamma),
                    "--expectile",
                    str(expectile),
                    "--temperature",
                    str(temperature),
                    "--entropy_alpha",
                    str(entropy_alpha),
                    "--cql_alpha",
                    str(cql_alpha),
                    "--bc_weight",
                    str(bc_weight),
                    "--conv_state_encoding",
                    conv_state_encoding,
                    "--action_dist_top_k",
                    str(action_dist_top_k),
                    "--action_dist_max_batches",
                    str(action_dist_max_batches),
                ]
                cmd += sft_init_args
                if init_from_sft:
                    cmd += ["--init_from_sft"]
                if init_critic_fusion_from_sft:
                    cmd += ["--init_critic_fusion_from_sft"]
                if plot_action_dist:
                    cmd += ["--plot_action_dist"]
                if use_conv_state:
                    cmd += [
                        "--use_conv_state",
                        "--conv_state_source",
                        offline_conv_state_source,
                        "--probe_batch_size",
                        str(offline_probe_batch_size),
                    ]
                    if offline_conv_state_source == "probe":
                        cmd += ["--probe_checkpoint", offline_probe_checkpoint]
                    elif offline_conv_state_source == "sft":
                        cmd += [
                            "--conv_state_sft_dir",
                            offline_conv_state_sft_dir or str(sft_runs / f"sft_conv_state_acts_L{int(layer)}"),
                        ]
                if state_repr == "activations":
                    cmd += [
                        "--activation_path",
                        activation_path,
                        "--activation_layer",
                        str(layer),
                        "--activation_key",
                        activation_key,
                    ]
                else:
                    cmd += [
                        "--encoder_name",
                        encoder_name,
                        "--encoder_batch_size",
                        str(encoder_batch_size),
                    ]
                    conv_adapter = sft_runs / "sft_conv_state" / "adapter"
                    conv_meta = sft_runs / "sft_conv_state" / "reports" / "conv_state_meta.json"
                    if use_conv_state and task == "policy_over_options" and conv_adapter.exists() and conv_meta.exists():
                        cmd += [
                            "--conv_state_adapter_dir",
                            str(conv_adapter),
                            "--conv_state_meta_path",
                            str(conv_meta),
                        ]
                return cmd

            cmds = []
            for layer in activation_layers:
                if run_policy_over_options_no_conv:
                    cmds.append(_offline_cmd("policy_over_options", None if layer is None else int(layer), "policy_over_options_no_conv"))
                if run_policy_over_options_with_conv:
                    cmds.append(_offline_cmd("policy_over_options", None if layer is None else int(layer), "policy_over_options_with_conv", use_conv_state=True))
                if run_flat_rl:
                    cmds.append(_offline_cmd("flat_rl", None if layer is None else int(layer), "flat_rl", use_conv_state=use_conv_state_for_intra_options))
                if run_intra_option_policies:
                    cmds.append(_offline_cmd("intra_option_all", None if layer is None else int(layer), "intra_option_policies", use_conv_state=use_conv_state_for_intra_options))

            out["offline_rl_commands"] = cmds
            for cmd in cmds:
                if "--seed" not in cmd:
                    cmd += ["--seed", str(seed)]
                subprocess.run(cmd, check=True, env=env)

        return out
