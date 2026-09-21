import os
import os.path as osp
from copy import deepcopy
from pathlib import Path
import json
import hashlib

from .utils import load_json
from .annotators import (
    MacroActionsAnnotator,
    MicroActionsAnnotator,
    MicroActionsDirectAnnotator,
    ConversationStateAnnotator,
    # UserActionsAnnotator,
    ProfileDimAnnotator,
    IntrinsicRewardAnnotator,
    CautionsAnnotator,
    ExtrinsicRewardAnnotator,
)
from .prompts.extrinsic_reward_prompt import set_reward_domain as _set_reward_domain
from ..utils import build_logger

logger = build_logger()

class AnnotationManager:
    """
    Runs the annotation pipeline over a TOPA-format dataset.
    - Annotations are saved as CSV per component.
    - Usage metadata is saved as JSON per component.
    """
    def __init__(self, domain: str, domain_params: dict, data_path: str, output_path: str, annotations_params: dict, api_key: str):
        self.domain = domain
        self.domain_params = domain_params
        self.data_path = data_path
        self.output_path = output_path
        self.api_key = api_key
        self.annotations_params = annotations_params
        self.components = annotations_params["components"]
        self.max_retries = annotations_params["max_retries"]

        os.makedirs(self.output_path, exist_ok=True)

        try:
            _set_reward_domain(self.domain)
        except Exception as e:
            logger.warning(f"[TOPA] Could not set reward prompt domain: {e}")

        self.annotators = {
            "macro_actions": MacroActionsAnnotator,
            "micro_actions": MicroActionsAnnotator,
            "micro_actions_direct": MicroActionsDirectAnnotator,
            "conv_state": ConversationStateAnnotator,
            "intrinsic_reward": IntrinsicRewardAnnotator,
            "cautions": CautionsAnnotator,
            "extrinsic_reward": ExtrinsicRewardAnnotator,
            "user_profile": ProfileDimAnnotator,
            # "user_actions": UserActionsAnnotator,
        }

        self.data = load_json(data_path, verify_struct=True)
    
    @staticmethod
    def _resolve_component_path(path: str, name: str) -> str:
        if name in ["macro_actions", "micro_actions", "micro_actions_direct", "intrinsic_reward"]:
            return osp.join(path, "micro_actions.json")
        elif name == "conv_state":
            return osp.join(path, "conversation_states.json")
        elif name == "cautions":
            return osp.join(path, "cautions.json")
        # elif name == "user_actions":
        #     return osp.join(path, "user_actions.json")
        elif name == "user_profile":
            root = Path(path) if Path(path).is_dir() else Path(path).parent
            direct = root / "user_profile.json"
            if direct.exists():
                return str(direct)
            matches = sorted(root.rglob("user_profile.json"))
            if matches:
                return str(matches[0])
            raise FileNotFoundError(f"user_profile.json not found under {root}")
        raise ValueError(f"Unkown component name {name}")

    @staticmethod
    def _prepare_component(loaded_comp, component_name, **kwargs):
        use_annotation_rules = bool(kwargs.get("use_annotation_rules", False))
        if component_name == "macro_actions":
            comp = []
            for c in loaded_comp:
                _c = deepcopy(c)
                _c.pop("micro_actions")
                _c.pop("confidence_score")
                _c["goal"] = _c["goal"]["objective"]
                if not use_annotation_rules:
                    _c.pop("states")
                comp.append(_c)
            return comp
        elif component_name in ["micro_actions", "micro_actions_direct"]:
            comp = []
            for c in loaded_comp:
                _c = deepcopy(c)
                _c.pop("confidence_score")
                _c["goal"] = _c["goal"]["objective"]
                _c.pop("states")
                _c["micro_actions"] = []
                for mi in c["micro_actions"]:
                    _mi = deepcopy(mi)
                    _mi.pop("confidence_score")
                    if not use_annotation_rules:
                        _mi.pop("states")
                    _c["micro_actions"].append(_mi)
                comp.append(_c)
            return comp
        elif component_name == "intrinsic_reward":
            comp = []
            for c in loaded_comp:
                _c = deepcopy(c)
                _c.pop("confidence_score")
                _c.pop("states")
                _c.pop("micro_actions")
                comp.append(_c)
            return comp
        elif component_name == "cautions":
            comp = []
            for c in loaded_comp:
                _c = deepcopy(c)
                _c.pop("confidence_score")
                comp.append(_c)
            return comp
        else:
            return loaded_comp

    @staticmethod
    def _resolve_annotation_path(path: str) -> str:
        p = Path(path)
        extracted_dir = p if p.is_dir() else p.parent  # extracted_components
        annotation_dir = extracted_dir
        while True:
            if annotation_dir.stem == "refined_components":
                annotation_dir = annotation_dir.parent
                break
            annotation_dir = annotation_dir.parent
            # reached filesystem root => stop (prevents infinite loop)
            if annotation_dir.parent == annotation_dir:
                raise Exception(f"Could not find the refined_components dir in this path {extracted_dir} !")
        annotation_dir = annotation_dir / "annotations"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        return str(annotation_dir)

    def annotate(self, components_path: str) -> str:
        annotation_path = self._resolve_annotation_path(components_path)

        # Optional post-processing hooks (configured in YAML under annotations_params)
        fpm_cfg = self.annotations_params.get("frequent_pattern_mining", {})
        asr_cfg = self.annotations_params.get("action_space_refinement", {})
        exp_cfg = self.annotations_params.get("expert_feedback", {})

        # Keep track of produced CSVs (so we can post-process them).
        produced_csv = {}

        # --- optional: annotate user_profile only on a validation subset ---
        val_cfg = self.annotations_params.get("validation_subset", {}) or {}
        val_enabled = bool(val_cfg.get("enabled", False))
        # Default to True when validation subset is enabled
        user_profile_val_only = bool(val_cfg.get("user_profile_only", True)) if val_enabled else False
        val_session_filter = None
        if val_enabled:
            data_path = Path(self.data_path).expanduser().resolve()
            data_root = data_path.parent if data_path.is_file() else data_path
            user_label = str(self.domain_params.get("user") or "user").strip().lower().replace(" ", "_")
            preferred_name = str(val_cfg.get("split_filename") or f"{user_label}_split.json")
            candidate_names = [preferred_name, f"{user_label}_split.json", "user_split.json"]
            split_path = None
            for name in candidate_names:
                p = data_root / name
                if p.exists():
                    split_path = p
                    break
            if split_path is None:
                split_path = data_root / preferred_name

            split_obj = None
            if split_path.exists():
                try:
                    split_obj = json.loads(split_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"[TOPA] Could not read existing user split file: {e}")
                    split_obj = None

            if split_obj and isinstance(split_obj.get("val_user_idxs"), list):
                # Use existing split to build validation sessions (all sessions for val users)
                try:
                    val_users = {int(u) for u in split_obj["val_user_idxs"]}
                    val_session_filter = set()
                    for user_idx, user in enumerate(self.data):
                        if user_idx not in val_users:
                            continue
                        for session_idx, _ in enumerate((user or {}).get("sessions", [])):
                            val_session_filter.add((user_idx, session_idx))
                except Exception as e:
                    logger.warning(f"[TOPA] Failed to use existing user split file: {e}")
                    val_session_filter = None

            if val_session_filter is None:
                # Build validation sessions from config
                val_session_filter = self._build_validation_session_filter(self.data, val_cfg)

                # If split file missing, write one based on val users
                if not split_path.exists():
                    try:
                        val_users = {u for (u, _s) in val_session_filter}
                        split_obj = self._build_user_split_from_users(
                            self.data,
                            val_users=val_users,
                            val_frac=float(val_cfg.get("ratio", 0.1)),
                        )
                        split_path.write_text(json.dumps(split_obj, indent=2), encoding="utf-8")
                        logger.info(f"[TOPA] Saved user split -> {split_path}")
                    except Exception as e:
                        logger.warning(f"[TOPA] Could not write user split: {e}")

            # Persist the chosen validation sessions for reproducibility
            try:
                export_path = val_cfg.get("export_path")
                if export_path:
                    out_split = Path(str(export_path)).expanduser()
                    if not out_split.is_absolute():
                        out_split = Path(annotation_path) / out_split
                else:
                    out_split = Path(annotation_path) / "validation_sessions.json"
                out_split.parent.mkdir(parents=True, exist_ok=True)
                out_split.write_text(json.dumps(sorted(list(val_session_filter)), indent=2), encoding="utf-8")
                logger.info(f"[TOPA] Saved validation subset indices -> {out_split}")
            except Exception as e:
                logger.warning(f"[TOPA] Could not write validation_sessions.json: {e}")

        for component_name in self.components:
            logger.info(f"[TOPA] Annotating {component_name.replace('_', ' ')}...")
            component_params = self.annotations_params.get(f"{component_name}_params", {})
            if "num_processes" not in component_params:
                component_params["num_processes"] = self.annotations_params.get("num_processes", 1)
            logger.info(f"[TOPA] params {str(component_params)}")
            if component_name != "extrinsic_reward":
                # Load extracted component
                loaded_comp = load_json(self._resolve_component_path(components_path, component_name))
                loaded_comp = self._prepare_component(loaded_comp, component_name, **component_params)
            else:
                loaded_comp = {}
            annotator = self.annotators[component_name](domain=self.domain, domain_params=self.domain_params, 
                                                        api_key=self.api_key, max_retries=self.max_retries,
                                                        **component_params)

            # Apply session_filter only for user_profile, as requested
            session_filter = val_session_filter if (component_name == "user_profile" and user_profile_val_only) else None
            csv_path = annotator.annotate(self.data, loaded_comp, component_name, annotation_path, session_filter=session_filter)
            produced_csv[component_name] = csv_path

            # Frequent pattern mining
            if component_name == "macro_actions" and fpm_cfg.get("enabled", False):
                from ..analysis.frequent_patterns import mine_frequent_macro_patterns

                mine_frequent_macro_patterns(
                    macro_csv_path=csv_path,
                    out_dir=annotation_path,
                    min_support_ratio=fpm_cfg.get("min_support_ratio", 0.5),
                    min_pattern_length=fpm_cfg.get("min_pattern_length", 2),
                    max_pattern_length=fpm_cfg.get("max_pattern_length", 10),
                    top_k=fpm_cfg.get("top_k", 30),
                    collapse_consecutive=fpm_cfg.get("collapse_consecutive", True),
                )

            # Refine action space + expert loop
            if component_name == "micro_actions" and asr_cfg.get("enabled", False):
                from ..refinement.action_space_refinement import refine_micro_action_space

                refine_micro_action_space(
                    api_key=self.api_key,
                    domain_params=self.domain_params,
                    micro_actions_json_path=self._resolve_component_path(components_path, "micro_actions"),
                    micro_actions_csv_path=csv_path,
                    data=self.data,
                    out_dir=annotation_path,
                    context_length=asr_cfg.get("context_length", 5),
                    conf_threshold=asr_cfg.get("conf_threshold", 0.65),
                    max_examples_per_macro=asr_cfg.get("max_examples_per_macro", 30),
                    max_new_actions_per_macro=asr_cfg.get("max_new_actions_per_macro", 6)
                )

            if component_name == "micro_actions" and exp_cfg.get("enabled", False):
                from ..analysis.expert_feedback import build_expert_review_set

                build_expert_review_set(
                    micro_actions_csv_path=csv_path,
                    data=self.data,
                    domain_params=self.domain_params,
                    out_dir=annotation_path,
                    conf_threshold=exp_cfg.get("conf_threshold", 0.65),
                    n_review=exp_cfg.get("n_review", 200),
                )

            # Conversation-state ambiguity review set (2 runs + disagreement mining)
            if component_name == "conv_state":
                amb_cfg = (self.annotations_params.get("conv_state_ambiguity") or {})
                if amb_cfg.get("enabled", False):
                    self._run_conv_state_ambiguity(
                        annotation_path=annotation_path,
                        components_path=components_path,
                        domain_params=self.domain_params,
                        amb_cfg=amb_cfg,
                    )

        return annotation_path


    @staticmethod
    def _resolve_pairs_from_session_ids(data, session_ids) -> set:
        """Resolve a set of (user_idx, session_idx) from session_id strings.

        This is more robust than relying on numeric indices because it survives
        re-ordering of users/sessions in the dataset file.
        """
        wanted = set([str(x) for x in (session_ids or [])])
        pairs = set()
        for user_idx, user in enumerate(data or []):
            sessions = (user or {}).get("sessions", []) or []
            for session_idx, sess in enumerate(sessions):
                sm = (sess or {}).get("session_metadata", {}) or {}
                sid = sm.get("session_id")
                if sid is None:
                    continue
                if str(sid) in wanted:
                    pairs.add((user_idx, session_idx))
        if wanted and not pairs:
            raise ValueError("No validation sessions matched the provided session_ids.")
        return pairs

    @staticmethod
    def _build_validation_session_filter(data, val_cfg: dict) -> set:
        """Return a set of (user_idx, session_idx) to be treated as validation.

        Priority:
          1) explicit_session_ids_path / explicit_pairs_path: JSON list of session_id strings OR [user_idx, session_idx]
          2) session_metadata split key
          3) deterministic hash sampling with ratio
        """
        mode = str(val_cfg.get("mode") or "by_session").strip().lower()
        # 1) explicit list (indices) OR explicit session IDs (recommended)
        explicit_path = (
            val_cfg.get("explicit_session_ids_path")
            or val_cfg.get("explicit_sessions_path")
            or val_cfg.get("session_ids_path")
            or val_cfg.get("explicit_pairs_path")
            or val_cfg.get("pairs_path")
        )
        if explicit_path:
            p = Path(str(explicit_path)).expanduser()
            payload = json.loads(p.read_text(encoding="utf-8"))

            # Accept:
            #  - list of [user_idx, session_idx] pairs
            #  - list of "session_id" strings
            #  - list of {"session_id": "..."} dicts
            #  - {"session_ids": [...]} dict
            if isinstance(payload, dict) and "session_ids" in payload:
                payload = payload["session_ids"]

            if isinstance(payload, list) and payload:
                first = payload[0]
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    return {tuple(map(int, x)) for x in payload}

                session_ids = []
                if isinstance(first, str):
                    session_ids = [str(x) for x in payload]
                elif isinstance(first, dict):
                    for item in payload:
                        sid = (item or {}).get("session_id") or (item or {}).get("id")
                        if sid is not None:
                            session_ids.append(str(sid))

                if session_ids:
                    return AnnotationManager._resolve_pairs_from_session_ids(data, session_ids)

            raise ValueError(f"Invalid validation subset file format at: {p}")
        
        # 2) split key in session_metadata
        split_key = val_cfg.get("split_key") or "split"
        split_vals = val_cfg.get("val_values") or val_cfg.get("split_values")
        val_values = set([v.lower() for v in (split_vals or ["val", "valid", "validation"])])
        pairs = set()
        if mode == "by_user":
            val_users = set()
            for user_idx, user in enumerate(data):
                for session_idx, sess in enumerate((user or {}).get("sessions", [])):
                    sm = (sess or {}).get("session_metadata", {}) or {}
                    v = sm.get(split_key)
                    if isinstance(v, str) and v.strip().lower() in val_values:
                        val_users.add(user_idx)
                        break
            if val_users:
                for user_idx, user in enumerate(data):
                    if user_idx not in val_users:
                        continue
                    for session_idx, _ in enumerate((user or {}).get("sessions", [])):
                        pairs.add((user_idx, session_idx))
                return pairs
        else:
            for user_idx, user in enumerate(data):
                for session_idx, sess in enumerate((user or {}).get("sessions", [])):
                    sm = (sess or {}).get("session_metadata", {}) or {}
                    v = sm.get(split_key)
                    if isinstance(v, str) and v.strip().lower() in val_values:
                        pairs.add((user_idx, session_idx))
            if pairs:
                return pairs

        # 3) deterministic sampling
        ratio = float(val_cfg.get("ratio", 0.1))
        seed = int(val_cfg.get("seed", 42))
        pairs = set()
        if mode == "by_user":
            for user_idx, user in enumerate(data):
                h = hashlib.md5(f"{seed}:user:{user_idx}".encode("utf-8")).hexdigest()
                r = int(h[:8], 16) / 0xFFFFFFFF
                if r < ratio:
                    for session_idx, _ in enumerate((user or {}).get("sessions", [])):
                        pairs.add((user_idx, session_idx))
        else:
            for user_idx, user in enumerate(data):
                for session_idx, _ in enumerate((user or {}).get("sessions", [])):
                    h = hashlib.md5(f"{seed}:{user_idx}:{session_idx}".encode("utf-8")).hexdigest()
                    # map to [0,1)
                    r = int(h[:8], 16) / 0xFFFFFFFF
                    if r < ratio:
                        pairs.add((user_idx, session_idx))
        return pairs

    @staticmethod
    def _build_user_split_from_users(data, val_users: set, val_frac: float) -> dict:
        """Build a split dict compatible with rl/sft_pretraining/split_users.py."""
        sess_counts = {u: len((user or {}).get("sessions", [])) for u, user in enumerate(data)}
        total_sessions = int(sum(sess_counts.values()))
        val_users_sorted = sorted([u for u in val_users if u in sess_counts])
        train_users_sorted = sorted([u for u in sess_counts.keys() if u not in set(val_users_sorted)])

        val_sessions = int(sum(sess_counts[u] for u in val_users_sorted))
        train_sessions = int(sum(sess_counts[u] for u in train_users_sorted))

        return {
            "train_user_idxs": train_users_sorted,
            "val_user_idxs": val_users_sorted,
            "sessions_per_user": {str(k): int(v) for k, v in sess_counts.items()},
            "total_sessions": total_sessions,
            "val_sessions": val_sessions,
            "train_sessions": train_sessions,
            "val_frac_requested": float(val_frac),
            "val_frac_actual_users": float(len(val_users_sorted) / len(sess_counts)) if sess_counts else 0.0,
            "val_frac_actual_sessions": float(val_sessions / total_sessions) if total_sessions else 0.0,
        }

    def _run_conv_state_ambiguity(self, annotation_path: str, components_path: str, domain_params: dict, amb_cfg: dict) -> None:
        """Run a 2nd conv_state annotation pass, then produce a disagreement-based review set."""
        from ..analysis.conv_state_ambiguity import run_conv_state_ambiguity_review

        # We assume the first run already produced annotations_conv_state.csv in annotation_path.
        ann_dir = Path(annotation_path)
        a1 = ann_dir / "annotations_conv_state.csv"
        if not a1.exists():
            logger.warning("[TOPA] conv_state_ambiguity enabled but first-run annotations_conv_state.csv not found.")
            return

        # Copy first run aside
        tmp_a1 = ann_dir / "annotations_conv_state__run1.tmp.csv"
        tmp_a1.write_bytes(a1.read_bytes())
        # and usage
        u1 = ann_dir / "usage_conv_state.json"
        tmp_u1 = ann_dir / "usage_conv_state__run1.tmp.json"
        if u1.exists():
            tmp_u1.write_bytes(u1.read_bytes())

        # Second run (overwrite)
        component_params = self.annotations_params.get("conv_state_params", {})
        if "num_processes" not in component_params:
            component_params["num_processes"] = self.annotations_params.get("num_processes", 1)
        loaded_comp = load_json(self._resolve_component_path(components_path, "conv_state"))
        annotator = self.annotators["conv_state"](
            domain=self.domain,
            domain_params=domain_params,
            api_key=self.api_key,
            max_retries=self.max_retries,
            **component_params,
        )
        annotator.annotate(self.data, loaded_comp, "conv_state", annotation_path)

        a2 = ann_dir / "annotations_conv_state.csv"
        a2_v2 = ann_dir / "annotations_conv_state_v2.csv"
        u2 = ann_dir / "usage_conv_state.json"
        u2_v2 = ann_dir / "usage_conv_state_v2.json"
        # move second run to v2
        if a2.exists():
            a2_v2.write_bytes(a2.read_bytes())
        if u2.exists():
            u2_v2.write_bytes(u2.read_bytes())

        # restore run1 to canonical name
        a1.write_bytes(tmp_a1.read_bytes())
        tmp_a1.unlink(missing_ok=True)
        if tmp_u1.exists():
            u1.write_bytes(tmp_u1.read_bytes())
            tmp_u1.unlink(missing_ok=True)

        # run review set generation
        conv_states_json = self._resolve_component_path(components_path, "conv_state")
        out_dir = str(ann_dir)
        run_conv_state_ambiguity_review(
            conversation_states_json=conv_states_json,
            ann_csv_1=str(a1),
            ann_csv_2=str(a2_v2),
            data=self.data,
            domain_params=domain_params,
            out_dir=out_dir,
            none_value=str(amb_cfg.get("none_value", "none")),
            n_review=int(amb_cfg.get("n_review", 200)),
            context_length=int(amb_cfg.get("context_length", 10)),
        )
        logger.info("[TOPA] conv_state ambiguity review set generated.")