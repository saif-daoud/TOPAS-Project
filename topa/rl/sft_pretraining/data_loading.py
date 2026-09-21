from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import os
import pandas as pd

from .utils import (
    load_json,
    is_none_like,
    norm_str,
    speaker_role,
    format_dialogue,
    role_labels,
    set_role_labels,
)

IGNORE_CONV_STATE_COLS = {"user_idx", "session_idx", "phase_idx", "from_id", "to_id", "from_idx", "to_idx", 
                          "selected_macro_action"}

def get_default_prefix() -> str:
    system_label, _ = role_labels()
    return f"Act as the {system_label} agent and continue the conversation: "

DEFAULT_PREFIX = ""

def set_default_prefix(prefix: str | None = None) -> str:
    global DEFAULT_PREFIX
    if prefix is None or not str(prefix).strip():
        prefix = os.environ.get("TOPA_RL_PREFIX") or get_default_prefix()
    DEFAULT_PREFIX = str(prefix).strip()
    return DEFAULT_PREFIX

set_default_prefix()

@dataclass
class DataRoot:
    root: Path

    def __post_init__(self) -> None:
        labels_path = self.root / "role_labels.json"
        if labels_path.exists():
            try:
                payload = load_json(labels_path)
                set_role_labels(payload.get("system"), payload.get("user"))
            except Exception:
                pass
        set_default_prefix()

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def components_dir(self) -> Path:
        return self.root / "components"

    @property
    def data_dir(self) -> Path:
        """Directory that contains the dataset JSON.

        Priority:
        1) <root>/data (recommended)
        2) env var TOPA_DATA_DIR (domain-agnostic)
        3) env var CBT_DATA_DIR (legacy compatibility)
        4) legacy hardcoded path if it exists
        """
        local = self.root / "data"
        if local.exists():
            return local
        env = (os.environ.get("TOPA_DATA_DIR") or os.environ.get("CBT_DATA_DIR") or "").strip()
        if env:
            p = Path(env)
            if p.exists():
                return p
        legacy = Path("/home/local/QCRI/saif.sedaoud/clean-env/TOPA/topa/data/datasets/CBT")
        if legacy.exists():
            return legacy
        # Fall back to <root>/data even if missing; downstream will raise a clear error.
        return local

    @property
    def dataset_json_path(self) -> Path:
        """Resolve the dataset JSON path.

        Supports:
          - TOPA_DATASET_PATH: absolute or relative file path
          - TOPA_DATASET_JSON: filename under data_dir
        If none are set, we fall back to AlexanderStreet_Dataset.json for
        backward compatibility.
        """

        p = (os.environ.get("TOPA_DATASET_PATH") or "").strip()
        if p:
            pp = Path(p).expanduser()
            if not pp.is_absolute():
                pp = (self.root / pp).resolve()
            return pp

        fname = (os.environ.get("TOPA_DATASET_JSON") or "").strip()
        if fname:
            return self.data_dir / fname

        # Backward compatible default
        return self.data_dir / "AlexanderStreet_Dataset.json"
    
def load_dataset_json(root: DataRoot) -> list[dict]:
    return load_json(root.dataset_json_path)

def load_conv_state_annotations(root: DataRoot) -> pd.DataFrame:
    """Load conversation-state annotations.

    Prefer the *updated* file produced by the conversation-state refinement
    phase, but fall back to the raw file if needed.
    """
    p = root.annotations_dir / (os.environ.get("TOPA_CONV_STATE_ANN_FILE", "updated_annotations_conv_state.csv"))
    if p.exists():
        return pd.read_csv(p)
    fallback = root.annotations_dir / "annotations_conv_state.csv"
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(
        f"Conversation-state annotations not found. Tried {p} and {fallback}."
    )

def _parse_csv_list_env(name: str) -> list[str]:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def load_macro_action_annotations(root: DataRoot) -> pd.DataFrame:
    """Load macro-action annotations with optional domain-agnostic filtering."""
    fname = (os.environ.get("TOPA_MACRO_ANN_FILE") or "annotations_macro_actions.csv").strip()
    df = pd.read_csv(root.annotations_dir / fname)

    # -----------------------------
    # Indexing compatibility (internal-representation mode)
    # -----------------------------
    # Some exports encode indices as strings like "utterance_12".
    # RL/SFT activation mode assumes indices are integers counting system turns.
    def _parse_utt_id(x):
        if pd.isna(x):
            return None
        s = str(x)
        try:
            return int(s)
        except Exception:
            pass
        m = None
        if "_" in s:
            try:
                m = int(s.split("_")[-1])
            except Exception:
                m = None
        return m

    if "from_idx" not in df.columns and "from_id" in df.columns:
        df["from_idx"] = df["from_id"].map(_parse_utt_id)
    if "to_idx" not in df.columns and "to_id" in df.columns:
        df["to_idx"] = df["to_id"].map(_parse_utt_id)

    # Optional filters
    ignore = set(_parse_csv_list_env("TOPA_RL_IGNORE_MACRO"))
    ignore.add("uncertain")
    if ignore:
        df = df[~df["selected_macro_action"].astype(str).isin(ignore)]

    try:
        min_conf = float(os.environ.get("TOPA_RL_MIN_CONF", "0.8"))
    except Exception:
        min_conf = 0.8
    if "confidence_score" in df.columns:
        df = df.loc[df["confidence_score"] > min_conf]

    return df.reset_index(drop=True)

def load_micro_action_annotations(root: DataRoot) -> pd.DataFrame:
    """Load micro-action annotations with optional domain-agnostic filtering."""
    # Prefer refined micro-action annotations if available.
    preferred = root.annotations_dir / (os.environ.get("TOPA_MICRO_ANN_FILE", "annotations_micro_actions_refined.csv"))
    if preferred.exists():
        df = pd.read_csv(preferred)
    else:
        df = pd.read_csv(root.annotations_dir / "annotations_micro_actions.csv")

    ignore = set(_parse_csv_list_env("TOPA_RL_IGNORE_MICRO"))
    if ignore:
        df = df[~df["selected_micro_action"].astype(str).isin(ignore)]

    try:
        min_conf = float(os.environ.get("TOPA_RL_MIN_CONF", "0.8"))
    except Exception:
        min_conf = 0.8
    if "confidence_score" in df.columns:
        df = df.loc[df["confidence_score"] > min_conf]

    df = df.dropna(subset=["selected_micro_action"])

    # -----------------------------
    # Indexing compatibility (internal-representation mode)
    # -----------------------------
    # Older exports may have "utterance_id" like "utterance_7" instead of an int.
    if "utterance_idx" not in df.columns and "utterance_id" in df.columns:
        def _parse(x):
            if pd.isna(x):
                return None
            s = str(x)
            try:
                return int(s)
            except Exception:
                pass
            if "_" in s:
                try:
                    return int(s.split("_")[-1])
                except Exception:
                    return None
            return None
        df["utterance_idx"] = df["utterance_id"].map(_parse)

    # Ensure ints if present
    if "utterance_idx" in df.columns:
        df["utterance_idx"] = pd.to_numeric(df["utterance_idx"], errors="coerce")
    df = df.reset_index(drop=True)
    return df

def load_conversation_state_spec(root: DataRoot) -> list[dict]:
    return load_json(root.components_dir / "conversation_states.json")

def load_action_space(root: DataRoot) -> list[dict]:
    # Prefer action_space.json, but fall back to micro_actions.json (produced by the pipeline).
    p = root.components_dir / "action_space.json"
    if p.exists():
        return load_json(p)
    fallback = root.components_dir / "micro_actions.json"
    if fallback.exists():
        return load_json(fallback)
    raise FileNotFoundError(f"Action space JSON not found. Tried {p} and {fallback}.")

def build_conv_state_label_maps(spec: list[dict]) -> tuple[list[str], Dict[str, Dict[str, int]]]:
    """Returns:
    - dim_names: in the same order as spec
    - dim_to_label2id: maps dim_name -> (normalized_label -> id)
    Adds a special 'none' label at the end for every dimension.
    """
    dim_names: list[str] = []
    dim_to_label2id: Dict[str, Dict[str, int]] = {}

    for d in spec:
        dim = d["Variable Name"]
        dim_names.append(dim)

        cats = d["Categorical values"]
        # normalize
        cats = [norm_str(c) for c in cats if not is_none_like(c)]
        uniq = []
        seen = set()
        for c in cats:
            key = c.lower()
            if key not in seen:
                uniq.append(c)
                seen.add(key)

        uniq.append("none")
        label2id = {c.lower(): i for i, c in enumerate(uniq)}
        dim_to_label2id[dim] = label2id

    return dim_names, dim_to_label2id

def build_macro_action_map(action_space: list[dict]) -> Dict[str, int]:
    names = [a["name"] for a in action_space]
    return {n: i for i, n in enumerate(names)}

def build_micro_action_maps(action_space: list[dict]) -> tuple[Dict[str, int], Dict[int, Dict[str, int]]]:
    """Returns:
    - macro_name_to_id
    - macro_id_to_micro_label2id (each macro has its own micro label space)
    """
    macro_name_to_id = build_macro_action_map(action_space)
    macro_id_to_micro: Dict[int, Dict[str, int]] = {}
    for macro in action_space:
        macro_id = macro_name_to_id[macro["name"]]
        micro_names = [m["name"] for m in macro["micro_actions"]]
        micro_label2id = {n: i for i, n in enumerate(micro_names)}
        macro_id_to_micro[macro_id] = micro_label2id
    return macro_name_to_id, macro_id_to_micro

def build_flat_micro_label_map(action_space: list[dict]) -> Tuple[List[str], Dict[str, int]]:
    """Return (class_names, label2id) for flat micro policy.

    We disambiguate micro actions by prefixing macro:
        label = "<macro>::<micro>"
    """
    class_names: List[str] = []
    label2id: Dict[str, int] = {}

    for m in action_space:
        macro = str(m["name"])
        micro_list = m["micro_actions"]

        for micro in micro_list:
            micro = str(micro["name"])
            lab = f"{macro}::{micro}"
            if lab not in label2id:
                label2id[lab] = len(class_names)
                class_names.append(lab)

    return class_names, label2id

def get_dialogue(data_json: list[dict], user_idx: int, session_idx: int) -> list[dict]:
    return data_json[user_idx]["sessions"][session_idx]["dialogue"]

def segment_end_with_last_user(dialogue: list[dict], to_idx: int) -> int:
    end = to_idx
    nxt = to_idx + 1
    if nxt < len(dialogue):
        _, user_label = role_labels()
        if speaker_role(dialogue[nxt]["speaker"]) == user_label:
            end = nxt
    return end

def build_macro_policy_context_text(data_json: list[dict], row: dict, prefix: str | None = None, max_turns: int = 64) -> str:
    """Macro-action policy context: ONLY utterances that precede the labeled system action.

    Here we assume the action label corresponds to the system utterance at `from_idx`.
    We therefore end the context at `from_idx - 1` (inclusive). We always prepend `prefix`
    so that early turns don't yield an empty input.
    """
    dialogue = get_dialogue(data_json, int(row["user_idx"]), int(row["session_idx"]))
    end = row["from_idx"] - 1
    if prefix is None:
        prefix = get_default_prefix()
    ctx = format_dialogue(dialogue, 0, end, max_turns=max_turns)
    if ctx:
        return prefix + "\n" + ctx
    return prefix

def build_conversation_text(data_json: list[dict], row: dict, max_turns: int = 64) -> str:
    dialogue = get_dialogue(data_json, row["user_idx"], row["session_idx"])
    # start = row["from_idx"]
    end = segment_end_with_last_user(dialogue, row["to_idx"])
    return format_dialogue(dialogue, 0, end, max_turns=max_turns)

def build_micro_action_text(
    data_json: list[dict],
    row: dict,
    prefix: str | None = None,
    context_turns: int | None = None,
) -> str:
    dialogue = get_dialogue(data_json, int(row["user_idx"]), int(row["session_idx"]))
    uidx = int(row["utterance_idx"])

    # Predict the micro action BEFORE the system utterance happens -> use context up to uidx-1.
    if prefix is None:
        prefix = get_default_prefix()
    ctx = format_dialogue(dialogue, 0, uidx - 1, max_turns=context_turns)
    if ctx:
        return prefix + "\n" + ctx
    return prefix