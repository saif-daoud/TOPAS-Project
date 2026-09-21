

from typing import Any, Dict, List, Optional, Sequence

from ...rl.simulation.roles import RoleLabels
from .prompts import P4G_Act, P4G_roleplay

OUTPUT_FIELDS = ["label", "reward", "raw"]

__all__ = [
    "P4G_Act",
    "P4G_roleplay",
    "OUTPUT_FIELDS",
    "build_prompt",
    "parse_output",
    "reward_fn",
]


def _transcript_to_conversation(transcript: str, roles: RoleLabels) -> List[Dict[str, str]]:
    convo: List[Dict[str, str]] = []
    if not transcript:
        return convo
    sys_label = roles.system.strip().lower()
    usr_label = roles.user.strip().lower()
    for line in transcript.splitlines():
        if ":" not in line:
            continue
        role_raw, text = line.split(":", 1)
        role_clean = role_raw.split("(", 1)[0].strip()
        role_norm = role_clean.lower()
        text = text.strip()
        if not text:
            continue
        if role_norm == sys_label:
            role = "Persuader"
        elif role_norm == usr_label:
            role = "Persuadee"
        else:
            role = role_clean if role_clean else role_raw.strip()
        convo.append({"role": role, "content": text})
    return convo


def _parse_p4g_label(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "refus" in s:
        return "refused"
    if "neutral" in s:
        return "neutral"
    if "positive" in s:
        return "positive"
    if "decided" in s or "donate" in s or "agree" in s:
        return "agree"
    return "unknown"


def build_prompt(
    *,
    session_metadata: Optional[Dict[str, Any]] = None,
    session_transcript: Optional[str] = None,
    conversation_transcript: Optional[str] = None,
    **_: Any,
) -> str:
    roles = RoleLabels(system="persuader", user="persuadee")
    transcript = session_transcript or conversation_transcript or ""
    convo = _transcript_to_conversation(transcript, roles)
    prompt = P4G_roleplay(
        case=session_metadata or {},
        role="critic",
        conversation=convo,
        emotion_states=[],
        action=None,
    )
    system = str(prompt[0].get("content", ""))
    user = str(prompt[1].get("content", ""))
    if system:
        return f"System:\n{system}\n\nUser:\n{user}"
    return user


def parse_output(raw: str) -> Dict[str, Any]:
    label = _parse_p4g_label(raw)
    reward_map = {
        "refused": -1.0,
        "neutral": -0.5,
        "positive": 0.1,
        "agree": 1.0,
        "unknown": 0.0,
    }
    reward = float(reward_map.get(label, 0.0))
    return {"label": label, "reward": reward, "overall_mean": reward, "raw": raw}


def reward_fn(
    *,
    generator,
    conversation_transcript: str,
    session_metadata: Optional[Dict[str, Any]] = None,
    session: Optional[Dict[str, Any]] = None,
    metadata_keys: Optional[Sequence[str]] = None,
    judge_system_prompt: str = "",
    **_: Any,
) -> Dict[str, Any]:
    """P4G extrinsic reward using the built-in critic prompt."""
    roles = RoleLabels(system="persuader", user="persuadee")
    convo = _transcript_to_conversation(conversation_transcript, roles)

    prompt = P4G_roleplay(case=session_metadata or {}, role="critic", conversation=convo, emotion_states=[], action=None)
    system = str(prompt[0].get("content", ""))
    user = str(prompt[1].get("content", ""))

    raw = generator.generate(system=system or judge_system_prompt, user=user)
    return parse_output(raw)
