import importlib
from typing import Any, Optional

ALL_ITEMS = []
PART1_ITEMS = []
PART2_ITEMS = []

_reward_mod = None

def _slugify(name: str) -> str:
    out = []
    for ch in str(name).strip().lower():
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch in {" ", "-"}:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "cbt"

def _load_reward_module(domain: Optional[str] = None):
    domain = _slugify(domain or "cbt")
    return importlib.import_module(f"topa.reward.{domain}")

def set_reward_domain(domain: str) -> None:
    """Set the active reward module for this prompt shim."""
    global _reward_mod, ALL_ITEMS, PART1_ITEMS, PART2_ITEMS
    global annotate_extrinsic_reward_prompt, build_prompt, format_anchors_for_prompt

    _reward_mod = _load_reward_module(domain)

    ALL_ITEMS = getattr(_reward_mod, "ALL_ITEMS", [])
    PART1_ITEMS = getattr(_reward_mod, "PART1_ITEMS", [])
    PART2_ITEMS = getattr(_reward_mod, "PART2_ITEMS", [])

    build_prompt = getattr(_reward_mod, "build_prompt", None)
    annotate_extrinsic_reward_prompt = getattr(_reward_mod, "annotate_extrinsic_reward_prompt", None)
    format_anchors_for_prompt = getattr(_reward_mod, "format_anchors_for_prompt", None)

    if annotate_extrinsic_reward_prompt is None:
        if callable(build_prompt):
            def _shim(session_metadata: dict, session_transcript: str) -> str:
                return build_prompt(session_metadata=session_metadata, session_transcript=session_transcript)
            annotate_extrinsic_reward_prompt = _shim
        else:
            def _err(*_: Any, **__: Any) -> str:
                raise ValueError("No prompt builder available in the reward module.")
            annotate_extrinsic_reward_prompt = _err

    if build_prompt is None and callable(annotate_extrinsic_reward_prompt):
        def _build_prompt(*, session_metadata: dict, session_transcript: str | None = None,
                          conversation_transcript: str | None = None, **_: Any) -> str:
            transcript = session_transcript or conversation_transcript or ""
            return annotate_extrinsic_reward_prompt(session_metadata=session_metadata, session_transcript=transcript)
        build_prompt = _build_prompt

    if format_anchors_for_prompt is None:
        def _no_anchors(_: str) -> str:
            return ""
        format_anchors_for_prompt = _no_anchors

def get_reward_module():
    global _reward_mod
    if _reward_mod is None:
        set_reward_domain("cbt")
    return _reward_mod

# Initialize with CBT for backward compatibility.
set_reward_domain("cbt")

__all__ = [
    "ALL_ITEMS",
    "PART1_ITEMS",
    "PART2_ITEMS",
    "annotate_extrinsic_reward_prompt",
    "build_prompt",
    "format_anchors_for_prompt",
    "set_reward_domain",
    "get_reward_module",
]
