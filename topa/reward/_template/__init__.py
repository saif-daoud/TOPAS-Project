"""Template reward module for new domains.

Copy this folder to `topa/reward/<your_domain>/` and customize.
"""



from typing import Any, Dict, Optional

# Optional: used by the extrinsic reward annotator to select which keys to export.
OUTPUT_FIELDS = ["reward", "raw"]


def build_prompt(
    *,
    session_metadata: Optional[Dict[str, Any]] = None,
    session_transcript: Optional[str] = None,
    conversation_transcript: Optional[str] = None,
    **_: Any,
) -> str:
    """Return the judge prompt for your domain."""
    transcript = session_transcript or conversation_transcript or ""
    meta = session_metadata or {}
    return (
        "You are an expert evaluator. Read the transcript and output JSON only.\n\n"
        f"Context: {meta}\n\nTranscript:\n{transcript}\n\n"
        "Return JSON: {\"reward\": <number>, \"raw\": \"...\"}"
    )


def parse_output(raw: str) -> Dict[str, Any]:
    """Parse the LLM output into a dict."""
    # Minimal fallback: store raw and a default reward.
    return {"reward": 0.0, "raw": raw}


def to_row(
    *,
    outputs: Dict[str, Any],
    user_idx: int,
    session_idx: int,
    **_: Any,
) -> Dict[str, Any]:
    """Optional: customize how annotation rows are written."""
    row = {"user_idx": user_idx, "session_idx": session_idx}
    row.update(outputs or {})
    return row