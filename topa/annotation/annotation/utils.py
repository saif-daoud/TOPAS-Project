import json
import os
from typing import Any, Dict, List

from .data_check import verify_topa_dataset, compute_dataset_stats

from ..utils import build_logger, format_transcript
logger = build_logger()

def load_json(data_path, verify_struct=False):
    # Load JSON from file
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if verify_struct:
        # Validate format
        if verify_topa_dataset(data):
            logger.info("✓ Dataset have a valid TOPA format.")
            stats = compute_dataset_stats(data)
            if stats:
                logger.info(
                    "[Dataset stats] "
                    + ", ".join([f"{k}={v}" for k, v in stats.items()])
                )
            return data
        else:
            raise Exception(f"✗ The Dataset in '{data_path}' is NOT valid TOPA format. Please check the format described in topa/annotation/readme.md")
    else:
        return data

def save_json(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path

def get_idx(utt_id: str) -> int:
    """Extract numeric index from 'speaker_XX'."""
    return int(utt_id.split("_")[-1])

def format_utterances(utterances: List[Dict[str, Any]]) -> str:
    """Human readable string block for prompts."""
    return format_transcript(utterances)
