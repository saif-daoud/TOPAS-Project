import json
from typing import Any, Dict, List


def verify_dialogue(dialogue: List[Dict[str, Any]]) -> bool:
    """
    Validate TOPA-style dialogue:
      - dialogue is a list of {"speaker": "system"|"user", "text": str}
    """
    if not isinstance(dialogue, list) or len(dialogue) == 0:
        return False

    speakers = set()
    allowed = {"user", "system"}

    for turn in dialogue:
        if not isinstance(turn, dict):
            return False
        if "speaker" not in turn or "text" not in turn:
            return False

        speaker = turn["speaker"]
        if speaker not in allowed:
            return False
        speakers.add(speaker)
    if len(allowed - speakers) != 0:
        print(dialogue)
        return False
    return True

def verify_session(session: Dict[str, Any]) -> bool:
    if not isinstance(session, dict):
        return False
    if "session_metadata" not in session:
        return False
    if "dialogue" not in session:
        return False
    if not isinstance(session["session_metadata"], dict):
        return False

    return verify_dialogue(session["dialogue"])

def verify_user_block(block: Dict[str, Any]) -> bool:
    if not isinstance(block, dict):
        return False
    if "user_metadata" not in block:
        return False
    if "sessions" not in block:
        return False

    if not isinstance(block["user_metadata"], dict):
        return False
    if not isinstance(block["sessions"], list):
        return False

    return all(verify_session(s) for s in block["sessions"])

def verify_topa_dataset(data: Any) -> bool:
    if not isinstance(data, list):
        return False
    return all(verify_user_block(item) for item in data)


def compute_dataset_stats(data: Any) -> dict:
    """Compute lightweight dataset stats (for sanity checks / reporting)."""
    if not isinstance(data, list):
        return {}

    n_users = len(data)
    n_sessions = 0
    turns_per_session: List[int] = []
    sys_turns = 0
    usr_turns = 0

    for user in data:
        sessions = user.get("sessions", []) if isinstance(user, dict) else []
        for sess in sessions:
            dialogue = (sess or {}).get("dialogue", []) if isinstance(sess, dict) else []
            n_sessions += 1
            turns_per_session.append(len(dialogue))
            for t in dialogue:
                sp = (t or {}).get("speaker")
                if sp == "system":
                    sys_turns += 1
                elif sp == "user":
                    usr_turns += 1

    if n_sessions == 0:
        return {
            "n_users": n_users,
            "n_sessions": 0,
        }

    turns_sorted = sorted(turns_per_session)
    mid = len(turns_sorted) // 2
    if len(turns_sorted) % 2 == 0:
        median_turns = 0.5 * (turns_sorted[mid - 1] + turns_sorted[mid])
    else:
        median_turns = turns_sorted[mid]

    total_turns = int(sum(turns_per_session))

    return {
        "n_users": n_users,
        "n_sessions": n_sessions,
        "total_turns": total_turns,
        "system_turns": sys_turns,
        "user_turns": usr_turns,
        "avg_turns_per_session": total_turns / n_sessions,
        "median_turns_per_session": median_turns,
        "min_turns_per_session": int(turns_sorted[0]),
        "max_turns_per_session": int(turns_sorted[-1]),
        "avg_sessions_per_user": n_sessions / max(1, n_users),
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_check.py <dataset.json>")
        exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if verify_topa_dataset(data):
        print("✓ Dataset is valid TOPA format.")
        stats = compute_dataset_stats(data)
        print("\nDataset statistics:")
        for k, v in stats.items():
            print(f"- {k}: {v}")
    else:
        print("✗ Dataset is NOT valid TOPA format.")