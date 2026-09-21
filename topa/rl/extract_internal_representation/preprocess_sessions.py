"""Write one role-tagged transcript per dataset session."""

import argparse
import os
from pathlib import Path

from ..sft_pretraining.data_loading import DataRoot, get_default_prefix, load_dataset_json
from ..sft_pretraining.utils import role_labels, safe_name, speaker_role


def build_transcript(dialogue: list[dict], prefix: str, system_tag: str, user_tag: str) -> tuple[str, int]:
    transcript = prefix
    system_label, _ = role_labels()
    system_turns = 0

    for utterance in dialogue:
        tag = system_tag if speaker_role(str(utterance["speaker"])) == system_label else user_tag
        transcript += f"{tag} {utterance['text']}\n"
        system_turns += int(tag == system_tag)

    return transcript, system_turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="extract_internal_representation/transcripts")
    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--system_tag", type=str, default="")
    ap.add_argument("--user_tag", type=str, default="")
    ap.add_argument("--id_dir_prefix", type=str, default="")
    args = ap.parse_args()

    root = DataRoot(Path(args.root).resolve())
    system_label, user_label = role_labels()

    system_tag = args.system_tag or f"<|{system_label}|>"
    user_tag = args.user_tag or f"<|{user_label}|>"
    id_dir_prefix = args.id_dir_prefix or safe_name(user_label)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (root.root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_prefix = os.environ["TOPA_RL_PREFIX"] if "TOPA_RL_PREFIX" in os.environ else ""
    prefix = (args.prefix or env_prefix or get_default_prefix()).strip()
    if prefix and not prefix.endswith(" "):
        prefix += " "

    data = load_dataset_json(root)
    n_sessions = 0
    n_system_turns = 0

    for user_idx, user in enumerate(data):
        user_out = out_dir / f"{id_dir_prefix}_{user_idx}"
        user_out.mkdir(parents=True, exist_ok=True)

        for session_idx, session in enumerate(user["sessions"]):
            transcript, system_turns = build_transcript(session["dialogue"], prefix, system_tag, user_tag)
            assert transcript.count(system_tag) == system_turns
            (user_out / f"session_{session_idx}.txt").write_text(transcript, encoding="utf-8")
            n_sessions += 1
            n_system_turns += system_turns

    print(f"Wrote {n_sessions} transcripts to: {out_dir}")
    print(f"Sanity check: found {n_system_turns} {system_tag} turns")


if __name__ == "__main__":
    main()
