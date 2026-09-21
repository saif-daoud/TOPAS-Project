"""Standalone P4G simulator validation protocol.

This script compares TOPA's user simulator vs P4G-sim (p4g_sim) with:
1) Turn-aligned generation using *real persuader turns*.
2) Independent rubric scoring per transcript.
3) Blind, randomized pairwise comparison against the real anchor.
4) Multiple segments + samples for robustness.

Example:
  python -m topa.rl.validate_simulators_p4g \
    --config topa/config/config_p4g.yaml \
    --out-dir outputs/P4G/simulator_validation \
    --num-segments 15 --min-turns 15 --max-turns 30 --num-samples 3
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from ..config import Config
from .simulation.llm import TextGenerator, build_generator_from_env
from .simulation.profiles import UserProfileStore
from .simulation.roles import RoleLabels
from .simulation.text_utils import select_session_metadata
from .simulation.user_simulators import TopaUserSimulator

try:
    from .simulation.p4g_sim import P4GPersuasionSimulator
except Exception:
    P4GPersuasionSimulator = None


RUBRIC = [
    ("linguistic_naturalness", "Linguistic naturalness (repairs, hedging, human-like phrasing)"),
    ("emotional_plausibility", "Emotional plausibility (appropriate intensity, non-theatrical)"),
    ("consistency", "Consistency (stable traits/goals without random flips)"),
    ("non_cooperativeness_realism", "Non-cooperativeness realism (doesn’t unrealistically comply)"),
    ("local_coherence", "Local coherence (responds to persuader turn appropriately)"),
    ("global_coherence", "Global coherence (topic/trajectory makes sense across turns)"),
]


@dataclass
class Segment:
    segment_id: str
    user_idx: int
    session_idx: int
    start_turn: int
    n_turns: int
    pairs: List[Tuple[str, str]]  # list of (system_text, user_text)

    def real_dialogue(self, roles: RoleLabels) -> List[Dict[str, str]]:
        dialogue: List[Dict[str, str]] = []
        for sys_text, user_text in self.pairs:
            dialogue.append({"speaker": roles.system, "text": sys_text})
            dialogue.append({"speaker": roles.user, "text": user_text})
        return dialogue


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_dataset(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("users"), list):
        return data["users"]
    if isinstance(data, list):
        return data
    raise ValueError("Dataset must be a list of users or {'users': [...]} dict.")


def _is_system_speaker(sp: str, roles: RoleLabels) -> bool:
    s = (sp or "").strip().lower()
    return s in {"system", "assistant", roles.system.strip().lower()}


def _is_user_speaker(sp: str, roles: RoleLabels) -> bool:
    s = (sp or "").strip().lower()
    return s in {"user", "human", roles.user.strip().lower()}


def _extract_pairs(dialogue: List[Dict[str, Any]], roles: RoleLabels) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pending_sys: List[str] = []
    for turn in dialogue or []:
        sp = str(turn.get("speaker", "") or "")
        tx = str(turn.get("text", "") or "")
        if _is_system_speaker(sp, roles):
            if tx.strip():
                pending_sys.append(tx.strip())
            continue
        if _is_user_speaker(sp, roles):
            if not pending_sys:
                continue
            sys_text = " ".join(pending_sys).strip()
            pending_sys = []
            if tx.strip():
                pairs.append((sys_text, tx.strip()))
    return pairs


def _format_dialogue(dialogue: List[Dict[str, str]], roles: RoleLabels) -> str:
    out: List[str] = []
    for turn in dialogue:
        sp = turn.get("speaker", "")
        txt = (turn.get("text") or "").strip()
        if not txt:
            continue
        if _is_system_speaker(sp, roles):
            out.append(f"{roles.system}: {txt}")
        else:
            out.append(f"{roles.user}: {txt}")
    return "\n".join(out)


def _extract_session_metadata(
    user_obj: Any,
    sess: Any,
    session_idx: int,
    metadata_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    md: Dict[str, Any] = {}
    if isinstance(sess, dict):
        for k in ["metadata", "session_metadata", "meta"]:
            v = sess.get(k)
            if isinstance(v, dict):
                md.update(v)
                break
    if isinstance(user_obj, dict):
        for k in ["metadata", "user_metadata", "meta"]:
            v = user_obj.get(k)
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk not in md:
                        md[kk] = vv
                break
    md["user_idx"] = int(
        md.get(
            "user_idx",
            user_obj.get("user_idx", user_obj.get("id", user_obj.get("user_id", -1))),
        )
    ) if isinstance(user_obj, dict) else -1
    md["session_idx"] = int(md.get("session_idx", session_idx))
    return select_session_metadata(md, metadata_keys)


def _parse_json_maybe(raw: str) -> Optional[dict]:
    if not raw:
        return None
    s = raw.strip()
    # strip code fences
    if "```" in s:
        s = s.replace("```json", "```").replace("```JSON", "```")
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1].strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def _maybe_set_seed(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _build_scoring_prompt(transcript: str, roles: RoleLabels) -> Tuple[str, str]:
    system = (
        "You are a strict but fair evaluator. Focus ONLY on the user turns; "
        "treat persuader turns as context."
    )
    dims = "\n".join([f"- {k}: {desc}" for k, desc in RUBRIC])
    user = (
        "You are evaluating whether this user dialogue could plausibly come from a real human client "
        "in a real P4G session. Score 0–4 for each dimension with one-sentence justification per dimension.\n\n"
        f"Dimensions (0–4 each):\n{dims}\n\n"
        "Return JSON ONLY with the following shape:\n"
        "{\n"
        '  "scores": {<dimension_key>: <int 0-4>, ...},\n'
        '  "justifications": {<dimension_key>: <string>, ...}\n'
        "}\n\n"
        "Transcript:\n"
        f"{transcript}\n"
    )
    return system, user


def _build_pairwise_prompt(real_t: str, sim_x: str, sim_y: str, roles: RoleLabels) -> Tuple[str, str]:
    system = (
        "You are a strict but fair evaluator. Focus ONLY on the user turns; "
        "treat persuader turns as context."
    )
    user = (
        "You are given a REAL transcript and two SIMULATED transcripts (X and Y). "
        "Your task is to decide which simulated transcript is closer to the real one "
        "in terms of the USER responses.\n\n"
        "REAL transcript:\n"
        f"{real_t}\n\n"
        "SIMULATED transcript X:\n"
        f"{sim_x}\n\n"
        "SIMULATED transcript Y:\n"
        f"{sim_y}\n\n"
        "Answer in JSON ONLY:\n"
        "{\n"
        '  "winner": "X" | "Y" | "tie",\n'
        '  "cues": ["cue1", "cue2", "cue3"],\n'
        '  "confidence": <float 0-1>\n'
        "}\n"
        "Give exactly 3 concrete textual cues."
    )
    return system, user


def _score_transcript(
    judge: TextGenerator, transcript: str, roles: RoleLabels
) -> Dict[str, Any]:
    system, user = _build_scoring_prompt(transcript, roles)
    raw = judge.generate(system=system, user=user)
    parsed = _parse_json_maybe(raw)
    return {
        "raw": raw,
        "parsed": parsed,
    }


def _pairwise_compare(
    judge: TextGenerator,
    real_t: str,
    sim_x: str,
    sim_y: str,
    roles: RoleLabels,
) -> Dict[str, Any]:
    system, user = _build_pairwise_prompt(real_t, sim_x, sim_y, roles)
    raw = judge.generate(system=system, user=user)
    parsed = _parse_json_maybe(raw)
    return {
        "raw": raw,
        "parsed": parsed,
    }


def _bootstrap_ci(values: List[float], n_boot: int, ci: float, seed: int) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    if len(values) == 1:
        v = float(values[0])
        return {"mean": v, "low": v, "high": v}
    rng = random.Random(int(seed))
    means = []
    for _ in range(int(n_boot)):
        sample = [values[rng.randrange(0, len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    mean = sum(values) / len(values)
    alpha = (1.0 - float(ci)) / 2.0
    lo_idx = max(0, int(alpha * len(means)))
    hi_idx = min(len(means) - 1, int((1.0 - alpha) * len(means)) - 1)
    return {"mean": mean, "low": means[lo_idx], "high": means[hi_idx]}


def _read_validation_pairs(obj: Any) -> List[Tuple[int, int]]:
    if isinstance(obj, dict):
        if "pairs" in obj and isinstance(obj["pairs"], list):
            obj = obj["pairs"]
        elif "validation_sessions" in obj and isinstance(obj["validation_sessions"], list):
            obj = obj["validation_sessions"]
        elif "val_user_idxs" in obj and isinstance(obj["val_user_idxs"], list):
            return [(int(u), -1) for u in obj["val_user_idxs"]]
    pairs: List[Tuple[int, int]] = []
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                pairs.append((int(it[0]), int(it[1])))
    return pairs


def _resolve_pairs(dataset: List[Dict[str, Any]], pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for u, s in pairs:
        if s != -1:
            out.append((u, s))
            continue
        try:
            sessions = dataset[u]["sessions"]
        except Exception:
            continue
        for si in range(len(sessions)):
            out.append((u, si))
    return out


def _build_segments(
    dataset: List[Dict[str, Any]],
    roles: RoleLabels,
    *,
    num_segments: int,
    min_turns: int,
    max_turns: int,
    seed: int,
    restrict_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    unique_sessions: bool = True,
) -> List[Segment]:
    rng = random.Random(int(seed))
    candidates: List[Tuple[int, int, List[Tuple[str, str]]]] = []
    restrict = set((int(u), int(s)) for (u, s) in restrict_pairs) if restrict_pairs else None

    for user_idx, user in enumerate(dataset):
        for session_idx, sess in enumerate(user.get("sessions", [])):
            if restrict is not None and (user_idx, session_idx) not in restrict:
                continue
            dialogue = sess.get("dialogue", [])
            pairs = _extract_pairs(dialogue, roles)
            if len(pairs) >= min_turns:
                candidates.append((user_idx, session_idx, pairs))

    if not candidates:
        raise ValueError("No sessions with enough turn pairs to sample segments.")

    # Sample sessions
    sampled: List[Segment] = []
    used_sessions: set[Tuple[int, int]] = set()
    attempts = 0
    while len(sampled) < int(num_segments) and attempts < int(num_segments) * 20:
        attempts += 1
        user_idx, session_idx, pairs = rng.choice(candidates)
        if unique_sessions and (user_idx, session_idx) in used_sessions:
            continue
        n_pairs = len(pairs)
        max_len = min(int(max_turns), n_pairs)
        if max_len < int(min_turns):
            continue
        n_turns = rng.randint(int(min_turns), int(max_len))
        start = rng.randint(0, n_pairs - n_turns)
        seg_pairs = pairs[start : start + n_turns]
        seg_id = f"u{user_idx}_s{session_idx}_t{start}_len{n_turns}"
        sampled.append(
            Segment(
                segment_id=seg_id,
                user_idx=int(user_idx),
                session_idx=int(session_idx),
                start_turn=int(start),
                n_turns=int(n_turns),
                pairs=seg_pairs,
            )
        )
        used_sessions.add((user_idx, session_idx))

    if len(sampled) < int(num_segments):
        # Fallback: allow multiple segments per session if needed.
        if unique_sessions:
            return _build_segments(
                dataset,
                roles,
                num_segments=num_segments,
                min_turns=min_turns,
                max_turns=max_turns,
                seed=seed + 1,
                restrict_pairs=restrict_pairs,
                unique_sessions=False,
            )
        raise ValueError("Could not sample enough segments.")

    return sampled


def _simulate_segment(
    *,
    pairs: List[Tuple[str, str]],
    user_simulator,
    roles: RoleLabels,
) -> List[Dict[str, str]]:
    dialogue: List[Dict[str, str]] = []
    for sys_text, _ in pairs:
        dialogue.append({"speaker": roles.system, "text": sys_text})
        convo = _format_dialogue(dialogue, roles)
        user_text = user_simulator.respond(convo)
        dialogue.append({"speaker": roles.user, "text": user_text})
    return dialogue


def _extract_scores(parsed: Optional[dict]) -> Dict[str, Optional[float]]:
    out = {k: None for k, _ in RUBRIC}
    if not isinstance(parsed, dict):
        return out
    scores = parsed.get("scores")
    if isinstance(scores, dict):
        for k, _ in RUBRIC:
            try:
                v = scores.get(k)
                if v is None:
                    continue
                out[k] = float(v)
            except Exception:
                continue
    return out


def _sum_scores(score_map: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [v for v in score_map.values() if v is not None]
    if len(vals) != len(score_map):
        return None
    return float(sum(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="P4G simulator validation (TOPA vs Persuadee-Ψ).")
    parser.add_argument("--config", default="topa/config/config_p4g.yaml", help="P4G config yaml.")
    parser.add_argument("--out-dir", required=True, help="Output directory for validation results.")
    parser.add_argument("--rl-root", default="", help="RL root (contains components/annotations).")
    parser.add_argument("--validation", default="", help="validation_sessions.json path (optional).")
    parser.add_argument("--num-segments", type=int, default=15)
    parser.add_argument("--min-turns", type=int, default=15)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-repeat-sessions", action="store_true", help="Allow multiple segments from the same session.")
    parser.add_argument("--metadata-keys", default="", help="Comma-separated metadata keys to include.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--profile-builder-max-new-tokens", type=int, default=256)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-ci", type=float, default=0.95)
    parser.add_argument("--allow-non-p4g", action="store_true", help="Override P4G-only guard.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if not args.allow_non_p4g:
        domain_l = str(cfg.domain or "").strip().lower()
        if domain_l and domain_l != "p4g":
            raise ValueError("This script is intended for P4G only. Use --allow-non-p4g to override.")
    roles = RoleLabels(
        system=str(cfg.domain_params.get("system") or "SYSTEM"),
        user=str(cfg.domain_params.get("user") or "USER"),
    )

    dataset = _normalize_dataset(_load_json(cfg.data_path))

    md_keys = [k.strip() for k in str(args.metadata_keys).split(",") if k.strip()] or None
    if md_keys is None:
        cfg_md = (cfg.simulation_params or {}).get("metadata_keys") or (cfg.simulation_params or {}).get("session_metadata_keys")
        if isinstance(cfg_md, str):
            md_keys = [k.strip() for k in cfg_md.split(",") if k.strip()] or None
        elif isinstance(cfg_md, list):
            md_keys = [str(k).strip() for k in cfg_md if str(k).strip()] or None
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_root = Path(args.rl_root).expanduser().resolve() if args.rl_root else None
    if rl_root is None or not rl_root.exists():
        domain_adj = cfg.get_adj() or cfg.domain
        artifact_dir = str((cfg.rl_params or {}).get("artifact_dir") or "rl")
        guess = cfg.output_path / domain_adj / artifact_dir
        if guess.exists():
            rl_root = guess

    profile_store: Optional[UserProfileStore] = None
    if rl_root is not None and rl_root.exists():
        profile_json = rl_root / "components" / "user_profile.json"
        profile_csv = rl_root / "annotations" / "annotations_user_profile.csv"
        if profile_json.exists() and profile_csv.exists():
            profile_store = UserProfileStore(profile_json, profile_csv, cache_dir=out_dir / "profile_cache")

    # Validation sessions (optional)
    restrict_pairs = None
    val_path = Path(args.validation).expanduser().resolve() if args.validation else None
    if val_path is None or not val_path.exists():
        if rl_root is not None:
            guess = rl_root / "annotations" / "validation_sessions.json"
            if guess.exists():
                val_path = guess
    if val_path is not None and val_path.exists():
        val_obj = _load_json(val_path)
        pairs = _read_validation_pairs(val_obj)
        restrict_pairs = _resolve_pairs(dataset, pairs)

    segments = _build_segments(
        dataset,
        roles,
        num_segments=args.num_segments,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        seed=args.seed,
        restrict_pairs=restrict_pairs,
        unique_sessions=not bool(args.allow_repeat_sessions),
    )

    _dump_json(out_dir / "segments.json", [s.__dict__ for s in segments])

    # Generators
    user_gen = build_generator_from_env("USER")
    judge_gen = build_generator_from_env("JUDGE")
    user_gen.max_new_tokens = int(args.max_new_tokens) if int(args.max_new_tokens) > 0 else None
    judge_gen.max_new_tokens = int(args.judge_max_new_tokens) if int(args.judge_max_new_tokens) > 0 else None

    # no builder/cache needed for P4G simulator

    results_path = out_dir / "results.jsonl"

    real_scores_cache: Dict[str, Dict[str, Any]] = {}
    wins_sample = {"topa": 0, "p4g_sim": 0, "tie": 0}
    wins_segment = {"topa": 0, "p4g_sim": 0, "tie": 0}

    segment_votes: Dict[str, Dict[str, int]] = {}

    for seg_idx, seg in enumerate(tqdm(segments, desc="Segments", unit="segment")):
        user_obj = dataset[seg.user_idx]
        sess = user_obj["sessions"][seg.session_idx]
        session_metadata = _extract_session_metadata(user_obj, sess, seg.session_idx, metadata_keys=md_keys)

        conv_profile = ""
        if profile_store is not None:
            try:
                conv_profile = profile_store.format_profile_text(seg.user_idx, seg.session_idx)
            except Exception:
                conv_profile = ""

        # Real transcript (scored once per segment)
        real_dialogue = seg.real_dialogue(roles)
        real_transcript = _format_dialogue(real_dialogue, roles)
        if seg.segment_id not in real_scores_cache:
            real_scores_cache[seg.segment_id] = _score_transcript(judge_gen, real_transcript, roles)

        segment_votes[seg.segment_id] = {"topa": 0, "p4g_sim": 0, "tie": 0}

        for sample_id in tqdm(
            range(int(args.num_samples)),
            desc=f"Samples (segment {seg_idx + 1}/{len(segments)})",
            unit="sample",
            leave=False,
        ):
            seed_val = int(args.seed) + seg_idx * 1000 + sample_id
            # Build simulators
            topa_sim = TopaUserSimulator(
                generator=user_gen,
                session_metadata=session_metadata,
                profile_text=conv_profile,
                roles=roles,
            )
            if P4GPersuasionSimulator is None:
                raise RuntimeError("P4G persuasion simulator is not available.")
            psi_sim = P4GPersuasionSimulator(generator=user_gen, roles=roles)

            # Generate turn-aligned transcripts
            _maybe_set_seed(seed_val + 1)
            topa_dialogue = _simulate_segment(pairs=seg.pairs, user_simulator=topa_sim, roles=roles)
            _maybe_set_seed(seed_val + 2)
            psi_dialogue = _simulate_segment(pairs=seg.pairs, user_simulator=psi_sim, roles=roles)
            topa_transcript = _format_dialogue(topa_dialogue, roles)
            psi_transcript = _format_dialogue(psi_dialogue, roles)

            # Save transcripts
            seg_dir = out_dir / "transcripts" / seg.segment_id
            seg_dir.mkdir(parents=True, exist_ok=True)
            (seg_dir / "real.txt").write_text(real_transcript, encoding="utf-8")
            (seg_dir / f"topa_sample_{sample_id}.txt").write_text(topa_transcript, encoding="utf-8")
            (seg_dir / f"p4g_sim_sample_{sample_id}.txt").write_text(psi_transcript, encoding="utf-8")

            # Score transcripts
            score_real = real_scores_cache[seg.segment_id]
            score_topa = _score_transcript(judge_gen, topa_transcript, roles)
            score_psi = _score_transcript(judge_gen, psi_transcript, roles)

            # Pairwise (blind, randomized)
            rng_pair = random.Random(seed_val + 999)
            if rng_pair.random() < 0.5:
                sim_x, sim_y = topa_transcript, psi_transcript
                map_x, map_y = "topa", "p4g_sim"
            else:
                sim_x, sim_y = psi_transcript, topa_transcript
                map_x, map_y = "p4g_sim", "topa"

            pairwise = _pairwise_compare(judge_gen, real_transcript, sim_x, sim_y, roles)
            winner = None
            if isinstance(pairwise.get("parsed"), dict):
                winner = str(pairwise["parsed"].get("winner", "")).strip().lower()

            if winner == "x":
                wins_sample[map_x] += 1
                segment_votes[seg.segment_id][map_x] += 1
            elif winner == "y":
                wins_sample[map_y] += 1
                segment_votes[seg.segment_id][map_y] += 1
            else:
                wins_sample["tie"] += 1
                segment_votes[seg.segment_id]["tie"] += 1

            # Persist sample result
            row = {
                "segment_id": seg.segment_id,
                "user_idx": seg.user_idx,
                "session_idx": seg.session_idx,
                "start_turn": seg.start_turn,
                "n_turns": seg.n_turns,
                "sample_id": int(sample_id),
                "seed": int(seed_val),
                "scores": {
                    "real": score_real,
                    "topa": score_topa,
                    "p4g_sim": score_psi,
                },
                "pairwise": {
                    "mapping": {"X": map_x, "Y": map_y},
                    "result": pairwise,
                },
            }
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Aggregate scores
    scores_by_sim: Dict[str, Dict[str, List[float]]] = {
        "topa": {k: [] for k, _ in RUBRIC},
        "p4g_sim": {k: [] for k, _ in RUBRIC},
        "real": {k: [] for k, _ in RUBRIC},
    }
    totals_by_sim: Dict[str, List[float]] = {"topa": [], "p4g_sim": [], "real": []}

    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        for sim in ["topa", "p4g_sim"]:
            parsed = (obj.get("scores", {}).get(sim) or {}).get("parsed")
            s = _extract_scores(parsed)
            for k in s:
                if s[k] is not None:
                    scores_by_sim[sim][k].append(float(s[k]))
            total = _sum_scores(s)
            if total is not None:
                totals_by_sim[sim].append(total)

    # Real scores: use one per segment (avoid inflating counts by samples).
    for seg_id, payload in real_scores_cache.items():
        parsed = (payload or {}).get("parsed")
        s = _extract_scores(parsed)
        for k in s:
            if s[k] is not None:
                scores_by_sim["real"][k].append(float(s[k]))
        total = _sum_scores(s)
        if total is not None:
            totals_by_sim["real"].append(total)

    # Segment-level win rates
    for seg_id, votes in segment_votes.items():
        if votes["topa"] > votes["p4g_sim"]:
            wins_segment["topa"] += 1
        elif votes["p4g_sim"] > votes["topa"]:
            wins_segment["p4g_sim"] += 1
        else:
            wins_segment["tie"] += 1

    summary: Dict[str, Any] = {
        "n_segments": len(segments),
        "num_samples": int(args.num_samples),
        "wins_sample": wins_sample,
        "wins_segment_majority": wins_segment,
        "scores": {},
        "totals": {},
    }

    for sim in ["real", "topa", "p4g_sim"]:
        sim_scores = {}
        for k, _ in RUBRIC:
            sim_scores[k] = _bootstrap_ci(
                scores_by_sim[sim][k],
                n_boot=int(args.bootstrap_iters),
                ci=float(args.bootstrap_ci),
                seed=int(args.seed) + 7,
            )
        summary["scores"][sim] = sim_scores
        summary["totals"][sim] = _bootstrap_ci(
            totals_by_sim[sim],
            n_boot=int(args.bootstrap_iters),
            ci=float(args.bootstrap_ci),
            seed=int(args.seed) + 11,
        )

    summary["results_jsonl"] = str(results_path)
    _dump_json(out_dir / "summary.json", summary)

    run_info = {
        "config": str(Path(args.config).resolve()),
        "rl_root": str(rl_root) if rl_root is not None else "",
        "validation": str(val_path) if val_path is not None else "",
        "metadata_keys": md_keys,
        "args": vars(args),
        "timestamp": int(time.time()),
    }
    _dump_json(out_dir / "run_config.json", run_info)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()