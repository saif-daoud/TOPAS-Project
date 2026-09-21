"""Standalone extrinsic reward evaluator.

This module provides:
  - A default prompt builder resolved from the domain reward module.
  - A pluggable interface so users can define their own reward logic.

Override options (preferred):
  - reward_fn: "your.module:reward_fn"
    A callable that returns a JSON-like dict. It receives:
      generator, conversation_transcript, session_metadata, session, metadata_keys, judge_system_prompt
    and optionally prompt_builder.

  - prompt_builder: "your.module:build_prompt"
    A callable that builds the judge prompt and returns a string.
    It receives:
      session_metadata, session_transcript (or conversation_transcript), metadata_keys, session

The reward definition used by the default simulation runner is:
    success = 1.0 if overall_mean >= 4.0 else 0.0

Optional success override:
  - success_fn: "your.module:success_fn"
    Receives (reward_json, threshold, reward_type, reward_key, min_value, max_value)
    and returns a boolean or numeric value (>=1 treated as success).
"""

import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

# from ..utils import parse_json

def parse_json(output: str):
    cleaned_output = output.strip()
    start_tag = "```json"
    end_tag = "```"

    start_index = cleaned_output.find(start_tag)
    if start_index == -1:
        start_index = 0
    else:
        start_index += len(start_tag)

    cleaned_output = cleaned_output[start_index:]

    end_index = cleaned_output.rfind(end_tag)
    if end_index != -1:
        cleaned_output = cleaned_output[:end_index]
    
    return json.loads(cleaned_output.strip())

CTRS_KEYS = [
    "agenda","feedback","understanding","interpersonal_effectiveness","collaboration","pacing_time_use",
    "guided_discovery","focusing_on_key_cognitions_behaviors","strategy_for_change",
    "application_of_cbt_techniques","homework"
]

def _slugify(name: str) -> str:
    out = []
    for ch in str(name or "").strip().lower():
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch in {" ", "-"}:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "cbt"


def _load_reward_module(
    reward_module: Optional[str | Any] = None,
    domain: Optional[str] = None,
):
    """Resolve a reward module by explicit spec or by domain slug."""
    if reward_module is not None and not isinstance(reward_module, str):
        return reward_module

    spec = (reward_module or "").strip()
    if not spec:
        spec = f"topa.reward.{_slugify(domain or 'cbt')}"

    try:
        return importlib.import_module(spec)
    except Exception as e:
        raise ImportError(
            f"Could not import reward module '{spec}'. "
            "Create topa/reward/<domain>/ or set simulation.extrinsic_reward.reward_module explicitly."
        ) from e

def _flatten_ctrs_scores(reward_json: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric CTRS scores from a judge JSON.

    Supports a few common shapes:
    - {"codes": {"agenda": {"score": 4, ...}, ...}}
    - {"ctrs_scores": {"agenda": 4, ...}}
    - {"codes": {"agenda": 4, ...}}
    """

    scores: Dict[str, float] = {}

    # Prefer an already-flattened dict if available.
    raw_flat = reward_json.get("ctrs_scores", None)
    if isinstance(raw_flat, dict):
        for k, v in raw_flat.items():
            try:
                scores[str(k)] = float(v)
            except Exception:
                continue

    codes = reward_json.get("codes", None)
    if isinstance(codes, dict):
        for k, v in codes.items():
            key = str(k)
            val = None
            if isinstance(v, dict) and "score" in v:
                val = v.get("score")
            elif isinstance(v, (int, float, str)):
                val = v
            if val is None:
                continue
            try:
                scores[key] = float(val)
            except Exception:
                continue

    # Keep only the expected CTRS keys when possible.
    filtered = {k: scores[k] for k in CTRS_KEYS if k in scores}
    return filtered if filtered else scores


def _compute_overall_mean(reward_json: Dict[str, Any]) -> Optional[float]:
    """Return overall_mean if present, else compute from extracted CTRS scores."""

    try:
        if reward_json.get("overall_mean", None) is not None:
            return float(reward_json["overall_mean"])
    except Exception:
        pass

    scores = _flatten_ctrs_scores(reward_json)
    if not scores:
        return None
    return float(sum(scores.values()) / float(len(scores)))

def _load_callable(spec: Optional[str]) -> Optional[Callable[..., Any]]:
    if spec is None:
        return None
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        return None
    spec = spec.strip()
    if not spec:
        return None

    if ":" not in spec:
        raise ValueError("Expected callable spec in the form 'module:function' or '/path/file.py:function'")

    mod_spec, attr = spec.split(":", 1)
    mod_spec = mod_spec.strip()
    attr = attr.strip()
    if not attr:
        raise ValueError("Callable spec must include a function name after ':'")

    module = None
    if mod_spec.endswith(".py") or Path(mod_spec).exists():
        path = Path(mod_spec).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Callable module file not found: {path}")
        name = f"topa_user_reward_{path.stem}"
        spec_obj = importlib.util.spec_from_file_location(name, str(path))
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Could not load module from {path}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_spec)

    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"Callable '{attr}' not found in module '{mod_spec}'")
    return fn

def _resolve_callable_from_module(
    spec: Optional[str | Callable[..., Any]],
    reward_module: Any,
) -> Optional[Callable[..., Any]]:
    if callable(spec):
        return spec
    if isinstance(spec, str) and spec.strip():
        s = spec.strip()
        # If it looks like a module:function spec (or a file path), load it.
        if ":" in s or s.endswith(".py") or Path(s).exists():
            return _load_callable(s)
        # Otherwise treat as attribute on the reward module.
        fn = getattr(reward_module, s, None)
        if callable(fn):
            return fn
    return None

def _call_with_known_args(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call fn with only the args it accepts (unless it accepts **kwargs)."""
    sig = inspect.signature(fn)
    params = sig.parameters.values()
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
    if accepts_kwargs:
        return fn(**kwargs)
    allowed = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    filt = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(**filt)

def _normalize_reward_output(out: Any) -> Dict[str, Any]:
    if isinstance(out, dict):
        # Best-effort: if a numeric "reward" exists, mirror to overall_mean for compatibility.
        if "overall_mean" not in out and "reward" in out:
            try:
                out["overall_mean"] = float(out["reward"])
            except Exception:
                pass
        return out
    # Scalar -> wrap
    try:
        val = float(out)
        return {"reward": val, "overall_mean": val}
    except Exception:
        return {"reward": out}


def compute_extrinsic_reward(
    *,
    generator,
    conversation_transcript: str,
    session_metadata: Optional[Dict[str, Any]] = None,
    session: Optional[Dict[str, Any]] = None,
    metadata_keys: Optional[Sequence[str]] = None,
    judge_system_prompt: str = "You are a strict evaluator of CBT therapy sessions.",
    reward_fn: Optional[str | Callable[..., Any]] = None,
    prompt_builder: Optional[str | Callable[..., Any]] = None,
    parse_output: Optional[str | Callable[..., Any]] = None,
    reward_module: Optional[str | Any] = None,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute extrinsic reward by LLM-judging.

    Parameters
    ----------
    generator:
        Any object with `generate(system=..., user=...) -> str`.
        (Matches `simulation.llm.TextGenerator`.)
    conversation_transcript:
        Full transcript text.
    session_metadata:
        Session metadata dict (preferred if already extracted).
    session:
        Optional session dict; if provided and session_metadata is None,
        we will try to read session["session_metadata"] / "metadata" / "meta".
    reward_module:
        Optional reward module spec (e.g., "topa.reward.cbt"). If not provided,
        we resolve from `domain` and default to CBT.
    domain:
        Domain name used to auto-resolve the reward module (e.g., "CBT", "P4G").
    """

    if session_metadata is None:
        if isinstance(session, dict):
            session_metadata = session.get("session_metadata") or session.get("metadata") or session.get("meta") or {}
        else:
            session_metadata = {}

    reward_mod = None
    need_module = (
        reward_module is not None
        or domain is not None
        or (reward_fn is None and prompt_builder is None and parse_output is None)
    )
    if need_module:
        try:
            reward_mod = _load_reward_module(reward_module=reward_module, domain=domain)
        except ImportError:
            # If the caller provided explicit overrides, allow running without a module.
            if reward_fn is None and prompt_builder is None and parse_output is None:
                raise
            reward_mod = None

    # Resolve optional overrides.
    if reward_mod is not None:
        reward_fn = _resolve_callable_from_module(reward_fn, reward_mod)
        prompt_builder_fn = _resolve_callable_from_module(prompt_builder, reward_mod)
        parse_output_fn = _resolve_callable_from_module(parse_output, reward_mod)

        if parse_output_fn is None:
            parse_output_fn = getattr(reward_mod, "parse_output", None)
            if not callable(parse_output_fn):
                parse_output_fn = None

        if prompt_builder_fn is None:
            for name in ("build_prompt", "annotate_extrinsic_reward_prompt"):
                fn = getattr(reward_mod, name, None)
                if callable(fn):
                    prompt_builder_fn = fn
                    break
    else:
        reward_fn = _load_callable(reward_fn) if reward_fn else None
        prompt_builder_fn = _load_callable(prompt_builder) if prompt_builder else None
        parse_output_fn = _load_callable(parse_output) if parse_output else None

    if reward_fn is not None:
        out = _call_with_known_args(
            reward_fn,
            generator=generator,
            conversation_transcript=conversation_transcript,
            session_metadata=session_metadata or {},
            session=session,
            metadata_keys=metadata_keys,
            judge_system_prompt=judge_system_prompt,
            prompt_builder=prompt_builder_fn,
        )
        return _normalize_reward_output(out)

    if prompt_builder_fn is None:
        raise ValueError("No extrinsic reward prompt builder available.")
    prompt = _call_with_known_args(
        prompt_builder_fn,
        session_metadata=session_metadata or {},
        session_transcript=conversation_transcript,
        conversation_transcript=conversation_transcript,
        metadata_keys=metadata_keys,
        session=session,
    )
    prompt = str(prompt)
    
    for attempt in range(3):
        raw = generator.generate(system=judge_system_prompt, user=prompt)
        try:
            if parse_output_fn is not None:
                out = _call_with_known_args(parse_output_fn, raw=raw, output=raw)
            else:
                out = parse_json(raw)
        except Exception as e:
            try:
                if parse_output_fn is not None:
                    out = _call_with_known_args(parse_output_fn, raw=raw.replace("'", '"'), output=raw)
                else:
                    out = parse_json(raw.replace("'", '"'))
            except:
                print(f"[WARNING] JSON parsing failed at attempt {attempt+1}/3, error: {e} retrying...")
                continue
        break
    else:
        raise ValueError(f"Could not parse this : {raw}")
        
    # Normalize / enrich output for downstream consumers.
    scores = _flatten_ctrs_scores(out)
    if scores:
        out["ctrs_scores"] = {k: float(scores[k]) for k in CTRS_KEYS if k in scores}
        out["n_ctrs_scores"] = int(len(out["ctrs_scores"]))
    else:
        out["n_ctrs_scores"] = 0

    overall_mean = _compute_overall_mean(out)
    if overall_mean is not None:
        out["overall_mean"] = float(overall_mean)
    elif "overall_mean" not in out and "reward" in out:
        try:
            out["overall_mean"] = float(out["reward"])
        except Exception:
            pass

    return out

def _extract_reward_value(reward_json: Any, *, reward_key: Optional[str] = None) -> Optional[float]:
    if isinstance(reward_json, (int, float, bool)):
        try:
            return float(reward_json)
        except Exception:
            return None
    if not isinstance(reward_json, dict):
        return None

    if reward_key and reward_key in reward_json:
        try:
            return float(reward_json[reward_key])
        except Exception:
            return None

    for k in ["overall_mean", "reward", "score", "value", "success"]:
        if k in reward_json:
            try:
                return float(reward_json[k])
            except Exception:
                continue
    return None


def success_from_reward(
    reward_json: Any,
    threshold: float = 4.0,
    *,
    reward_type: str = "score",
    reward_key: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    rtype = str(reward_type or "score").strip().lower()

    if rtype in {"binary", "bool", "boolean"}:
        val = None
        if isinstance(reward_json, dict):
            for k in ["success", "is_success", "binary", "reward"]:
                if k in reward_json:
                    val = reward_json[k]
                    break
        else:
            val = reward_json
        return 1.0 if bool(val) else 0.0

    val = _extract_reward_value(reward_json, reward_key=reward_key)
    if val is None:
        return 0.0

    if min_value is not None:
        try:
            val = max(float(min_value), float(val))
        except Exception:
            pass
    if max_value is not None:
        try:
            val = min(float(max_value), float(val))
        except Exception:
            pass

    return 1.0 if float(val) >= float(threshold) else 0.0


def compute_success(
    reward_json: Any,
    *,
    success_fn: Optional[str | Callable[..., Any]] = None,
    threshold: float = 4.0,
    reward_type: str = "score",
    reward_key: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """Compute success using an optional user-defined function."""
    if success_fn:
        fn = _load_callable(success_fn) if not callable(success_fn) else success_fn
        out = _call_with_known_args(
            fn,
            reward_json=reward_json,
            reward=reward_json,
            threshold=threshold,
            reward_type=reward_type,
            reward_key=reward_key,
            min_value=min_value,
            max_value=max_value,
        )
        try:
            return float(out)
        except Exception:
            return 1.0 if bool(out) else 0.0

    return success_from_reward(
        reward_json,
        threshold=threshold,
        reward_type=reward_type,
        reward_key=reward_key,
        min_value=min_value,
        max_value=max_value,
    )
