import importlib
import inspect
from typing import Any, Callable, Dict, Optional

from .base import BaseAnnotator
from ..utils import format_utterances

from ...utils import build_logger, parse_json, run_llm_query

logger = build_logger()

class ExtrinsicRewardAnnotator(BaseAnnotator):
    def __init__(
        self,
        domain: str,
        domain_params: Dict,
        api_key: str,
        max_retries: int = 3,
        inp_cost: float = 3.0,
        out_cost: float = 12.0,
        num_processes: int = 1,
        reward_module: Optional[str] = None,
        prompt_builder: Optional[str] = None,
        parse_output: Optional[str] = None,
        to_row: Optional[str] = None,
        output_fields: Optional[list] = None,
    ):
        super().__init__(
            domain=domain,
            domain_params=domain_params,
            api_key=api_key,
            max_retries=max_retries,
            inp_cost=inp_cost,
            out_cost=out_cost,
            num_processes=num_processes,
        )
        self.reward_module_spec = reward_module
        self.prompt_builder_spec = prompt_builder
        self.parse_output_spec = parse_output
        self.to_row_spec = to_row
        self.output_fields = output_fields

        self.reward_module = self._load_reward_module()
        self.prompt_builder = self._resolve_callable(
            self.prompt_builder_spec,
            default_names=("build_prompt", "annotate_extrinsic_reward_prompt"),
        )
        self.parse_output_fn = self._resolve_callable(self.parse_output_spec, default_names=("parse_output",))
        self.to_row_fn = self._resolve_callable(self.to_row_spec, default_names=("to_row",))
        if self.output_fields is None:
            self.output_fields = getattr(self.reward_module, "OUTPUT_FIELDS", None)

        # CTRS-style modules expose these items; used for strict sanity checks and scoring.
        self.ctrs_items = getattr(self.reward_module, "ALL_ITEMS", None)
        self.ctrs_part1 = getattr(self.reward_module, "PART1_ITEMS", None)
        self.ctrs_part2 = getattr(self.reward_module, "PART2_ITEMS", None)

    def _load_reward_module(self):
        if self.reward_module_spec is not None and not isinstance(self.reward_module_spec, str):
            return self.reward_module_spec

        spec = (self.reward_module_spec or "").strip()
        if not spec:
            domain = str(self.domain or "").strip().lower()
            safe = []
            for ch in domain:
                if ch.isalnum() or ch == "_":
                    safe.append(ch)
                elif ch in {" ", "-"}:
                    safe.append("_")
            suffix = "".join(safe).strip("_") or "cbt"
            spec = f"topa.reward.{suffix}"

        try:
            return importlib.import_module(spec)
        except Exception as e:
            raise ImportError(
                f"Could not import reward module '{spec}'. "
                "Create a reward module under topa/reward/<domain>/ or set "
                "annotations_params.extrinsic_reward_params.reward_module explicitly."
            ) from e

    @staticmethod
    def _call_with_known_args(fn: Callable[..., Any], **kwargs: Any) -> Any:
        sig = inspect.signature(fn)
        params = sig.parameters.values()
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
        if accepts_kwargs:
            return fn(**kwargs)
        allowed = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        filt = {k: v for k, v in kwargs.items() if k in allowed}
        return fn(**filt)

    def _resolve_callable(
        self,
        spec: Optional[str | Callable[..., Any]],
        default_names: tuple[str, ...] = (),
    ) -> Optional[Callable[..., Any]]:
        if callable(spec):
            return spec
        if isinstance(spec, str) and spec.strip():
            if ":" in spec:
                mod, attr = spec.split(":", 1)
                module = importlib.import_module(mod.strip())
                fn = getattr(module, attr.strip(), None)
                if callable(fn):
                    return fn
            # Treat as attribute name on the reward module.
            fn = getattr(self.reward_module, spec.strip(), None)
            if callable(fn):
                return fn
        for name in default_names:
            fn = getattr(self.reward_module, name, None)
            if callable(fn):
                return fn
        return None

    def build_prompt(self, input_content: dict) -> str:
        if not self.prompt_builder:
            raise ValueError("No extrinsic reward prompt builder available.")
        return self._call_with_known_args(
            self.prompt_builder,
            session_metadata=input_content["session_metadata"],
            session_transcript=input_content["session_transcript"],
            conversation_transcript=input_content["session_transcript"],
            session=input_content.get("session"),
            metadata_keys=input_content.get("metadata_keys"),
        )

    @staticmethod
    def coerce_to_int_0_6(x) -> int:
        try:
            v = int(round(float(x)))
        except Exception:
            v = 0
        return max(0, min(6, v))

    @staticmethod
    def coerce_to_conf(x) -> float:
        try:
            v = float(x)
        except Exception:
            return None
        return max(0.0, min(1.0, v))

    def sanity_check(self, outputs, data):
        if not self.ctrs_items:
            if not isinstance(outputs, (dict, int, float, bool, str)):
                logger.info("Top-level output is not a supported type.")
                return False
            return True

        if not isinstance(outputs, dict):
            logger.info("Top-level output is not an object.")
            return False
        if self.ctrs_items:
            codes = outputs.get("codes", {})
            if not isinstance(codes, dict):
                logger.info('"codes" must be an object.')
                return False
            for it in self.ctrs_items:
                k = it["key"]
                if k not in codes or not isinstance(codes[k], dict):
                    logger.info(f'Missing or invalid "codes.{k}".')
                    return False
        return True

    def _postprocess(self, outputs):
        if not self.ctrs_items:
            return outputs

        codes = outputs.get("codes", {})
        cleaned_codes = {}
        for it in self.ctrs_items:
            k = it["key"]
            entry = codes[k]
            score = self.coerce_to_int_0_6(entry.get("score"))
            rationale = str(entry.get("rationale", "")).strip()[:300]
            conf = entry.get("confidence", None)
            conf = self.coerce_to_conf(conf) if conf is not None else None
            cleaned_codes[k] = {
                "score": score,
                "rationale": rationale,
                "confidence": conf,
            }

        part1_keys = [it["key"] for it in (self.ctrs_part1 or [])]
        part2_keys = [it["key"] for it in (self.ctrs_part2 or [])]
        part1_sum = sum(cleaned_codes[k]["score"] for k in part1_keys)
        part2_sum = sum(cleaned_codes[k]["score"] for k in part2_keys)
        overall_sum = part1_sum + part2_sum
        part1_mean = round(part1_sum / len(part1_keys), 2) if part1_keys else 0.0
        part2_mean = round(part2_sum / len(part2_keys), 2) if part2_keys else 0.0
        overall_mean = round(
            overall_sum / (len(part1_keys) + len(part2_keys)),
            2,
        ) if (part1_keys or part2_keys) else 0.0

        return {
            "codes": cleaned_codes,
            "part1_sum": part1_sum,
            "part1_mean": part1_mean,
            "part2_sum": part2_sum,
            "part2_mean": part2_mean,
            "overall_sum": overall_sum,
            "overall_mean": overall_mean,
        }

    def fallback(self, last_outputs, data):
        return last_outputs

    def _fallback(self):
        # Fallback if parsing keeps failing
        if not self.ctrs_items:
            return {"reward": 0, "raw": "Parse failure fallback."}
        zero_codes = {
            it["key"]: {"score": 0, "rationale": "Parse failure fallback.", "confidence": None}
            for it in self.ctrs_items
        }
        return {
            "codes": zero_codes,
            "part1_sum": 0, "part1_mean": 0.0,
            "part2_sum": 0, "part2_mean": 0.0,
            "overall_sum": 0, "overall_mean": 0.0,
        }

    def run_with_retries(self, prompt: str, data) -> tuple[Dict, dict]:
        last_outputs = None
        usage = {"inp_t": 0.0, "out_t": 0.0}

        for attempt in range(self.max_retries):
            raw, inp_t, out_t = run_llm_query(prompt, self.api_key)

            usage["inp_t"] += (inp_t / 1_000_000)
            usage["out_t"] += (out_t / 1_000_000)
            try:
                if self.parse_output_fn:
                    outputs = self._call_with_known_args(self.parse_output_fn, raw=raw, output=raw, data=data)
                else:
                    try:
                        outputs = parse_json(raw)
                    except Exception:
                        outputs = parse_json(raw.replace("'", '"'))
            except Exception as e:
                logger.warning(f"[WARNING] parsing failed at attempt {attempt+1}, error: {e} retrying...")
                continue

            outputs = self.postprocess(outputs, data)
            last_outputs = outputs
            if self.sanity_check(outputs, data):
                return outputs, usage
            logger.warning(f"[WARNING] Sanity check failed at attempt {attempt+1}, retrying...")

        logger.warning("[ERROR] Max retries reached. Using fallback.")
        fixed_outputs = self.fallback(last_outputs, data)
        return fixed_outputs, usage

    def annotate_single_session(self, user_idx, session_idx, session, session_metadata, *args):
        utterances, system_counter, user_counter = self.process_dialogue(session)

        input_content = {
            "session_metadata": session_metadata, 
            "session_transcript": format_utterances(utterances),
            "session": session,
        }
        prompt = self.build_prompt(input_content)
        parsed, usage = self.run_with_retries(prompt, {})
        if self.sanity_check(parsed, {}):
            cleaned_parsed = self._postprocess(parsed)
        else:
            cleaned_parsed = self._fallback()

        if self.to_row_fn:
            row = self._call_with_known_args(
                self.to_row_fn,
                outputs=cleaned_parsed,
                parsed=cleaned_parsed,
                user_idx=user_idx,
                session_idx=session_idx,
                session=session,
                session_metadata=session_metadata,
            )
            if isinstance(row, dict):
                row.setdefault("user_idx", user_idx)
                row.setdefault("session_idx", session_idx)
                return [row], usage

        if self.ctrs_items:
            row = {
                "user_idx": user_idx,
                "session_idx": session_idx,
                "part1_sum": cleaned_parsed["part1_sum"],
                "part1_mean": cleaned_parsed["part1_mean"],
                "part2_sum": cleaned_parsed["part2_sum"],
                "part2_mean": cleaned_parsed["part2_mean"],
                "overall_sum": cleaned_parsed["overall_sum"],
                "overall_mean": cleaned_parsed["overall_mean"],
            }
            for it in self.ctrs_items:
                k = it["key"]
                r = cleaned_parsed["codes"][k]
                row[f"{k}_score"] = r["score"]
                row[f"{k}_confidence"] = r["confidence"]
                row[f"{k}_rationale"] = r["rationale"]
            return [row], usage

        # Generic fallback: keep selected fields or full dict.
        row = {"user_idx": user_idx, "session_idx": session_idx}
        if isinstance(cleaned_parsed, dict):
            fields = self.output_fields or list(cleaned_parsed.keys())
            for key in fields:
                row[key] = cleaned_parsed.get(key)
        else:
            row["reward"] = cleaned_parsed
        return [row], usage