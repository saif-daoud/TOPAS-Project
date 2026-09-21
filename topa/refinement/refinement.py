import json
from pathlib import Path
from typing import Any, Dict
import shutil
from tqdm import tqdm

from .prompts import build_refine_conversation_state_prompt
from ..utils import build_logger, parse_json, run_llm_query

logger = build_logger()

class RefinementPhase:
    """
    Refines conversation_states by iterating over the action space (macro + micro actions)
    and asking the LLM to add only missing dimensions.
    """
    def __init__(self, domain: str, api_key: str, max_retries: int = 3):
        self.domain = domain
        self.api_key = api_key
        self.max_retries = max_retries

    def _load_inputs(self, path: str | Path) -> Dict[str, Any]:
        p = Path(path)
        if p.is_dir():
            micro_path = p / "micro_actions.json"
            conv_path = p / "conversation_states.json"
            micro_actions = json.loads(micro_path.read_text(encoding="utf-8"))
            conversation_states = json.loads(conv_path.read_text(encoding="utf-8"))
            return {
                "micro_actions": micro_actions,
                "conversation_states": conversation_states,
            }
        assert p.stem == "all_components.json", f"missing file all_components.json"
        # assume JSON file containing all components
        data = json.loads(p.read_text(encoding="utf-8"))
        return data

    def _copy_components(self, extracted_dir: Path, refined_dir: Path) -> None:
        if extracted_dir.resolve() == refined_dir.resolve():
            return
        for src in extracted_dir.glob("*.json"):
            dst = refined_dir / src.name
            shutil.copy2(src, dst)

    def _dump_refined(self, path: str, data: Dict[str, Any], reports: list, summary: Dict[str, Any]) -> Path:
        p = Path(path)
        extracted_dir = p if p.is_dir() else p.parent # extracted_components
        refined_dir = extracted_dir
        while True:
            if refined_dir.stem == "extracted_components":
                refined_dir = refined_dir.parent
                break
            refined_dir = refined_dir.parent
            # reached filesystem root => stop (prevents infinite loop)
            if refined_dir.parent == refined_dir:
                raise Exception(f"Could not find the extracted_components dir in this path {extracted_dir} !")

        refined_dir = refined_dir / "refined_components"
        refined_dir.mkdir(parents=True, exist_ok=True)
        # Copy everything from extracted_components -> refined_components
        self._copy_components(extracted_dir, refined_dir)

        (refined_dir / "refinement_report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Summary of only NEW dimensions + count + rationales
        (refined_dir / "refinement_new_dimensions_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if p.is_dir():
            (refined_dir / "conversation_states.json").write_text(
                json.dumps(data["conversation_states"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return refined_dir
        else:
            out_path = refined_dir / "all_components.json"
            out_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return out_path

    @staticmethod
    def _extract_missing_states(parsed_resp: Dict[str, Any]) -> list:
        """
        Returns a list like:
        [
            {
                "action": "...",
                "missing": [
                {"state_phrase": "...", "gap_reason": "..."},
                ...
                ]
            },
            ...
        ]
        """
        out = []
        for pa in parsed_resp["per_action"]:
            action = pa["action"]
            missing = []
            for sm in pa["states_mappings"]:
                gap = sm["gap_reason"]
                # treat as missing if the model wrote a gap reason
                if isinstance(gap, str) and gap.strip():
                    missing.append({
                        "state_phrase": sm["state_phrase"],
                        "gap_reason": gap,
                    })
            if missing:
                out.append({"action": action, "missing": missing})
        return out

    def run(self, path: str | Dict[str, Any]) -> str:
        """
        path can be:
        1) /path/all_components.json     (expects keys: "micro_actions", "conversation_states")
        2) /path/to/dir                  (expects files: micro_actions.json and conversation_states.json)
        3) {"system": "...", "user": "..."} mapping of role -> extracted path
        """
        extra_paths: list[Path] = []
        target_path = path

        if isinstance(path, dict):
            candidates = [Path(str(v)) for v in path.values() if v]
            target_path = None
            if "system" in path and path["system"]:
                target_path = Path(str(path["system"]))
            if target_path is None:
                for p in candidates:
                    if (p.is_dir() and (p / "micro_actions.json").exists() and (p / "conversation_states.json").exists()) or (
                        p.is_file() and p.stem == "all_components"
                    ):
                        target_path = p
                        break
            if target_path is None and candidates:
                target_path = candidates[0]
            if target_path is None:
                raise ValueError("Refinement requires a valid extracted-components path.")
            extra_paths = [p for p in candidates if p != target_path]

        data = self._load_inputs(str(target_path))

        action_space = data["micro_actions"] # list[macro_action]
        conversation_states = data["conversation_states"] # list[dimensions]

        reports = [] # store the full rationals per macro iteration
        added_dimensions = []
        added_by_macro = []
        parsed = None
        last_err = None

        for macro_action in tqdm(action_space, total=len(action_space)):
            prompt = build_refine_conversation_state_prompt(self.domain, macro_action, conversation_states)
            for attempt in range(self.max_retries):
                resp, _, _ = run_llm_query(prompt, self.api_key)
                try:
                    parsed_resp = parse_json(resp)
                    new_dims = parsed_resp["new_dimensions"]

                    added_dimensions.extend(new_dims)
                    added_by_macro.append({
                        "macro_action": macro_action["name"],
                        "num_added": len(new_dims),
                        "new_dimensions": new_dims,
                        "missing_states": self._extract_missing_states(parsed_resp),
                    })

                    # extend conversation_states (drop Rationale)
                    for d in new_dims:
                        conversation_states.append({
                            "Variable Name": d["Variable Name"],
                            "Description": d["Description"],
                            "Categorical values": d["Categorical values"],
                            "Numerical values": d["Numerical values"],
                        })

                    # keep Rationales in a report list
                    parsed_resp["_macro_action_name"] = macro_action["name"]
                    reports.append(parsed_resp)

                    parsed = parsed_resp
                    break

                except Exception as e:
                    last_err = e
                    logger.warning(f"[WARN] Refinement parse failed (attempt {attempt+1}/{self.max_retries}): {e}")

            if parsed is None:
                logger.warning(f"[ERROR] Skipping macro '{macro_action['name']}' after {self.max_retries} attempts. Last error: {last_err}")
                continue

        summary = {
            "domain": self.domain,
            "num_added_dimensions": len(added_dimensions),
            "new_dimensions": added_dimensions,
            "by_macro_action": added_by_macro,
        }
        data["conversation_states"] = conversation_states
        out_path = self._dump_refined(str(target_path), data, reports, summary)

        # If we were given role -> path mappings, copy over any extra JSON artifacts (e.g., user_profile.json)
        if extra_paths:
            refined_dir = Path(out_path)
            if refined_dir.is_file():
                refined_dir = refined_dir.parent
            for p in extra_paths:
                if p.is_file():
                    if p.suffix.lower() == ".json":
                        dst = refined_dir / p.name
                        if not dst.exists():
                            shutil.copy2(p, dst)
                elif p.is_dir():
                    for src in p.glob("*.json"):
                        dst = refined_dir / src.name
                        if not dst.exists():
                            shutil.copy2(src, dst)

        return str(out_path)