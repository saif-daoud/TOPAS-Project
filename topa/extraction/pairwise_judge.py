import json
import os
import random
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Optional

from .metric_utils import load_outputs
from . import specs as _specs
from ..utils import get_llm_response, parse_json

def _get_component_spec(component: str, *, mode: str, system: str, user: str, domain: str, interaction_unit: str) -> str:
    """Return components definitions from specs.py."""
    is_task = mode == "task"

    if component == "action_space":
        definition1 = _specs.macro_action_definition_task if is_task else _specs.macro_action_definition
        definition1 = definition1.format(system=system, user=user)
        definition2 = _specs.micro_action_definition_task if is_task else _specs.micro_action_definition
        definition2 = definition2.format(system=system, user=user)
        definition = definition1 + definition2
    elif component == "conversation_states":
        definition = _specs.conversation_states_definition_task if is_task else _specs.conversation_states_definition
        definition = definition.format(system=system, domain=domain, interaction_unit=interaction_unit)
    elif component == "knowledge_graph":
        definition = _specs.knowledge_graph_definition_task if is_task else _specs.knowledge_graph_definition
        definition = definition.format(user=user, interaction_unit=interaction_unit)
    elif component == "cautions":
        definition = _specs.cautions_definition_task if is_task else _specs.cautions_definition
        definition = definition.format(system=system, user=user, interaction_unit=interaction_unit, domain=domain)
    elif component == "user_profile":
        definition = _specs.user_profile_definition
        definition = definition.format(system=system, user=user, interaction_unit=interaction_unit, domain=domain)
    else:
        raise Exception(f"Unknown component {component}")

    return definition

class PairwiseJudge:
    """LLM-based pairwise judge."""
    def __init__(self, api_key: str, *, mode: str, domain: str, interaction_unit: str, system_name: str, user_name: str, 
                 debias_randomize_order: bool = True, max_parse_retries: int = 2):
        self.api_key = api_key
        self.mode = mode
        self.domain = domain
        self.interaction_unit = interaction_unit
        self.system_name = system_name
        self.user_name = user_name
        self.debias_randomize_order = debias_randomize_order
        self.max_parse_retries = max_parse_retries

    def _build_prompt(self, component: str, out_a, out_b) -> str:
        # Pull only the component definition from specs.py
        definition = _get_component_spec(component, mode=self.mode, system=self.system_name, user=self.user_name, 
                                   domain=self.domain, interaction_unit=self.interaction_unit)

        prompt = f"""You are an impartial expert judge in {self.domain}.

Task: Compare Output A vs Output B for the SAME component: "{component}".

Component definition:
{definition}

Instructions:
- Relevance: Prefer the output that most directly and accurately addresses what the component definition asks for.
- Completeness: Prefer the output that includes the key required details and does not omit important elements of the definition.
- Clarity: Prefer the output that is easiest to understand and is clinically meaningful (clear wording, minimal ambiguity).
- Overall preference: If one output is superior across these criteria, choose it.
- Tie rule: If both outputs are genuinely indistinguishable in relevance, completeness, and clarity, answer "tie".
- Justification: Provide a brief reason (1–3 sentences), referring to the candidates as "Output A" and "Output B".

Return ONLY valid JSON (no markdown, no extra text):
{{
  "winner": "A" | "B" | "tie",
  "reason": "<1-3 sentences>"
}}

Output A:
{json.dumps(out_a, ensure_ascii=False, indent=2)}

Output B:
{json.dumps(out_b, ensure_ascii=False, indent=2)}
"""
        return prompt

    def judge(self, component: str, out_a: Any, out_b: Any, name_a: str, name_b: str) -> Dict[str, Any]:
        # Debias: randomize order of A/B in the prompt, then map back.
        swap = bool(self.debias_randomize_order and random.random() < 0.5)
        pa_out_a, pa_out_b = (out_b, out_a) if swap else (out_a, out_b)
        pa_name_a, pa_name_b = (name_b, name_a) if swap else (name_a, name_b)

        prompt = self._build_prompt(component, pa_out_a, pa_out_b)

        last_err: Optional[str] = None
        for attempt in range(self.max_parse_retries + 1):
            if attempt > 0:
                # Repair prompt: ask for valid JSON only.
                prompt2 = (
                    prompt
                    + "\n\nIMPORTANT: Your previous answer was not valid JSON or missed required keys."
                      " Return ONLY valid JSON matching the schema, no markdown, no extra text."
                )
            else:
                prompt2 = prompt

            resp = get_llm_response(prompt2, self.api_key)
            try:
                parsed = parse_json(resp.replace("'", '"'))
                if parsed["winner"] not in ("A", "B", "tie"):
                    parsed["winner"] = "tie"
                if swap:
                    # Swap winner value (prompt labels -> original labels)
                    w = parsed["winner"]
                    if w == "A":
                        parsed["winner"] = "B"
                    elif w == "B":
                        parsed["winner"] = "A"

                parsed.setdefault("meta", {})
                parsed["meta"].update({"component": component, "swap_order_in_prompt": swap})
                return parsed
            except Exception as e:
                last_err = str(e)

        # Fallback if repeated parse failures
        return {
            "winner": "tie",
            "reason": f"Judge failed to return valid JSON after retries. Last error: {last_err}",
            "meta": {"component": component, "swap_order_in_prompt": swap},
        }

    def run(self, outputs: Dict[str, str], components: List[str], save_path: str) -> Dict[str, Any]:
        methods = sorted(outputs.keys())
        results = {"methods": methods, "components": components, "pairs": {}}

        for i in tqdm(range(len(methods)), total=len(methods)):
            for j in range(i + 1, len(methods)):
                A = methods[i]
                B = methods[j]
                oa = load_outputs(outputs[A])
                ob = load_outputs(outputs[B])
                pair_key = f"{A}__vs__{B}"
                results["pairs"][pair_key] = {"wins": {"A": 0, "B": 0, "tie": 0}, "by_component": {}}

                for c in components:
                    r = self.judge(c, oa[c], ob[c], A, B)
                    w = r["winner"]
                    if w not in ("A", "B"):
                        w = "tie"
                    results["pairs"][pair_key]["wins"][w] += 1
                    results["pairs"][pair_key]["by_component"][c] = r

        os.makedirs(Path(save_path).parent, exist_ok=True)
        Path(save_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return results