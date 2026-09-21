import json
from copy import deepcopy

from .base import BaseAnnotator
from ..prompts import annotate_user_profile_dim_prompt
from ..utils import format_utterances

from ...utils import build_logger
logger = build_logger()

class ProfileDimAnnotator(BaseAnnotator):
    def build_prompt(self, input_content: str) -> str:
        return annotate_user_profile_dim_prompt(
            domain=self.domain_params["adj"],
            user=self.domain_params["user"],
            system=self.domain_params["system"],
            interaction_unit=self.domain_params["interaction_unit"],
            input_content=input_content
        )

    @staticmethod
    def _normalize_profile_spec(component):
        """Normalize user_profile spec to a dict with categorical_dimensions/free_form_dimensions."""
        cat = component.get("categorical_dimensions")
        free = component.get("free_form_dimensions")

        def _norm_cat(d):
            return {
                "dimension_name": d.get("dimension_name"),
                "description": d.get("description"),
                "options": d.get("options"),
            }

        def _norm_free(d):
            return {
                "dimension_name": d.get("dimension_name"),
                "description": d.get("description"),
            }

        cat = [_norm_cat(d) for d in cat if isinstance(d, dict)]
        free = [_norm_free(d) for d in free if isinstance(d, dict)]
        cat = [d for d in cat if d.get("dimension_name")]
        free = [d for d in free if d.get("dimension_name")]

        return {"categorical_dimensions": cat, "free_form_dimensions": free}

    def sanity_check(self, outputs, data):
        spec = data["spec"]
        cat_dims = spec["categorical_dimensions"]
        free_dims = spec["free_form_dimensions"]

        if not isinstance(outputs, dict):
            logger.info("User profile output is not a dict.")
            return False

        out_cat = outputs.get("categorical_dimensions")
        out_free = outputs.get("free_form_dimensions")
        if not isinstance(out_cat, list) or not isinstance(out_free, list):
            logger.info("User profile output missing categorical/free_form lists.")
            return False

        if len(out_cat) != len(cat_dims) or len(out_free) != len(free_dims):
            logger.info(
                f"mismatched length cat/free: got {len(out_cat)}/{len(out_free)} expected {len(cat_dims)}/{len(free_dims)}"
            )
            return False

        # categorical: each value must be in allowed options OR "Unknown"
        for v, dim in zip(out_cat, cat_dims):
            allowed = set([str(x) for x in (dim.get("options"))] + ["Unknown", "unknown"])
            val = v.get("value")
            if val not in allowed and len(allowed) > 2:
                logger.info(f"Inconsistent category in dim {dim['dimension_name']}, predicted {val}, allowed {allowed}")
                return False

        return True

    def postprocess(self, outputs, data):
        spec = data["spec"]
        cat_dims = spec["categorical_dimensions"]
        free_dims = spec["free_form_dimensions"]

        # Build maps by name for robust matching
        out_cat = outputs.get("categorical_dimensions")
        out_free = outputs.get("free_form_dimensions")
        out_cat_map = {str(x.get("dimension_name")): x for x in out_cat if isinstance(x, dict)}
        out_free_map = {str(x.get("dimension_name")): x for x in out_free if isinstance(x, dict)}

        norm_cat = []
        for d in cat_dims:
            name = str(d.get("dimension_name"))
            v = out_cat_map.get(name, {})
            norm_cat.append({"dimension_name": name, "value": v.get("value", "Unknown")})

        norm_free = []
        for d in free_dims:
            name = str(d.get("dimension_name"))
            v = out_free_map.get(name, {})
            norm_free.append({"dimension_name": name, "value": v.get("value", "Unknown")})

        return {"categorical_dimensions": norm_cat, "free_form_dimensions": norm_free}

    def fallback(self, last_outputs, data):
        logger.warning("[FALLBACK] Returning last outputs for profile dimensions.")
        spec = data["spec"]
        cat_dims = spec["categorical_dimensions"]
        free_dims = spec["free_form_dimensions"]

        out = {"categorical_dimensions": [], "free_form_dimensions": []}

        last_cat = (last_outputs or {}).get("categorical_dimensions") or []
        last_free = (last_outputs or {}).get("free_form_dimensions") or []
        last_cat_map = {str(x.get("dimension_name")): x for x in last_cat if isinstance(x, dict)}
        last_free_map = {str(x.get("dimension_name")): x for x in last_free if isinstance(x, dict)}

        for d in cat_dims:
            name = str(d.get("dimension_name"))
            allowed = set([str(x) for x in (d.get("options") or [])] + ["Unknown", "unknown"])
            v = last_cat_map.get(name, {})
            val = v.get("value", "Unknown")
            if val not in allowed and len(allowed) > 2:
                val = "Unknown"
            out["categorical_dimensions"].append({"dimension_name": name, "value": val})

        for d in free_dims:
            name = str(d.get("dimension_name"))
            v = last_free_map.get(name, {})
            val = v.get("value", "Unknown") or "Unknown"
            out["free_form_dimensions"].append({"dimension_name": name, "value": val})

        return out

    def annotate_single_session(self, user_idx, session_idx, session, session_metadata, component, annotated_data):
        utterances, system_counter, user_counter = self.process_dialogue(session)
        spec = self._normalize_profile_spec(component)
        _component = {
            "categorical_dimensions": deepcopy(spec["categorical_dimensions"]),
            "free_form_dimensions": deepcopy(spec["free_form_dimensions"]),
        }
        input_content = json.dumps(
            {
                **_component,
                "session_script": format_utterances(utterances)
            },
            ensure_ascii=False,
            indent=2,
        )
        prompt = self.build_prompt(input_content)
        parsed, usage = self.run_with_retries(prompt, {"spec": spec})

        row = {
            "user_idx": user_idx,
            "session_idx": session_idx,
        }
        for dim in spec["categorical_dimensions"]:
            name = dim["dimension_name"]
            val = next((x for x in parsed["categorical_dimensions"] if x.get("dimension_name") == name), {})
            row[f"{name}_value"] = val.get("value", "Unknown")
        for dim in spec["free_form_dimensions"]:
            name = dim["dimension_name"]
            val = next((x for x in parsed["free_form_dimensions"] if x.get("dimension_name") == name), {})
            row[f"{name}_value"] = val.get("value", "Unknown")

        return [row], usage