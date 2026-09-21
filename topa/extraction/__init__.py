from pathlib import Path
import random
from typing import Optional

class ExtractionPhase:
    def __init__(self, domain_name, domain_params, textbooks_path, data_path, output_path, api_key, mode, seed_base):
        from .baseline_methods import LC_Full, Chap_Seq, Chunk_RAG, Rules, ChatGPT, DeepSeek, Merge_RAG, Mamba
        from .topa_ours import TOPAPerBookExtractor, TOPAOurExtractor
        self.domain_name = domain_name
        self.adj = domain_params["adj"]
        self.system = domain_params["system"]
        self.user = domain_params["user"]
        self.interaction_unit = domain_params["interaction_unit"]
        self.textbooks_path = textbooks_path
        self.data_path = data_path
        self.output_path = output_path
        self.api_key = api_key
        self.mode = mode
        self.seed_base = seed_base

        self.methods = {
            "ChatGPT": ChatGPT, "DeepSeek": DeepSeek,
            "LC_Full": LC_Full, "Rules": Rules, "Chap_Seq": Chap_Seq, "Chunk_RAG": Chunk_RAG,
            "Mamba": Mamba, "topa_per_book": TOPAPerBookExtractor,
            "Merge_RAG": Merge_RAG, "topa_ours": TOPAOurExtractor
        }

    def filter_components(self, role: str, components: list, params: dict) -> list:
        return [component for component in components if component in params["role_components"][role]]

    def _resolve_textbooks_path_for_role(self, role: str):
        """Resolve role-specific textbooks path if split folders exist."""
        p = Path(self.textbooks_path).expanduser()
        if not p.exists():
            return p
        if p.is_dir():
            cand_names = [
                role,
                f"{role}_books",
                f"{role} books",
                f"{role}-books",
                f"{role}_textbooks",
                f"{role} textbooks",
            ]
            for name in cand_names:
                cand = p / name
                if cand.exists():
                    return cand
        return p

    def _save_usage(self, extractor: BaseExtractor, output_path: str):
        p = Path(output_path)
        out_dir = p.parent if p.is_file() else p
        extractor.save_usage(str(out_dir))

    def _compute_and_save_metrics(self, method: str, role: str, outputs_path: str, components: list, 
                                  sc_outputs_by_seed: Optional[dict] = None, cs_shuffled_by_seed: Optional[dict] = None,
                                  textbooks_path: Optional[str | Path] = None):
        metrics_root = Path(self.output_path) / self.adj / "metrics"
        mm = MetricsManager(domain=self.adj, metrics_root=str(metrics_root), api_key=self.api_key)

        outputs = load_outputs(outputs_path)
        usage = None
        up = (Path(outputs_path).parent if Path(outputs_path).is_file() else Path(outputs_path)) / "usage.json"
        if up.exists():
            usage = load_json(up).get("usage")

        tb_path = str(textbooks_path) if textbooks_path is not None else str(self.textbooks_path)
        cov = mm.compute_coverage(method, outputs_path, tb_path, components)
        stab = mm.compute_stability(sc_outputs_by_seed, components)
        cs = mm.compute_curriculum_sensitivity(cs_shuffled_by_seed, components)
        gr = mm.compute_graph_rag(outputs, str(self.data_path))
        comp = mm.compute_complexity(method, usage)

        metrics = {
            "CC": cov.get("CC"), "SWC": cov.get("SWC"), # For book coverage
            "SC": stab.get("SC"), "CS": cs.get("CS"), # For stabilty and sensitivity
            "ES": gr.get("ES"), "NSR": gr.get("NSR"), # For KG robustness
            "MEM": comp.get("MEM"), "TOKi": comp.get("TOKi"), "TOKo": comp.get("TOKo"),
            "LAT": comp.get("LAT"), "COST": comp.get("COST")
        }

        per_component = {
            "coverage": cov.get("per_component"),
            "stability": stab.get("details"),
            "curriculum_sensitivity": cs.get("details")
        }

        extra = {"graph_rag_details": gr.get("details")}
        mm.save_metrics(method, role, metrics, per_component, outputs_path, extra=extra)
        mm.update_summary_csv()

    def _run_sc(self, role: str, params: dict, extractor: BaseExtractor, output_path: str) -> dict:
        # SC: 3 seeds (seed_base + 0 (already run),1,2)
        seeds = [self.seed_base + i for i in range(3)]
        out_by_seed = {}
        out_by_seed[seeds[0]] = load_outputs(output_path) # Already run
        for seed in seeds[1:]:
            extractor.reset_usage()
            extractor.output_path = str(Path(self.output_path) / "self_consistency" / str(seed))
            p = dict(params)
            p["role"] = role
            out_path = extractor.extract(**p)
            out_by_seed[seed] = load_outputs(out_path)
        return out_by_seed

    def _run_cs(self, method: str, role: str, params: dict, extractor: BaseExtractor, output_path: str, textbooks_path: Optional[str | Path] = None) -> tuple:
        # CS only for multi-book textbooks_path and multi-book methods
        p_books = Path(textbooks_path if textbooks_path is not None else self.textbooks_path)
        if not p_books.is_dir():
            return None
        pdfs = sorted([str(x) for x in p_books.glob("*.pdf")])
        if len(pdfs) <= 1 or method not in ("Merge_RAG", "topa_ours"):
            return None

        # shuffled runs for 3 seeds
        seeds = [self.seed_base + i for i in range(3)]
        shuffled = {}
        shuffled[seeds[0]] = load_outputs(output_path) # Already run
        for seed in seeds[1:]:
            extractor.reset_usage()
            rnd = random.Random(seed)
            order = list(pdfs)
            rnd.shuffle(order)
            extractor.output_path = str(Path(self.output_path) / "curriculum_sensitivity" / str(seed))
            p2 = dict(params)
            p2["role"] = role
            p2["pdf_files"] = order
            out_path = extractor.extract(**p2)
            shuffled[seed] = load_outputs(out_path)

        return shuffled

    def run(self, extraction_params):
        params = dict(extraction_params)
        method = params.pop("method")
        self_consistency = bool(params.pop("self_consistency", False))
        curriculum_sensitivity = bool(params.pop("curriculum_sensitivity", False))
        compute_metrics = bool(params.pop("compute_metrics", True))

        extractor = self.methods[method](self.adj, self.system, self.user, self.interaction_unit, self.textbooks_path, 
                                         self.output_path, self.api_key, self.mode)
        role = params["role"]
        if isinstance(role, str):
            role = [role]
        results = {}
        for _single_role in role:
            params["role"] = _single_role
            role_textbooks_path = self._resolve_textbooks_path_for_role(_single_role)
            extractor.textbooks_path = role_textbooks_path
            extractor.components[_single_role] = self.filter_components(_single_role, extractor.components[_single_role], params)
            out_path = extractor.extract(**params)
            # if role == "system":
            #     out_path = "/home/local/QCRI/saif.sedaoud/clean-env/TOPA/outputs/cognitive behavioral therapy/Mamba/extracted_components/system/Cognitive Behavior Therapy, Second Edition/all_components.json"
            # else:
            #     out_path = "/home/local/QCRI/saif.sedaoud/clean-env/TOPA/outputs/cognitive behavioral therapy/Mamba/extracted_components/user/Dsm-5-Trt Clinical Cases/all_components.json"
            self._save_usage(extractor, out_path)

            sc = self._run_sc(_single_role, params, extractor, out_path) if self_consistency else None
            cs_shuf = self._run_cs(method, _single_role, params, extractor, out_path, textbooks_path=role_textbooks_path) if curriculum_sensitivity else None

            if compute_metrics:
                self._compute_and_save_metrics(method, _single_role, out_path, extractor.components[_single_role], sc, cs_shuf, textbooks_path=role_textbooks_path)
            results[_single_role] = out_path
        return results if len(results) > 1 else list(results.values())[0]
