import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .metric_utils import (average_pairwise_similarity, chapter_lengths_words, compute_cc_and_swc, llm_chapter_sources, 
                           run_graph_rag_eval, load_json, dump_json, is_none_summary, load_outputs, load_eval_utterances)
from ..utils import OpenAIEmbedder, JinaLocalEmbedder

class MetricsManager:
    def __init__(self, domain: str, metrics_root: str, api_key: str, embed_model: str = "text-embedding-3-small"):
        self.domain = domain
        self.metrics_root = Path(metrics_root)
        self.api_key = api_key
        self.embed_model = embed_model
        # self.openai_embedding_key_env = os.environ["OPENAI_API_KEY"]

        self.metrics_root.mkdir(parents=True, exist_ok=True)
        # self.embedder = OpenAIEmbedder(key_env=self.openai_embedding_key_env, model=embed_model)
        self.embedder = JinaLocalEmbedder()

    def _metrics_dir(self, method: str, role: str) -> Path:
        return self.metrics_root / method / role

    def compute_coverage(self, method: str, outputs_path: str, textbooks_path: str, components: List[str]) -> Dict:
        # Methods without coverage
        if method in ("ChatGPT", "DeepSeek"):
            return {"CC": None, "SWC": None, "per_component": {}}

        p_books = Path(textbooks_path)
        # Determine single-book vs multi-book
        pdfs = []
        if p_books.is_dir():
            pdfs = sorted([str(x) for x in p_books.glob("*.pdf")])
        elif p_books.is_file() and p_books.suffix.lower() == ".pdf":
            pdfs = [str(p_books)]

        per_cc = {}
        per_swc = {}

        # TOPA summaries-based coverage
        if method.startswith("topa"):
            # Locate summaries files near outputs_path
            base_dir = Path(outputs_path).parent
            summaries = [base_dir / f for f in sorted([x.name for x in base_dir.glob("summaries_book_*.json")])]

            # Build chapter lengths per summary item
            lengths = []
            for b, pdf in enumerate(pdfs):
                _, lens_b = chapter_lengths_words(pdf)
                lengths.extend(lens_b)

            for c in components:
                presence = []
                for sp in summaries:
                    book_sum = load_json(sp)
                    for ch in book_sum:
                        val = ch[c].get("summary")
                        presence.append(0 if is_none_summary(val) else 1)

                cc, swc = compute_cc_and_swc(presence, lengths)
                per_cc[c] = cc
                per_swc[c] = swc

        # RAG retrieval-log coverage
        outdir = Path(outputs_path) if Path(outputs_path).is_dir() else Path(outputs_path).parent
        rlog = outdir / "retrieval_log.json"
        if rlog.exists():
            log = load_json(rlog)
            comps = log.get("components", {})
            if not isinstance(comps, dict):
                comps = {}
            # compute total chapter lengths
            if len(pdfs) == 1:
                _, lengths = chapter_lengths_words(pdfs[0])
                n = len(lengths)
                for c in components:
                    counts_map = {}
                    if c in comps and isinstance(comps[c], dict):
                        counts_map = comps[c].get("chapter_counts", {}) or {}
                    counts = [int(counts_map.get(f"{0}:{str(i)}", 0)) for i in range(n)]
                    cc, swc = compute_cc_and_swc(counts, lengths)
                    per_cc[c] = cc
                    per_swc[c] = swc
            else:
                # multi-book: treat chapter key "b:ch"
                lengths_keys = {}
                all_keys = []
                for b, pdf in enumerate(pdfs):
                    _, lens = chapter_lengths_words(pdf)
                    for ch, L in enumerate(lens):
                        key = f"{b}:{ch}"
                        lengths_keys[key] = L
                        all_keys.append(key)
                for c in components:
                    counts_map = {}
                    if c in comps and isinstance(comps[c], dict):
                        counts_map = comps[c].get("chapter_counts", {}) or {}
                    counts = [int(counts_map.get(k, 0)) for k in all_keys]
                    lengths = [int(lengths_keys[k]) for k in all_keys]
                    cc, swc = compute_cc_and_swc(counts, lengths)
                    per_cc[c] = cc
                    per_swc[c] = swc

        # LLM-based chapter attribution (ToC titles + extracted output only)
        # Only valid for single-book methods
        if len(pdfs) == 1:
            titles, lengths = chapter_lengths_words(pdfs[0])
            n = len(lengths)
            outputs = load_outputs(outputs_path)
            for c in components:
                covered = llm_chapter_sources(self.api_key, titles, json.dumps(outputs.get(c), ensure_ascii=False))
                presence = [1 if i in covered else 0 for i in range(n)]
                cc, swc = compute_cc_and_swc(presence, lengths)
                per_cc[c] = cc
                per_swc[c] = swc

        CC = sum(per_cc.values()) / max(len(per_cc), 1)
        SWC = sum(per_swc.values()) / max(len(per_swc), 1)
        return {"CC": CC, "SWC": SWC, "per_component": {"CC": per_cc, "SWC": per_swc}}

    def compute_graph_rag(self, outputs: dict, data_path: str) -> dict:
        kg = outputs.get("knowledge_graph")
        if not kg:
            return {"ES": None, "NSR": None, "details": None}
        eval_utts, eval_utts_str = load_eval_utterances(data_path)
        if not eval_utts or not eval_utts_str:
            return {"ES": None, "NSR": None, "details": None}
        res = run_graph_rag_eval(self.embedder, kg, eval_utts, eval_utts_str, self.api_key, top_k=2, min_sim=0.18)
        return {"ES": res["overall_ES"], "NSR": res["overall_NSR"], "details": res}

    def compute_complexity(self, method: str, usage: Optional[dict]) -> dict:
        # Methods with no usage/cost
        if method in ("ChatGPT", "DeepSeek") or not usage:
            return {"MEM": None, "TOKi": None, "TOKo": None, "LAT": None, "COST": None}

        toki = int(usage.get("tok_in", 0))
        toko = int(usage.get("tok_out", 0))
        lat = float(usage.get("lat_sec", 0.0))
        mem = int(usage.get("mem_peak", 0))

        cost = (toki / 1_000_000.0) * 3.0 + (toko / 1_000_000.0) * 12.0
        return {"MEM": mem, "TOKi": toki, "TOKo": toko, "LAT": lat, "COST": cost}

    def compute_stability(self, outputs_by_seed: Optional[dict], components: List[str]) -> dict:
        if outputs_by_seed is None:
            return {"SC": None, "details": None}
        overall, per_comp = average_pairwise_similarity(self.embedder, outputs_by_seed, components)
        return {"SC": overall, "details": {"per_component": per_comp}}

    def compute_curriculum_sensitivity(self, shuffled_by_seed: Optional[dict], components: List[str]) -> dict:
        if shuffled_by_seed is None:
            return {"CS": None, "details": None}
        overall, per_comp = average_pairwise_similarity(self.embedder, shuffled_by_seed, components)
        return {"CS": overall, "details": {"per_component": per_comp}}

    def save_metrics(self, method: str, role: str, metrics: dict, per_component: dict, outputs_path: str, 
                     extra: Optional[dict] = None) -> Path:
        out = {"method": method, "role": role, "outputs_path": outputs_path, "metrics": metrics, "per_component": per_component}
        if extra:
            out.update(extra)

        mdir = self._metrics_dir(method, role)
        mpath = mdir / "metrics.json"
        dump_json(mpath, out)
        return mpath

    def update_summary_csv(self) -> Path:
        # Collect all metrics.json under metrics_root/*/*/metrics.json
        rows = []
        for mpath in sorted(self.metrics_root.glob("*/*/metrics.json")):
            data = load_json(mpath)
            method = data.get("method")
            role = data.get("role")
            met = data.get("metrics", {})
            rows.append({
                "method": method,
                "role": role,
                "CC": met.get("CC"),
                "SWC": met.get("SWC"),
                "SC": met.get("SC"),
                "CS": met.get("CS"),
                "ES": met.get("ES"),
                "NSR": met.get("NSR"),
                "MEM": met.get("MEM"),
                "TOKi": met.get("TOKi"),
                "TOKo": met.get("TOKo"),
                "LAT": met.get("LAT"),
                "COST": met.get("COST"),
            })

        out_csv = self.metrics_root / "summary.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method", "role"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return out_csv