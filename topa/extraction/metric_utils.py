import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base_extractor import BaseExtractor
from ..utils import get_llm_response, parse_json

# ========================
# Coverage helpers
# ========================
def chapter_lengths_words(pdf_path: str) -> List[int]:
    _, chapter_texts = BaseExtractor.extract_chapters_and_toc(pdf_path)
    lengths = []
    titles = []
    for title, chapter_text in chapter_texts:
        lengths.append(len(chapter_text.split()))
        titles.append(title)
    return titles, lengths

def _weighted_coverage(presence: List[int], lengths: List[int]) -> float:
    if not lengths:
        return 0.0
    Lbar = sum(lengths)
    vals = []
    for c, L in zip(presence, lengths):
        if L <= 0:
            vals.append(0.0)
        else:
            vals.append(int(c > 0) * (L / Lbar))
    return float(sum(vals) / max(len(vals), 1))

def compute_cc_and_swc(presence: List[int], lengths: List[int]) -> Tuple[float, float]:
    if not presence:
        return 0.0, 0.0
    cc = float(sum(1 for x in presence if x > 0) / len(presence))
    swc = _weighted_coverage(presence, lengths)
    return cc, swc

def llm_chapter_sources(api_key: str, toc_titles: List[str], extracted_output: str) -> List[int]:
    """Ask LLM to return source chapter indices using ONLY ToC titles + extracted output."""
    prompt = """You will be given:
(1) a textbook Table of Contents (chapter titles only)
(2) an extracted output (for ONE component)

Task:
Return the list of chapter indices (0-based) that most likely contain the information used in the extracted output.

Rules:
- You MUST use ONLY the chapter titles and the extracted output.
- Return JSON only: {"covered_chapters": [0, 2, ...]}
- If unsure, return an empty list.

ToC (0-based):
""" + '\n'.join([f"{i}: {title}" for i, title in enumerate(toc_titles)]) + "\n\nExtracted output:\n" + extracted_output

    resp = get_llm_response(prompt, api_key)
    try:
        data = parse_json(resp.replace("'", '"'))
        chs = data.get("covered_chapters", [])
        out = []
        for x in chs:
            try:
                xi = int(x)
                if 0 <= xi < len(toc_titles):
                    out.append(xi)
            except Exception:
                pass
        return sorted(list(set(out)))
    except Exception:
        return []
    
# ========================
# Similarity helpers
# ========================
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)

def _to_text(x) -> str:
    # Normalize input into a list of dicts
    if x is None:
        return ""
    if isinstance(x, dict):
        _x = [x.get("macro_actions") if isinstance(x.get("macro_actions"), dict) else x]
    elif isinstance(x, list):
        _x = x
    else:
        return str(x)

    if not _x:
        return ""

    # Some components store dicts inside list, others store nested dicts
    head = _x[0] if isinstance(_x[0], dict) else None
    if head is None:
        return ", ".join([str(it) for it in _x])

    values = ["Variable Name", "negative_action", "nodes", "name"]
    value = None
    for v in values:
        if v in head:
            value = v
            break
    if value is None:
        # Fallback: join stringified dicts
        return ", ".join([json.dumps(it, ensure_ascii=False) if isinstance(it, dict) else str(it) for it in _x])
    return ", ".join([str(it.get(value, "")) for it in _x])

def similarity_between_outputs(embedder, out_a: dict, out_b: dict, components: List[str]) -> Dict[str, float]:
    sims = {}
    for c in components:
        ta = _to_text(out_a[c])
        tb = _to_text(out_b[c])
        va = embedder.embed(ta)
        vb = embedder.embed(tb)
        sims[c] = cosine(va, vb)
    return sims

def average_pairwise_similarity(embedder, outputs_by_seed: dict, components: List[str]) -> Tuple[float, Dict[str, float]]:
    seeds = sorted(outputs_by_seed.keys())
    pairs = []
    per_comp = {c: [] for c in components}

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a = outputs_by_seed[seeds[i]]
            b = outputs_by_seed[seeds[j]]
            sims = similarity_between_outputs(embedder, a, b, components)
            pairs.append(float(np.mean(list(sims.values()))))
            for c, v in sims.items():
                per_comp[c].append(v)

    if not pairs:
        return None, {c: None for c in components}
    overall = float(np.mean(pairs))
    per_comp_avg = {c: float(np.mean(vs)) if vs else None for c, vs in per_comp.items()}
    return overall, per_comp_avg

# ========================
# Graph RAG helpers
# ========================
def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T

def build_graph(kg_components, session, api_key):
    node_types = [x["type"] for x in kg_components["nodes"]]
    relation_types = [x["relation"] for x in kg_components["edges"]]

    allowed_nodes = ", ".join(sorted(set(node_types)))
    allowed_rels  = ", ".join(sorted(set(relation_types)))

    prompt = f"""
    You are an information extraction system. Read the THERAPY SESSION and output ONLY a single JSON object
    with two arrays: "nodes" and "relations". Include nothing else.

    ALLOWED NODE TYPES: {allowed_nodes}
    ALLOWED RELATION TYPES: {allowed_rels}

    INPUT:
    <<<TRANSCRIPT_START>>>
    {session}
    <<<TRANSCRIPT_END>>>

    OUTPUT (JSON ONLY, no markdown, no comments, no trailing text):
    ```json
    {{
    "nodes": [
        {{
        "type": "<OneOf [{allowed_nodes}]>",
        "content": "<short canonical surface form>"
        }}
    ],
    "relations": [
        {{
        "type": "<OneOf [{allowed_rels}]>",
        "source": <int>,   // 0-based index into the nodes array
        "target": <int>    // 0-based index into the nodes array
        }}
    ]
    }}
    ```

    RULES:
    - Output ONLY the keys shown above. No other keys anywhere (no ids, no qualifiers, no confidence, no evidence).
    - "nodes" items MUST have exactly: "type", "content".
    - "relations" items MUST have exactly: "type", "source", "target".
    - "type" values MUST be from the allowed lists.
    - "source"/"target" MUST reference valid node indexes.
    - Deduplicate nodes so each real-world concept appears once; reuse its index in relations.
    - If nothing matches, output: {{"nodes": [], "relations": []}}
    """

    output = get_llm_response(prompt, api_key)
    parsed_output = parse_json(output)
    return parsed_output

def map_utterances_to_nodes(embedder, utterances: List[str], node_texts: List[str], top_k: int = 2,
                            min_sim: float = 0.18) -> List[List[int]]:
    node_emb = embedder.embed_many(node_texts)
    utt_emb = embedder.embed_many(utterances)
    S = cosine_sim_matrix(utt_emb, node_emb)

    mappings = []
    for i in range(S.shape[0]):
        sims = S[i]
        top_idx = np.argsort(-sims)[:top_k]
        keep = [int(j) for j in top_idx if float(sims[j]) >= min_sim]
        mappings.append(keep)
    return mappings

def compute_nsr(mappings: List[List[int]], num_nodes: int) -> List[float]:
    U = len(mappings)
    counts = np.zeros(num_nodes, dtype=np.float32)
    for idxs in mappings:
        for j in set(idxs): # count a node at most once per utterance
            counts[j] += 1.0
    return (counts / max(U, 1)).tolist()

def compute_es(mappings: List[List[int]], edges: List[Tuple[int, int]]) -> List[float]:
    U = len(mappings)
    ES = []
    for (s, t) in edges:
        c = 0.0
        for idxs in mappings:
            if (s in idxs) and (t in idxs):
                c += 1.0
        ES.append(c / max(U, 1))
    return ES

def run_graph_rag_eval(embedder, kg: Dict[str, Any], eval_utterances: List[str], eval_utterances_str: str, api_key: str,
                       top_k: int = 2, min_sim: float = 0.18) -> Dict[str, Any]:
    graph = build_graph(kg, eval_utterances_str, api_key)
    nodes = graph["nodes"]
    relations = graph["relations"]

    node_texts = [n["content"] for n in nodes]
    edges = [(r["source"], r["target"]) for r in relations]

    mappings = map_utterances_to_nodes(embedder, eval_utterances, node_texts, top_k=top_k, min_sim=min_sim)
    nsr = compute_nsr(mappings, num_nodes=len(nodes))
    es = compute_es(mappings, edges)

    model_name = getattr(embedder, "model_id", None)
    if not model_name:
        model_obj = getattr(embedder, "model", None)
        if isinstance(model_obj, str):
            model_name = model_obj
        elif model_obj is not None:
            model_name = getattr(model_obj, "name_or_path", None) or model_obj.__class__.__name__
        else:
            model_name = embedder.__class__.__name__

    return {
        "params": {"top_k": top_k, "min_sim": min_sim, "embed_model": str(model_name)},
        "overall_NSR": float(sum(nsr)),
        "overall_ES": float(sum(es)),
        "NSR": nsr,
        "ES": es,
        "utterance_node_mappings": mappings,
        "num_nodes": len(node_texts),
        "num_edges": len(edges),
    }

# Utils
def load_json(path: Path | str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))

def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def is_none_summary(x) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() == "none":
        return True
    return False

def load_outputs(output_path: str) -> dict:
    p = Path(output_path)
    if p.is_file() and p.suffix.lower() == ".json":
        return load_json(p)
    if p.is_dir():
        out = {}
        for fp in sorted(p.glob("*.json")):
            out[fp.stem] = load_json(fp)
        return out
    return {}

def load_eval_utterances(data_path: str) -> List[str]:
    data = load_json(data_path)
    if isinstance(data, dict) and isinstance(data.get("users"), list):
        data = data["users"]
    if not data:
        return [], ""
    sessions = data[0].get("sessions") or []
    if not sessions:
        return [], ""
    turns = sessions[0].get("dialogue") or []
    if not turns:
        return [], ""
    utts = [t.get("text", "") for t in turns]
    utts_str = "\n".join([f"{t.get('speaker', '')}: {t.get('text', '')}" for t in turns])
    return utts, utts_str