import json
from typing import Any, Dict, List


def _kg_schema(kg_components: Dict[str, Any]) -> str:
    schema = {
        "nodes": kg_components.get("nodes", []),
        "edges": kg_components.get("edges", []),
    }
    return json.dumps(schema, ensure_ascii=False, indent=2)


def _allowed_text(kg_components: Dict[str, Any]):
    node_types = [x.get("type", "") for x in kg_components.get("nodes", [])]
    relation_types = [x.get("relation", "") for x in kg_components.get("edges", [])]
    allowed_nodes = ", ".join(sorted({x for x in node_types if x}))
    allowed_rels = ", ".join(sorted({x for x in relation_types if x}))
    return allowed_nodes, allowed_rels


def annotate_knowledge_graph_prompt(
    *,
    domain: str,
    user: str,
    system: str,
    interaction_unit: str,
    kg_components: Dict[str, Any],
    session_utterances: List[Dict[str, Any]],
    current_session_idx: int,
) -> str:
    """Build the first-session KG prompt.

    Nodes use session_idx rather than system_id because the KG is now a
    longitudinal per-user memory. For session 0, all extracted information is
    first available after session 0.
    """
    allowed_nodes, allowed_rels = _allowed_text(kg_components)

    return f"""
You are an information extraction system for a {domain} {interaction_unit}.
Read the CURRENT SESSION and build the initial per-{user} knowledge graph.
Output ONLY a single JSON object with two arrays: "nodes" and "relations".
Include nothing else.

CURRENT SESSION INDEX: {current_session_idx}
ALLOWED NODE TYPES: {allowed_nodes}
ALLOWED RELATION TYPES: {allowed_rels}

KG SCHEMA WITH DESCRIPTIONS:
{_kg_schema(kg_components)}

CURRENT SESSION:
<<<TRANSCRIPT_START>>>
{json.dumps(session_utterances, ensure_ascii=False, indent=2)}
<<<TRANSCRIPT_END>>>

OUTPUT JSON ONLY:
{{
  "nodes": [
    {{
      "type": "<one allowed node type>",
      "content": "<short canonical surface form>",
      "session_idx": {current_session_idx}
    }}
  ],
  "relations": [
    {{
      "type": "<one allowed relation type>",
      "source": 0,
      "target": 1
    }}
  ]
}}

RULES:
- Output ONLY the keys shown above. No markdown, no comments, no trailing text.
- "nodes" items MUST have exactly: "type", "content", "session_idx".
- For this initial KG, every node "session_idx" MUST be {current_session_idx}.
- "relations" items MUST have exactly: "type", "source", "target".
- Node "type" values MUST be from ALLOWED NODE TYPES.
- Relation "type" values MUST be from ALLOWED RELATION TYPES.
- "source" and "target" MUST be integer 0-based indexes into the nodes array.
- Deduplicate nodes so each real-world concept appears once; reuse its index in relations.
- If nothing matches, output exactly: {{"nodes": [], "relations": []}}
""".strip()


def refine_knowledge_graph_prompt(
    *,
    domain: str,
    user: str,
    system: str,
    interaction_unit: str,
    kg_components: Dict[str, Any],
    previous_kg: Dict[str, Any],
    session_utterances: List[Dict[str, Any]],
    current_session_idx: int,
) -> str:
    """Build the longitudinal KG refinement prompt for session >= 1."""
    allowed_nodes, allowed_rels = _allowed_text(kg_components)

    return f"""
You are an information extraction system for a {domain} {interaction_unit}.
You maintain one longitudinal knowledge graph for the same {user} across sessions.
Given the PREVIOUS USER KG and the CURRENT SESSION, output the FULL UPDATED USER KG.
Output ONLY a single JSON object with two arrays: "nodes" and "relations".
Include nothing else.

CURRENT SESSION INDEX: {current_session_idx}
ALLOWED NODE TYPES: {allowed_nodes}
ALLOWED RELATION TYPES: {allowed_rels}

KG SCHEMA WITH DESCRIPTIONS:
{_kg_schema(kg_components)}

PREVIOUS USER KG:
{json.dumps(previous_kg, ensure_ascii=False, indent=2)}

CURRENT SESSION:
<<<TRANSCRIPT_START>>>
{json.dumps(session_utterances, ensure_ascii=False, indent=2)}
<<<TRANSCRIPT_END>>>

OUTPUT JSON ONLY:
{{
  "nodes": [
    {{
      "type": "<one allowed node type>",
      "content": "<short canonical surface form>",
      "session_idx": 0
    }}
  ],
  "relations": [
    {{
      "type": "<one allowed relation type>",
      "source": 0,
      "target": 1
    }}
  ]
}}

RULES:
- Output the FULL UPDATED USER KG, not only newly added facts.
- Output ONLY the keys shown above. No markdown, no comments, no trailing text.
- "nodes" items MUST have exactly: "type", "content", "session_idx".
- For old nodes from PREVIOUS USER KG, preserve their original "session_idx".
- For newly introduced nodes from CURRENT SESSION, set "session_idx" to {current_session_idx}.
- Never use a node "session_idx" greater than {current_session_idx}.
- "relations" items MUST have exactly: "type", "source", "target".
- Node "type" values MUST be from ALLOWED NODE TYPES.
- Relation "type" values MUST be from ALLOWED RELATION TYPES.
- "source" and "target" MUST be integer 0-based indexes into the returned nodes array.
- Deduplicate nodes so each real-world concept appears once; reuse its index in relations.
- Refine wording or merge duplicates when useful, but do not delete clinically relevant previous-session information.
- If nothing matches and there is no previous KG, output exactly: {{"nodes": [], "relations": []}}
""".strip()
