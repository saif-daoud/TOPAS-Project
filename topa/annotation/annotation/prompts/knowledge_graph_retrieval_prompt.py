import json
from typing import Any, Dict, List


def annotate_knowledge_graph_retrieval_prompt(
    *,
    domain: str,
    user: str,
    system: str,
    interaction_unit: str,
    session_utterances: List[Dict[str, Any]],
    target_utterances: List[Dict[str, Any]],
    candidate_nodes: List[Dict[str, Any]],
    candidate_relations: List[Dict[str, Any]],
    memory_session_idx: int,
) -> str:
    return f"""
You are an information-use annotator for a {domain} {interaction_unit}.
Your task is to identify which PREVIOUS-SESSION knowledge-graph information the {system} used to produce each TARGET {system} utterance in the current session.

Important: the current session dialogue is already visible at inference time. This prompt may include only the relevant excerpt around the target utterances to keep long sessions tractable. Do NOT select facts that are only needed from the current session. Select only facts from the previous-session KG memory.

The candidate KG below is the user KG available after session {memory_session_idx}; it contains no current-session facts.

CURRENT SESSION DIALOGUE EXCERPT ALREADY VISIBLE:
{json.dumps(session_utterances, ensure_ascii=False, indent=2)}

TARGET {system} UTTERANCES TO ANNOTATE:
{json.dumps(target_utterances, ensure_ascii=False, indent=2)}

CANDIDATE PREVIOUS-SESSION KG NODES:
{json.dumps(candidate_nodes, ensure_ascii=False, indent=2)}

CANDIDATE PREVIOUS-SESSION KG RELATIONS:
{json.dumps(candidate_relations, ensure_ascii=False, indent=2)}

OUTPUT JSON ONLY:
{{
  "utterances": [
    {{
      "system_id": "{system}_0",
      "nodes": [0, 2],
      "relations": [1]
    }}
  ]
}}

RULES:
- Output ONLY a single JSON object with exactly one key: "utterances".
- Include one item for every target {system} utterance.
- Each item MUST have exactly: "system_id", "nodes", "relations".
- "system_id" MUST be one of the target utterance ids.
- "nodes" must be a list of node_idx integers from the candidate KG nodes.
- "relations" must be a list of relation_idx integers from the candidate KG relations.
- Select a KG item only if the target utterance uses that information, refers to it, reasons from it, follows up on it, or depends on it.
- Do not select generic background facts or facts merely related to the topic.
- Do not select information that is directly present only in the current session dialogue.
- If a relation is selected, also include its source and target nodes when they are candidate nodes.
- If no previous-session KG information is used for a target utterance, use empty lists for that utterance.
""".strip()
