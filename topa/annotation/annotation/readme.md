# Annotation

This package annotates a TOPA-format dataset with extracted/refined components.

## Entrypoint

- Class: `topa.annotation.AnnotationManager`
- CLI (typical):

```bash
python run_topa.py --config topa/config/config_cbt.yaml --skip-extraction --skip-refinement
```

## Inputs

- Refined components path (from refinement)
- Dataset JSON: `paths.data`
- `annotations_params` config

## Outputs

Annotations are written under:

```
<paths.output>/<domain_adj>/<ExtractorName>/annotations/
```

Common files:
- `annotations_macro_actions.csv`
- `annotations_micro_actions.csv`
- `annotations_conv_state.csv`
- `annotations_user_profile.csv`
- `annotations_extrinsic_reward.csv` (if enabled)
- `validation_sessions.json` (if validation subset enabled)

Per-component usage logs:
- `annotations_<component>_usage.json`

## Extrinsic reward plug-in (domain-specific)

The extrinsic reward annotator is **not hard-coded**. It loads its prompt and
parsing logic from a domain module under:

```
topa/reward/<domain>/
```

Minimum requirement for a new domain:
- `build_prompt(session_metadata=..., session_transcript=...) -> str`

Optional helpers:
- `parse_output(raw) -> dict`
- `to_row(outputs, user_idx, session_idx, ...) -> dict`
- `OUTPUT_FIELDS = [...]`

Override the module explicitly if needed:

```yaml
annotations_params:
  extrinsic_reward_params:
    reward_module: topa.reward.<your_domain>
```

Starter template:
- `topa/reward/_template/`

Auto-generate:

```bash
python -m topa.reward.generate_reward_module --domain <your_domain>
```

## Validation subset

If `annotations_params.validation_subset.enabled: true`, TOPA writes:

```
<annotations_dir>/validation_sessions.json
```

This is used by the simulation phase.

---

## Expert review of high‑uncertainty cases

Low‑confidence annotation cases can be surfaced for expert review. This is controlled by
`annotations_params.expert_feedback` (micro actions) and `annotations_params.conv_state_ambiguity`
(conversation states).

### With an expert available (recommended)

```yaml
annotations_params:
  expert_feedback:
    enabled: true
    conf_threshold: 0.65
    n_review: 200
    n_clusters: 25
    similarity_delta: 0.85
  conv_state_ambiguity:
    enabled: true
    n_review: 200
    context_length: 10
    none_value: none
```

Outputs (in the annotations/ or analysis/ folder):
- `expert_review_micro_actions.csv`
- `expert_review_conversation_states.csv`

### Without an expert

```yaml
annotations_params:
  expert_feedback:
    enabled: false
  conv_state_ambiguity:
    enabled: false
```

This skips review‑set creation and uses the automatic labels only.

---

# TOPA dataset format

A TOPA dataset is either:
- a list of users, or
- a dict with key `users` containing the list.

Minimal schema (list form):

```json
[
  {
    "user_id": "optional",
    "sessions": [
      {
        "session_metadata": {
          "summary": "...",
          "presenting conditions": ["..."],
          "split": "train"
        },
        "dialogue": [
          {"speaker": "system", "text": "..."},
          {"speaker": "user", "text": "..."}
        ]
      }
    ]
  }
]
```

Required fields:
- `sessions`
- each session has `dialogue` and `session_metadata`

Speaker labels:
- Default: `system` and `user`
- If your dataset uses different labels, set env:
  - `TOPA_SYSTEM_LABEL`
  - `TOPA_USER_LABEL`
