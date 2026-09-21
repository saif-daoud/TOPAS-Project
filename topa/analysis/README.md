# Analysis

This package contains post-annotation analysis utilities (review set creation, ambiguity mining, diagnostics).

## Typical inputs

- Annotations directory:

`
<paths.output>/<domain_adj>/<ExtractorName>/annotations/
`

Common required files:
- Annotations_macro_actions.csv
- Annotations_micro_actions.csv
- Annotations_conv_state.csv
- Annotations_extrinsic_reward.csv (optional)

## Typical outputs

Most scripts write outputs under the same annotations directory or under:

`
<paths.output>/<domain_adj>/analysis/
`

Examples:
- expert review CSVs (e.g., `expert_review_micro_actions.csv`)
- ambiguity review sets (e.g., `expert_review_conversation_states.csv`)
- diagnostic plots or summary JSON

## Notes

These scripts are optional. They do not modify core pipeline artifacts unless explicitly configured.

## How review sets are triggered

Review sets are produced by the annotation phase when you enable:
- `annotations_params.expert_feedback` (micro actions)
- `annotations_params.conv_state_ambiguity` (conversation states)
