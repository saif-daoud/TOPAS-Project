# RL (SFT + Offline PPO + Simulation)

This package provides:
- SFT (supervised) training for encoder policies or activation-MLP policies
- Offline PPO for macro/micro/termination policies
- Validation simulation
- Activation extraction for latent methods

## Data layout (RL root)

All RL training and simulation operates on a self-contained root:

`
<paths.output>/<domain_adj>/<artifact_dir>/
  data/<dataset>.json
  components/*.json
  annotations/*.csv
`

This root is prepared by 	opa.rl.training_phase.RLTrainingPhase.prepare.

User split file (optional, for user-level train/val splits):
- Place `user_split.json` (or `<user_label>_split.json`) next to the dataset JSON.
- When running the pipeline, it is copied into the RL root `data/` folder.
- SFT scripts will automatically fall back to `root/data/user_split.json` if the default path is missing.

## SFT outputs

Encoders (state_repr=encoder):
- uns/sft_macro_no_conv/
- uns/sft_conv_state/
- uns/sft_macro_with_conv/
- uns/sft_micro_flat/
- uns/sft_micro_independent/
- uns/sft_termination/

Activations (state_repr=activations):
- uns/sft_macro_no_conv_acts/
- uns/sft_macro_with_conv_acts/
- uns/sft_micro_independent_acts/
- uns/sft_termination_acts/

## PPO outputs

Encoders:
- uns/ppo_macro_no_conv/
- uns/ppo_macro_with_conv/
- uns/ppo_micro_flat/
- uns/ppo_micro_all/
- uns/ppo_termination_rl/

Activations:
- uns/ppo_macro_no_conv_acts/
- uns/ppo_macro_with_conv_acts/
- uns/ppo_micro_flat_acts/
- uns/ppo_micro_all_acts/
- uns/ppo_termination_rl_acts/

## Simulation outputs

Default:

`
<paths.output>/<domain_adj>/simulation/
  results.jsonl
  progress.json
  summary.json
  conversations/
  conversations_meta/
`

## Experiment isolation

To avoid overwriting older runs:
- Set l.runs_dir to a unique folder per experiment, e.g. uns/encoders_2026_01_27.
- Set simulation.output_dir with a unique run tag.

The provided configs keep encoder and activation runs separate:
- config_cbt_encoders.yaml
- config_cbt_activations.yaml
- config_cbt_simulation.yaml

## Activation extraction (latent methods)

Use 	opa/rl/extract_internal_representation/ to build activation caches:
- Input: transcripts under extract_internal_representation/transcripts/
- Output: extract_internal_representation/activations_<model>/user_<idx>/session_<idx>.pt

These activations are used by SFT/PPO activation training and latent simulation methods.
