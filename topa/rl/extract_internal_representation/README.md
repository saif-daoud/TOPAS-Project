# Internal Representation Extraction

This module extracts one target-LLM activation vector per system turn.

The TOPA pipeline reads roles from `domain_params` in the config:

- CBT: `system: therapist`, `user: patient`
- P4G: `system: persuader`, `user: persuadee`

Those roles become transcript tags such as `<|therapist|>` and `<|patient|>`.
The user-role directory prefix is derived from `domain_params.user`.
The activation cache is saved as:

```text
<activation_out_dir>/<user_role>_<user_idx>/session_<session_idx>.pt
```

Each file contains `activations` with shape `[layers, system_turns, hidden_dim]`.

For activation-based SFT/PPO, set:

```yaml
rl:
  activation_model_id: Qwen/Qwen2.5-7B-Instruct
  activation_dtype: float16
  activation_device_map: auto
  activation_transcripts_dir: extract_internal_representation/transcripts
  activation_out_dir: extract_internal_representation/activations_{model}
  sft:
    state_repr: activations
  ppo:
    state_repr: activations
```

Activation extraction is always available for activation-based SFT/PPO; there is no
separate enable flag.

If `activation_out_dir` already contains a complete cache, TOPA checks it and reuses it.
If files are missing, TOPA regenerates the cache at that path.
Sanity checks run during transcript creation, activation extraction, and cache reuse.
