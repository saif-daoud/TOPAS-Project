# SFT pretraining (Conversation State -> Macro Action -> Micro Action)

This folder contains **simple** scripts to pretrain 3 RoBERTa-based encoders using **LoRA adapters**:

1. **Conversation-state encoder** (multi-head classification, one head per conv-state dimension)
2. **Macro-action encoder** (macro action classification, conditioned on gold conv-state labels)
3. **Micro-action encoder** (shared backbone + **one head per macro action**)

All scripts assume this directory lives inside your `encoders_RL/` folder:

```
encoders_RL/
  data/AlexanderStreet_Dataset.json
  annotations/*.csv
  components/*.json
  sft_pretraining/   <-- this folder
```

## Domain-agnostic usage

All training scripts take `--root <dir>` where `<dir>` contains:

```
<dir>/data/
<dir>/annotations/
<dir>/components/
```

The dataset JSON can be set with either:

- `TOPA_DATASET_PATH=/abs/path/to/dataset.json` (absolute path), or
- `TOPA_DATASET_JSON=dataset.json` (relative to `<dir>/data/`)

The default prompt prefix used when building text inputs can be overridden with:
`TOPA_RL_PREFIX="..."`.

## Latent-representation (activation) SFT

In addition to RoBERTa+LoRA encoders, this folder includes activation-based SFT
scripts that train an MLP directly on cached LLM activations (compatible with PPO
activation training via `--mlp_init`):

- `train_macro_action_activations.py`
- `train_micro_action_activations.py`
- `train_termination_activations.py`

These expect cached activations at:
`<activation_root>/user_<user_idx>/session_<session_idx>.pt`.

## Install

```bash
cd encoders_RL/sft_pretraining
pip install -r requirements.txt
```

## 1) User split (80/20) with similar session proportions

This uses **enumerate(session)** logic (not `session_metadata.Number`) implicitly because
all training reads sessions by `data[user_idx]['sessions'][session_idx]`.

```bash
python split_users.py --root .. --val_frac 0.2
```

This writes: `<root>/data/user_split.json` (use `--out_dir` to override).

## 2) Train conversation-state encoder (SFT)

Input segment is:
- from system `from_idx`
- to **last user utterance**: `to_idx + 1` if exists and is user, else `to_idx`

```bash
python train_conv_state.py --root .. --output_dir outputs/conv_state
```

Adapter is saved to: `outputs/conv_state/adapter/`

> In your later HRL stage, you can load this adapter and **freeze it**.

## 3) Train macro-action encoder (SFT)

This conditions on gold conv-state labels (compact: only non-`none` dims):

```bash
python train_macro_action.py --root .. --output_dir outputs/macro_action
```

Adapter saved to: `outputs/macro_action/adapter/`

## 4) Train micro-action encoder (SFT)

For each system micro-action label, the input context is the turns **before** that system utterance
(up to `--context_turns` turns), and we append `MACRO_ACTION: <name>`.

```bash
python train_micro_action.py --root .. --output_dir outputs/micro_action
```

Adapter saved to: `outputs/micro_action/adapter/`

## Activation-based SFT (latent vectors)

If you have cached activation vectors (see `topa/rl/activations.py` for the expected
layout), you can train *MLP* policies directly on those vectors. The saved
`*.pt` files are compatible with PPO activation training via `--mlp_init`.

```bash
# Macro policy on activations (optionally fuse conversation-state one-hot vector)
python -m topa.rl.sft_pretraining.train_macro_action_activations \
  --root .. --activation_path <activations_dir> --activation_layer -1 \
  --output_dir outputs/macro_action_acts --use_conv_state

# Micro policy on activations (per-macro)
python -m topa.rl.sft_pretraining.train_micro_action_activations \
  --root .. --activation_path <activations_dir> --activation_layer -1 \
  --output_dir outputs/micro_action_acts --mode per_macro

# Termination function on activations
python -m topa.rl.sft_pretraining.train_termination_activations \
  --root .. --activation_path <activations_dir> --activation_layer -1 \
  --output_dir outputs/termination_acts
```

---

### Notes

- All adapters include the task heads via `modules_to_save`, so `adapter/` is sufficient to reload later.
- Defaults are intentionally simple; tune batch size/epochs/lr as needed.
