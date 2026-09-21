#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SFT_DIR}"

AMP=${AMP:-auto}

python split_users.py --root "${ROOT_DIR}" --val_frac 0.2

python train_conv_state.py --root "${ROOT_DIR}" --output_dir outputs/conv_state --amp "${AMP}"

python train_macro_action.py --root "${ROOT_DIR}" --output_dir outputs/macro_action --amp "${AMP}"

python train_micro_action.py --root "${ROOT_DIR}" --output_dir outputs/micro_action --amp "${AMP}"
