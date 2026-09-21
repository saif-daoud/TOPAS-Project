"""Backward-compatible shim for the P4G reward definition."""

from ..reward.p4g import build_prompt, parse_output, reward_fn

__all__ = ["build_prompt", "parse_output", "reward_fn"]
