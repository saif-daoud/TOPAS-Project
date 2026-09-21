import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def make_writer(log_dir: Path):
    if SummaryWriter is None:
        logger.warning("tensorboard is not installed; continuing without SummaryWriter")
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach()
    return {
        f"{prefix}.mean": float(x.mean().cpu()),
        f"{prefix}.min": float(x.min().cpu()),
        f"{prefix}.max": float(x.max().cpu()),
    }
