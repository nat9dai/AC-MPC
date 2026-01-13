"""Utilities for converting rollout batches into training-ready tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor

from .rollout import RolloutBatch


@dataclass
class FlattenedBatch:
    tensors: Dict[str, Tensor]
    num_envs: int
    horizon: int


def rollout_to_training(batch: RolloutBatch, *, device: torch.device) -> FlattenedBatch:
    """Flatten a rollout batch into per-sample tensors for PPO/GAE.

    Returns a dictionary containing:
      * observations/states/actions/log_probs/rewards/dones
      * mask (0 for padded steps)
      * episode_start/episode_id/token_offset for TXL masking
    along with metadata about the original env/horizon sizes.
    """

    tensors = batch.flatten_time()
    flattened = {key: value.to(device=device) for key, value in tensors.items()}

    return FlattenedBatch(
        tensors=flattened,
        num_envs=batch.num_envs,
        horizon=batch.horizon,
    )
