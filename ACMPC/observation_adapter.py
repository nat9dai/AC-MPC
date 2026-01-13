"""Observation preprocessing utilities for ACMPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor


@dataclass
class ObservationSpec:
    """Describes the structure of raw environment observations."""

    state_dim: int
    history: int = 1

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.history <= 0:
            self.history = 1


class ObservationAdapter:
    """Stateless utilities for constructing observation sequences."""

    def __init__(self, spec: ObservationSpec, *, device: str = "cpu") -> None:
        self.spec = spec
        self.device = torch.device(device)
        self.history = max(1, spec.history)

    def initial_history(self) -> Tensor:
        """Return an empty history tensor ``[history, state_dim]`` filled with zeros."""

        return torch.zeros(self.history, self.spec.state_dim, device=self.device)

    def update(self, history: Tensor, raw_obs: Iterable[float]) -> Tensor:
        """Return a new history tensor that includes ``raw_obs`` as the latest entry."""

        expected_shape = (self.history, self.spec.state_dim)
        if history.shape != expected_shape:
            raise ValueError(f"Expected history shape {expected_shape}, received {tuple(history.shape)}")

        obs_tensor = torch.as_tensor(list(raw_obs), dtype=torch.float32, device=self.device)
        if obs_tensor.numel() != self.spec.state_dim:
            raise ValueError(f"Expected observation of size {self.spec.state_dim}, received {obs_tensor.numel()}")

        obs_tensor = obs_tensor.view(1, self.spec.state_dim)
        if self.history == 1:
            return obs_tensor.clone()

        new_history = torch.cat([history[1:], obs_tensor], dim=0)
        return new_history

    def process(self, history: Tensor, raw_obs: Iterable[float]) -> Tuple[Tensor, Tensor, Tensor]:
        """Update history and return ``(new_history, sequence, current_state)``."""

        new_history = self.update(history, raw_obs)
        seq = new_history.unsqueeze(0)
        current = new_history[-1].unsqueeze(0)
        return new_history, seq, current

    def __call__(self, history: Tensor, raw_obs: Iterable[float]) -> Tuple[Tensor, Tensor, Tensor]:
        return self.process(history, raw_obs)

    @staticmethod
    def collate_sequences(sequences: Sequence[Tensor], *, device: Optional[str | torch.device] = None) -> Tensor:
        """Stack a list of ``[T, state_dim]`` sequences into ``[B, T, state_dim]``."""

        if len(sequences) == 0:
            raise ValueError("At least one sequence is required to collate.")
        target_device = torch.device(device) if device is not None else sequences[0].device
        seqs = [seq.to(device=target_device) for seq in sequences]
        lengths = {seq.size(0) for seq in seqs}
        if len(lengths) != 1:
            raise ValueError("All sequences must share the same temporal length.")
        return torch.stack(seqs, dim=0)
