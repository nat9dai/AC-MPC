"""Normalization utilities for PPO training targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class RunningMeanStd:
    """Track running mean and variance of scalar streams."""

    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        self.mean = torch.tensor(0.0)
        self.var = torch.tensor(1.0)
        self.count = torch.tensor(self.epsilon)

    def to(self, device: torch.device) -> RunningMeanStd:
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)
        return self

    def update(self, values: Tensor) -> None:
        if values.numel() == 0:
            return
        flat = values.detach().reshape(-1).to(self.mean.device, dtype=torch.float32)
        batch_count = torch.tensor(float(flat.numel()), device=flat.device)
        batch_mean = flat.mean()
        batch_var = flat.var(unbiased=False)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = torch.clamp(new_var, min=self.epsilon)
        self.count = total_count

    @property
    def std(self) -> Tensor:
        return torch.sqrt(self.var + self.epsilon)

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "mean": self.mean.clone(),
            "var": self.var.clone(),
            "count": self.count.clone(),
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for key in ("mean", "var", "count"):
            if key not in state:
                raise KeyError(f"Missing '{key}' in RunningMeanStd state dict.")
        self.mean = state["mean"].to(self.mean.device)
        self.var = state["var"].to(self.var.device)
        self.count = state["count"].to(self.count.device)


@dataclass
class ObservationNormalizer:
    """Track running mean/std for vector observations and provide normalization."""

    epsilon: float = 1e-8
    feature_dims: Optional[int] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        # Lazy init to match observation dimensionality on first update.
        self.mean = torch.tensor(0.0)
        self.var = torch.tensor(1.0)
        self.count = torch.tensor(self.epsilon)

    def _maybe_init(self, dim: int, device: torch.device) -> None:
        label = self.name or "ObservationNormalizer"
        if self.feature_dims is not None and self.feature_dims != dim:
            raise ValueError(f"{label} expected dim {self.feature_dims}, got {dim}.")
        if self.mean.dim() > 0 and self.mean.numel() != 1 and self.mean.numel() != dim:
            raise ValueError(f"{label} already initialised with dim {self.mean.numel()}, got {dim}.")
        if self.mean.dim() == 0 and self.mean.numel() == 1:
            self.mean = torch.zeros(dim, device=device)
            self.var = torch.ones(dim, device=device)
            self.count = torch.as_tensor(self.epsilon, device=device)
        if self.feature_dims is None:
            self.feature_dims = dim

    def to(self, device: torch.device) -> ObservationNormalizer:
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)
        return self

    def update(self, values: Tensor, mask: Optional[Tensor] = None) -> None:
        if values.numel() == 0:
            return
        if values.dim() < 2:
            raise ValueError("ObservationNormalizer expects inputs with last dimension = features.")

        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            while mask_bool.dim() < values.dim():
                mask_bool = mask_bool.unsqueeze(-1)
            values = values[mask_bool.expand_as(values)].view(-1, values.size(-1))
        else:
            values = values.view(-1, values.size(-1))

        if values.numel() == 0:
            return

        self._maybe_init(values.size(-1), values.device)
        batch_count = torch.tensor(float(values.size(0)), device=values.device)
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = torch.clamp(m2 / total_count, min=self.epsilon)

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self) -> Tensor:
        return torch.sqrt(self.var + self.epsilon)

    def normalize(self, values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if values.numel() == 0:
            return values
        if values.dim() < 2:
            return values
        self._maybe_init(values.size(-1), values.device)
        mean = self.mean.to(values.device)
        std = self.std.to(values.device)
        normed = (values - mean) / std
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            while mask_bool.dim() < values.dim():
                mask_bool = mask_bool.unsqueeze(-1)
            normed = torch.where(mask_bool, normed, values)
        return normed

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "mean": self.mean.clone(),
            "var": self.var.clone(),
            "count": self.count.clone(),
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for key in ("mean", "var", "count"):
            if key not in state:
                raise KeyError(f"Missing '{key}' in ObservationNormalizer state dict.")
        self.mean = state["mean"].clone()
        self.var = state["var"].clone()
        self.count = state["count"].clone()
        if self.mean.dim() > 0 and self.mean.numel() > 1:
            self.feature_dims = int(self.mean.numel())


@dataclass
class RewardNormalizer:
    """Normalise value targets using running statistics."""

    epsilon: float = 1e-8
    clip_value: Optional[float] = 10.0

    def __post_init__(self) -> None:
        self._running = RunningMeanStd(epsilon=self.epsilon)

    def normalize(self, targets: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask_tensor = torch.ones_like(targets, dtype=torch.bool)
        else:
            mask_tensor = mask.to(dtype=torch.bool)

        valid = mask_tensor
        if not bool(valid.any()):
            return targets

        self._running.to(targets.device)
        self._running.update(targets[valid])

        mean = self._running.mean.to(targets.device)
        std = self._running.std.to(targets.device)
        normalised = targets.clone()
        normalised[valid] = (targets[valid] - mean) / std
        if self.clip_value is not None:
            normalised[valid] = torch.clamp(normalised[valid], -self.clip_value, self.clip_value)
        normalised = normalised * mask_tensor.to(normalised.dtype)
        return normalised

    def state_dict(self) -> dict[str, Tensor]:
        return self._running.state_dict()

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self._running.load_state_dict(state)
