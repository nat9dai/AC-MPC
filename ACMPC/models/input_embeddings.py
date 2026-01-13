"""Shared sequence embedding utilities for the ACMPC actor and critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class HistoryEmbeddingConfig:
    state_dim: int
    d_model: int
    dropout: float
    include_prev_action: bool = False
    prev_action_dim: int = 0
    include_lidar: bool = False
    lidar_dim: int = 0


class HistoryTokenEncoder(nn.Module):
    """Projects sequences of absolute states (and optional context) into TXL tokens."""

    def __init__(self, config: HistoryEmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.state_dim
        if config.include_prev_action:
            in_dim += config.prev_action_dim
        if config.include_lidar:
            in_dim += config.lidar_dim
        self.in_dim = in_dim

        self.proj = nn.Linear(in_dim, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        state_seq: Tensor,
        *,
        prev_actions: Optional[Tensor] = None,
        lidar_features: Optional[Tensor] = None,
    ) -> Tensor:
        if state_seq.dim() != 3:
            raise ValueError("state_seq must have shape [batch, time, state_dim].")

        batch, time, state_dim = state_seq.shape
        if state_dim != self.config.state_dim:
            raise ValueError(
                f"Expected state dimension {self.config.state_dim}, received {state_dim}."
            )

        features = [state_seq]

        if self.config.include_prev_action:
            if prev_actions is None:
                raise ValueError("prev_actions must be provided when include_prev_action=True.")
            if prev_actions.dim() != 3 or prev_actions.shape[:2] != (batch, time):
                raise ValueError("prev_actions must have shape [batch, time, action_dim].")
            if prev_actions.shape[-1] != self.config.prev_action_dim:
                raise ValueError(
                    f"Expected prev_action_dim {self.config.prev_action_dim}, received {prev_actions.shape[-1]}."
                )
            features.append(prev_actions)
        elif prev_actions is not None:
            raise ValueError("prev_actions provided but include_prev_action is False.")

        if self.config.include_lidar:
            if lidar_features is None:
                raise ValueError("lidar_features must be provided when include_lidar=True.")
            if lidar_features.dim() == 2:
                lidar_features = lidar_features.unsqueeze(1).expand(-1, time, -1)
            if lidar_features.dim() != 3 or lidar_features.shape[:2] != (batch, time):
                raise ValueError("lidar_features must have shape [batch, time, lidar_dim].")
            if lidar_features.shape[-1] != self.config.lidar_dim:
                raise ValueError(
                    f"Expected lidar_dim {self.config.lidar_dim}, received {lidar_features.shape[-1]}."
                )
            features.append(lidar_features)
        elif lidar_features is not None:
            raise ValueError("lidar_features provided but include_lidar is False.")

        token_features = torch.cat(features, dim=-1)
        if token_features.shape[-1] != self.in_dim:
            raise RuntimeError("Constructed feature dimension mismatch.")

        projected = self.proj(token_features)
        projected = self.norm(projected)
        return self.dropout(projected)


class WaypointSequenceEmbedding(nn.Module):
    """Embeds sequences of absolute waypoint targets."""

    def __init__(self, waypoint_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.waypoint_dim = waypoint_dim
        self.d_model = d_model
        self.proj = nn.Linear(waypoint_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, waypoint_seq: Tensor) -> Tensor:
        if waypoint_seq.dim() != 3:
            raise ValueError("waypoint_seq must have shape [batch, num_waypoints, waypoint_dim].")
        if waypoint_seq.shape[-1] != self.waypoint_dim:
            raise ValueError(
                f"Expected waypoint_dim {self.waypoint_dim}, received {waypoint_seq.shape[-1]}."
            )
        projected = self.proj(waypoint_seq)
        projected = self.norm(projected)
        return self.dropout(projected)


def compute_episode_ids(
    episode_starts: Optional[Tensor],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    initial_ids: Optional[Tensor] = None,
) -> Tensor:
    """Return monotonically increasing episode identifiers for each timestep."""

    if seq_len == 0:
        return torch.zeros(batch_size, 0, dtype=torch.long, device=device)

    if episode_starts is None:
        starts = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    else:
        if episode_starts.shape != (batch_size, seq_len):
            raise ValueError("episode_starts must have shape [batch, seq_len].")
        starts = episode_starts.to(device=device)
        starts = (starts > 0).to(torch.long)

    if initial_ids is None:
        initial_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    else:
        initial_ids = initial_ids.to(device=device, dtype=torch.long)
        if initial_ids.dim() == 1:
            initial_ids = initial_ids.view(batch_size, 1)
        elif initial_ids.shape != (batch_size, 1):
            raise ValueError("initial_ids must have shape [batch, 1] if provided.")

    cumulative = torch.cumsum(starts, dim=1)
    episode_ids = initial_ids + cumulative
    return episode_ids


__all__ = [
    "HistoryEmbeddingConfig",
    "HistoryTokenEncoder",
    "WaypointSequenceEmbedding",
    "compute_episode_ids",
]
