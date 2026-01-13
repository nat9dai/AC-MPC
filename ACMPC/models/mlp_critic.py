"""MLP-based critic model for ACMPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .mlp_backbone import MLPBackbone, MLPMemories


@dataclass
class MLPCriticOutput:
    """Container for MLP critic outputs."""

    value: Tensor
    memories: MLPMemories


class MLPCritic(nn.Module):
    """MLP-based critic producing scalar value estimates."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 1,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        include_prev_action: bool = False,
        prev_action_dim: int = 0,
        include_lidar: bool = False,
        lidar_dim: int = 0,
        waypoint_dim: int = 0,
        waypoint_sequence_len: int = 0,
    ) -> None:
        super().__init__()
        self.state_dim = input_dim
        self.include_prev_action = include_prev_action
        self.include_lidar = include_lidar
        self.prev_action_dim = prev_action_dim
        self.lidar_dim = lidar_dim
        self.waypoint_dim = waypoint_dim
        self.waypoint_sequence_len = waypoint_sequence_len

        # Build input dimension for MLP
        mlp_input_dim = input_dim
        if include_prev_action:
            mlp_input_dim += prev_action_dim
        if include_lidar:
            mlp_input_dim += lidar_dim
        if waypoint_dim > 0 and waypoint_sequence_len > 0:
            mlp_input_dim += waypoint_dim * waypoint_sequence_len

        # MLP backbone: 2 layers of 512-ReLU (as per paper)
        self.backbone = MLPBackbone(
            input_dim=mlp_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output of backbone is hidden_dim
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

        # Value head: single linear layer to output scalar
        self.value_head = nn.Linear(hidden_dim, output_dim)

    def init_memories(self, batch_size: int, device: torch.device) -> MLPMemories:
        """Initialise memories for the MLP backbone."""
        return self.backbone.init_memories(batch_size, device)

    def forward(
        self,
        history: Tensor,
        *,
        memories: Optional[MLPMemories] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,
        raw_state: Optional[Tensor] = None,
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> MLPCriticOutput:
        if history.dim() == 2:
            history = history.unsqueeze(0)
        if history.dim() != 3:
            raise ValueError("history must have shape [batch, time, state_dim].")

        batch, seq_len, state_dim = history.shape
        if state_dim != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, received {state_dim}."
            )

        # Extract current state (last timestep)
        current_state = history[:, -1, :]  # [batch, state_dim]

        # Build MLP input by concatenating features
        mlp_inputs = [current_state]

        if self.include_prev_action:
            if prev_actions is None:
                raise ValueError("prev_actions must be provided when include_prev_action=True.")
            if prev_actions.dim() == 2:
                prev_actions = prev_actions.unsqueeze(1)
            if prev_actions.dim() == 3:
                prev_actions = prev_actions[:, -1, :]
            mlp_inputs.append(prev_actions)

        if self.include_lidar:
            if lidar is None:
                raise ValueError("lidar must be provided when include_lidar=True.")
            if lidar.dim() == 2:
                lidar = lidar.unsqueeze(1)
            if lidar.dim() == 3:
                lidar = lidar[:, -1, :]
            mlp_inputs.append(lidar)

        if self.waypoint_dim > 0 and self.waypoint_sequence_len > 0:
            waypoint_input = waypoint_seq if waypoint_seq is not None else raw_waypoint_seq
            if waypoint_input is not None:
                if waypoint_input.dim() == 2:
                    waypoint_input = waypoint_input.unsqueeze(1)
                if waypoint_input.dim() == 3:
                    waypoint_flat = waypoint_input.view(batch, -1)
                    mlp_inputs.append(waypoint_flat)

        mlp_input = torch.cat(mlp_inputs, dim=-1)  # [batch, mlp_input_dim]

        # Forward through MLP backbone
        features, new_memories = self.backbone(mlp_input, memories=memories)

        # Value head: output scalar value
        value = self.value_head(features).squeeze(-1)  # [batch]

        return MLPCriticOutput(value=value, memories=new_memories)

