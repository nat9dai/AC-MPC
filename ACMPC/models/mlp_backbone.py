"""Simple MLP backbone for actor and critic networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class MLPMemories:
    """Dummy memory container for MLP backbone (for compatibility with Transformer interface)."""

    pass

    def detach(self) -> "MLPMemories":
        return MLPMemories()


class MLPBackbone(nn.Module):
    """Simple MLP backbone that processes current state (and optional history) into latent features."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output projection to desired dimension
        if output_dim != hidden_dim:
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def init_memories(self, batch_size: int, device: torch.device) -> MLPMemories:
        """Return dummy memories for compatibility with Transformer interface."""
        return MLPMemories()

    def forward(
        self,
        x: Tensor,
        *,
        episode_ids: Optional[Tensor] = None,
        memories: Optional[MLPMemories] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, MLPMemories]:
        """
        Forward pass through MLP backbone.

        Parameters
        ----------
        x: Tensor
            Input tensor of shape [batch, time, input_dim] or [batch, input_dim].
            If 3D, only the last timestep is used (current state).
        episode_ids: Optional[Tensor]
            Ignored (for compatibility with Transformer interface).
        memories: Optional[MLPMemories]
            Ignored (for compatibility with Transformer interface).
        attn_mask: Optional[Tensor]
            Ignored (for compatibility with Transformer interface).

        Returns
        -------
        features: Tensor
            Output features of shape [batch, output_dim] or [batch, time, output_dim].
        memories: MLPMemories
            Dummy memories for compatibility.
        """
        if x.dim() == 3:
            # For sequence input, use only the last timestep (current state)
            x = x[:, -1, :]  # [batch, input_dim]
            squeeze_time = False
        elif x.dim() == 2:
            squeeze_time = False
        else:
            raise ValueError(f"Input must be 2D or 3D, got shape {x.shape}")

        features = self.net(x)  # [batch, output_dim]

        # Return dummy memories for compatibility
        return features, MLPMemories()

