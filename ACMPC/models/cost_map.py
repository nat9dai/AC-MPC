"""Neural cost map that generates quadratic cost parameters for the MPC head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch
from torch import Tensor, nn

from ..model_config import CostMapConfig


@dataclass
class CostMapParameters:
    """Container holding per-step quadratic and linear cost coefficients."""

    running_C: Tensor  # [B, H, nx+nu, nx+nu]
    running_c: Tensor  # [B, H, nx+nu]
    terminal_C: Tensor  # [B, nx+nu, nx+nu]
    terminal_c: Tensor  # [B, nx+nu]


class CostMapNetwork(nn.Module):
    """Maps latent features into diagonal Q/R and linear terms as in AC-MPC."""

    def __init__(
        self,
        *,
        latent_dim: int,
        state_dim: int,
        action_dim: int,
        horizon: int,
        config: CostMapConfig,
        include_state: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.config = config
        self.include_state = include_state

        hidden_dim = config.hidden_dim
        num_layers = max(int(config.num_layers), 1)
        dropout = float(config.dropout)

        input_dim = latent_dim + (state_dim if include_state else 0)

        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)

        stage_diag_dim = horizon * (state_dim + action_dim)
        stage_linear_dim = horizon * (state_dim + action_dim)
        terminal_diag_dim = state_dim + action_dim
        terminal_linear_dim = state_dim + action_dim
        output_dim = stage_diag_dim + stage_linear_dim + terminal_diag_dim + terminal_linear_dim
        self.head = nn.Linear(hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, latent: Tensor, state: Tensor | None = None) -> CostMapParameters:
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Cost map expected latent dimension {self.latent_dim}, received {latent.shape[-1]}"
            )

        if self.include_state:
            if state is None:
                raise ValueError("state must be provided when include_state=True")
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if state.shape[0] != latent.shape[0]:
                raise ValueError(
                    f"Cost map state batch size {state.shape[0]} does not match latent batch {latent.shape[0]}"
                )
            if state.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Cost map expected state dimension {self.state_dim}, received {state.shape[-1]}"
                )
            features = torch.cat([latent, state], dim=-1)
        else:
            features = latent

        x = self.backbone(features)
        raw = self.head(x)

        stage_diag_dim = self.horizon * (self.state_dim + self.action_dim)
        stage_linear_dim = stage_diag_dim
        terminal_diag_dim = self.state_dim + self.action_dim
        terminal_linear_dim = terminal_diag_dim

        cursor = 0
        stage_diag_raw = raw[:, cursor : cursor + stage_diag_dim]
        cursor += stage_diag_dim
        stage_linear_raw = raw[:, cursor : cursor + stage_linear_dim]
        cursor += stage_linear_dim
        terminal_diag_raw = raw[:, cursor : cursor + terminal_diag_dim]
        cursor += terminal_diag_dim
        terminal_linear_raw = raw[:, cursor : cursor + terminal_linear_dim]

        cfg = self.config.bounds
        eps = 1e-6

        # Running diagonals
        stage_diag = stage_diag_raw.view(-1, self.horizon, self.state_dim + self.action_dim)
        stage_q_raw = stage_diag[..., : self.state_dim]
        stage_r_raw = stage_diag[..., self.state_dim :]
        stage_q = cfg.q_min + (cfg.q_max - cfg.q_min) * self.sigmoid(stage_q_raw)
        stage_r = cfg.r_min + (cfg.r_max - cfg.r_min) * self.sigmoid(stage_r_raw)
        # Optional exploration noise on running Q/R during training
        if self.training and getattr(self.config, "noise_scale_diag", 0.0) > 0.0:
            noise_q = torch.randn_like(stage_q) * float(self.config.noise_scale_diag)
            noise_r = torch.randn_like(stage_r) * float(self.config.noise_scale_diag)
            stage_q = stage_q + noise_q
            stage_r = stage_r + noise_r
        stage_q = torch.clamp(stage_q, min=cfg.q_min + eps, max=cfg.q_max)
        stage_r = torch.clamp(stage_r, min=cfg.r_min + eps, max=cfg.r_max)

        # Running linear terms (state/action separated bounds)
        # FIX: Use sigmoid to guarantee c > 0 (attracts to tau=0 waypoint)
        stage_linear = stage_linear_raw.view(-1, self.horizon, self.state_dim + self.action_dim)
        stage_linear_state = 0.1 + cfg.linear_state_bound * self.sigmoid(stage_linear[..., : self.state_dim])
        stage_linear_action = 0.1 + cfg.linear_action_bound * self.sigmoid(stage_linear[..., self.state_dim :])
        if self.training and getattr(self.config, "noise_scale_linear", 0.0) > 0.0:
            noise_ls = torch.randn_like(stage_linear_state) * float(self.config.noise_scale_linear)
            noise_la = torch.randn_like(stage_linear_action) * float(self.config.noise_scale_linear)
            stage_linear_state = stage_linear_state + noise_ls
            stage_linear_action = stage_linear_action + noise_la
        stage_linear_full = torch.cat([stage_linear_state, stage_linear_action], dim=-1)

        running_diag = torch.cat([stage_q, stage_r], dim=-1)
        running_C = torch.diag_embed(running_diag)
        running_c = stage_linear_full

        # Terminal terms
        terminal_q_raw = terminal_diag_raw[:, : self.state_dim]
        terminal_r_raw = terminal_diag_raw[:, self.state_dim :]
        terminal_q = cfg.q_min + (cfg.q_max - cfg.q_min) * self.sigmoid(terminal_q_raw)
        terminal_r = cfg.r_min + (cfg.r_max - cfg.r_min) * self.sigmoid(terminal_r_raw)
        if self.training and getattr(self.config, "noise_scale_diag", 0.0) > 0.0:
            noise_tq = torch.randn_like(terminal_q) * float(self.config.noise_scale_diag)
            noise_tr = torch.randn_like(terminal_r) * float(self.config.noise_scale_diag)
            terminal_q = terminal_q + noise_tq
            terminal_r = terminal_r + noise_tr
        terminal_q = torch.clamp(terminal_q, min=cfg.q_min + eps, max=cfg.q_max)
        terminal_r = torch.clamp(terminal_r, min=cfg.r_min + eps, max=cfg.r_max)
        terminal_diag = torch.cat([terminal_q, terminal_r], dim=-1)
        terminal_C = torch.diag_embed(terminal_diag)

        # FIX: Use sigmoid to guarantee c > 0 (attracts to tau=0 waypoint)
        terminal_linear_state = 0.1 + cfg.linear_state_bound * self.sigmoid(terminal_linear_raw[:, : self.state_dim])
        terminal_linear_action = 0.1 + cfg.linear_action_bound * self.sigmoid(terminal_linear_raw[:, self.state_dim :])
        terminal_c = torch.cat([terminal_linear_state, terminal_linear_action], dim=-1)

        return CostMapParameters(
            running_C=running_C,
            running_c=running_c,
            terminal_C=terminal_C,
            terminal_c=terminal_c,
        )
