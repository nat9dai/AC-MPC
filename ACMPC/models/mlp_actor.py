"""MLP-based actor model for ACMPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import warnings
from torch import Tensor, nn
from torch.distributions import Normal, TransformedDistribution, transforms as dist_transforms

from ..model_config import CostMapConfig, TransformerConfig
from ..mpc import CostMapMPCHead, EconomicMPCConfig, EconomicMPCHead
from .cost_map import CostMapNetwork
from .input_embeddings import HistoryEmbeddingConfig, HistoryTokenEncoder
from .mlp_backbone import MLPBackbone, MLPMemories


@dataclass
class MLPActorOutput:
    """Container for MLP actor results."""

    action: Tensor
    memories: MLPMemories
    plan: Optional[Tuple[Tensor, Tensor]] = None
    log_prob: Optional[Tensor] = None
    entropy: Optional[Tensor] = None


class MLPActor(nn.Module):
    """MLP-based actor whose decisions are produced by an economic MPC head."""

    def __init__(
        self,
        *,
        input_dim: int,
        mpc_config: EconomicMPCConfig,
        cost_map_config: CostMapConfig | None,
        dynamics_fn,
        dynamics_jacobian_fn=None,
        include_prev_action: bool = False,
        prev_action_dim: Optional[int] = None,
        include_lidar: bool = False,
        lidar_dim: int = 0,
        waypoint_dim: int = 0,
        waypoint_sequence_len: int = 0,
        use_waypoint_as_ref: bool = False,
        tanh_rescale_actions: bool = False,
        mlp_hidden_dim: int = 512,
        mlp_output_dim: int = 512,
        mlp_num_layers: int = 2,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mlp_output_dim != mpc_config.latent_dim:
            raise ValueError("mlp_output_dim must match mpc_config.latent_dim")

        self.input_dim = input_dim
        self.state_dim = mpc_config.state_dim
        self.action_dim = mpc_config.action_dim
        self.include_prev_action = include_prev_action
        self.include_lidar = include_lidar
        self.prev_action_dim = prev_action_dim if prev_action_dim is not None else self.action_dim
        self.lidar_dim = lidar_dim
        self.waypoint_dim = waypoint_dim
        self.waypoint_sequence_len = waypoint_sequence_len
        self.use_waypoint_as_ref = use_waypoint_as_ref
        self.tanh_rescale_actions = tanh_rescale_actions

        # Build input dimension for MLP
        mlp_input_dim = input_dim
        if include_prev_action:
            mlp_input_dim += prev_action_dim
        if include_lidar:
            mlp_input_dim += lidar_dim
        if waypoint_dim > 0 and waypoint_sequence_len > 0:
            # Concatenate waypoint sequence
            mlp_input_dim += waypoint_dim * waypoint_sequence_len

        self.mlp_backbone = MLPBackbone(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_output_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
        )

        self.cost_map: Optional[CostMapNetwork]
        if cost_map_config is not None:
            self.cost_map = CostMapNetwork(
                latent_dim=mlp_output_dim,
                state_dim=mpc_config.state_dim,
                action_dim=mpc_config.action_dim,
                horizon=mpc_config.horizon,
                config=cost_map_config,
                include_state=False,  # Match transformer actor behavior
            )
            self.mpc_head: nn.Module = CostMapMPCHead(
                mpc_config,
                dynamics_fn=dynamics_fn,
                dynamics_jacobian_fn=dynamics_jacobian_fn,
            )
        else:
            self.cost_map = None
            self.mpc_head = EconomicMPCHead(
                mpc_config,
                dynamics_fn=dynamics_fn,
                dynamics_jacobian_fn=dynamics_jacobian_fn,
            )

        self.log_std = nn.Parameter(torch.zeros(mpc_config.action_dim))

        self._log_std_min = -3.0
        self._log_std_max = 2.0
        if self.tanh_rescale_actions:
            if isinstance(self.mpc_head, (EconomicMPCHead, CostMapMPCHead)):
                warnings.warn(
                    "tanh_rescale_actions is not supported when using an MPC head; leave it disabled.",
                    UserWarning,
                )
            if mpc_config.u_min is None or mpc_config.u_max is None:
                raise ValueError("tanh_rescale_actions requires u_min and u_max in mpc_config.")
            low = torch.as_tensor(mpc_config.u_min, dtype=torch.float32).view(-1)
            high = torch.as_tensor(mpc_config.u_max, dtype=torch.float32).view(-1)
            if low.shape[0] != self.action_dim or high.shape[0] != self.action_dim:
                raise ValueError("Action bounds must match action_dim when using tanh_rescale_actions.")
            self.register_buffer("_action_low", low)
            self.register_buffer("_action_high", high)
        else:
            self._action_low = None  # type: ignore[assignment]
            self._action_high = None  # type: ignore[assignment]

    def init_memories(self, batch_size: int, device: torch.device) -> MLPMemories:
        """Initialise memories for the MLP backbone."""
        return self.mlp_backbone.init_memories(batch_size, device)

    def forward(
        self,
        history: Tensor,
        *,
        state: Tensor,
        raw_state: Optional[Tensor] = None,
        memories: Optional[MLPMemories] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        warm_start: Optional[Tensor] = None,
        return_plan: bool = False,
        stochastic: bool = False,
    ) -> MLPActorOutput:
        """Run the actor forward pass with optional stochastic sampling."""

        mpc_state = raw_state if raw_state is not None else state
        mpc_waypoint = raw_waypoint_seq if raw_waypoint_seq is not None else waypoint_seq

        if history.dim() == 2:
            history = history.unsqueeze(0)
        if history.dim() != 3:
            raise ValueError("history must have shape [batch, time, state_dim].")

        batch, seq_len, state_dim = history.shape
        if state_dim != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, received {state_dim}.")

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
                # Use last timestep
                prev_actions = prev_actions[:, -1, :]
            mlp_inputs.append(prev_actions)

        if self.include_lidar:
            if lidar is None:
                raise ValueError("lidar must be provided when include_lidar=True.")
            if lidar.dim() == 2:
                lidar = lidar.unsqueeze(1)
            if lidar.dim() == 3:
                # Use last timestep
                lidar = lidar[:, -1, :]
            mlp_inputs.append(lidar)

        if self.waypoint_dim > 0 and self.waypoint_sequence_len > 0:
            waypoint_input = waypoint_seq if waypoint_seq is not None else raw_waypoint_seq
            if waypoint_input is not None:
                if waypoint_input.dim() == 2:
                    waypoint_input = waypoint_input.unsqueeze(1)
                if waypoint_input.dim() == 3:
                    # Flatten waypoint sequence
                    waypoint_flat = waypoint_input.view(batch, -1)  # [batch, waypoint_dim * waypoint_sequence_len]
                    mlp_inputs.append(waypoint_flat)

        mlp_input = torch.cat(mlp_inputs, dim=-1)  # [batch, mlp_input_dim]

        # Forward through MLP backbone
        policy_latent, new_memories = self.mlp_backbone(mlp_input, memories=memories)

        if mpc_state.dim() == 1:
            state_batch = mpc_state.unsqueeze(0)
        elif mpc_state.dim() == 2:
            state_batch = mpc_state
        else:
            raise ValueError("State tensor must be rank 1 or 2.")

        if self.cost_map is not None:
            cost_params = self.cost_map(policy_latent)

            # Handle SE2 rotation if needed (similar to transformer actor)
            if self.state_dim == 3 and raw_state is not None:
                geom_state = raw_state if raw_state.dim() == 2 else raw_state.unsqueeze(0)
                theta = geom_state[:, 2]
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                zero = torch.zeros_like(theta)
                one = torch.ones_like(theta)

                horizon = cost_params.running_C.shape[1]
                R = torch.stack([
                    torch.stack([cos_t, -sin_t, zero], dim=-1),
                    torch.stack([sin_t, cos_t, zero], dim=-1),
                    torch.stack([zero, zero, one], dim=-1)
                ], dim=-2).unsqueeze(1).expand(-1, horizon, -1, -1)

                Q_local = cost_params.running_C[..., :3, :3]
                Q_global = torch.matmul(R, torch.matmul(Q_local, R.transpose(-1, -2)))
                cost_params.running_C[..., :3, :3] = Q_global

                N_local = cost_params.running_C[..., :3, 3:]
                N_global = torch.matmul(R, N_local)
                cost_params.running_C[..., :3, 3:] = N_global
                cost_params.running_C[..., 3:, :3] = N_global.transpose(-1, -2)

                R_term = R[:, 0, :, :]
                Qt_local = cost_params.terminal_C[..., :3, :3]
                Qt_global = torch.matmul(R_term, torch.matmul(Qt_local, R_term.transpose(-1, -2)))
                cost_params.terminal_C[..., :3, :3] = Qt_global

                Nt_local = cost_params.terminal_C[..., :3, 3:]
                Nt_global = torch.matmul(R_term, Nt_local)
                cost_params.terminal_C[..., :3, 3:] = Nt_global
                cost_params.terminal_C[..., 3:, :3] = Nt_global.transpose(-1, -2)

            x_ref_tensor: Optional[Tensor] = None
            u_ref_tensor: Optional[Tensor] = None

            if self.use_waypoint_as_ref:
                if mpc_waypoint is None:
                    raise ValueError("use_waypoint_as_ref is True but waypoint_seq is None.")
                wp = mpc_waypoint
                if wp.dim() == 2:
                    wp = wp.unsqueeze(1)
                current_wp = wp[:, 0, :]
                desired_state = state_batch.clone()
                desired_state[:, : self.waypoint_dim] = current_wp
                if self.state_dim == 3 and self.waypoint_dim == 2:
                    dx = current_wp[:, 0] - state_batch[:, 0]
                    dy = current_wp[:, 1] - state_batch[:, 1]
                    desired_theta = torch.atan2(dy, dx)
                    desired_state[:, 2] = desired_theta
                horizon = self.mpc_head.config.horizon
                batch = state_batch.size(0)
                x_ref_tensor = desired_state.unsqueeze(1).repeat(1, horizon + 1, 1)
                u_ref_tensor = torch.zeros(
                    batch,
                    horizon,
                    self.action_dim,
                    device=state_batch.device,
                    dtype=state_batch.dtype,
                )
            else:
                if mpc_waypoint is not None and self.waypoint_dim > 0:
                    wp = mpc_waypoint[:, 0, :] if mpc_waypoint.dim() == 3 else mpc_waypoint
                    mpc_state_rel = mpc_state.clone()
                    mpc_state_rel[:, :self.waypoint_dim] -= wp
                    mpc_state = mpc_state_rel

            head_output = self.mpc_head(
                state=mpc_state if mpc_state.dim() == 2 else mpc_state.unsqueeze(0),
                cost=cost_params,
                x_ref=x_ref_tensor,
                u_ref=u_ref_tensor,
                warm_start=warm_start,
                return_plan=return_plan,
            )
        else:
            head_output = self.mpc_head(
                policy_latent,
                mpc_state if mpc_state.dim() == 2 else mpc_state.unsqueeze(0),
                warm_start=warm_start,
                return_plan=return_plan,
            )

        if isinstance(head_output, tuple):
            mean_action, plan = head_output
        else:
            mean_action = head_output
            plan = None

        if not torch.isfinite(mean_action).all():
            mean_action = torch.where(
                torch.isfinite(mean_action),
                mean_action,
                torch.zeros_like(mean_action),
            )

        log_std_param = self.log_std
        if not torch.isfinite(log_std_param).all():
            with torch.no_grad():
                self.log_std.data.zero_()
        log_std_param = self.log_std

        log_std = log_std_param.clamp(self._log_std_min, self._log_std_max)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean_action)
        if not torch.isfinite(std).all() or (std <= 0).any():
            std = torch.where(torch.isfinite(std) & (std > 0), std, torch.ones_like(std))
        action, log_prob, entropy = self._sample_action(mean_action, std, stochastic)

        if not return_plan:
            plan = None

        return MLPActorOutput(
            action=action,
            memories=new_memories,
            plan=plan,
            log_prob=log_prob,
            entropy=entropy,
        )

    def evaluate_actions(
        self,
        history: Tensor,
        *,
        state: Tensor,
        raw_state: Optional[Tensor] = None,
        actions: Tensor,
        memories: Optional[MLPMemories] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        warm_start: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, MLPMemories]:
        output = self.forward(
            history,
            state=state,
            raw_state=raw_state,
            memories=memories,
            waypoint_seq=waypoint_seq,
            raw_waypoint_seq=raw_waypoint_seq,
            prev_actions=prev_actions,
            lidar=lidar,
            episode_starts=episode_starts,
            attn_mask=attn_mask,
            warm_start=warm_start,
            return_plan=False,
            stochastic=False,
        )
        mean_action = output.action
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.shape != mean_action.shape:
            raise ValueError(
                f"Actions shape {actions.shape} incompatible with policy mean {mean_action.shape}"
            )
        log_std = self.log_std.clamp(self._log_std_min, self._log_std_max)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean_action)
        action, log_prob, entropy = self._sample_action(mean_action, std, stochastic=False, provided_actions=actions)
        return log_prob, entropy, output.memories

    def _sample_action(
        self,
        mean_action: Tensor,
        std: Tensor,
        stochastic: bool,
        provided_actions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Build distribution (optionally tanh-rescaled) and return action/log_prob/entropy."""
        if self.tanh_rescale_actions:
            if self._action_low is None or self._action_high is None:
                raise RuntimeError("Action bounds not initialised for tanh rescaling.")
            base_dist = Normal(mean_action, std)
            low = self._action_low.to(device=mean_action.device, dtype=mean_action.dtype)
            high = self._action_high.to(device=mean_action.device, dtype=mean_action.dtype)
            scale = (high - low) / 2.0
            loc = (high + low) / 2.0
            transform = dist_transforms.ComposeTransform(
                [dist_transforms.TanhTransform(cache_size=1), dist_transforms.AffineTransform(loc=loc, scale=scale)]
            )
            dist = TransformedDistribution(base_dist, transform)
            if provided_actions is not None:
                action = provided_actions
            elif stochastic:
                action = dist.rsample()
            else:
                action = transform(base_dist.mean)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = base_dist.entropy().sum(dim=-1)
        else:
            dist = Normal(mean_action, std)
            if provided_actions is not None:
                action = provided_actions
            elif stochastic:
                action = dist.rsample()
            else:
                action = mean_action
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

