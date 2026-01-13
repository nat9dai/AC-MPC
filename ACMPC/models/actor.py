"""Actor model built on top of the Transformer-XL backbone."""

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
from .input_embeddings import (
    HistoryEmbeddingConfig,
    HistoryTokenEncoder,
    WaypointSequenceEmbedding,
    compute_episode_ids,
)
from .transformer_xl import TransformerXLBackbone, TransformerXLMemories


_HISTORY_TOKEN_TYPE = 0
_CURRENT_WAYPOINT_TOKEN_TYPE = 1
_FUTURE_WAYPOINT_TOKEN_TYPE = 2


@dataclass
class ActorOutput:
    """Container for actor results."""

    action: Tensor
    memories: TransformerXLMemories
    plan: Optional[Tuple[Tensor, Tensor]] = None
    log_prob: Optional[Tensor] = None
    entropy: Optional[Tensor] = None


class TransformerActor(nn.Module):
    """Transformer-XL actor whose decisions are produced by an economic MPC head."""

    def __init__(
        self,
        *,
        input_dim: int,
        transformer_config: TransformerConfig,
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
        kv_cache_max_tokens: Optional[int] = None,
        tanh_rescale_actions: bool = False,
    ) -> None:
        super().__init__()
        if transformer_config.d_model != mpc_config.latent_dim:
            raise ValueError("mpc_config.latent_dim must match transformer_config.d_model")

        self.input_dim = input_dim
        self.transformer_config = transformer_config
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

        self.backbone = TransformerXLBackbone(
            d_model=transformer_config.d_model,
            n_heads=transformer_config.n_heads,
            d_inner=transformer_config.d_inner,
            n_layers=transformer_config.n_layers,
            dropout=transformer_config.dropout,
            mem_len=transformer_config.mem_len,
            cache_limit=kv_cache_max_tokens,
        )

        history_cfg = HistoryEmbeddingConfig(
            state_dim=self.state_dim,
            d_model=transformer_config.d_model,
            dropout=transformer_config.dropout,
            include_prev_action=self.include_prev_action,
            prev_action_dim=self.prev_action_dim,
            include_lidar=self.include_lidar,
            lidar_dim=self.lidar_dim,
        )
        self.history_encoder = HistoryTokenEncoder(history_cfg)
        if self.waypoint_dim > 0 and self.waypoint_sequence_len > 0:
            self.waypoint_embedding: Optional[WaypointSequenceEmbedding] = WaypointSequenceEmbedding(
                waypoint_dim=self.waypoint_dim,
                d_model=transformer_config.d_model,
                dropout=transformer_config.dropout,
            )
        else:
            self.waypoint_embedding = None
        self.token_type_embeddings = nn.Embedding(3, transformer_config.d_model)

        self.latent_to_ref = nn.Sequential(
            nn.Linear(transformer_config.d_model, mpc_config.state_dim + mpc_config.action_dim),
            nn.Tanh(),
        )

        self.cost_map: Optional[CostMapNetwork]
        if cost_map_config is not None:
            self.cost_map = CostMapNetwork(
                latent_dim=transformer_config.d_model,
                state_dim=mpc_config.state_dim,
                action_dim=mpc_config.action_dim,
                horizon=mpc_config.horizon,
                config=cost_map_config,
                include_state=False, # FIXED: Disable global state injection for SE2 invariance
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

        # Initialize latent_to_ref with proper weights instead of zeros
        linear_layer = self.latent_to_ref[0]
        nn.init.xavier_uniform_(linear_layer.weight, gain=0.1)  # Small gain for stability
        nn.init.zeros_(linear_layer.bias)  # Keep bias at zero initially

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

    def init_memories(self, batch_size: int, device: torch.device) -> TransformerXLMemories:
        """Initialise recurrent memories for the actor backbone."""

        return self.backbone.init_memories(batch_size, device)

    def forward(
        self,
        history: Tensor,
        *,
        state: Tensor,
        raw_state: Optional[Tensor] = None,
        memories: Optional[TransformerXLMemories] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        warm_start: Optional[Tensor] = None,
        return_plan: bool = False,
        stochastic: bool = False,
    ) -> ActorOutput:
        """Run the actor forward pass with optional stochastic sampling."""

        mpc_state = raw_state if raw_state is not None else state
        mpc_waypoint = raw_waypoint_seq if raw_waypoint_seq is not None else waypoint_seq
        if history.dim() == 2:
            history = history.unsqueeze(0)
        if history.dim() != 3:
            raise ValueError("history must have shape [batch, time, state_dim].")

        batch, seq_len, state_dim = history.shape
        if state_dim != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, received {state_dim}."
            )
        if seq_len == 0:
            raise ValueError("history sequence length must be > 0.")

        # Use raw (unnormalised) state for geometric constructions (SE(2) transforms)
        # when available, to avoid mixing normalization with frame transforms.
        geom_state_src = raw_state if raw_state is not None else state
        if geom_state_src.dim() == 1:
            geom_state = geom_state_src.unsqueeze(0)
        elif geom_state_src.dim() == 2:
            geom_state = geom_state_src
        else:
            raise ValueError("State tensor must be rank 1 or 2.")
        if geom_state.shape[0] != batch or geom_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Geometric state shape {geom_state.shape} incompatible with history batch/state_dim "
                f"({batch}, {self.state_dim})."
            )

        encoded_history = self.history_encoder(
            history,
            prev_actions=prev_actions,
            lidar_features=lidar,
        )
        history_type_ids = torch.full(
            (batch, seq_len),
            _HISTORY_TOKEN_TYPE,
            dtype=torch.long,
            device=history.device,
        )
        history_tokens = encoded_history + self.token_type_embeddings(history_type_ids)

        initial_episode_ids = None
        if memories is not None and memories.layers:
            base_layer = memories.layers[0]
            if base_layer.episode_ids is not None and base_layer.episode_ids.shape[1] > 0:
                initial_episode_ids = base_layer.episode_ids[:, -1:].detach()

        history_episode_ids = compute_episode_ids(
            episode_starts,
            batch_size=batch,
            seq_len=seq_len,
            device=history.device,
            initial_ids=initial_episode_ids,
        )

        tokens = [history_tokens]
        full_episode_ids = history_episode_ids

        waypoint_input = waypoint_seq if waypoint_seq is not None else raw_waypoint_seq
        if waypoint_input is not None:
            if self.waypoint_embedding is None:
                raise ValueError("Waypoint inputs provided but waypoint embedding is disabled.")
            if waypoint_input.dim() == 2:
                waypoint_input = waypoint_input.unsqueeze(1)
            if waypoint_input.dim() != 3:
                raise ValueError("waypoint_seq must have shape [batch, W, waypoint_dim].")
            if waypoint_input.shape[-1] != self.waypoint_dim:
                raise ValueError(
                    f"Expected waypoint dimension {self.waypoint_dim}, received {waypoint_input.shape[-1]}."
                )
            if self.waypoint_sequence_len > 0 and waypoint_input.size(1) > self.waypoint_sequence_len:
                raise ValueError(
                    f"waypoint_seq length {waypoint_input.size(1)} exceeds configured waypoint_sequence_len {self.waypoint_sequence_len}."
                )

            geom_waypoint = waypoint_input
            if raw_state is not None and raw_waypoint_seq is not None:
                geom_waypoint = raw_waypoint_seq
                if geom_waypoint.dim() == 2:
                    geom_waypoint = geom_waypoint.unsqueeze(1)
                if geom_waypoint.dim() != 3:
                    raise ValueError("raw_waypoint_seq must have shape [batch, W, waypoint_dim].")
                if geom_waypoint.shape[-1] != self.waypoint_dim:
                    raise ValueError(
                        f"Expected raw waypoint dimension {self.waypoint_dim}, received {geom_waypoint.shape[-1]}."
                    )
                if self.waypoint_sequence_len > 0 and geom_waypoint.size(1) > self.waypoint_sequence_len:
                    raise ValueError(
                        f"raw_waypoint_seq length {geom_waypoint.size(1)} exceeds configured waypoint_sequence_len {self.waypoint_sequence_len}."
                    )

            # RELATIVE COORDINATES TRANSFORMATION (Full SE2 Invariance)
            # Extract current position (x, y) and orientation (theta) from geometric state.
            current_pos = geom_state[:, :2]  # [Batch, 2]

            # Broadcast current_pos to match waypoint sequence length
            # rel_pos_global: [Batch, W, 2]
            rel_pos_global = geom_waypoint - current_pos.unsqueeze(1)

            # If SE2 (state_dim == 3), rotate into body frame
            if self.state_dim == 3:
                theta = geom_state[:, 2]  # [Batch]
                # Prepare rotation components
                # theta needs to be broadcasted to [Batch, W]
                theta = theta.unsqueeze(1).expand(-1, rel_pos_global.size(1))

                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)

                dx = rel_pos_global[..., 0]
                dy = rel_pos_global[..., 1]

                # Body frame rotation:
                # x_local =  dx * cos + dy * sin
                # y_local = -dx * sin + dy * cos
                local_x = dx * cos_t + dy * sin_t
                local_y = -dx * sin_t + dy * cos_t

                rel_waypoint_seq = torch.stack([local_x, local_y], dim=-1)
            else:
                # Fallback for Point Mass / Double Integrator (Translation only)
                rel_waypoint_seq = rel_pos_global

            waypoint_tokens = self.waypoint_embedding(rel_waypoint_seq)
            type_ids = torch.full(
                (batch, rel_waypoint_seq.size(1)),
                _FUTURE_WAYPOINT_TOKEN_TYPE,
                dtype=torch.long,
                device=history.device,
            )
            type_ids[:, 0] = _CURRENT_WAYPOINT_TOKEN_TYPE
            waypoint_tokens = waypoint_tokens + self.token_type_embeddings(type_ids)
            tokens.append(waypoint_tokens)

            last_episode = history_episode_ids[:, -1:].expand(-1, rel_waypoint_seq.size(1))
            full_episode_ids = torch.cat([history_episode_ids, last_episode], dim=1)
        else:
            full_episode_ids = history_episode_ids

        token_seq = torch.cat(tokens, dim=1)
        features, new_memories = self.backbone(
            token_seq,
            episode_ids=full_episode_ids,
            memories=memories,
            attn_mask=attn_mask,
        )
        policy_latent = features[:, -1]

        if mpc_state.dim() == 1:
            state_batch = mpc_state.unsqueeze(0)
        elif mpc_state.dim() == 2:
            state_batch = mpc_state
        else:
            raise ValueError("State tensor must be rank 1 or 2.")
        if state_batch.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, received {state_batch.shape[-1]}"
            )

        if self.cost_map is not None:
            cost_params = self.cost_map(policy_latent)

            # OUTPUT ROTATION: Transform predicted Local Costs to Global Frame for MPC
            # The network thinks in Body Frame (Local), MPC acts in Global Frame.
            if self.state_dim == 3:
                theta = geom_state[:, 2]  # [Batch]
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                zero = torch.zeros_like(theta)
                one = torch.ones_like(theta)

                # Construct Batch Rotation Matrix R [B, 3, 3]
                # R = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
                # This rotates vectors from Body to Global.
                # Note: cost vector q transforms as q_global = R @ q_local
                # cost matrix Q transforms as Q_global = R @ Q_local @ R.T

                # We need to apply this to every step in the horizon.
                # running_C shape: [B, H, nx+nu, nx+nu]
                # running_c shape: [B, H, nx+nu]

                horizon = cost_params.running_C.shape[1]

                # Prepare R expanded for horizon: [B, H, 3, 3]
                R = torch.stack([
                    torch.stack([cos_t, -sin_t, zero], dim=-1),
                    torch.stack([sin_t, cos_t, zero], dim=-1),
                    torch.stack([zero, zero, one], dim=-1)
                ], dim=-2).unsqueeze(1).expand(-1, horizon, -1, -1) # [B, H, 3, 3]

                # 1. Rotate Running Linear Cost (q) - DISABLED
                # FIX: Do NOT rotate linear term c - causes rotating attractor
                # With waypoint-relative state, c points toward tau=0 (waypoint)
                # q_local = cost_params.running_c[..., :3].unsqueeze(-1) # [B, H, 3, 1]
                # q_global = torch.matmul(R, q_local).squeeze(-1)        # [B, H, 3]
                # cost_params.running_c[..., :3] = q_global

                # 2. Rotate Running Quadratic Cost (Q) - State-State
                Q_local = cost_params.running_C[..., :3, :3]       # [B, H, 3, 3]
                Q_global = torch.matmul(R, torch.matmul(Q_local, R.transpose(-1, -2)))
                cost_params.running_C[..., :3, :3] = Q_global

                # 3. Rotate Running Cross-Terms (N) - State-Action
                # N_global = R @ N_local
                # Assumes action_dim is the rest of the matrix columns
                N_local = cost_params.running_C[..., :3, 3:]       # [B, H, 3, nu]
                N_global = torch.matmul(R, N_local)
                cost_params.running_C[..., :3, 3:] = N_global
                cost_params.running_C[..., 3:, :3] = N_global.transpose(-1, -2)

                # 4. Rotate Terminal Costs (No horizon dim)
                # terminal_c: [B, nx+nu]
                R_term = R[:, 0, :, :] # [B, 3, 3]

                # FIX: Do NOT rotate terminal linear term - causes rotating attractor
                # qt_local = cost_params.terminal_c[..., :3].unsqueeze(-1)
                # qt_global = torch.matmul(R_term, qt_local).squeeze(-1)
                # cost_params.terminal_c[..., :3] = qt_global

                Qt_local = cost_params.terminal_C[..., :3, :3]
                Qt_global = torch.matmul(R_term, torch.matmul(Qt_local, R_term.transpose(-1, -2)))
                cost_params.terminal_C[..., :3, :3] = Qt_global

                Nt_local = cost_params.terminal_C[..., :3, 3:]
                Nt_global = torch.matmul(R_term, Nt_local)
                cost_params.terminal_C[..., :3, 3:] = Nt_global
                cost_params.terminal_C[..., 3:, :3] = Nt_global.transpose(-1, -2)

            # TRUE ECONOMIC MPC: State remains absolute [x, y, theta]
            # The CostMap learns c that points toward waypoint based on
            # the Transformer's understanding of relative waypoint position.
            # No state transformation needed - direction comes purely from costs.

            x_ref_tensor: Optional[Tensor] = None
            u_ref_tensor: Optional[Tensor] = None

            if self.use_waypoint_as_ref:
                if mpc_waypoint is None:
                    raise ValueError("use_waypoint_as_ref is True but waypoint_seq is None.")
                wp = mpc_waypoint
                if wp.dim() == 2:
                    wp = wp.unsqueeze(1)
                if wp.dim() != 3:
                    raise ValueError("waypoint_seq must have shape [batch, W, waypoint_dim] or [batch, waypoint_dim].")
                if wp.shape[0] != state_batch.shape[0]:
                    raise ValueError("waypoint_seq batch size must match state batch size when using tracking mode.")
                if self.waypoint_dim <= 0 or wp.shape[-1] != self.waypoint_dim:
                    raise ValueError("waypoint_dim must be configured and match waypoint_seq size when using tracking.")
                if self.waypoint_dim > self.state_dim:
                    raise ValueError("waypoint_dim must not exceed state_dim when using tracking mode.")

                # Use the first waypoint token as the current tracking target for each batch element.
                current_wp = wp[:, 0, :]  # [B, waypoint_dim]

                # Build desired state by copying the current state and replacing the position components
                # (first waypoint_dim entries) with the waypoint coordinates.
                desired_state = state_batch.clone()
                desired_state[:, : self.waypoint_dim] = current_wp

                # SE2 Kinematic Fix:
                # If state_dim is 3 (x, y, theta) and we are tracking a 2D waypoint (x, y),
                # the reference orientation (theta) should point TOWARDS the waypoint,
                # not just copy the current orientation (which causes spinning).
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
                # COORDINATE RELATIVE FIX: Economic MPC with waypoint-relative state
                # When NOT using tracking mode (use_waypoint_as_ref=False), we transform
                # the MPC state to be relative to the first waypoint.
                #
                # Mathematical justification:
                # - Economic MPC minimizes: L(τ) = 0.5 * τ'Cτ + c'τ  where τ = x - x_ref
                # - With x_ref = 0 (default), attractor x* = -C^{-1}c < 0 (wrong direction)
                # - With state_rel = x - waypoint and x_ref = 0:
                #   τ = state_rel - 0 = x - waypoint
                #   Attractor: τ* ≈ 0 means x* ≈ waypoint (correct!)
                #
                # This makes the Economic MPC naturally drive toward the waypoint
                # without needing tracking reference (pure economic mode).
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

        # Guard against NaNs / infs from the MPC head.
        if not torch.isfinite(mean_action).all():
            mean_action = torch.where(
                torch.isfinite(mean_action),
                mean_action,
                torch.zeros_like(mean_action),
            )

        # Ensure a valid, finite log-std before constructing the Normal distribution.
        log_std_param = self.log_std
        if not torch.isfinite(log_std_param).all():
            # Reset corrupted log_std to a safe default.
            with torch.no_grad():
                self.log_std.data.zero_()
        log_std_param = self.log_std

        log_std = log_std_param.clamp(self._log_std_min, self._log_std_max)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean_action)
        # Guard against numerical issues that could still create invalid scales.
        if not torch.isfinite(std).all() or (std <= 0).any():
            std = torch.where(torch.isfinite(std) & (std > 0), std, torch.ones_like(std))
        action, log_prob, entropy = self._sample_action(mean_action, std, stochastic)

        if not return_plan:
            plan = None

        return ActorOutput(
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
        memories: Optional[TransformerXLMemories] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        warm_start: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, TransformerXLMemories]:
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
            # Entropy of transformed distribution is not analytic; fallback to base entropy.
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
