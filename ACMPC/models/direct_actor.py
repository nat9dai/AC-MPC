"""Transformer-XL actor that outputs actions directly without an MPC head."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Normal, TransformedDistribution, transforms as dist_transforms

from ..model_config import TransformerConfig
from ..mpc import EconomicMPCConfig
from .actor import ActorOutput
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


class DirectTransformerActor(nn.Module):
    """Transformer-XL actor that samples actions directly from a Gaussian policy."""

    def __init__(
        self,
        *,
        input_dim: int,
        transformer_config: TransformerConfig,
        mpc_config: EconomicMPCConfig,
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

        self.latent_to_action = nn.Linear(transformer_config.d_model, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        self._log_std_min = -3.0
        self._log_std_max = 2.0

        if self.tanh_rescale_actions:
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
        raw_state: Optional[Tensor] = None,  # unused but kept for API parity
        memories: Optional[TransformerXLMemories] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,  # unused but kept for API parity
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        warm_start: Optional[Tensor] = None,  # unused placeholder for interface compatibility
        return_plan: bool = False,
        stochastic: bool = False,
    ) -> ActorOutput:
        """Run the actor forward pass and sample an action."""

        if history.dim() == 2:
            history = history.unsqueeze(0)
        if history.dim() != 3:
            raise ValueError("history must have shape [batch, time, state_dim].")

        batch, seq_len, state_dim = history.shape
        if state_dim != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, received {state_dim}.")
        if seq_len == 0:
            raise ValueError("history sequence length must be > 0.")

        if state.dim() == 1:
            state_batch = state.unsqueeze(0)
        elif state.dim() == 2:
            state_batch = state
        else:
            raise ValueError("state must have shape [batch, state_dim] or [state_dim].")
        if state_batch.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, received {state_batch.shape[-1]}.")

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

        if waypoint_seq is not None:
            if self.waypoint_embedding is None:
                raise ValueError("Waypoint inputs provided but waypoint embedding is disabled.")
            if waypoint_seq.dim() == 2:
                waypoint_seq = waypoint_seq.unsqueeze(1)
            if waypoint_seq.dim() != 3:
                raise ValueError("waypoint_seq must have shape [batch, W, waypoint_dim].")
            if waypoint_seq.shape[-1] != self.waypoint_dim:
                raise ValueError(
                    f"Expected waypoint dimension {self.waypoint_dim}, received {waypoint_seq.shape[-1]}."
                )
            if self.waypoint_sequence_len > 0 and waypoint_seq.size(1) > self.waypoint_sequence_len:
                raise ValueError(
                    f"waypoint_seq length {waypoint_seq.size(1)} exceeds configured waypoint_sequence_len {self.waypoint_sequence_len}."
                )
            waypoint_tokens = self.waypoint_embedding(waypoint_seq)
            type_ids = torch.full(
                (batch, waypoint_seq.size(1)),
                _FUTURE_WAYPOINT_TOKEN_TYPE,
                dtype=torch.long,
                device=history.device,
            )
            type_ids[:, 0] = _CURRENT_WAYPOINT_TOKEN_TYPE
            waypoint_tokens = waypoint_tokens + self.token_type_embeddings(type_ids)
            tokens.append(waypoint_tokens)

            last_episode = history_episode_ids[:, -1:].expand(-1, waypoint_seq.size(1))
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

        mean_action = self.latent_to_action(policy_latent)
        log_std = self.log_std.clamp(self._log_std_min, self._log_std_max)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean_action)
        action, log_prob, entropy = self._sample_action(mean_action, std, stochastic)

        return ActorOutput(
            action=action,
            memories=new_memories,
            plan=None,
            log_prob=log_prob,
            entropy=entropy,
        )

    def evaluate_actions(
        self,
        history: Tensor,
        *,
        state: Tensor,
        raw_state: Optional[Tensor] = None,  # unused but kept for API parity
        actions: Tensor,
        memories: Optional[TransformerXLMemories] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,  # unused but kept for API parity
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        warm_start: Optional[Tensor] = None,  # unused placeholder
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
            raise ValueError(f"Actions shape {actions.shape} incompatible with policy mean {mean_action.shape}")
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
