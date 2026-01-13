"""Critic model built on top of the shared Transformer-XL backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ..model_config import TransformerConfig
from .input_embeddings import (
    HistoryEmbeddingConfig,
    HistoryTokenEncoder,
    WaypointSequenceEmbedding,
    compute_episode_ids,
)
from .transformer_xl import TransformerXLBackbone, TransformerXLMemories


@dataclass
class CriticOutput:
    """Container for critic outputs."""

    value: Tensor
    memories: TransformerXLMemories


class TransformerCritic(nn.Module):
    """Transformer-XL critic producing scalar value estimates."""

    def __init__(
        self,
        *,
        input_dim: int,
        transformer_config: TransformerConfig,
        hidden_dim: int = 128,
        include_prev_action: bool = False,
        prev_action_dim: int = 0,
        include_lidar: bool = False,
        lidar_dim: int = 0,
        waypoint_dim: int = 0,
        waypoint_sequence_len: int = 0,
        kv_cache_max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.state_dim = input_dim
        self.include_prev_action = include_prev_action
        self.include_lidar = include_lidar
        self.prev_action_dim = prev_action_dim
        self.lidar_dim = lidar_dim
        self.waypoint_dim = waypoint_dim
        self.waypoint_sequence_len = waypoint_sequence_len

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

        self.value_head = nn.Sequential(
            nn.Linear(transformer_config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def init_memories(self, batch_size: int, device: torch.device) -> TransformerXLMemories:
        """Initialise recurrent memories for the critic backbone."""

        return self.backbone.init_memories(batch_size, device)

    def forward(
        self,
        history: Tensor,
        *,
        memories: Optional[TransformerXLMemories] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,
        raw_state: Optional[Tensor] = None,
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> CriticOutput:
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

        encoded_history = self.history_encoder(
            history,
            prev_actions=prev_actions,
            lidar_features=lidar,
        )
        history_type_ids = torch.zeros(
            batch,
            seq_len,
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

            # RELATIVE COORDINATES TRANSFORMATION (Full SE2 Invariance) - Critic Sync
            # Must match Actor logic perfectly.
            # FIX: Use raw_state for geometric transforms (matches actor behavior)
            geom_state = raw_state if raw_state is not None else history[:, -1, :]
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

            current_pos = geom_state[:, :2]
            rel_pos_global = geom_waypoint - current_pos.unsqueeze(1)

            if self.state_dim == 3: # SE2 Heuristic
                theta = geom_state[:, 2]
                theta = theta.unsqueeze(1).expand(-1, rel_pos_global.size(1))
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                dx = rel_pos_global[..., 0]
                dy = rel_pos_global[..., 1]
                local_x = dx * cos_t + dy * sin_t
                local_y = -dx * sin_t + dy * cos_t
                rel_waypoint_seq = torch.stack([local_x, local_y], dim=-1)
            else:
                rel_waypoint_seq = rel_pos_global

            waypoint_tokens = self.waypoint_embedding(rel_waypoint_seq)
            type_ids = torch.full(
                (batch, rel_waypoint_seq.size(1)),
                2,
                dtype=torch.long,
                device=history.device,
            )
            type_ids[:, 0] = 1
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
        pooled = features[:, -1]
        value = self.value_head(pooled).squeeze(-1)

        return CriticOutput(value=value, memories=new_memories)
