"""Transformer-XL style backbone used by actor and critic networks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


def _shape_projection(tensor: Tensor, n_heads: int) -> Tensor:
    """Reshape a projected tensor into ``[batch, heads, time, d_head]``."""

    batch, time, channels = tensor.shape
    d_head = channels // n_heads
    return tensor.view(batch, time, n_heads, d_head).transpose(1, 2)


def _rel_shift(tensor: Tensor) -> Tensor:
    """
    Performs the relative shift used in Transformer-XL attention.
    Uses the standard pad→view→slice→view sequence from the paper.
    """
    batch, n_head, time_q, time_k = tensor.shape

    zero_pad = tensor.new_zeros(batch, n_head, time_q, 1)
    padded = torch.cat([zero_pad, tensor], dim=-1)  # [B, H, T_q, T_k+1]
    padded = padded.view(batch, n_head, time_k + 1, time_q)
    shifted = padded[:, :, 1:, :].view(batch, n_head, time_q, time_k)
    return shifted


def _build_episodic_attention_mask(
    current_episode_ids: Optional[Tensor],
    memory_episode_ids: Optional[Tensor],
    *,
    seq_len: int,
) -> Optional[Tensor]:
    """Return a causal attention mask that prevents cross-episode attention."""

    if current_episode_ids is None:
        return None
    if current_episode_ids.dim() != 2:
        raise ValueError("episode ids must have shape [batch, time].")

    batch, time = current_episode_ids.shape
    if time != seq_len:
        raise ValueError(
            f"episode ids length {time} does not match sequence length {seq_len}."
        )

    current_episode_ids = current_episode_ids.to(dtype=torch.long)
    device = current_episode_ids.device
    key_ids: List[Tensor] = []
    mem_len = 0
    if memory_episode_ids is not None:
        if memory_episode_ids.dim() != 2 or memory_episode_ids.size(0) != batch:
            raise ValueError("memory episode ids must have shape [batch, mem_len].")
        mem_len = memory_episode_ids.size(1)
        if mem_len > 0:
            key_ids.append(memory_episode_ids.to(dtype=torch.long, device=device))
    key_ids.append(current_episode_ids)
    key_cat = torch.cat(key_ids, dim=1)

    same_episode = key_cat.unsqueeze(1) == current_episode_ids.unsqueeze(-1)

    if mem_len > 0:
        memory_mask = same_episode[:, :, :mem_len]
    else:
        memory_mask = same_episode.new_zeros(batch, time, 0, dtype=torch.bool)

    current_mask = same_episode[:, :, mem_len:]
    causal = torch.tril(torch.ones(time, time, dtype=torch.bool, device=device))
    current_mask = current_mask & causal.unsqueeze(0)

    if mem_len > 0:
        full_mask = torch.cat([memory_mask, current_mask], dim=-1)
    else:
        full_mask = current_mask
    return full_mask.unsqueeze(1)


class RelativePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with descending indices for TXL."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("Transformer-XL expects an even model dimension.")
        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.d_model = d_model

    def forward(self, length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if length <= 0:
            raise ValueError("Sequence length must be positive.")
        pos_seq = torch.arange(length - 1, -1, -1.0, device=device, dtype=self.inv_freq.dtype)
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb.to(dtype=dtype)


class PositionwiseFeedForward(nn.Module):
    """Standard two-layer feed-forward module."""

    def __init__(self, d_model: int, d_inner: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RelMultiHeadSelfAttention(nn.Module):
    """Relative multi-head self-attention with Transformer-XL style memory."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.r_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.drop_attn = nn.Dropout(dropout)
        self.drop_proj = nn.Dropout(dropout)

        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_head))

    def forward(
        self,
        x: Tensor,
        *,
        memory: Optional[Tensor],
        pos_emb: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch, time, _ = x.shape
        if memory is None:
            mem = x.new_zeros(batch, 0, x.size(-1))
        else:
            mem = memory

        cat = torch.cat([mem, x], dim=1)
        k = _shape_projection(self.k_proj(cat), self.n_heads)
        v = _shape_projection(self.v_proj(cat), self.n_heads)
        q = _shape_projection(self.q_proj(x), self.n_heads)

        r = self.r_proj(pos_emb)  # [L, d_model]
        r = r.view(-1, self.n_heads, self.d_head).transpose(0, 1)  # [heads, L, d_head]

        q_with_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)

        content_score = torch.matmul(q_with_u, k.transpose(-1, -2))  # [B, heads, T, L]
        rel_score = torch.einsum("bnid,njd->bnij", q_with_v, r)
        rel_score = _rel_shift(rel_score)

        scores = (content_score + rel_score) * self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop_attn(attn)

        output = torch.matmul(attn, v)  # [B, heads, T, d_head]
        output = output.transpose(1, 2).contiguous().view(batch, time, -1)
        output = self.drop_proj(self.out_proj(output))
        return output


class TransformerXLBlock(nn.Module):
    """Single Transformer-XL block with memory support."""

    def __init__(self, d_model: int, n_heads: int, d_inner: int, dropout: float, mem_len: int) -> None:
        super().__init__()
        self.mem_len = mem_len
        self.attn = RelMultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_inner, dropout)
        self.drop = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.positional_encoding = RelativePositionalEncoding(d_model)

    def forward(
        self,
        x: Tensor,
        *,
        memory: LayerMemory,
        episode_ids: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, LayerMemory]:
        residual = x
        norm_x = self.norm_attn(x)
        mem_states = memory.states
        total_len = norm_x.size(1) + (0 if mem_states is None else mem_states.size(1))
        pos_emb = self.positional_encoding(
            total_len,
            device=norm_x.device,
            dtype=norm_x.dtype,
        )
        attn_out = self.attn(norm_x, memory=mem_states, pos_emb=pos_emb, attn_mask=attn_mask)
        x = residual + self.drop(attn_out)

        residual_ff = x
        x = self.norm_ff(x)
        x = residual_ff + self.drop(self.ff(x))

        new_memory = self._update_memory(memory, residual, episode_ids)
        return x, new_memory

    def _update_memory(
        self,
        memory: LayerMemory,
        input_states: Tensor,
        episode_ids: Optional[Tensor],
    ) -> LayerMemory:
        if self.mem_len == 0:
            return LayerMemory(states=None, episode_ids=None)

        input_states = input_states.detach()
        mem_states = memory.states
        mem_episode_ids = memory.episode_ids

        if episode_ids is not None:
            if episode_ids.dim() != 2 or episode_ids.size(1) != input_states.size(1):
                raise ValueError("episode ids must match the temporal dimension of the input.")
            input_episode_ids = episode_ids.detach().to(dtype=torch.long)
        else:
            input_episode_ids = None

        if mem_states is None or mem_states.numel() == 0:
            combined_states = input_states
            combined_episode_ids = input_episode_ids
        else:
            combined_states = torch.cat([mem_states, input_states], dim=1)
            if input_episode_ids is not None:
                if mem_episode_ids is None:
                    raise ValueError("memory episode ids missing despite episodic inputs being provided.")
                combined_episode_ids = torch.cat([mem_episode_ids, input_episode_ids], dim=1)
            else:
                combined_episode_ids = mem_episode_ids

        if combined_states.size(1) > self.mem_len:
            combined_states = combined_states[:, -self.mem_len :]
            if combined_episode_ids is not None:
                combined_episode_ids = combined_episode_ids[:, -self.mem_len :]

        return LayerMemory(
            states=combined_states.detach(),
            episode_ids=None if combined_episode_ids is None else combined_episode_ids.detach(),
        )


@dataclass
class LayerMemory:
    """Caches the hidden states and metadata for a single TXL layer."""

    states: Optional[Tensor]
    episode_ids: Optional[Tensor]

    def detach(self) -> "LayerMemory":
        return LayerMemory(
            None if self.states is None else self.states.detach(),
            None if self.episode_ids is None else self.episode_ids.detach(),
        )


@dataclass
class TransformerXLMemories:
    """Container for cached Transformer-XL states."""

    layers: Tuple[LayerMemory, ...]

    def detach(self) -> "TransformerXLMemories":
        return TransformerXLMemories(tuple(layer.detach() for layer in self.layers))


class TransformerXLBackbone(nn.Module):
    """Stack of Transformer-XL blocks with shared configuration."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_inner: int,
        n_layers: int,
        dropout: float,
        mem_len: int,
        cache_limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mem_len = max(0, mem_len)
        if cache_limit is not None:
            self.mem_len = min(self.mem_len, cache_limit)
        effective_mem_len = self.mem_len
        self.layers = nn.ModuleList(
            TransformerXLBlock(d_model, n_heads, d_inner, dropout, effective_mem_len)
            for _ in range(n_layers)
        )

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def init_memories(self, batch_size: int, device: torch.device) -> TransformerXLMemories:
        memories: List[LayerMemory] = []
        for _ in range(self.n_layers):
            if self.mem_len == 0:
                memories.append(LayerMemory(states=None, episode_ids=None))
            else:
                memories.append(LayerMemory(states=None, episode_ids=None))
        return TransformerXLMemories(tuple(memories))

    def forward(
        self,
        x: Tensor,
        *,
        episode_ids: Optional[Tensor] = None,
        memories: Optional[TransformerXLMemories] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, TransformerXLMemories]:
        batch, time, _ = x.shape
        if episode_ids is not None:
            if episode_ids.dim() != 2 or episode_ids.shape != (batch, time):
                raise ValueError("episode ids must have shape [batch, time].")

        if memories is None:
            mems: Sequence[LayerMemory] = tuple(LayerMemory(states=None, episode_ids=None) for _ in range(self.n_layers))
        else:
            mems = memories.layers
            if len(mems) != self.n_layers:
                raise ValueError("Incorrect number of memory tensors supplied.")

        new_mems: List[LayerMemory] = []
        h = x
        for block, mem in zip(self.layers, mems):
            episodic_mask = _build_episodic_attention_mask(episode_ids, mem.episode_ids, seq_len=time)
            mask = episodic_mask
            if attn_mask is not None:
                attn_bool = attn_mask.bool()
                mask = attn_bool if mask is None else (mask & attn_bool)
            h, new_mem = block(h, memory=mem, episode_ids=episode_ids, attn_mask=mask)
            new_mems.append(new_mem)

        return h, TransformerXLMemories(tuple(new_mems))
