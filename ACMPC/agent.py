"""High-level agent that wires together actor, critic, and MPC head."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn

from .model_config import AgentConfig
from .models import (
    ActorOutput,
    CriticOutput,
    DirectTransformerActor,
    MLPActor,
    MLPCritic,
    MLPCriticOutput,
    MLPMemories,
    MLPActorOutput,
    TransformerActor,
    TransformerCritic,
    TransformerXLMemories,
)


@dataclass
class ActorCriticState:
    """Container for recurrent memories of actor and critic."""

    actor: Optional[TransformerXLMemories | MLPMemories]
    critic: Optional[TransformerXLMemories | MLPMemories]


class ActorCriticAgent(nn.Module):
    """Couples Transformer-XL actor and critic along with MPC policy head."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        dynamics_fn,
        dynamics_jacobian_fn: Optional[Callable[[Tensor, Tensor, float], Tuple[Tensor, Tensor]]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        actor_mpc_config = config.actor.mpc
        if actor_mpc_config.device != config.device:
            actor_mpc_config = replace(actor_mpc_config, device=config.device)

        actor_backbone_type = getattr(config.actor, "backbone_type", "transformer")
        actor_policy = getattr(config.actor, "policy_head", "mpc")

        if actor_backbone_type == "mlp":
            if actor_policy != "mpc":
                raise ValueError("MLP backbone only supports 'mpc' policy_head.")
            self.actor = MLPActor(
                input_dim=config.actor.input_dim,
                mpc_config=actor_mpc_config,
                cost_map_config=config.actor.cost_map,
                dynamics_fn=dynamics_fn,
                dynamics_jacobian_fn=dynamics_jacobian_fn,
                include_prev_action=config.actor.include_prev_action,
                prev_action_dim=config.actor.prev_action_dim,
                include_lidar=config.actor.include_lidar,
                lidar_dim=config.actor.lidar_dim,
                waypoint_dim=config.actor.waypoint_dim,
                waypoint_sequence_len=config.actor.waypoint_sequence_len,
                use_waypoint_as_ref=config.actor.use_waypoint_as_ref,
                tanh_rescale_actions=config.actor.tanh_rescale_actions,
                mlp_hidden_dim=config.actor.mlp.hidden_dim,
                mlp_output_dim=config.actor.mlp.output_dim,
                mlp_num_layers=config.actor.mlp.num_layers,
                mlp_activation=config.actor.mlp.activation,
                mlp_dropout=config.actor.mlp.dropout,
            )
        else:  # transformer
            if actor_policy == "mpc":
                self.actor = TransformerActor(
                    input_dim=config.actor.input_dim,
                    transformer_config=config.actor.transformer,
                    mpc_config=actor_mpc_config,
                    cost_map_config=config.actor.cost_map,
                    dynamics_fn=dynamics_fn,
                    dynamics_jacobian_fn=dynamics_jacobian_fn,
                    include_prev_action=config.actor.include_prev_action,
                    prev_action_dim=config.actor.prev_action_dim,
                    include_lidar=config.actor.include_lidar,
                    lidar_dim=config.actor.lidar_dim,
                    waypoint_dim=config.actor.waypoint_dim,
                    waypoint_sequence_len=config.actor.waypoint_sequence_len,
                    use_waypoint_as_ref=config.actor.use_waypoint_as_ref,
                    kv_cache_max_tokens=config.actor.kv_cache_max_tokens,
                    tanh_rescale_actions=config.actor.tanh_rescale_actions,
                )
            elif actor_policy == "direct":
                self.actor = DirectTransformerActor(
                    input_dim=config.actor.input_dim,
                    transformer_config=config.actor.transformer,
                    mpc_config=actor_mpc_config,
                    include_prev_action=config.actor.include_prev_action,
                    prev_action_dim=config.actor.prev_action_dim,
                    include_lidar=config.actor.include_lidar,
                    lidar_dim=config.actor.lidar_dim,
                    waypoint_dim=config.actor.waypoint_dim,
                    waypoint_sequence_len=config.actor.waypoint_sequence_len,
                    use_waypoint_as_ref=config.actor.use_waypoint_as_ref,
                    kv_cache_max_tokens=config.actor.kv_cache_max_tokens,
                    tanh_rescale_actions=config.actor.tanh_rescale_actions,
                )
            else:
                raise ValueError(f"Unsupported actor policy_head='{actor_policy}'.")

        critic_backbone_type = getattr(config.critic, "backbone_type", "transformer")
        if critic_backbone_type == "mlp":
            self.critic = MLPCritic(
                input_dim=config.critic.input_dim,
                hidden_dim=config.critic.mlp.hidden_dim,
                output_dim=1,
                num_layers=config.critic.mlp.num_layers,
                activation=config.critic.mlp.activation,
                dropout=config.critic.mlp.dropout,
                include_prev_action=config.critic.include_prev_action,
                prev_action_dim=config.critic.prev_action_dim,
                include_lidar=config.critic.include_lidar,
                lidar_dim=config.critic.lidar_dim,
                waypoint_dim=config.critic.waypoint_dim,
                waypoint_sequence_len=config.critic.waypoint_sequence_len,
            )
        else:  # transformer
            self.critic = TransformerCritic(
                input_dim=config.critic.input_dim,
                transformer_config=config.critic.transformer,
                hidden_dim=config.critic.value_head_hidden,
                include_prev_action=config.critic.include_prev_action,
                prev_action_dim=config.critic.prev_action_dim,
                include_lidar=config.critic.include_lidar,
                lidar_dim=config.critic.lidar_dim,
                waypoint_dim=config.critic.waypoint_dim,
                waypoint_sequence_len=config.critic.waypoint_sequence_len,
                kv_cache_max_tokens=config.critic.kv_cache_max_tokens,
            )

        self.to(self.device)

    def init_state(self, batch_size: int) -> ActorCriticState:
        """Initialise recurrent memories for both actor and critic."""

        actor_mem = self.actor.init_memories(batch_size, self.device)
        critic_mem = self.critic.init_memories(batch_size, self.device)
        return ActorCriticState(actor=actor_mem, critic=critic_mem)

    def act(
        self,
        history: Tensor,
        *,
        state: Tensor,
        raw_state: Optional[Tensor] = None,
        memories: Optional[ActorCriticState] = None,
        warm_start: Optional[Tensor] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        return_plan: bool = False,
        stochastic: bool = False,
        return_log_prob: bool = False,
    ) -> Tuple[Tensor, ActorCriticState, Optional[Tuple[Tensor, Tensor]]] | Tuple[Tensor, Tensor, ActorCriticState, Optional[Tuple[Tensor, Tensor]]]:
        actor_mem = None if memories is None else memories.actor
        actor_output = self.actor(
            history.to(self.device),
            state=state.to(self.device),
            raw_state=raw_state.to(self.device) if raw_state is not None else None,
            memories=actor_mem,
            waypoint_seq=waypoint_seq.to(self.device) if waypoint_seq is not None else None,
            raw_waypoint_seq=raw_waypoint_seq.to(self.device) if raw_waypoint_seq is not None else None,
            prev_actions=prev_actions.to(self.device) if prev_actions is not None else None,
            lidar=lidar.to(self.device) if lidar is not None else None,
            episode_starts=episode_starts.to(self.device) if episode_starts is not None else None,
            attn_mask=attn_mask.to(self.device) if attn_mask is not None else None,
            warm_start=warm_start.to(self.device) if warm_start is not None else None,
            return_plan=return_plan,
            stochastic=stochastic,
        )

        new_state = ActorCriticState(
            actor=actor_output.memories,
            critic=None if memories is None else memories.critic,
        )
        if return_log_prob:
            if actor_output.log_prob is None:
                raise RuntimeError("Actor did not provide log probability.")
            return actor_output.action, actor_output.log_prob, new_state, actor_output.plan
        return actor_output.action, new_state, actor_output.plan

    def value(
        self,
        history: Tensor,
        *,
        memories: Optional[TransformerXLMemories | MLPMemories] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,
        raw_state: Optional[Tensor] = None,
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> CriticOutput | MLPCriticOutput:
        return self.critic(
            history.to(self.device),
            memories=memories,
            waypoint_seq=waypoint_seq.to(self.device) if waypoint_seq is not None else None,
            raw_waypoint_seq=raw_waypoint_seq.to(self.device) if raw_waypoint_seq is not None else None,
            raw_state=raw_state.to(self.device) if raw_state is not None else None,
            prev_actions=prev_actions.to(self.device) if prev_actions is not None else None,
            lidar=lidar.to(self.device) if lidar is not None else None,
            episode_starts=episode_starts.to(self.device) if episode_starts is not None else None,
            attn_mask=attn_mask.to(self.device) if attn_mask is not None else None,
        )

    def evaluate(
        self,
        history: Tensor,
        *,
        state: Tensor,
        raw_state: Optional[Tensor] = None,
        memories: Optional[ActorCriticState] = None,
        warm_start: Optional[Tensor] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        return_plan: bool = False,
    ) -> Tuple[Tensor, Tensor, ActorCriticState, Optional[Tuple[Tensor, Tensor]]]:
        actor_mem = None if memories is None else memories.actor
        critic_mem = None if memories is None else memories.critic

        actor_output = self.actor(
            history.to(self.device),
            state=state.to(self.device),
            raw_state=raw_state.to(self.device) if raw_state is not None else None,
            memories=actor_mem,
            waypoint_seq=waypoint_seq.to(self.device) if waypoint_seq is not None else None,
            raw_waypoint_seq=raw_waypoint_seq.to(self.device) if raw_waypoint_seq is not None else None,
            prev_actions=prev_actions.to(self.device) if prev_actions is not None else None,
            lidar=lidar.to(self.device) if lidar is not None else None,
            episode_starts=episode_starts.to(self.device) if episode_starts is not None else None,
            attn_mask=attn_mask.to(self.device) if attn_mask is not None else None,
            warm_start=warm_start.to(self.device) if warm_start is not None else None,
            return_plan=return_plan,
            stochastic=False,
        )
        critic_output = self.critic(
            history.to(self.device),
            memories=critic_mem,
            waypoint_seq=waypoint_seq.to(self.device) if waypoint_seq is not None else None,
            raw_waypoint_seq=raw_waypoint_seq.to(self.device) if raw_waypoint_seq is not None else None,
            raw_state=raw_state.to(self.device) if raw_state is not None else None,
            prev_actions=prev_actions.to(self.device) if prev_actions is not None else None,
            lidar=lidar.to(self.device) if lidar is not None else None,
            episode_starts=episode_starts.to(self.device) if episode_starts is not None else None,
            attn_mask=attn_mask.to(self.device) if attn_mask is not None else None,
        )
        new_state = ActorCriticState(
            actor=actor_output.memories,
            critic=critic_output.memories,
        )
        return actor_output.action, critic_output.value, new_state, actor_output.plan

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        state_dict = dict(state_dict)
        latent_keys = [
            "actor.latent_to_ref.0.weight",
            "actor.latent_to_ref.0.bias",
        ]
        named_params = dict(self.named_parameters())
        named_buffers = dict(self.named_buffers())
        for key in latent_keys:
            if key not in state_dict:
                tensor = named_params.get(key)
                if tensor is None:
                    tensor = named_buffers.get(key)
                if tensor is not None:
                    state_dict[key] = tensor.detach().clone()
        return super().load_state_dict(state_dict, strict=strict)

    def evaluate_actions(
        self,
        history: Tensor,
        *,
        state: Tensor,
        raw_state: Optional[Tensor] = None,
        actions: Tensor,
        memories: Optional[ActorCriticState] = None,
        warm_start: Optional[Tensor] = None,
        waypoint_seq: Optional[Tensor] = None,
        raw_waypoint_seq: Optional[Tensor] = None,
        prev_actions: Optional[Tensor] = None,
        lidar: Optional[Tensor] = None,
        episode_starts: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        actor_mem = None if memories is None else memories.actor
        critic_mem = None if memories is None else memories.critic

        log_prob, entropy, new_actor_mem = self.actor.evaluate_actions(
            history.to(self.device),
            state=state.to(self.device),
            raw_state=raw_state.to(self.device) if raw_state is not None else None,
            actions=actions.to(self.device),
            memories=actor_mem,
            waypoint_seq=waypoint_seq.to(self.device) if waypoint_seq is not None else None,
            raw_waypoint_seq=raw_waypoint_seq.to(self.device) if raw_waypoint_seq is not None else None,
            prev_actions=prev_actions.to(self.device) if prev_actions is not None else None,
            lidar=lidar.to(self.device) if lidar is not None else None,
            episode_starts=episode_starts.to(self.device) if episode_starts is not None else None,
            attn_mask=attn_mask.to(self.device) if attn_mask is not None else None,
            warm_start=warm_start.to(self.device) if warm_start is not None else None,
        )
        critic_output = self.critic(
            history.to(self.device),
            memories=critic_mem,
            waypoint_seq=waypoint_seq.to(self.device) if waypoint_seq is not None else None,
            raw_waypoint_seq=raw_waypoint_seq.to(self.device) if raw_waypoint_seq is not None else None,
            raw_state=raw_state.to(self.device) if raw_state is not None else None,
            prev_actions=prev_actions.to(self.device) if prev_actions is not None else None,
            lidar=lidar.to(self.device) if lidar is not None else None,
            episode_starts=episode_starts.to(self.device) if episode_starts is not None else None,
            attn_mask=attn_mask.to(self.device) if attn_mask is not None else None,
        )
        return log_prob, entropy, critic_output.value
