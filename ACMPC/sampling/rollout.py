"""Rollout collection utilities for the ACMPC training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..agent import ActorCriticState


@dataclass
class EnvBatch:
    """Container produced by environment managers for vectorised steps.

    All tensors are shaped ``[num_envs, *]`` and expected to live on CPU.
    Observations, states, and waypoints must already be expressed in the
    absolute/world frame – the collector never performs relative transforms.
    """

    observation: Tensor
    state: Tensor
    waypoint_seq: Tensor
    reward: Tensor
    done: Tensor
    episode_start: Tensor
    info: Sequence[Dict]

    def to(self, device: torch.device) -> EnvBatch:
        return EnvBatch(
            observation=self.observation.to(device),
            state=self.state.to(device),
            waypoint_seq=self.waypoint_seq.to(device),
            reward=self.reward.to(device),
            done=self.done.to(device),
            episode_start=self.episode_start.to(device),
            info=self.info,
        )


@dataclass
class RolloutBatch:
    """Batched rollout data aligned per environment and timestep.

    The collector stores every quantity required downstream by PPO/GAE as well
    as additional metadata (token offsets, episode ids, MPC plans).  All tensors
    follow the ``[env, time, ...]`` convention unless noted otherwise.
    """

    history: Tensor  # [E, T, history_window, obs_dim]
    observation: Tensor  # [E, T, obs_dim]
    state: Tensor  # [E, T, state_dim]
    next_history: Tensor  # [E, history_window, obs_dim]
    next_state: Tensor  # [E, state_dim]
    action: Tensor  # [E, T, action_dim]
    log_prob: Tensor  # [E, T]
    reward: Tensor  # [E, T]
    done: Tensor  # [E, T]
    episode_start: Tensor  # [E, T]
    next_episode_start: Tensor  # [E]
    episode_id: Tensor  # [E, T]
    token_offset: Tensor  # [E, T]
    trial_length: Tensor  # [E]
    waypoint_seq: Tensor  # [E, T, W, waypoint_dim]
    next_waypoint_seq: Tensor  # [E, W, waypoint_dim]
    plan_states: Tensor  # [E, T, H+1, state_dim]
    plan_actions: Tensor  # [E, T, H, action_dim]
    warm_start_source: Sequence[Sequence[str]]
    info: Sequence[Sequence[Dict]]
    mask: Tensor  # [E, T], 1 for valid timesteps, 0 for padding
    old_value: Tensor  # [E, T]
    plan_rewards: Optional[Tensor] = None  # [E, T, H]
    plan_observations: Optional[Tensor] = None  # [E, T, H+1, obs_dim]
    lidar: Optional[Tensor] = None  # [E, T, lidar_dim]
    next_lidar: Optional[Tensor] = None  # [E, lidar_dim]

    def to(self, device: torch.device) -> RolloutBatch:
        return RolloutBatch(
            history=self.history.to(device),
            observation=self.observation.to(device),
            state=self.state.to(device),
            next_history=self.next_history.to(device),
            next_state=self.next_state.to(device),
            action=self.action.to(device),
            log_prob=self.log_prob.to(device),
            reward=self.reward.to(device),
            done=self.done.to(device),
            episode_start=self.episode_start.to(device),
            next_episode_start=self.next_episode_start.to(device),
            episode_id=self.episode_id.to(device),
            token_offset=self.token_offset.to(device),
            trial_length=self.trial_length.to(device),
            waypoint_seq=self.waypoint_seq.to(device),
            next_waypoint_seq=self.next_waypoint_seq.to(device),
            plan_states=self.plan_states.to(device),
            plan_actions=self.plan_actions.to(device),
            warm_start_source=self.warm_start_source,
            info=self.info,
            mask=self.mask.to(device),
            old_value=self.old_value.to(device),
            plan_rewards=None if self.plan_rewards is None else self.plan_rewards.to(device),
            plan_observations=(
                None if self.plan_observations is None else self.plan_observations.to(device)
            ),
            lidar=None if self.lidar is None else self.lidar.to(device),
            next_lidar=None if self.next_lidar is None else self.next_lidar.to(device),
        )

    @property
    def num_envs(self) -> int:
        return int(self.history.size(0))

    @property
    def horizon(self) -> int:
        return int(self.history.size(1))

    def flatten_time(self) -> Dict[str, Tensor]:
        """Flatten env/time dimensions for per-sample processing."""

        E, T, _, obs_dim = self.history.shape
        flattened: Dict[str, Tensor] = {
            "history": self.history.view(E * T, *self.history.shape[2:]),
            "observation": self.observation.view(E * T, obs_dim),
            "state": self.state.view(E * T, self.state.shape[-1]),
            "action": self.action.view(E * T, self.action.shape[-1]),
            "log_prob": self.log_prob.reshape(E * T),
            "reward": self.reward.reshape(E * T),
            "done": self.done.reshape(E * T),
            "episode_start": self.episode_start.reshape(E * T),
            "episode_id": self.episode_id.reshape(E * T),
            "token_offset": self.token_offset.reshape(E * T),
            "mask": self.mask.reshape(E * T),
        }
        return flattened


class RolloutCollector:
    """Collects fixed-length batches of rollouts in absolute coordinates."""

    def __init__(
        self,
        *,
        agent,
        env_manager,
        history_window: int,
        horizon: int,
        device: torch.device,
        collect_plan_rewards: bool = False,
        collect_plan_observations: bool = False,
        observation_normalizer=None,
        waypoint_normalizer=None,
        lidar_normalizer=None,
    ) -> None:
        self.agent = agent
        self.env = env_manager
        self.history_window = int(history_window)
        self.horizon = int(horizon)
        self.device = device
        self.collect_plan_rewards = bool(collect_plan_rewards)
        self.collect_plan_observations = bool(collect_plan_observations)
        self.observation_normalizer = observation_normalizer
        self.waypoint_normalizer = waypoint_normalizer
        self.lidar_normalizer = lidar_normalizer

        initial = self.env.reset()
        self.num_envs = int(initial.observation.size(0))
        self.obs_dim = int(initial.observation.size(-1))
        self.state_dim = int(initial.state.size(-1))
        if initial.waypoint_seq.ndim != 3:
            raise ValueError("waypoint_seq must have shape [env, seq, dim]")
        self.waypoint_len = int(initial.waypoint_seq.size(1))
        self.waypoint_dim = int(initial.waypoint_seq.size(2))
        self.action_dim = getattr(self.env, "action_dim", None)
        if self.action_dim is None:
            raise ValueError("Environment manager must expose 'action_dim'.")

        self.action_dim = getattr(self.env, "action_dim", None)
        if self.action_dim is None:
            raise ValueError("Environment manager must expose an 'action_dim' attribute.")

        self._history = torch.zeros(
            self.num_envs,
            self.history_window,
            self.state_dim,
            dtype=initial.state.dtype,
        )
        # Initialise history with the absolute state rather than raw observation.
        self._history[:, -1] = initial.state
        self._state = initial.state
        self._waypoint_seq = initial.waypoint_seq
        self._episode_flags = initial.episode_start.bool()
        self._episode_counters = torch.zeros(self.num_envs, dtype=torch.long)
        self._token_offsets = torch.zeros(self.num_envs, dtype=torch.long)
        self._warm_start_cache: Optional[Tensor] = None
        if hasattr(self.agent, "init_state"):
            self._actor_state = self.agent.init_state(self.num_envs)
        else:
            self._actor_state = None

        # Optional lidar features (fed as separate inputs to the agent).
        self._lidar_enabled: bool = False
        self._lidar_dim: int = 0
        self._lidar: Optional[Tensor] = None
        self._initialise_lidar(initial)

        if self.collect_plan_rewards and not getattr(self.env, "supports_reward_prediction", lambda: False)():
            raise ValueError(
                "collect_plan_rewards=True but the environment manager does not expose reward prediction support."
            )
        if self.collect_plan_observations and not getattr(self.env, "supports_observation_prediction", lambda: False)():
            raise ValueError(
                "collect_plan_observations=True but the environment manager does not expose observation prediction support."
            )

    def collect(self, horizon: Optional[int] = None) -> RolloutBatch:
        T = self.horizon if horizon is None else int(horizon)

        def _normalize_obs(tensor: Tensor) -> Tensor:
            if self.observation_normalizer is None:
                return tensor
            self.observation_normalizer.to(tensor.device)
            return self.observation_normalizer.normalize(tensor)

        def _normalize_waypoint(tensor: Tensor) -> Tensor:
            if self.waypoint_normalizer is None:
                return tensor
            self.waypoint_normalizer.to(tensor.device)
            return self.waypoint_normalizer.normalize(tensor)

        def _normalize_lidar(tensor: Tensor) -> Tensor:
            if self.lidar_normalizer is None:
                return tensor
            self.lidar_normalizer.to(tensor.device)
            return self.lidar_normalizer.normalize(tensor)

        history_records: List[Tensor] = []
        obs_records: List[Tensor] = []
        state_records: List[Tensor] = []
        action_records: List[Tensor] = []
        log_prob_records: List[Tensor] = []
        reward_records: List[Tensor] = []
        done_records: List[Tensor] = []
        episode_start_records: List[Tensor] = []
        episode_id_records: List[Tensor] = []
        token_offset_records: List[Tensor] = []
        waypoint_records: List[Tensor] = []
        plan_state_records: List[Tensor] = []
        plan_action_records: List[Tensor] = []
        plan_reward_records: List[Tensor] = []
        plan_observation_records: List[Tensor] = []
        warm_start_source_records: List[List[str]] = []
        info_records: List[List[Dict]] = []
        lidar_records: List[Tensor] = []
        old_value_records: List[Tensor] = []

        warm_start_tensor, warm_start_source = self._prepare_warm_start()

        for _ in range(T):
            history_records.append(self._history.clone())
            obs_records.append(self._state.clone())
            state_records.append(self._state.clone())
            # Update obs normalizer with current state/history before acting.
            if self.observation_normalizer is not None:
                self.observation_normalizer.to(self._state.device)
                self.observation_normalizer.update(self._state, mask=None)
                norm_history = _normalize_obs(self._history)
                norm_state = _normalize_obs(self._state)
            else:
                norm_history = self._history
                norm_state = self._state
            if self.waypoint_normalizer is not None and self._waypoint_seq.numel() > 0:
                self.waypoint_normalizer.to(self._waypoint_seq.device)
                self.waypoint_normalizer.update(self._waypoint_seq, mask=None)
                norm_waypoint = _normalize_waypoint(self._waypoint_seq)
            else:
                norm_waypoint = self._waypoint_seq
            norm_lidar: Optional[Tensor] = self._lidar
            if self._lidar_enabled and self._lidar is not None and self.lidar_normalizer is not None:
                self.lidar_normalizer.to(self._lidar.device)
                self.lidar_normalizer.update(self._lidar, mask=None)
                norm_lidar = _normalize_lidar(self._lidar)
            episode_start_records.append(self._episode_flags.clone())
            episode_id_records.append(self._episode_counters.clone())
            token_offset_records.append(self._token_offsets.clone())
            waypoint_records.append(self._waypoint_seq.clone())
            if self._lidar_enabled and self._lidar is not None:
                lidar_records.append(self._lidar.clone())

            actor_device = getattr(self.agent, "device", norm_history.device)
            norm_history_act = norm_history.to(actor_device)
            norm_state_act = norm_state.to(actor_device)
            norm_waypoint_act = norm_waypoint.to(actor_device)
            norm_lidar_act = (
                None if norm_lidar is None else norm_lidar.to(actor_device, dtype=norm_state_act.dtype)
            )
            warm_start_act = warm_start_tensor.to(actor_device, dtype=norm_state_act.dtype) if warm_start_tensor is not None else None

            act_kwargs = {
                "state": norm_state_act,
                "raw_state": self._state.to(actor_device),
                "memories": self._actor_state,
                "warm_start": warm_start_act,
                "return_plan": True,
                "return_log_prob": True,
                "waypoint_seq": norm_waypoint_act,
                "raw_waypoint_seq": self._waypoint_seq.to(actor_device),
            }
            if self._lidar_enabled and self._lidar is not None:
                act_kwargs["lidar"] = norm_lidar_act

            with torch.no_grad():
                action, log_prob, self._actor_state, plan = self.agent.act(
                    norm_history_act,
                    stochastic=True,
                    **act_kwargs,
                )
                if hasattr(self.agent, "critic"):
                    critic_value = self.agent.critic(
                        norm_history_act,
                        waypoint_seq=norm_waypoint_act,
                        raw_state=self._state.to(actor_device),
                        raw_waypoint_seq=self._waypoint_seq.to(actor_device),
                        lidar=norm_lidar_act if self._lidar_enabled else None,
                    ).value.view(self.num_envs)
                else:
                    critic_value = torch.zeros(
                        self.num_envs,
                        device=norm_history.device,
                        dtype=norm_history.dtype,
                    )
            if plan is None:
                raise RuntimeError("RolloutCollector requires the agent to return MPC plans.")

            plan_states, plan_actions = plan
            plan_state_records.append(plan_states.detach())
            plan_action_records.append(plan_actions.detach())
            action_records.append(action.detach())
            log_prob_records.append(log_prob.detach())
            old_value_records.append(critic_value.detach().clone())
            if self.collect_plan_rewards:
                rewards = self.env.predict_rewards(
                    plan_states.detach().cpu(),
                    plan_actions.detach().cpu(),
                )
                plan_reward_records.append(rewards.detach())
            if self.collect_plan_observations:
                predicted_obs = self.env.predict_observations(
                    plan_states.detach().cpu(),
                    plan_actions.detach().cpu(),
                )
                plan_observation_records.append(predicted_obs.detach())

            warm_start_source_records.append(list(warm_start_source))

            step_batch: EnvBatch = self.env.step(action)

            reward_records.append(step_batch.reward)
            done_records.append(step_batch.done.bool())
            info_records.append(list(step_batch.info))

            if self._lidar_enabled:
                self._update_lidar(step_batch)

            warm_start_tensor = plan_actions.detach().clone()
            continue_mask = (~step_batch.episode_start.bool()).view(self.num_envs, 1, 1)
            continue_mask = continue_mask.to(device=warm_start_tensor.device, dtype=warm_start_tensor.dtype)
            warm_start_tensor = warm_start_tensor * continue_mask
            warm_start_source = ["zeros" if flag else "cache" for flag in step_batch.episode_start.bool().tolist()]

            # Update episode counters for next iteration
            self._update_episode_counters(step_batch.episode_start.bool())

            self._update_history(step_batch)

            self._state = step_batch.state
            self._waypoint_seq = step_batch.waypoint_seq

            new_episode_flags = step_batch.episode_start.bool()
            self._token_offsets = torch.where(
                new_episode_flags,
                torch.zeros_like(self._token_offsets),
                self._token_offsets + 1,
            )
            self._episode_flags = new_episode_flags

        history_tensor = torch.stack(history_records, dim=1)
        observation_tensor = torch.stack(obs_records, dim=1)
        state_tensor = torch.stack(state_records, dim=1)
        action_tensor = torch.stack(action_records, dim=1)
        log_prob_tensor = torch.stack(log_prob_records, dim=1)
        reward_tensor = torch.stack(reward_records, dim=1)
        done_tensor = torch.stack(done_records, dim=1)
        episode_start_tensor = torch.stack(episode_start_records, dim=1)
        episode_id_tensor = torch.stack(episode_id_records, dim=1)
        token_offset_tensor = torch.stack(token_offset_records, dim=1)
        waypoint_tensor = torch.stack(waypoint_records, dim=1)
        plan_state_tensor = torch.stack(plan_state_records, dim=1)
        plan_action_tensor = torch.stack(plan_action_records, dim=1)
        old_value_tensor = torch.stack(old_value_records, dim=1)
        plan_reward_tensor = (
            torch.stack(plan_reward_records, dim=1) if plan_reward_records else None
        )
        plan_observation_tensor = (
            torch.stack(plan_observation_records, dim=1) if plan_observation_records else None
        )

        next_history_tensor = self._history.clone()
        next_state_tensor = self._state.clone()
        next_waypoint_tensor = self._waypoint_seq.clone()
        next_episode_start_tensor = self._episode_flags.clone()

        lidar_tensor: Optional[Tensor]
        next_lidar_tensor: Optional[Tensor]
        if self._lidar_enabled and lidar_records:
            lidar_tensor = torch.stack(lidar_records, dim=1)
            next_lidar_tensor = self._lidar.clone() if self._lidar is not None else None
        else:
            lidar_tensor = None
            next_lidar_tensor = None

        batch = RolloutBatch(
            history=history_tensor,
            observation=observation_tensor,
            state=state_tensor,
            next_history=next_history_tensor,
            next_state=next_state_tensor,
            action=action_tensor,
            log_prob=log_prob_tensor,
            reward=reward_tensor,
            done=done_tensor,
            episode_start=episode_start_tensor,
            next_episode_start=next_episode_start_tensor,
            episode_id=episode_id_tensor,
            token_offset=token_offset_tensor,
            trial_length=torch.full((self.num_envs,), T, dtype=torch.long),
            waypoint_seq=waypoint_tensor,
            next_waypoint_seq=next_waypoint_tensor,
            plan_states=plan_state_tensor,
            plan_actions=plan_action_tensor,
            warm_start_source=warm_start_source_records,
            info=info_records,
            mask=torch.ones(self.num_envs, T, dtype=torch.float32),
            old_value=old_value_tensor,
            plan_rewards=plan_reward_tensor,
            plan_observations=plan_observation_tensor,
            lidar=lidar_tensor,
            next_lidar=next_lidar_tensor,
        )
        self._warm_start_cache = warm_start_tensor.detach().clone()
        return batch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialise_lidar(self, initial: EnvBatch) -> None:
        """Configure lidar support based on agent config and initial env infos."""
        actor_cfg = getattr(getattr(self.agent, "config", None), "actor", None)
        if actor_cfg is None or not getattr(actor_cfg, "include_lidar", False):
            return
        lidar_dim = int(getattr(actor_cfg, "lidar_dim", 0) or 0)
        if lidar_dim <= 0:
            return

        # Extract initial lidar features from env info if available; otherwise zeros.
        lidar_list: List[Tensor] = []
        for info in initial.info:
            lidar_raw = info.get("lidar")
            if lidar_raw is None:
                lidar_list.append(torch.zeros(lidar_dim, dtype=initial.observation.dtype))
                continue
            lidar_tensor = torch.as_tensor(lidar_raw, dtype=initial.observation.dtype)
            if lidar_tensor.numel() != lidar_dim:
                raise ValueError(
                    f"Environment lidar dimension {lidar_tensor.numel()} does not match actor.lidar_dim={lidar_dim}."
                )
            lidar_tensor = lidar_tensor.view(lidar_dim)
            lidar_list.append(lidar_tensor)

        self._lidar_enabled = True
        self._lidar_dim = lidar_dim
        self._lidar = torch.stack(lidar_list, dim=0)

    def _update_lidar(self, step_batch: EnvBatch) -> None:
        """Update cached lidar features from the latest EnvBatch."""
        if not self._lidar_enabled:
            return
        lidar_list: List[Tensor] = []
        for info in step_batch.info:
            lidar_raw = info.get("lidar")
            if lidar_raw is None:
                lidar_list.append(torch.zeros(self._lidar_dim, dtype=step_batch.observation.dtype))
                continue
            lidar_tensor = torch.as_tensor(lidar_raw, dtype=step_batch.observation.dtype)
            if lidar_tensor.numel() != self._lidar_dim:
                raise ValueError(
                    f"Environment lidar dimension {lidar_tensor.numel()} does not match configured lidar_dim={self._lidar_dim}."
                )
            lidar_list.append(lidar_tensor.view(self._lidar_dim))
        self._lidar = torch.stack(lidar_list, dim=0)

    def _prepare_warm_start(self) -> Tuple[Optional[Tensor], List[str]]:
        if self._warm_start_cache is None:
            zeros = torch.zeros(
                self.num_envs,
                self.horizon,
                self.action_dim,
                dtype=self._state.dtype,
            )
            return zeros, ["zeros"] * self.num_envs
        return self._warm_start_cache, ["cache"] * self.num_envs

    def _update_episode_counters(self, next_episode_flags: Tensor) -> None:
        finished = next_episode_flags.bool()
        self._episode_counters += finished.to(self._episode_counters.dtype)

    def _update_history(self, step_batch: EnvBatch) -> None:
        obs_state = step_batch.state
        episode_start = step_batch.episode_start.bool()
        for idx in range(self.num_envs):
            if episode_start[idx]:
                self._history[idx].zero_()
            else:
                prev = self._history[idx, 1:].clone()
                self._history[idx, :-1] = prev
            self._history[idx, -1] = obs_state[idx]
