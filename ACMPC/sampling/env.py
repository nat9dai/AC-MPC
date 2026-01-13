"""Environment wrappers and managers ensuring absolute-frame observations."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .rollout import EnvBatch


def _to_tensor(array, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


class AbsoluteEnvWrapper:
    """Wraps a Gym-like environment enforcing absolute-frame outputs.

    Parameters
    ----------
    env:
        Environment providing ``reset`` and ``step`` methods following the Gym API.
    state_fn:
        Callable mapping ``(observation, info)`` to the absolute state tensor.
        Defaults to the observation itself.
    waypoint_fn:
        Callable mapping ``(observation, info)`` to a waypoint sequence tensor of
        shape ``[waypoint_len, waypoint_dim]``.  Defaults to all zeros.
    waypoint_len:
        Length of waypoint sequence produced by ``waypoint_fn`` when the default
        implementation is used.
    action_dim:
        Dimension of continuous action space; inferred from the environment when
        available.
    device/dtype:
        Torch device and dtype used for returned tensors.
    """

    def __init__(
        self,
        env,
        *,
        state_fn: Optional[Callable[[Tensor, dict], Tensor]] = None,
        waypoint_fn: Optional[Callable[[Tensor, dict], Tensor]] = None,
        waypoint_len: int = 1,
        action_dim: Optional[int] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        reward_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        observation_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        self.env = env
        self.device = torch.device(device)
        self.dtype = dtype
        self.state_fn = state_fn or (lambda obs, info: obs)
        self._default_waypoint_len = waypoint_len
        self.waypoint_fn = waypoint_fn or self._default_waypoint
        self._reward_fn = reward_fn
        self._observation_fn = observation_fn

        if action_dim is not None:
            self.action_dim = int(action_dim)
        elif hasattr(env, "action_space") and getattr(env.action_space, "shape", None):
            self.action_dim = int(env.action_space.shape[0])
        elif hasattr(env, "action_dim"):
            self.action_dim = int(env.action_dim)
        else:
            raise ValueError("Unable to infer action dimension; please specify action_dim explicitly.")

        self._seed_sequence: Optional[np.random.SeedSequence] = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def seed(self, seed: int) -> None:
        self._seed_sequence = np.random.SeedSequence(seed)

    def reset(self, *, seed: Optional[int] = None):
        if seed is not None:
            self._seed_sequence = np.random.SeedSequence(seed)
            reset_seed = seed
        elif self._seed_sequence is not None:
            child = self._seed_sequence.spawn(1)[0]
            reset_seed = int(child.entropy)
        else:
            reset_seed = None

        if reset_seed is not None:
            result = self.env.reset(seed=reset_seed)
        else:
            result = self.env.reset()

        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        return self._build_output(obs, info, reward=0.0, done=False, episode_start=True)

    def step(self, action: Tensor):
        action_np = np.asarray(action.detach().cpu().numpy())
        result = self.env.step(action_np)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        elif len(result) == 4:
            obs, reward, done, info = result
            done = bool(done)
        else:
            raise ValueError("Environment step output not recognised.")

        if done:
            # Preserve terminal info separately to avoid double-counting in the new episode.
            terminal_collision = bool(isinstance(info, dict) and info.get("collision"))
            terminal_min_dist = info.get("min_obstacle_distance") if isinstance(info, dict) else None

            reset_output = self.reset()
            obs_tensor = reset_output["observation"].cpu().numpy()
            info = dict(reset_output["info"])
            if terminal_collision:
                info["terminal_collision"] = True
            if terminal_min_dist is not None:
                info["terminal_min_obstacle_distance"] = terminal_min_dist
            episode_start = True
        else:
            obs_tensor = obs
            episode_start = False

        return self._build_output(obs_tensor, info, reward, done, episode_start)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _default_waypoint(self, obs: Tensor, info: dict) -> Tensor:
        state_dim = int(obs.shape[-1])
        return torch.zeros(self._default_waypoint_len, state_dim, dtype=obs.dtype, device=obs.device)

    def _build_output(
        self,
        observation,
        info: dict,
        reward: float,
        done: bool,
        episode_start: bool,
    ) -> dict:
        obs_tensor = _to_tensor(observation, dtype=self.dtype, device=self.device)
        state_tensor = self.state_fn(obs_tensor, info)
        if not isinstance(state_tensor, torch.Tensor):
            state_tensor = _to_tensor(state_tensor, dtype=self.dtype, device=self.device)

        waypoint_tensor = self.waypoint_fn(obs_tensor, info)
        if not isinstance(waypoint_tensor, torch.Tensor):
            waypoint_tensor = _to_tensor(waypoint_tensor, dtype=self.dtype, device=self.device)
        if waypoint_tensor.dim() == 1:
            waypoint_tensor = waypoint_tensor.unsqueeze(0)

        return {
            "observation": obs_tensor,
            "state": state_tensor,
            "waypoint_seq": waypoint_tensor,
            "reward": torch.as_tensor(reward, dtype=self.dtype, device=self.device),
            "done": torch.tensor(done, dtype=torch.bool),
            "episode_start": torch.tensor(episode_start, dtype=torch.bool),
            "info": info,
        }

    # ------------------------------------------------------------------
    # MPC reward prediction interface
    # ------------------------------------------------------------------
    def has_reward_fn(self) -> bool:
        return self._reward_fn is not None or hasattr(self.env, "predict_reward")

    def has_observation_fn(self) -> bool:
        return self._observation_fn is not None or hasattr(self.env, "predict_observations")

    def predict_reward(self, states: Tensor, actions: Tensor) -> Tensor:
        """Evaluate environment reward on predicted MPC trajectories.

        Parameters
        ----------
        states:
            Tensor with shape ``[horizon + 1, state_dim]`` representing the MPC
            state rollout starting at the current state.
        actions:
            Tensor with shape ``[horizon, action_dim]`` containing MPC control
            inputs applied between consecutive states.
        """

        if states.dim() != 2:
            raise ValueError("states must have shape [horizon + 1, state_dim].")
        if actions.dim() != 2:
            raise ValueError("actions must have shape [horizon, action_dim].")
        if states.size(0) != actions.size(0) + 1:
            raise ValueError("states length must be one greater than actions length.")

        states = states.to(device=self.device, dtype=self.dtype)
        actions = actions.to(device=self.device, dtype=self.dtype)

        if self._reward_fn is not None:
            reward = self._reward_fn(states, actions)
        elif hasattr(self.env, "predict_reward"):
            reward = self.env.predict_reward(states, actions)
        else:
            raise RuntimeError(
                "Environment does not provide a reward prediction function; supply reward_fn when MPVE is enabled."
            )

        if not isinstance(reward, torch.Tensor):
            reward = _to_tensor(reward, dtype=self.dtype, device=self.device)

        if reward.dim() == 0:
            reward = reward.unsqueeze(0).expand(actions.size(0))
        if reward.dim() != 1 or reward.size(0) != actions.size(0):
            raise ValueError(
                "Reward prediction must return a 1D tensor with length equal to the number of actions."
            )
        return reward

    def predict_observations(self, states: Tensor, actions: Tensor) -> Tensor:
        """Return predicted observation trajectory for MPC rollouts.

        The returned tensor must have shape ``[horizon + 1, obs_dim]`` and match
        the observation format expected by the rollout collector.
        """

        if states.dim() != 2:
            raise ValueError("states must have shape [horizon + 1, state_dim].")
        if actions.dim() != 2:
            raise ValueError("actions must have shape [horizon, action_dim].")
        if states.size(0) != actions.size(0) + 1:
            raise ValueError("states length must be actions length + 1.")

        states = states.to(device=self.device, dtype=self.dtype)
        actions = actions.to(device=self.device, dtype=self.dtype)

        if self._observation_fn is not None:
            observation = self._observation_fn(states, actions)
        elif hasattr(self.env, "predict_observations"):
            observation = self.env.predict_observations(states, actions)
        else:
            raise RuntimeError(
                "Environment does not provide an observation prediction function; supply observation_fn when MPVE is enabled."
            )

        if not isinstance(observation, torch.Tensor):
            observation = _to_tensor(observation, dtype=self.dtype, device=self.device)

        if observation.dim() != 2 or observation.size(0) != states.size(0):
            raise ValueError(
                "Observation prediction must have shape [horizon + 1, obs_dim]."
            )
        return observation


class VectorEnvManager:
    """Manages a list of wrapped environments synchronously."""

    def __init__(
        self,
        env_fns: Sequence[Callable[[], AbsoluteEnvWrapper]],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        base_seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.envs: List[AbsoluteEnvWrapper] = []
        for idx, fn in enumerate(env_fns):
            wrapper = fn()
            wrapper.seed(base_seed + idx)
            self.envs.append(wrapper)

        if len(self.envs) == 0:
            raise ValueError("VectorEnvManager requires at least one environment.")

        self.num_envs = len(self.envs)
        self.action_dim = self.envs[0].action_dim

        self._episode_start_flags = torch.ones(self.num_envs, dtype=torch.bool)

    def reset(self) -> EnvBatch:
        observations: List[Tensor] = []
        states: List[Tensor] = []
        waypoints: List[Tensor] = []
        infos: List[dict] = []

        for env in self.envs:
            output = env.reset()
            observations.append(output["observation"].to("cpu"))
            states.append(output["state"].to("cpu"))
            waypoints.append(output["waypoint_seq"].to("cpu"))
            infos.append(output["info"])

        return EnvBatch(
            observation=torch.stack(observations, dim=0),
            state=torch.stack(states, dim=0),
            waypoint_seq=torch.stack(waypoints, dim=0),
            reward=torch.zeros(self.num_envs),
            done=torch.zeros(self.num_envs, dtype=torch.bool),
            episode_start=torch.ones(self.num_envs, dtype=torch.bool),
            info=infos,
        )

    def step(self, actions: Tensor) -> EnvBatch:
        if actions.shape[0] != self.num_envs:
            raise ValueError("Action batch dimension mismatch.")

        observations: List[Tensor] = []
        states: List[Tensor] = []
        waypoints: List[Tensor] = []
        rewards: List[Tensor] = []
        dones: List[Tensor] = []
        episode_starts: List[Tensor] = []
        infos: List[dict] = []

        for idx, env in enumerate(self.envs):
            output = env.step(actions[idx])
            observations.append(output["observation"].to("cpu"))
            states.append(output["state"].to("cpu"))
            waypoints.append(output["waypoint_seq"].to("cpu"))
            rewards.append(output["reward"].to("cpu"))
            dones.append(output["done"].to("cpu"))
            episode_starts.append(output["episode_start"].to("cpu"))
            infos.append(output["info"])

        return EnvBatch(
            observation=torch.stack(observations, dim=0),
            state=torch.stack(states, dim=0),
            waypoint_seq=torch.stack(waypoints, dim=0),
            reward=torch.stack(rewards, dim=0),
            done=torch.stack(dones, dim=0),
            episode_start=torch.stack(episode_starts, dim=0),
            info=infos,
        )

    def close(self) -> None:
        for env in self.envs:
            if hasattr(env.env, "close"):
                env.env.close()

    # ------------------------------------------------------------------
    # MPC reward prediction helpers
    # ------------------------------------------------------------------
    def supports_reward_prediction(self) -> bool:
        return all(env.has_reward_fn() for env in self.envs)

    def supports_observation_prediction(self) -> bool:
        return all(env.has_observation_fn() for env in self.envs)

    def predict_rewards(self, states: Tensor, actions: Tensor) -> Tensor:
        """Evaluate rewards for MPC rollouts across environments.

        Parameters
        ----------
        states:
            Tensor with shape ``[num_envs, horizon + 1, state_dim]``.
        actions:
            Tensor with shape ``[num_envs, horizon, action_dim]``.
        """

        if states.dim() != 3:
            raise ValueError("states must have shape [env, horizon + 1, state_dim].")
        if actions.dim() != 3:
            raise ValueError("actions must have shape [env, horizon, action_dim].")
        if states.shape[0] != self.num_envs or actions.shape[0] != self.num_envs:
            raise ValueError("states/actions batch dimension must equal manager.num_envs.")
        if states.size(1) != actions.size(1) + 1:
            raise ValueError("states horizon must be actions horizon + 1.")

        rewards = []
        for idx, env in enumerate(self.envs):
            env_reward = env.predict_reward(states[idx], actions[idx])
            rewards.append(env_reward.to("cpu"))

        return torch.stack(rewards, dim=0)

    def predict_observations(self, states: Tensor, actions: Tensor) -> Tensor:
        """Evaluate observation trajectories for MPC rollouts."""

        if states.dim() != 3:
            raise ValueError("states must have shape [env, horizon + 1, state_dim].")
        if actions.dim() != 3:
            raise ValueError("actions must have shape [env, horizon, action_dim].")
        if states.shape[0] != self.num_envs or actions.shape[0] != self.num_envs:
            raise ValueError("states/actions batch dimension must equal manager.num_envs.")
        if states.size(1) != actions.size(1) + 1:
            raise ValueError("states horizon must be actions horizon + 1.")

        observations = []
        for idx, env in enumerate(self.envs):
            obs = env.predict_observations(states[idx], actions[idx])
            observations.append(obs.to("cpu"))
        return torch.stack(observations, dim=0)
