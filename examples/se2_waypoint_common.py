"""Shared helpers for SE(2) kinematic waypoint training/evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch

from ACMPC import AbsoluteEnvWrapper, VectorEnvManager
from ACMPC.envs import SE2WaypointEnv
from ACMPC.model_config import CostMapConfig


@dataclass
class DimensionSpec:
    state_dim: int
    action_dim: int
    waypoint_dim: int = 2


class StateObservationAdapter(gym.Env):
    """Expose only the physical state [x, y, theta] to the policy."""

    metadata = {"render_modes": []}

    def __init__(self, **env_kwargs):
        super().__init__()
        self._base_env = SE2WaypointEnv(**env_kwargs)
        self.nx = int(self._base_env.nx)
        self._obs_shape = (self.nx,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._obs_shape,
            dtype=np.float32,
        )
        self.action_space = self._base_env.action_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        result = self._base_env.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return self._extract_state(obs), self._augment_info(info, obs)

    def step(self, action):
        result = self._base_env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError("Unexpected step signature from base environment.")
        return (
            self._extract_state(obs),
            reward,
            terminated,
            truncated,
            self._augment_info(info, obs),
        )

    def close(self):
        self._base_env.close()

    def _extract_state(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs[: self.nx], dtype=np.float32)

    def _augment_info(self, info: Optional[Dict], obs: np.ndarray) -> Dict:
        data = dict(info) if info else {}
        data["target_waypoint"] = np.asarray(
            obs[self.nx : self.nx + 2],
            dtype=np.float32,
        ).copy()
        return data


def waypoint_from_info(obs: torch.Tensor, info: Dict) -> torch.Tensor:
    waypoint = info.get("target_waypoint")
    if waypoint is None:
        raise KeyError("Environment info missing 'target_waypoint'.")
    tensor = torch.as_tensor(waypoint, dtype=obs.dtype, device=obs.device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def build_env_manager(
    *,
    num_envs: int,
    env_kwargs: Dict,
    seed: int,
    device: str = "cpu",
) -> VectorEnvManager:
    env_fns = []
    for idx in range(num_envs):

        def _make(idx=idx):
            kwargs = dict(env_kwargs)
            kwargs["env_id"] = idx
            adapter = StateObservationAdapter(**kwargs)
            return AbsoluteEnvWrapper(
                adapter,
                state_fn=lambda obs, _info: obs,
                waypoint_fn=waypoint_from_info,
                waypoint_len=1,
                device=device,
            )

        env_fns.append(_make)
    return VectorEnvManager(env_fns, device=device, base_seed=seed)


def probe_dimensions(env_kwargs: Dict) -> DimensionSpec:
    env = SE2WaypointEnv(**env_kwargs)
    dims = DimensionSpec(state_dim=int(env.nx), action_dim=int(env.nu))
    env.close()
    return dims


def prepare_config(
    config,
    *,
    dims: DimensionSpec,
    history_window: int,
    rollout_len: int,
    mpc_horizon: int,
    dt: float,
    action_limit: float,
    device: str,
) -> None:
    model = config.model
    actor_cfg = model.actor
    critic_cfg = model.critic
    if actor_cfg.cost_map is None:
        actor_cfg.cost_map = CostMapConfig()

    model.history_window = min(model.history_window, history_window)
    model.segment_len = min(model.segment_len, model.history_window)
    model.mem_len = max(model.segment_len, min(model.mem_len, model.history_window * 2))
    model.max_history_tokens = max(
        model.history_window,
        min(model.max_history_tokens, 4 * model.history_window),
    )
    model.waypoint_sequence_len = 1
    model.waypoint_dim = dims.waypoint_dim
    model.include_prev_action = False
    model.include_lidar = False

    actor_cfg.input_dim = dims.state_dim
    actor_cfg.mpc.state_dim = dims.state_dim
    actor_cfg.mpc.action_dim = dims.action_dim
    actor_cfg.mpc.horizon = mpc_horizon
    actor_cfg.mpc.dt = dt
    actor_cfg.mpc.u_min = [-action_limit] * dims.action_dim
    actor_cfg.mpc.u_max = [action_limit] * dims.action_dim
    actor_cfg.mpc.device = device
    actor_cfg.waypoint_dim = dims.waypoint_dim
    actor_cfg.waypoint_sequence_len = 1
    actor_cfg.include_prev_action = False
    actor_cfg.include_lidar = False

    critic_cfg.input_dim = dims.state_dim
    critic_cfg.waypoint_dim = dims.waypoint_dim
    critic_cfg.waypoint_sequence_len = 1
    critic_cfg.include_prev_action = False
    critic_cfg.include_lidar = False

    config.sampler.num_envs = max(1, config.sampler.num_envs)
    config.sampler.rollout_steps = rollout_len
    config.sampler.episode_len = max(config.sampler.episode_len, rollout_len)

    config.training.mini_batch_size = min(
        config.training.mini_batch_size,
        config.sampler.num_envs * rollout_len,
    )
    config.training.device = device
    config.training.log_interval = max(1, config.training.log_interval)

    model.synchronise()
    config.validate()


__all__ = [
    "DimensionSpec",
    "StateObservationAdapter",
    "build_env_manager",
    "prepare_config",
    "probe_dimensions",
    "waypoint_from_info",
]

