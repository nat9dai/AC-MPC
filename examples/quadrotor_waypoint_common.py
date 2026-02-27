"""Shared helpers for 3-D quadrotor waypoint training/evaluation scripts.

The MPC plans using a 6-D double-integrator model (position + velocity,
3-D velocity commands) while the simulation runs the full 15-D quadrotor
physics.  A geometric low-level controller converts the velocity commands
into [thrust, ωx, ωy, ωz] for the underlying QuadrotorWaypointEnv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch

from ACMPC import AbsoluteEnvWrapper, VectorEnvManager
from ACMPC.envs import QuadrotorWaypointEnv
from ACMPC.model_config import CostMapConfig


@dataclass
class DimensionSpec:
    state_dim: int
    action_dim: int
    waypoint_dim: int = 3


# ---------------------------------------------------------------------------
# Geometric low-level controller: 3-D velocity command → [thrust, ω]
# ---------------------------------------------------------------------------

def _velocity_to_thrust_rates(
    v_cmd: np.ndarray,
    v_current: np.ndarray,
    R: np.ndarray,
    *,
    mass: float,
    gravity: float,
    max_thrust: float,
    max_body_rate: float,
    kv: float = 5.0,
    kR: float = 10.0,
) -> np.ndarray:
    """Convert a 3-D velocity command into [thrust, ωx, ωy, ωz].

    Uses a standard geometric tracking controller:
    1. PD on velocity error → desired acceleration
    2. Feedforward gravity → total thrust vector
    3. Project onto body z-axis → scalar thrust
    4. Cross-product rotation error → angular rates
    """
    # Desired acceleration (velocity P-controller + gravity feedforward)
    a_des = kv * (v_cmd - v_current) + np.array([0.0, 0.0, gravity], dtype=np.float32)

    # Thrust magnitude = mass * projection of desired accel onto body z-axis
    z_body = R[:, 2]
    thrust = float(mass * np.dot(a_des, z_body))
    thrust = np.clip(thrust, 0.0, max_thrust)

    # Desired body z direction
    a_norm = float(np.linalg.norm(a_des))
    if a_norm > 1e-6:
        z_des = a_des / a_norm
    else:
        z_des = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Rotation error via cross product (small-angle approximation)
    e_R = np.cross(z_body, z_des)

    # Proportional angular rate controller
    omega = kR * e_R
    omega = np.clip(omega, -max_body_rate, max_body_rate)

    return np.array([thrust, omega[0], omega[1], omega[2]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Environment adapter: full quadrotor env with 6-D MPC interface
# ---------------------------------------------------------------------------

class QuadrotorVelocityAdapter(gym.Env):
    """Wraps QuadrotorWaypointEnv exposing a 6-D state and 3-D velocity actions.

    Internally runs the full 15-D quadrotor physics.  Accepts 3-D velocity
    commands as actions, converts them to [thrust, ω] via a geometric
    controller, and returns only [position, velocity] as the observation.
    """

    def __init__(
        self,
        *,
        kv: float = 5.0,
        kR: float = 10.0,
        max_speed: float = 3.0,
        **env_kwargs,
    ):
        super().__init__()
        self._base_env = QuadrotorWaypointEnv(**env_kwargs)
        self.metadata = self._base_env.metadata
        self.mass = self._base_env.mass
        self.gravity = self._base_env.gravity
        self.max_thrust = self._base_env.max_thrust
        self.max_body_rate = self._base_env.max_body_rate
        self.kv = kv
        self.kR = kR
        self.max_speed = max_speed

        # MPC-facing dimensions
        self.nx = 6   # [x, y, z, vx, vy, vz]
        self.nu = 3   # [vx_cmd, vy_cmd, vz_cmd]

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.nx,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.nu, dtype=np.float32) * self.max_speed,
            high=np.ones(self.nu, dtype=np.float32) * self.max_speed,
            dtype=np.float32,
        )

        # Cache the full 15-D state for the geometric controller
        self._full_state: np.ndarray = np.zeros(15, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs_full, info = self._base_env.reset(seed=seed, options=options)
        self._full_state = obs_full[:15].copy()
        return self._extract_6d(obs_full), self._augment_info(info, obs_full)

    def step(self, action: np.ndarray):
        v_cmd = np.asarray(action, dtype=np.float32)
        v_cmd = np.clip(v_cmd, -self.max_speed, self.max_speed)

        # Convert velocity command → [thrust, ω] using current full state
        v_current = self._full_state[3:6]
        R = self._full_state[6:15].reshape(3, 3)
        action_4d = _velocity_to_thrust_rates(
            v_cmd, v_current, R,
            mass=self.mass,
            gravity=self.gravity,
            max_thrust=self.max_thrust,
            max_body_rate=self.max_body_rate,
            kv=self.kv,
            kR=self.kR,
        )

        obs_full, reward, terminated, truncated, info = self._base_env.step(action_4d)
        self._full_state = obs_full[:15].copy()
        return (
            self._extract_6d(obs_full),
            reward,
            terminated,
            truncated,
            self._augment_info(info, obs_full),
        )

    def render(self, **kwargs):
        return self._base_env.render(**kwargs)

    def close(self):
        self._base_env.close()

    def _extract_6d(self, obs: np.ndarray) -> np.ndarray:
        """Extract [position, velocity] from the full 18-D observation."""
        return np.concatenate([obs[:3], obs[3:6]]).astype(np.float32)

    def _augment_info(self, info: Optional[Dict], obs: np.ndarray) -> Dict:
        data = dict(info) if info else {}
        data["target_waypoint"] = np.asarray(obs[15:18], dtype=np.float32).copy()
        return data


# Keep the old adapter available for backward compatibility
class QuadrotorStateObservationAdapter(gym.Env):
    """Expose only the 15-D physical state; waypoint goes into info."""

    metadata = {"render_modes": []}

    def __init__(self, **env_kwargs):
        super().__init__()
        self._base_env = QuadrotorWaypointEnv(**env_kwargs)
        self.nx = int(self._base_env.nx)
        self._obs_shape = (self.nx,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=np.float32,
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
            obs[self.nx : self.nx + 3], dtype=np.float32,
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
    substeps: int = 1,
) -> VectorEnvManager:
    env_fns = []
    for idx in range(num_envs):
        def _make(idx=idx):
            kwargs = dict(env_kwargs)
            kwargs["env_id"] = idx
            adapter = QuadrotorVelocityAdapter(**kwargs)
            return AbsoluteEnvWrapper(
                adapter,
                state_fn=lambda obs, _info: obs,
                waypoint_fn=waypoint_from_info,
                waypoint_len=1,
                device=device,
                substeps=substeps,
            )
        env_fns.append(_make)
    return VectorEnvManager(env_fns, device=device, base_seed=seed)


def probe_dimensions(env_kwargs: Dict) -> DimensionSpec:
    env = QuadrotorVelocityAdapter(**env_kwargs)
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
    max_speed: float,
    device: str,
    mpc_dt: float | None = None,
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
        model.history_window, min(model.max_history_tokens, 4 * model.history_window),
    )
    model.waypoint_sequence_len = 1
    model.waypoint_dim = dims.waypoint_dim
    model.include_prev_action = False
    model.include_lidar = False

    actor_cfg.input_dim = dims.state_dim
    actor_cfg.mpc.state_dim = dims.state_dim
    actor_cfg.mpc.action_dim = dims.action_dim
    actor_cfg.mpc.horizon = mpc_horizon
    actor_cfg.mpc.dt = mpc_dt if mpc_dt is not None else dt
    actor_cfg.mpc.u_min = [-max_speed] * dims.action_dim
    actor_cfg.mpc.u_max = [max_speed] * dims.action_dim
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
    "QuadrotorVelocityAdapter",
    "QuadrotorStateObservationAdapter",
    "build_env_manager",
    "prepare_config",
    "probe_dimensions",
    "waypoint_from_info",
]
