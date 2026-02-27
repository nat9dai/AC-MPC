"""3-D waypoint-tracking double integrator with velocity commands.

Extends the 2-D double integrator to three dimensions.  The environment
models a point-mass robot moving in 3-D space with state
``[x, y, z, vx, vy, vz]`` and actions interpreted as desired velocities
``[vx_cmd, vy_cmd, vz_cmd]``.  The commanded velocities are clamped to
``[-max_speed, max_speed]`` and blended with the previous velocity through
a first-order response.  After each waypoint is reached (within
``goal_radius``) a new one is sampled uniformly inside a bounded cube.
Rewards encourage progress towards the active waypoint while penalising
high commanded velocities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


@dataclass
class DoubleIntegrator3DEnvConfig:
    """Convenience container for environment hyper-parameters."""

    dt: float = 0.05
    episode_len: int = 400
    waypoint_range: float = 1.0
    goal_radius: float = 0.15
    min_start_radius: float = 0.2
    max_speed: float = 3.0
    velocity_response: float = 0.5
    progress_gain: float = 15.0
    action_penalty: float = 0.05
    living_penalty: float = 0.01
    goal_bonus: float = 5.0
    control_gain: float = 0.0


class DoubleIntegrator3DWaypointEnv(gym.Env):
    """Gymnasium environment: velocity-controlled 3-D double integrator."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        config: DoubleIntegrator3DEnvConfig | None = None,
        dt: float = 0.05,
        episode_len: int = 400,
        waypoint_range: float = 1.0,
        goal_radius: float = 0.15,
        min_start_radius: float = 0.2,
        max_speed: float = 3.0,
        velocity_response: float = 0.5,
        progress_gain: float = 15.0,
        action_penalty: float = 0.05,
        living_penalty: float = 0.01,
        goal_bonus: float = 5.0,
        control_gain: float = 0.0,
        env_id: int = 0,
    ) -> None:
        super().__init__()
        if config is not None:
            dt = config.dt
            episode_len = config.episode_len
            waypoint_range = config.waypoint_range
            goal_radius = config.goal_radius
            min_start_radius = config.min_start_radius
            max_speed = config.max_speed
            velocity_response = config.velocity_response
            progress_gain = config.progress_gain
            action_penalty = config.action_penalty
            living_penalty = config.living_penalty
            goal_bonus = config.goal_bonus
            control_gain = config.control_gain

        if not 0.0 < velocity_response <= 1.0:
            raise ValueError("velocity_response must be in (0, 1].")
        if waypoint_range <= 0.0:
            raise ValueError("waypoint_range must be positive.")
        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive.")

        self.dt = float(dt)
        self.episode_len = int(episode_len)
        self.waypoint_range = float(waypoint_range)
        self.goal_radius = float(goal_radius)
        self.min_start_radius = float(min_start_radius)
        self.max_speed = float(max_speed)
        self.velocity_response = float(velocity_response)
        self.progress_gain = float(progress_gain)
        self.action_penalty = float(action_penalty)
        self.living_penalty = float(living_penalty)
        self.goal_bonus = float(goal_bonus)
        self.control_gain = float(control_gain)
        self.env_id = int(env_id)

        self.nx = 6   # [x, y, z, vx, vy, vz]
        self.nu = 3   # [vx_cmd, vy_cmd, vz_cmd]
        obs_dim = self.nx + 3  # state + 3-D waypoint
        high = np.full(obs_dim, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(self.nu, dtype=np.float32) * self.max_speed,
            high=np.ones(self.nu, dtype=np.float32) * self.max_speed,
            dtype=np.float32,
        )

        self.state = np.zeros(self.nx, dtype=np.float32)
        self.target_waypoint = np.zeros(3, dtype=np.float32)
        self.prev_distance = 0.0
        self.steps = 0
        self.waypoints_reached = 0
        self._np_random = np.random.default_rng()

        # Rendering
        self.render_mode = None
        self._trajectory: List[np.ndarray] = []

    # Gymnasium API ---------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            base_seed = (seed + self.env_id) % (2**32 - 1)
            self._np_random = np.random.default_rng(base_seed)

        self.steps = 0
        self.waypoints_reached = 0
        self.state = np.zeros(self.nx, dtype=np.float32)
        self.target_waypoint = self._sample_waypoint()
        self.prev_distance = self._distance_to_waypoint(self.state[:3], self.target_waypoint)
        self._trajectory = [self.state[:3].copy()]

        observation = self._get_obs()
        info = {
            "target_waypoint": self.target_waypoint.copy(),
            "waypoints_reached": self.waypoints_reached,
        }
        return observation, info

    def step(self, action: np.ndarray):
        self.steps += 1
        action = np.asarray(action, dtype=np.float32)
        clipped_action = np.clip(action, -self.max_speed, self.max_speed)

        vel = self.state[3:]
        vel_next = (1.0 - self.velocity_response) * vel + self.velocity_response * clipped_action
        pos_next = self.state[:3] + vel_next * self.dt

        next_state = np.concatenate((pos_next, vel_next)).astype(np.float32)
        distance = self._distance_to_waypoint(pos_next, self.target_waypoint)
        progress = self.prev_distance - distance

        reward = self.progress_gain * progress
        control_mag = float(np.linalg.norm(clipped_action))
        if self.action_penalty > 0.0:
            reward -= self.action_penalty * control_mag
        elif self.control_gain > 0.0:
            reward += self.control_gain * control_mag

        waypoint_reached = distance <= self.goal_radius
        if waypoint_reached:
            reward += self.goal_bonus
            self.waypoints_reached += 1
            self.target_waypoint = self._sample_waypoint()
            distance = self._distance_to_waypoint(pos_next, self.target_waypoint)
            self.prev_distance = distance
        else:
            self.prev_distance = distance

        self.state = next_state

        truncated = self.steps >= self.episode_len
        terminated = False

        info = {
            "target_waypoint": self.target_waypoint.copy(),
            "waypoints_reached": self.waypoints_reached,
            "distance_to_waypoint": distance,
            "action_clipped": float(np.any(action != clipped_action)),
        }
        observation = self._get_obs()
        return observation, float(reward), terminated, truncated, info

    # Helpers ---------------------------------------------------------------
    def _sample_waypoint(self) -> np.ndarray:
        """Sample a 3-D waypoint within the arena.

        The first waypoint of each episode is drawn at a fixed radius
        of 0.5 from the origin, while subsequent waypoints follow uniform
        sampling inside the cube with a minimum radius exclusion.
        """
        if self.steps == 0 and self.waypoints_reached == 0:
            radius = 0.5
            # Sample a random direction on the unit sphere
            vec = self._np_random.standard_normal(3).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            return (radius * vec).astype(np.float32)

        for _ in range(512):
            waypoint = self._np_random.uniform(
                low=-self.waypoint_range,
                high=self.waypoint_range,
                size=(3,),
            ).astype(np.float32)
            if np.linalg.norm(waypoint) >= self.min_start_radius:
                return waypoint
        raise RuntimeError("Failed to sample waypoint outside the exclusion radius.")

    def _distance_to_waypoint(self, position: np.ndarray, waypoint: np.ndarray) -> float:
        return float(np.linalg.norm(position - waypoint))

    def _get_obs(self) -> np.ndarray:
        return np.concatenate((self.state, self.target_waypoint)).astype(np.float32)

    def render(self, mode: str = "human"):
        """Render the 3-D environment with matplotlib."""
        if mode not in self.metadata["render_modes"]:
            return None

        try:
            import matplotlib
            if mode == "human":
                try:
                    matplotlib.use("TkAgg")
                except Exception:
                    try:
                        matplotlib.use("Qt5Agg")
                    except Exception:
                        pass
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available for rendering")
            return None

        if not hasattr(self, '_fig') or self._fig is None:
            self._fig = plt.figure(figsize=(9, 9))
            self._ax = self._fig.add_subplot(111, projection="3d")
            lim = self.waypoint_range * 1.2
            self._ax.set_xlim(-lim, lim)
            self._ax.set_ylim(-lim, lim)
            self._ax.set_zlim(-lim, lim)
            self._ax.set_xlabel("X")
            self._ax.set_ylabel("Y")
            self._ax.set_zlabel("Z")
            self._ax.set_box_aspect([1, 1, 1])
            plt.ion()
            plt.show(block=False)

        pos = self.state[:3].copy()
        if len(self._trajectory) == 0 or not np.allclose(self._trajectory[-1], pos):
            self._trajectory.append(pos)

        self._ax.cla()
        lim = self.waypoint_range * 1.2
        self._ax.set_xlim(-lim, lim)
        self._ax.set_ylim(-lim, lim)
        self._ax.set_zlim(-lim, lim)
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_zlabel("Z")
        self._ax.set_title(
            f"3D Double Integrator (Env {self.env_id}) | "
            f"Steps: {self.steps} | Waypoints: {self.waypoints_reached}"
        )

        # Trajectory
        if len(self._trajectory) > 1:
            traj = np.array(self._trajectory)
            self._ax.plot(
                traj[:, 0], traj[:, 1], traj[:, 2],
                "b-", alpha=0.5, linewidth=1, label="Path",
            )

        # Current position
        self._ax.scatter(*pos, c="green", s=60, zorder=7, label="Robot")

        # Velocity arrow
        vel = self.state[3:]
        vel_norm = np.linalg.norm(vel)
        if vel_norm > 0.01:
            scale = min(0.3, vel_norm * 0.1)
            v_dir = vel / vel_norm * scale
            self._ax.quiver(
                pos[0], pos[1], pos[2],
                v_dir[0], v_dir[1], v_dir[2],
                color="green", linewidth=1.5, arrow_length_ratio=0.25, label="Vel",
            )

        # Target waypoint + goal sphere
        wp = self.target_waypoint
        self._ax.scatter(*wp, c="red", s=80, marker="*", zorder=5, label="Target")
        u_sp = np.linspace(0, 2 * np.pi, 16)
        v_sp = np.linspace(0, np.pi, 12)
        xs = wp[0] + self.goal_radius * np.outer(np.cos(u_sp), np.sin(v_sp))
        ys = wp[1] + self.goal_radius * np.outer(np.sin(u_sp), np.sin(v_sp))
        zs = wp[2] + self.goal_radius * np.outer(np.ones_like(u_sp), np.cos(v_sp))
        self._ax.plot_wireframe(xs, ys, zs, color="orange", alpha=0.15, linewidth=0.4)

        # Dashed line from robot to waypoint
        self._ax.plot(
            [pos[0], wp[0]], [pos[1], wp[1]], [pos[2], wp[2]],
            "r--", alpha=0.3, linewidth=1,
        )

        self._ax.legend(loc="upper right", fontsize=8)

        if mode == "human":
            plt.draw()
            plt.pause(0.01)
            return None
        elif mode == "rgb_array":
            self._fig.canvas.draw()
            buf = np.asarray(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
            return buf[:, :, :3].copy()
        return None

    def close(self):
        """Close rendering window."""
        if hasattr(self, '_fig') and self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig.number if hasattr(self._fig, 'number') else self._fig)
            except Exception:
                pass
            finally:
                self._fig = None
                self._ax = None
        self._trajectory = []


# -------------------------------------------------------------------------
# Torch dynamics helpers
# -------------------------------------------------------------------------
def _prepare_inputs_3d(
    x: torch.Tensor,
    u: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if u.ndim == 1:
        u = u.unsqueeze(0)
    if x.shape[0] != u.shape[0]:
        raise ValueError("x and u batch dimensions must match.")
    if x.shape[-1] != 6:
        raise ValueError("Expected state dimension 6.")
    if u.shape[-1] != 3:
        raise ValueError("Expected action dimension 3.")
    return x, u


def build_velocity_dynamics_3d(
    *,
    max_speed: float = 3.0,
    velocity_response: float = 0.5,
) -> Tuple[
    Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor]],
]:
    """Factory returning (f_dyn, f_dyn_jac) closures for 3-D double integrator."""

    if not 0.0 < velocity_response <= 1.0:
        raise ValueError("velocity_response must be in (0, 1].")
    max_speed = float(max_speed)
    alpha = float(velocity_response)

    def f_dyn_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        x, u = _prepare_inputs_3d(x, u)
        dt_tensor = torch.as_tensor(dt, dtype=x.dtype, device=x.device)
        clipped = torch.clamp(u, -max_speed, max_speed)
        vel = x[..., 3:]
        vel_next = (1.0 - alpha) * vel + alpha * clipped
        pos_next = x[..., :3] + vel_next * dt_tensor
        next_state = torch.cat([pos_next, vel_next], dim=-1)
        if next_state.shape[0] == 1:
            return next_state.squeeze(0)
        return next_state

    def f_dyn_jac_torch(
        x: torch.Tensor,
        u: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, u = _prepare_inputs_3d(x, u)
        batch = x.shape[0]
        dtype, device = x.dtype, x.device
        dt_tensor = torch.as_tensor(dt, dtype=dtype, device=device)

        vel_mask = (u.abs() < (max_speed - 1e-5)).to(dtype=dtype)

        A = torch.zeros(batch, 6, 6, dtype=dtype, device=device)
        B = torch.zeros(batch, 6, 3, dtype=dtype, device=device)

        # Position derivatives w.r.t position: identity
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 1.0

        # Position derivatives w.r.t velocity
        A[:, 0, 3] = dt_tensor * (1.0 - alpha)
        A[:, 1, 4] = dt_tensor * (1.0 - alpha)
        A[:, 2, 5] = dt_tensor * (1.0 - alpha)

        # Velocity derivatives w.r.t velocity
        A[:, 3, 3] = 1.0 - alpha
        A[:, 4, 4] = 1.0 - alpha
        A[:, 5, 5] = 1.0 - alpha

        # Velocity derivatives w.r.t actions
        B[:, 3, 0] = alpha * vel_mask[:, 0]
        B[:, 4, 1] = alpha * vel_mask[:, 1]
        B[:, 5, 2] = alpha * vel_mask[:, 2]

        # Position derivatives w.r.t actions (through velocity)
        B[:, 0, 0] = dt_tensor * alpha * vel_mask[:, 0]
        B[:, 1, 1] = dt_tensor * alpha * vel_mask[:, 1]
        B[:, 2, 2] = dt_tensor * alpha * vel_mask[:, 2]

        if batch == 1:
            return A.squeeze(0), B.squeeze(0)
        return A, B

    return f_dyn_torch, f_dyn_jac_torch


__all__ = [
    "DoubleIntegrator3DEnvConfig",
    "DoubleIntegrator3DWaypointEnv",
    "build_velocity_dynamics_3d",
]
