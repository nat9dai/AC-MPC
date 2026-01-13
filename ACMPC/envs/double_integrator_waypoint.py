"""Waypoint-tracking planar double integrator with velocity commands.

The environment models a point-mass robot moving in the plane with state
``[x, y, vx, vy]`` and actions interpreted as desired velocities
``[vx_cmd, vy_cmd]``. The commanded velocities are clamped to ``[-max_speed,
max_speed]`` and blended with the previous velocity through a first-order
response. After each `waypoint` is reached (within ``goal_radius``) a new one
is sampled uniformly inside a bounded square arena. Rewards encourage progress
towards the active waypoint while penalising high commanded velocities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


@dataclass
class WaypointEnvConfig:
    """Convenience container for environment hyper-parameters."""

    dt: float = 0.05
    episode_len: int = 400
    waypoint_range: float = 1.0
    goal_radius: float = 0.15
    min_start_radius: float = 0.2
    max_speed: float = 3.0
    velocity_response: float = 0.5  # how quickly the commanded velocity is applied
    progress_gain: float = 15.0
    action_penalty: float = 0.05
    living_penalty: float = 0.01
    goal_bonus: float = 5.0
    control_gain: float = 0.0


class DoubleIntegratorWaypointEnvV2(gym.Env):
    """Gymnasium environment exposing the velocity-controlled double integrator."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        config: WaypointEnvConfig | None = None,
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

        self.nx = 4
        self.nu = 2
        obs_dim = self.nx + 2
        high = np.full(obs_dim, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(self.nu, dtype=np.float32) * self.max_speed,
            high=np.ones(self.nu, dtype=np.float32) * self.max_speed,
            dtype=np.float32,
        )

        self.state = np.zeros(self.nx, dtype=np.float32)
        self.target_waypoint = np.zeros(2, dtype=np.float32)
        self.prev_distance = 0.0
        self.steps = 0
        self.waypoints_reached = 0
        self._np_random = np.random.default_rng()
        
        # Rendering
        self.render_mode = None
        self._trajectory = []  # Store trajectory for rendering

    # Gymnasium API -----------------------------------------------------
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
        self.prev_distance = self._distance_to_waypoint(self.state[:2], self.target_waypoint)
        self._trajectory = [self.state[:2].copy()]  # Reset trajectory

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

        vel = self.state[2:]
        vel_next = (1.0 - self.velocity_response) * vel + self.velocity_response * clipped_action
        pos_next = self.state[:2] + vel_next * self.dt

        next_state = np.concatenate((pos_next, vel_next)).astype(np.float32)
        distance = self._distance_to_waypoint(pos_next, self.target_waypoint)
        progress = self.prev_distance - distance

        # Reward shaping:
        # - always reward progress towards the waypoint;
        # - optionally penalise control magnitude via action_penalty
        #   (economic-style reward as in AC-MPC paper);
        # - or, if no penalty is configured, optionally reward control
        #   magnitude via control_gain to break the \"do-nothing\" symmetry.
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

    # Helpers -----------------------------------------------------------
    def _sample_waypoint(self) -> np.ndarray:
        """Sample a waypoint within the arena.

        The first waypoint of each episode is drawn at a fixed radius
        of 0.5 from the origin (for more controlled initial conditions),
        while subsequent waypoints follow the original uniform sampling
        inside the square with a minimum radius ``min_start_radius``.
        """
        # First waypoint of the episode: fixed radius 0.5 from origin.
        if self.steps == 0 and self.waypoints_reached == 0:
            radius = 0.5
            theta = float(self._np_random.uniform(0.0, 2.0 * np.pi))
            waypoint = np.array(
                [radius * np.cos(theta), radius * np.sin(theta)],
                dtype=np.float32,
            )
            return waypoint

        # Subsequent waypoints: original sampling strategy.
        for _ in range(512):
            waypoint = self._np_random.uniform(
                low=-self.waypoint_range,
                high=self.waypoint_range,
                size=(2,),
            ).astype(np.float32)
            if np.linalg.norm(waypoint) >= self.min_start_radius:
                return waypoint
        raise RuntimeError("Failed to sample waypoint outside the exclusion radius.")

    def _distance_to_waypoint(self, position: np.ndarray, waypoint: np.ndarray) -> float:
        return float(np.linalg.norm(position - waypoint))

    def _get_obs(self) -> np.ndarray:
        return np.concatenate((self.state, self.target_waypoint)).astype(np.float32)
    
    def render(self, mode: str = "human"):
        """Render the environment.
        
        Args:
            mode: 'human' for interactive display, 'rgb_array' for numpy array
        """
        if mode not in self.metadata["render_modes"]:
            return None
        
        try:
            import matplotlib
            # Use TkAgg backend for interactive display
            if mode == "human":
                try:
                    matplotlib.use("TkAgg")
                except Exception:
                    # Fallback to other backends
                    try:
                        matplotlib.use("Qt5Agg")
                    except Exception:
                        pass  # Use default backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Warning: matplotlib not available for rendering")
            return None
        
        if not hasattr(self, '_fig') or self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            self._ax.set_xlim(-self.waypoint_range * 1.2, self.waypoint_range * 1.2)
            self._ax.set_ylim(-self.waypoint_range * 1.2, self.waypoint_range * 1.2)
            self._ax.set_aspect('equal')
            self._ax.grid(True, alpha=0.3)
            self._ax.set_title(f"Double Integrator Waypoint (Env {self.env_id})")
            plt.ion()
            plt.show(block=False)
        
        # Store current position in trajectory
        if len(self._trajectory) == 0 or not np.allclose(self._trajectory[-1], self.state[:2]):
            self._trajectory.append(self.state[:2].copy())
        
        self._ax.clear()
        self._ax.set_xlim(-self.waypoint_range * 1.2, self.waypoint_range * 1.2)
        self._ax.set_ylim(-self.waypoint_range * 1.2, self.waypoint_range * 1.2)
        self._ax.set_aspect('equal')
        self._ax.grid(True, alpha=0.3)
        self._ax.set_title(
            f"Double Integrator Waypoint (Env {self.env_id}) | "
            f"Steps: {self.steps} | Waypoints: {self.waypoints_reached}"
        )
        
        # Draw trajectory
        if len(self._trajectory) > 1:
            traj = np.array(self._trajectory)
            self._ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=1, label='Path')
        
        # Draw current position
        pos = self.state[:2]
        vel = self.state[2:]
        self._ax.plot(pos[0], pos[1], 'go', markersize=10, label='Robot')
        
        # Draw velocity vector
        if np.linalg.norm(vel) > 0.01:
            self._ax.arrow(pos[0], pos[1], vel[0] * 0.1, vel[1] * 0.1,
                          head_width=0.05, head_length=0.05, fc='green', ec='green')
        
        # Draw target waypoint
        wp = self.target_waypoint
        circle = patches.Circle(wp, self.goal_radius, color='orange', alpha=0.3, label='Goal radius')
        self._ax.add_patch(circle)
        self._ax.plot(wp[0], wp[1], 'ro', markersize=12, label='Target')
        
        # Draw distance line
        self._ax.plot([pos[0], wp[0]], [pos[1], wp[1]], 'r--', alpha=0.3, linewidth=1)
        
        self._ax.legend(loc='upper right')
        self._ax.set_xlabel('X')
        self._ax.set_ylabel('Y')
        
        if mode == "human":
            plt.draw()
            plt.pause(0.01)
            return None
        elif mode == "rgb_array":
            self._fig.canvas.draw()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return buf
    
    def close(self):
        """Close rendering window."""
        if hasattr(self, '_fig') and self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                # Use close('all') to safely close figures from any thread
                plt.close(self._fig.number if hasattr(self._fig, 'number') else self._fig)
            except Exception:
                # Ignore errors when closing from non-main thread
                pass
            finally:
                self._fig = None
                self._ax = None
        self._trajectory = []


# -------------------------------------------------------------------------
# Torch dynamics helpers
# -------------------------------------------------------------------------
def _prepare_inputs(
    x: torch.Tensor,
    u: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if u.ndim == 1:
        u = u.unsqueeze(0)
    if x.shape[0] != u.shape[0]:
        raise ValueError("x and u batch dimensions must match.")
    if x.shape[-1] != 4:
        raise ValueError("Expected state dimension 4.")
    if u.shape[-1] != 2:
        raise ValueError("Expected action dimension 2.")
    return x, u


def build_velocity_dynamics(
    *,
    max_speed: float = 3.0,
    velocity_response: float = 0.5,
) -> Tuple[
    Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor]],
]:
    """Factory returning (f_dyn, f_dyn_jac) closures for ActorMPC."""

    if not 0.0 < velocity_response <= 1.0:
        raise ValueError("velocity_response must be in (0, 1].")
    max_speed = float(max_speed)
    alpha = float(velocity_response)

    def f_dyn_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        x, u = _prepare_inputs(x, u)
        dt_tensor = torch.as_tensor(dt, dtype=x.dtype, device=x.device)
        clipped = torch.clamp(u, -max_speed, max_speed)
        vel = x[..., 2:]
        vel_next = (1.0 - alpha) * vel + alpha * clipped
        pos_next = x[..., :2] + vel_next * dt_tensor
        next_state = torch.cat([pos_next, vel_next], dim=-1)
        if next_state.shape[0] == 1:
            return next_state.squeeze(0)
        return next_state

    def f_dyn_jac_torch(
        x: torch.Tensor,
        u: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, u = _prepare_inputs(x, u)
        batch = x.shape[0]
        dtype, device = x.dtype, x.device
        dt_tensor = torch.as_tensor(dt, dtype=dtype, device=device)

        vel_mask = (u.abs() < (max_speed - 1e-5)).to(dtype=dtype)

        one = torch.ones(batch, dtype=dtype, device=device)
        zero = torch.zeros(batch, dtype=dtype, device=device)

        A = torch.zeros(batch, 4, 4, dtype=dtype, device=device)
        B = torch.zeros(batch, 4, 2, dtype=dtype, device=device)

        # Position derivatives
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 0, 2] = dt_tensor * (1.0 - alpha)
        A[:, 1, 3] = dt_tensor * (1.0 - alpha)

        # Velocity derivatives w.r.t velocity state
        A[:, 2, 2] = 1.0 - alpha
        A[:, 3, 3] = 1.0 - alpha

        # Velocity derivatives w.r.t actions
        B[:, 2, 0] = alpha * vel_mask[:, 0]
        B[:, 3, 1] = alpha * vel_mask[:, 1]

        # Position derivatives w.r.t actions (through velocity)
        B[:, 0, 0] = dt_tensor * alpha * vel_mask[:, 0]
        B[:, 1, 1] = dt_tensor * alpha * vel_mask[:, 1]

        if batch == 1:
            return A.squeeze(0), B.squeeze(0)
        return A, B

    return f_dyn_torch, f_dyn_jac_torch


__all__ = [
    "WaypointEnvConfig",
    "DoubleIntegratorWaypointEnvV2",
    "build_velocity_dynamics",
]
