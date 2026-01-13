"""Waypoint-tracking planar SE(2) kinematic model with velocity commands.

The environment models a planar robot with pose ``[x, y, theta]`` and control
inputs interpreted as body-frame velocities ``[v_s, v_l, omega]``:

    dx = v_s * cos(theta) - v_l * sin(theta)
    dy = v_s * sin(theta) + v_l * cos(theta)
    dtheta = omega

The commanded velocities are clamped to ``[-max_speed, max_speed]`` and the
agent is rewarded for progress towards a moving waypoint, with optional
economic-style penalties on control effort.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


@dataclass
class CircleObstacle:
    """Simple circular obstacle in world-frame coordinates."""

    center: np.ndarray  # shape (2,)
    radius: float


@dataclass
class SE2WaypointEnvConfig:
    """Hyper-parameters for the SE(2) waypoint environment."""

    dt: float = 0.05
    episode_len: int = 400
    waypoint_range: float = 1.0
    goal_radius: float = 0.15
    min_start_radius: float = 0.2
    max_speed: float = 3.0
    progress_gain: float = 15.0
    action_penalty: float = 0.05
    living_penalty: float = 0.01
    goal_bonus: float = 5.0
    control_gain: float = 0.0


class SE2WaypointEnv(gym.Env):
    """Gymnasium environment exposing the SE(2) kinematic waypoint task."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        config: SE2WaypointEnvConfig | None = None,
        dt: float = 0.05,
        episode_len: int = 400,
        waypoint_range: float = 1.0,
        goal_radius: float = 0.15,
        min_start_radius: float = 0.2,
        max_speed: float = 3.0,
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
            progress_gain = config.progress_gain
            action_penalty = config.action_penalty
            living_penalty = config.living_penalty
            goal_bonus = config.goal_bonus
            control_gain = config.control_gain

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
        self.progress_gain = float(progress_gain)
        self.action_penalty = float(action_penalty)
        self.living_penalty = float(living_penalty)
        self.goal_bonus = float(goal_bonus)
        self.control_gain = float(control_gain)
        self.env_id = int(env_id)

        # State: [x, y, theta]
        self.nx = 3
        # Action: [v_s, v_l, omega]
        self.nu = 3

        obs_dim = self.nx + 2  # [state, waypoint_xy]
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

        # Sample initial pose: position away from origin, random heading.
        radius = float(self._np_random.uniform(self.min_start_radius, self.waypoint_range))
        angle = float(self._np_random.uniform(0.0, 2.0 * np.pi))
        x0 = radius * np.cos(angle)
        y0 = radius * np.sin(angle)
        theta0 = float(self._np_random.uniform(-np.pi, np.pi))
        self.state = np.array([x0, y0, theta0], dtype=np.float32)

        self.target_waypoint = self._sample_waypoint()
        self.prev_distance = self._distance_to_waypoint(self.state[:2], self.target_waypoint)

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

        v_s, v_l, omega = clipped_action
        x, y, theta = self.state

        dx = v_s * np.cos(theta) - v_l * np.sin(theta)
        dy = v_s * np.sin(theta) + v_l * np.cos(theta)
        dtheta = omega

        x_next = x + dx * self.dt
        y_next = y + dy * self.dt
        theta_next = theta + dtheta * self.dt

        # Normalise heading to [-pi, pi] for numerical stability.
        theta_next = (theta_next + np.pi) % (2.0 * np.pi) - np.pi

        next_state = np.array([x_next, y_next, theta_next], dtype=np.float32)
        distance = self._distance_to_waypoint(next_state[:2], self.target_waypoint)
        progress = self.prev_distance - distance

        # Economic-style reward: progress towards waypoint plus optional
        # penalty on control magnitude or positive gain when no penalty is set.
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
            distance = self._distance_to_waypoint(next_state[:2], self.target_waypoint)
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
        """Sample a waypoint within the arena."""
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


class SE2WaypointObstacleEnv(gym.Env):
    """SE(2) waypoint environment with static circular obstacles and lidar.

    This variant extends the basic SE2WaypointEnv by adding:
      - a set of circular obstacles placed in world-frame coordinates;
      - a 2D lidar that returns normalised distances to the closest obstacle
        along each beam direction;
      - episode termination when the robot collides with any obstacle.

    The physical state and dynamics remain identical to SE2WaypointEnv:
      state = [x, y, theta]
      action = [v_s, v_l, omega]

    Observations are kept in the same format as SE2WaypointEnv to remain
    compatible with existing wrappers:
      obs = [x, y, theta, target_x, target_y]
    Lidar readings are exposed via the info dict under the key ``\"lidar\"``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        config: SE2WaypointEnvConfig | None = None,
        dt: float = 0.05,
        episode_len: int = 400,
        waypoint_range: float = 1.0,
        goal_radius: float = 0.15,
        min_start_radius: float = 0.2,
        max_speed: float = 3.0,
        progress_gain: float = 15.0,
        action_penalty: float = 0.05,
        living_penalty: float = 0.01,
        goal_bonus: float = 5.0,
        control_gain: float = 0.0,
        env_id: int = 0,
        # Obstacle configuration
        enable_obstacles: bool = True,
        num_obstacles: int = 3,
        obstacle_radius_min: float = 0.3,
        obstacle_radius_max: float = 0.7,
        arena_radius: float = 4.0,
        # Lidar configuration
        enable_lidar: bool = True,
        lidar_num_beams: int = 36,
        lidar_fov: float = 2.0 * np.pi,
        lidar_max_range: float = 4.0,
        # Collision handling
        collision_penalty: float = 25.0,
        proximity_gain: float = 5.0,
        proximity_margin: float = 0.5,
    ) -> None:
        super().__init__()

        if config is not None:
            dt = config.dt
            episode_len = config.episode_len
            waypoint_range = config.waypoint_range
            goal_radius = config.goal_radius
            min_start_radius = config.min_start_radius
            max_speed = config.max_speed
            progress_gain = config.progress_gain
            action_penalty = config.action_penalty
            living_penalty = config.living_penalty
            goal_bonus = config.goal_bonus
            control_gain = config.control_gain

        if waypoint_range <= 0.0:
            raise ValueError("waypoint_range must be positive.")
        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive.")
        if obstacle_radius_min <= 0.0 or obstacle_radius_max <= 0.0:
            raise ValueError("Obstacle radii must be positive.")
        if obstacle_radius_max < obstacle_radius_min:
            raise ValueError("obstacle_radius_max must be >= obstacle_radius_min.")
        if arena_radius <= 0.0:
            raise ValueError("arena_radius must be positive.")
        if enable_lidar and lidar_num_beams <= 0:
            raise ValueError("lidar_num_beams must be positive when enable_lidar is True.")
        if enable_lidar and lidar_max_range <= 0.0:
            raise ValueError("lidar_max_range must be positive when enable_lidar is True.")
        if proximity_margin <= 0.0:
            raise ValueError("proximity_margin must be positive.")
        if proximity_gain < 0.0:
            raise ValueError("proximity_gain must be non-negative.")

        self.dt = float(dt)
        self.episode_len = int(episode_len)
        self.waypoint_range = float(waypoint_range)
        self.goal_radius = float(goal_radius)
        self.min_start_radius = float(min_start_radius)
        self.max_speed = float(max_speed)
        self.progress_gain = float(progress_gain)
        self.action_penalty = float(action_penalty)
        self.living_penalty = float(living_penalty)
        self.goal_bonus = float(goal_bonus)
        self.control_gain = float(control_gain)
        self.env_id = int(env_id)

        # State: [x, y, theta]
        self.nx = 3
        # Action: [v_s, v_l, omega]
        self.nu = 3

        obs_dim = self.nx + 2  # [state, waypoint_xy]
        high = np.full(obs_dim, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(self.nu, dtype=np.float32) * self.max_speed,
            high=np.ones(self.nu, dtype=np.float32) * self.max_speed,
            dtype=np.float32,
        )

        # Obstacle configuration
        self.enable_obstacles = bool(enable_obstacles)
        self.num_obstacles = int(num_obstacles)
        self.obstacle_radius_min = float(obstacle_radius_min)
        self.obstacle_radius_max = float(obstacle_radius_max)
        self.arena_radius = float(arena_radius)
        self.obstacles: List[CircleObstacle] = []

        # Lidar configuration
        self.enable_lidar = bool(enable_lidar)
        self.lidar_num_beams = int(lidar_num_beams)
        self.lidar_fov = float(lidar_fov)
        self.lidar_max_range = float(lidar_max_range)

        # Collision / reward
        self.collision_penalty = float(collision_penalty)
        self.proximity_gain = float(proximity_gain)
        self.proximity_margin = float(proximity_margin)
        self.collision_occurred = False

        self.state = np.zeros(self.nx, dtype=np.float32)
        self.target_waypoint = np.zeros(2, dtype=np.float32)
        self.prev_distance = 0.0
        self.steps = 0
        self.waypoints_reached = 0
        self._np_random = np.random.default_rng()

    # Gymnasium API -----------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        del options  # unused, kept for API compatibility

        if seed is not None:
            base_seed = (seed + self.env_id) % (2**32 - 1)
            self._np_random = np.random.default_rng(base_seed)

        self.steps = 0
        self.waypoints_reached = 0
        self.collision_occurred = False

        self._generate_obstacles()

        # Sample initial pose away from obstacles.
        for _ in range(512):
            max_r = max(self.min_start_radius, min(self.waypoint_range, self.arena_radius - 0.05))
            radius = float(self._np_random.uniform(self.min_start_radius, max_r))
            angle = float(self._np_random.uniform(0.0, 2.0 * np.pi))
            x0 = radius * np.cos(angle)
            y0 = radius * np.sin(angle)
            theta0 = float(self._np_random.uniform(-np.pi, np.pi))
            candidate = np.array([x0, y0, theta0], dtype=np.float32)
            if not self._point_in_obstacle(candidate[:2]):
                self.state = candidate
                break
        else:
            raise RuntimeError("Failed to sample collision-free initial state.")

        self.target_waypoint = self._sample_waypoint()
        self.prev_distance = self._distance_to_waypoint(self.state[:2], self.target_waypoint)

        observation = self._get_obs()
        lidar = self._compute_lidar_distances()
        info = {
            "target_waypoint": self.target_waypoint.copy(),
            "waypoints_reached": self.waypoints_reached,
            "distance_to_waypoint": self.prev_distance,
            "collision": False,
            "lidar": lidar.copy(),
        }
        return observation, info

    def step(self, action: np.ndarray):
        self.steps += 1
        action = np.asarray(action, dtype=np.float32)
        clipped_action = np.clip(action, -self.max_speed, self.max_speed)

        v_s, v_l, omega = clipped_action
        x, y, theta = self.state

        dx = v_s * np.cos(theta) - v_l * np.sin(theta)
        dy = v_s * np.sin(theta) + v_l * np.cos(theta)
        dtheta = omega

        x_next = x + dx * self.dt
        y_next = y + dy * self.dt
        theta_next = theta + dtheta * self.dt

        # Normalise heading to [-pi, pi] for numerical stability.
        theta_next = (theta_next + np.pi) % (2.0 * np.pi) - np.pi

        next_state = np.array([x_next, y_next, theta_next], dtype=np.float32)
        distance = self._distance_to_waypoint(next_state[:2], self.target_waypoint)
        progress = self.prev_distance - distance

        # Base waypoint tracking reward.
        reward = self.progress_gain * progress
        control_mag = float(np.linalg.norm(clipped_action))
        if self.action_penalty > 0.0:
            reward -= self.action_penalty * control_mag
        elif self.control_gain > 0.0:
            reward += self.control_gain * control_mag

        # Soft penalty when approaching obstacles (dominant close to collision).
        if self.enable_obstacles and self.proximity_gain > 0.0 and self.obstacles:
            dist = self._min_distance_to_obstacles(next_state[:2])
            if dist < self.proximity_margin:
                proximity_term = (self.proximity_margin - dist) / self.proximity_margin
                reward -= self.proximity_gain * (proximity_term ** 2)
        else:
            dist = float("inf")

        collision = self._point_in_obstacle(next_state[:2]) if self.enable_obstacles else False
        if collision:
            reward -= self.collision_penalty
            self.collision_occurred = True

        waypoint_reached = (distance <= self.goal_radius) and not collision
        if waypoint_reached:
            reward += self.goal_bonus
            self.waypoints_reached += 1
            self.target_waypoint = self._sample_waypoint()
            distance = self._distance_to_waypoint(next_state[:2], self.target_waypoint)

        self.prev_distance = distance
        self.state = next_state

        truncated = self.steps >= self.episode_len
        terminated = bool(collision)

        observation = self._get_obs()
        lidar = self._compute_lidar_distances()
        info = {
            "target_waypoint": self.target_waypoint.copy(),
            "waypoints_reached": self.waypoints_reached,
            "distance_to_waypoint": distance,
            "action_clipped": float(np.any(action != clipped_action)),
            "collision": collision,
            "lidar": lidar.copy(),
            "min_obstacle_distance": dist,
            "obstacles": [
                {"center": obs.center.tolist(), "radius": float(obs.radius)} for obs in self.obstacles
            ],
        }
        return observation, float(reward), terminated, truncated, info

    # Helpers -----------------------------------------------------------
    def _sample_waypoint(self) -> np.ndarray:
        """Sample a waypoint within the arena, avoiding the inner exclusion radius and obstacles."""
        max_radius = min(self.waypoint_range, self.arena_radius - 0.05)
        if max_radius <= self.min_start_radius:
            raise RuntimeError("waypoint_range too small for the arena radius.")
        for _ in range(512):
            r = float(self._np_random.uniform(self.min_start_radius, max_radius))
            angle = float(self._np_random.uniform(0.0, 2.0 * np.pi))
            waypoint = np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
            if self.enable_obstacles and self._point_in_obstacle(waypoint):
                continue
            return waypoint
        raise RuntimeError("Failed to sample waypoint outside exclusion radius and obstacles.")

    def _distance_to_waypoint(self, position: np.ndarray, waypoint: np.ndarray) -> float:
        return float(np.linalg.norm(position - waypoint))

    def _get_obs(self) -> np.ndarray:
        return np.concatenate((self.state, self.target_waypoint)).astype(np.float32)

    def save_map_png(self, path: str, extra_waypoints: int = 0) -> None:
        """Save a PNG snapshot of the arena with obstacles, robot and waypoint."""
        if getattr(self, "_plt_unavailable", False):
            self._save_map_with_pil(path, extra_waypoints)
            return
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect("equal")
            extent = self.arena_radius
            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
            ax.set_title(f"SE2 arena with obstacles (env {self.env_id})")

            # Arena boundary
            arena_circle = plt.Circle(
                (0.0, 0.0),
                self.arena_radius,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(arena_circle)

            # Obstacles
            for obs in self.obstacles:
                circle = plt.Circle(
                    obs.center,
                    obs.radius,
                    facecolor="grey",
                    alpha=0.55,
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax.add_patch(circle)

            # Robot pose
            x, y, theta = self.state
            ax.plot(x, y, marker="o", color="green", label="Robot")
            heading = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            ax.arrow(
                x,
                y,
                0.5 * heading[0],
                0.5 * heading[1],
                head_width=0.1,
                head_length=0.2,
                fc="green",
                ec="green",
            )

            # Current waypoint
            ax.plot(
                self.target_waypoint[0],
                self.target_waypoint[1],
                marker="*",
                color="red",
                markersize=14,
                label="Current waypoint",
            )

            # Optional extra sampled waypoints (for map visualisation only)
            sampled: List[np.ndarray] = []
            if extra_waypoints > 0:
                rng_state = self._np_random.bit_generator.state
                try:
                    for _ in range(extra_waypoints):
                        wp = self._sample_waypoint()
                        sampled.append(wp)
                except RuntimeError:
                    pass
                self._np_random.bit_generator.state = rng_state

            if sampled:
                stacked = np.stack(sampled)
                ax.scatter(stacked[:, 0], stacked[:, 1], marker="x", color="blue", label="Preview waypoint")

            # Optional beam lines
            self._plot_lidar_lines(ax)

            ax.legend(loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.4)
            self._annotate_lidar_text(ax)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        except (ImportError, RuntimeError):
            self._plt_unavailable = True
            self._save_map_with_pil(path, extra_waypoints)


    def _plot_lidar_lines(self, ax) -> None:
        if not (self.enable_lidar and self.lidar_num_beams > 0):
            return
        lidar = self._compute_lidar_distances()
        origin = self.state[:2]
        max_range = self.lidar_max_range
        for i in range(self.lidar_num_beams):
            vx = float(lidar[3 * i + 0])
            vy = float(lidar[3 * i + 1])
            d_norm = float(lidar[3 * i + 2])
            direction = np.array([vx, vy], dtype=np.float32)
            dir_norm = float(np.linalg.norm(direction)) or 1.0
            direction = direction / dir_norm
            d = d_norm * max_range
            end = origin + d * direction
            # Fade long beams and emphasise close returns for quicker visual inspection.
            alpha = 0.15 + 0.65 * (1.0 - d_norm)
            color = (1.0, 0.55, 0.0, alpha)
            ax.plot([origin[0], end[0]], [origin[1], end[1]], color=color, linewidth=1.5)

    def _annotate_lidar_text(self, ax) -> None:
        lines = self._lidar_info_lines()
        ax.text(
            1.02,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            family="monospace",
        )

    def _lidar_info_lines(self, max_beams: int = 6) -> List[str]:
        lidar = self._compute_lidar_distances()
        heading_deg = math.degrees(float(self.state[2]))
        lines = [f"heading {heading_deg:+6.1f}°"]
        for i in range(min(self.lidar_num_beams, max_beams)):
            cos_v = float(lidar[3 * i + 0])
            sin_v = float(lidar[3 * i + 1])
            d_norm = float(lidar[3 * i + 2])
            angle_deg = math.degrees(math.atan2(sin_v, cos_v))
            distance = d_norm * self.lidar_max_range
            lines.append(f"beam {i:02d}: {angle_deg:+6.1f}° {distance:5.2f}m")
        if self.lidar_num_beams > max_beams:
            lines.append("...")
        if self.lidar_num_beams > 0:
            distances = lidar[2::3] * self.lidar_max_range
            closest_idx = int(np.argmin(distances))
            lines.append(
                f"closest beam {closest_idx:02d}: {distances[closest_idx]:5.2f}m"
                f" (norm {float(lidar[2 + 3 * closest_idx]):.3f})"
            )
        return lines

    def _save_map_with_pil(self, path: str, extra_waypoints: int = 0) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("Matplotlib/PIL unavailable: map skipped.")
            return

        size = 600
        scale = size / (2 * self.arena_radius)
        canvas = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(canvas)

        def world_to_pixel(point):
            return (
                int((point[0] + self.arena_radius) * scale),
                int((self.arena_radius - point[1]) * scale),
            )

        # Arena border
        bbox = [
            0,
            0,
            size - 1,
            size - 1,
        ]
        draw.ellipse(bbox, outline="black", width=2)

        # Obstacles
        for obs in self.obstacles:
            center_px = world_to_pixel(obs.center)
            r_px = int(obs.radius * scale)
            draw.ellipse(
                [
                    center_px[0] - r_px,
                    center_px[1] - r_px,
                    center_px[0] + r_px,
                    center_px[1] + r_px,
                ],
                fill="grey",
                outline="black",
            )

        # Robot
        origin_px = world_to_pixel(self.state[:2])
        draw.ellipse(
            [
                origin_px[0] - 6,
                origin_px[1] - 6,
                origin_px[0] + 6,
                origin_px[1] + 6,
            ],
            fill="green",
        )
        theta = float(self.state[2])
        heading = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        heading_end = self.state[:2] + 0.6 * heading
        draw.line([origin_px, world_to_pixel(heading_end)], fill="darkgreen", width=3)

        # Waypoint
        wp_px = world_to_pixel(self.target_waypoint)
        draw.ellipse(
            [
                wp_px[0] - 8,
                wp_px[1] - 8,
                wp_px[0] + 8,
                wp_px[1] + 8,
            ],
            fill="red",
        )

        # Extra waypoints
        sampled: List[np.ndarray] = []
        if extra_waypoints > 0:
            rng_state = self._np_random.bit_generator.state
            try:
                for _ in range(extra_waypoints):
                    sampled.append(self._sample_waypoint())
            except RuntimeError:
                pass
            self._np_random.bit_generator.state = rng_state
        for wp in sampled:
            wp_px = world_to_pixel(wp)
            draw.ellipse(
                [
                    wp_px[0] - 4,
                    wp_px[1] - 4,
                    wp_px[0] + 4,
                    wp_px[1] + 4,
                ],
                fill="blue",
            )

        # Lidar beams
        if self.enable_lidar and self.lidar_num_beams > 0:
            lidar = self._compute_lidar_distances()
            for i in range(self.lidar_num_beams):
                vx = float(lidar[3 * i + 0])
                vy = float(lidar[3 * i + 1])
                d_norm = float(lidar[3 * i + 2])
                direction = np.array([vx, vy], dtype=np.float32)
                dir_norm = float(np.linalg.norm(direction)) or 1.0
                direction = direction / dir_norm
                d = d_norm * self.lidar_max_range
                end = world_to_pixel(self.state[:2] + d * direction)
                draw.line([origin_px, end], fill="orange", width=1)

        canvas.save(path, format="PNG")
        self._draw_pil_text(canvas)

    def _draw_pil_text(self, canvas) -> None:
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            return
        font = ImageFont.load_default()
        draw = ImageDraw.Draw(canvas)
        lines = self._lidar_info_lines(max_beams=6)
        text = "\n".join(lines)
        margin = 10
        draw.text(
            (margin, margin),
            text,
            fill="black",
            font=font,
        )

    # Obstacle utilities -----------------------------------------------
    def _generate_obstacles(self) -> None:
        self.obstacles = []
        if not self.enable_obstacles or self.num_obstacles <= 0:
            return

        attempts = 0
        max_attempts = self.num_obstacles * 20
        while len(self.obstacles) < self.num_obstacles and attempts < max_attempts:
            attempts += 1
            radius = float(self._np_random.uniform(self.obstacle_radius_min, self.obstacle_radius_max))
            # Sample centre in a disc of radius (arena_radius - radius)
            r = float(self._np_random.uniform(0.0, max(self.arena_radius - radius, 0.1)))
            angle = float(self._np_random.uniform(0.0, 2.0 * np.pi))
            cx = r * np.cos(angle)
            cy = r * np.sin(angle)
            center = np.array([cx, cy], dtype=np.float32)

            # Keep obstacles away from origin to ease initial placement.
            if np.linalg.norm(center) < self.min_start_radius + radius:
                continue

            # Simple non-overlap constraint.
            too_close = False
            for obs in self.obstacles:
                if np.linalg.norm(center - obs.center) < (radius + obs.radius + 0.1):
                    too_close = True
                    break
            if too_close:
                continue

            self.obstacles.append(CircleObstacle(center=center, radius=radius))

    def _point_in_obstacle(self, position: np.ndarray) -> bool:
        if not self.enable_obstacles:
            return False
        for obs in self.obstacles:
            if float(np.linalg.norm(position - obs.center)) <= obs.radius:
                return True
        return False

    def _min_distance_to_obstacles(self, position: np.ndarray) -> float:
        if not (self.enable_obstacles and self.obstacles):
            return float("inf")
        distances = [float(np.linalg.norm(position - obs.center) - obs.radius) for obs in self.obstacles]
        return min(distances)

    # Lidar utilities --------------------------------------------------
    def _compute_lidar_distances(self) -> np.ndarray:
        """Return lidar features for each beam as [cos(angle_world), sin(angle_world), dist_norm]."""
        if not self.enable_lidar or self.lidar_num_beams <= 0:
            return np.zeros(0, dtype=np.float32)

        theta = float(self.state[2])
        origin = self.state[:2].astype(np.float32)
        max_range = float(self.lidar_max_range)
        rel_angles = np.linspace(
            -0.5 * self.lidar_fov,
            0.5 * self.lidar_fov,
            self.lidar_num_beams,
            dtype=np.float32,
        )
        distances = np.full(self.lidar_num_beams, max_range, dtype=np.float32)

        obstacles: List[CircleObstacle] = self.obstacles if self.enable_obstacles else []
        for i, rel in enumerate(rel_angles):
            direction = np.array(
                [np.cos(theta + float(rel)), np.sin(theta + float(rel))],
                dtype=np.float32,
            )
            best = max_range
            for obs in obstacles:
                d = self._ray_circle_intersection(origin, direction, obs.center, obs.radius, max_range)
                if d is not None:
                    best = min(best, d)
            distances[i] = best

        # Build per-beam features: [cos(angle_world), sin(angle_world), distance_norm]
        features = np.zeros(3 * self.lidar_num_beams, dtype=np.float32)
        for i, rel in enumerate(rel_angles):
            angle_world = theta + float(rel)
            d_norm = float(np.clip(distances[i] / max_range, 0.0, 1.0))
            features[3 * i + 0] = np.cos(angle_world)
            features[3 * i + 1] = np.sin(angle_world)
            features[3 * i + 2] = d_norm
        return features

    @staticmethod
    def _ray_circle_intersection(
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
        max_range: float,
    ) -> Optional[float]:
        """Return the smallest non-negative intersection distance along the ray, or None."""
        # Ensure direction is treated as unit-length; small deviations are acceptable.
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm <= 1e-6:
            return None
        d = direction / dir_norm
        m = origin - center
        b = float(np.dot(m, d))
        c = float(np.dot(m, m) - radius * radius)
        disc = b * b - c
        if disc < 0.0:
            return None
        sqrt_disc = float(np.sqrt(disc))
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc
        hits = [t for t in (t1, t2) if t >= 0.0]
        if not hits:
            return None
        t_hit = min(hits)
        if t_hit > max_range:
            return None
        return float(t_hit)


# -------------------------------------------------------------------------
# Torch dynamics helpers
# -------------------------------------------------------------------------
def _prepare_inputs_se2(
    x: torch.Tensor,
    u: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensure batched inputs with correct shapes for SE(2) dynamics."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if u.ndim == 1:
        u = u.unsqueeze(0)
    if x.shape[0] != u.shape[0]:
        raise ValueError("x and u batch dimensions must match.")
    if x.shape[-1] != 3:
        raise ValueError("Expected state dimension 3.")
    if u.shape[-1] != 3:
        raise ValueError("Expected action dimension 3.")
    return x, u


def build_se2_kinematic_dynamics(
    *,
    max_speed: float = 3.0,
) -> Tuple[
    Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor]],
]:
    """Factory returning (f_dyn, f_dyn_jac) closures for the SE(2) kinematics."""

    max_speed = float(max_speed)

    def f_dyn_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        x, u = _prepare_inputs_se2(x, u)
        dt_tensor = torch.as_tensor(dt, dtype=x.dtype, device=x.device)

        clipped = torch.clamp(u, -max_speed, max_speed)
        v_s = clipped[..., 0]
        v_l = clipped[..., 1]
        omega = clipped[..., 2]

        x_pos = x[..., 0]
        y_pos = x[..., 1]
        theta = x[..., 2]

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        dx = v_s * cos_t - v_l * sin_t
        dy = v_s * sin_t + v_l * cos_t
        dtheta = omega

        x_next = x_pos + dx * dt_tensor
        y_next = y_pos + dy * dt_tensor
        theta_next = theta + dtheta * dt_tensor

        # Normalise heading to [-pi, pi] for numerical stability.
        pi = torch.pi
        theta_next = (theta_next + pi) % (2.0 * pi) - pi

        next_state = torch.stack((x_next, y_next, theta_next), dim=-1)
        if next_state.shape[0] == 1:
            return next_state.squeeze(0)
        return next_state

    def f_dyn_jac_torch(
        x: torch.Tensor,
        u: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, u = _prepare_inputs_se2(x, u)
        batch = x.shape[0]
        dtype, device = x.dtype, x.device
        dt_tensor = torch.as_tensor(dt, dtype=dtype, device=device)

        clipped = torch.clamp(u, -max_speed, max_speed)
        v_s = clipped[..., 0]
        v_l = clipped[..., 1]
        omega = clipped[..., 2]

        theta = x[..., 2]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Mask for gradients through clamping
        ctrl_mask = (u.abs() < (max_speed - 1e-5)).to(dtype=dtype)
        mask_vs = ctrl_mask[..., 0]
        mask_vl = ctrl_mask[..., 1]
        mask_omega = ctrl_mask[..., 2]

        A = torch.zeros(batch, 3, 3, dtype=dtype, device=device)
        B = torch.zeros(batch, 3, 3, dtype=dtype, device=device)

        # Identity contributions
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 1.0

        # Derivatives w.r.t. theta
        A[:, 0, 2] = dt_tensor * (-v_s * sin_t - v_l * cos_t)
        A[:, 1, 2] = dt_tensor * (v_s * cos_t - v_l * sin_t)

        # Derivatives w.r.t. controls
        B[:, 0, 0] = dt_tensor * cos_t * mask_vs
        B[:, 0, 1] = -dt_tensor * sin_t * mask_vl
        B[:, 1, 0] = dt_tensor * sin_t * mask_vs
        B[:, 1, 1] = dt_tensor * cos_t * mask_vl
        B[:, 2, 2] = dt_tensor * mask_omega

        if batch == 1:
            return A.squeeze(0), B.squeeze(0)
        return A, B

    return f_dyn_torch, f_dyn_jac_torch


__all__ = [
    "SE2WaypointEnvConfig",
    "SE2WaypointEnv",
    "SE2WaypointObstacleEnv",
    "CircleObstacle",
    "build_se2_kinematic_dynamics",
]
