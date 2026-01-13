"""Train ACMPC on a SE(2) kinematic waypoint task with obstacles and lidar.

Questo esempio estende il setup SE(2) esistente introducendo:
  - ostacoli statici circolari in frame mondo;
  - un sensore lidar 2D che misura distanze normalizzate agli ostacoli;
  - terminazione dell'episodio alla collisione;
  - utilizzo del lidar da parte dell'actor (via include_lidar).

La pipeline usa le nuove API ACMPC (ActorCriticAgent, TrainingLoop, RolloutCollector).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import gymnasium as gym

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ACMPC import (  # noqa: E402
    AbsoluteEnvWrapper,
    ActorCriticAgent,
    TrainingLoop,
    VectorEnvManager,
    load_experiment_config,
)
from ACMPC.training.normalization import ObservationNormalizer  # noqa: E402
from ACMPC.envs import (  # noqa: E402
    SE2WaypointObstacleEnv,
    build_se2_kinematic_dynamics,
)
from ACMPC.experiment_config import ExperimentConfig  # noqa: E402
from ACMPC.sampling import RolloutBatch, RolloutCollector  # noqa: E402
from ACMPC.training.checkpoint import CheckpointConfig, CheckpointManager  # noqa: E402
from examples.se2_kinematic_AC_DRL import extract_cost_diagonals  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "double_integrator_waypoint.yaml"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "se2_kinematic_obstacles"


@dataclass
class DimensionSpec:
    state_dim: int
    action_dim: int
    waypoint_dim: int = 2


@dataclass
class EvalEpisodeLog:
    positions: np.ndarray  # [steps+1, 2]
    headings: np.ndarray  # [steps+1]
    target_history: np.ndarray  # [steps+1, 2]
    visited_waypoints: np.ndarray  # [num_reached, 2]
    rewards: np.ndarray  # [steps]
    actions: np.ndarray  # [steps, action_dim]
    distances: np.ndarray  # [steps] distance to current target at each step
    obstacle_distances: np.ndarray  # [steps] distance to closest obstacle (<= margin triggers penalty)
    q_history: np.ndarray  # [steps, state_dim]
    r_history: np.ndarray  # [steps, action_dim]
    q_terminal: np.ndarray  # [state_dim]
    r_terminal: np.ndarray  # [action_dim]
    dt: float
    goal_radius: float
    obstacles: List[Tuple[np.ndarray, float]]
    total_reward: float
    steps: int
    waypoints_reached: int
    collision: bool
    collision_step: Optional[int]
    lidar_min: float
    lidar_max: float
    goal_radius: float
    arena_radius: float


class StateObservationObstacleAdapter(gym.Env):
    """Expose only the physical state [x, y, theta] to the policy.

    Wrappa SE2WaypointObstacleEnv mantenendo l'osservazione ridotta allo stato
    (usato da MPC e Transformer), mentre waypoint e lidar sono esposti tramite
    il dict info.
    """

    metadata = {"render_modes": []}

    def __init__(self, **env_kwargs):
        super().__init__()
        self._base_env = SE2WaypointObstacleEnv(**env_kwargs)
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
        # Ricostruisci sempre il waypoint a partire dall'osservazione completa.
        data["target_waypoint"] = np.asarray(
            obs[self.nx : self.nx + 2],
            dtype=np.float32,
        ).copy()
        return data


def waypoint_from_info(_obs: torch.Tensor, info: Dict) -> torch.Tensor:
    waypoint = info.get("target_waypoint")
    if waypoint is None:
        raise KeyError("Environment info missing 'target_waypoint'.")
    tensor = torch.as_tensor(waypoint, dtype=_obs.dtype, device=_obs.device)
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
            adapter = StateObservationObstacleAdapter(**kwargs)
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
    env = SE2WaypointObstacleEnv(**env_kwargs)
    dims = DimensionSpec(state_dim=int(env.nx), action_dim=int(env.nu))
    env.close()
    return dims


def prepare_config_with_lidar(
    config: ExperimentConfig,
    *,
    dims: DimensionSpec,
    history_window: int,
    rollout_len: int,
    mpc_horizon: int,
    dt: float,
    action_limit: float,
    device: str,
    lidar_dim: int,
) -> None:
    model = config.model
    actor_cfg = model.actor
    critic_cfg = model.critic

    if actor_cfg.cost_map is None:
        from ACMPC.model_config import CostMapConfig  # lazy import to avoid cycles

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

    # Lidar abilitato per actor e critic.
    model.include_prev_action = False
    model.include_lidar = False  # usato solo per validazione globale

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
    actor_cfg.use_waypoint_as_ref = True  # tracking-style MPC like se2_kinematic_tracking_AC_DRL
    actor_cfg.include_prev_action = False
    actor_cfg.include_lidar = True
    actor_cfg.lidar_dim = int(lidar_dim)

    critic_cfg.input_dim = dims.state_dim
    critic_cfg.waypoint_dim = dims.waypoint_dim
    critic_cfg.waypoint_sequence_len = 1
    critic_cfg.include_prev_action = False
    critic_cfg.include_lidar = True
    critic_cfg.lidar_dim = int(lidar_dim)

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


def _rollout_stats(
    batches: Iterable[RolloutBatch],
    *,
    lidar_dim: int,
) -> Dict[str, float]:
    """Aggregate rollout diagnostics for quick logging."""
    total_reward = 0.0
    total_steps = 0
    total_done = 0
    total_waypoints = 0
    total_collisions = 0
    lidar_min = float("inf")
    lidar_max = 0.0
    lidar_ok = True

    for batch in batches:
        reward = batch.reward
        total_reward += float(reward.sum().item())
        total_steps += int(reward.numel())
        total_done += int(batch.done.sum().item())

        # Track waypoints and collisions per-episode using done flags.
        if batch.info:
            num_envs = batch.num_envs
            horizon = batch.horizon
            last_wp = [0 for _ in range(num_envs)]
            collided = [False for _ in range(num_envs)]
            for t in range(horizon):
                infos_t = batch.info[t] if t < len(batch.info) else [{} for _ in range(num_envs)]
                for env_idx in range(num_envs):
                    info = infos_t[env_idx] if env_idx < len(infos_t) else {}
                    wp = int(info.get("waypoints_reached", last_wp[env_idx]))
                    if wp > last_wp[env_idx]:
                        total_waypoints += wp - last_wp[env_idx]
                    last_wp[env_idx] = wp
                    if (info.get("collision") or info.get("terminal_collision")) and not collided[env_idx]:
                        collided[env_idx] = True
                # If episode ended at this t for an env, account collision once.
                for env_idx in range(num_envs):
                    if bool(batch.done[env_idx, t]):
                        if collided[env_idx]:
                            total_collisions += 1
                        collided[env_idx] = False
                        last_wp[env_idx] = 0

        if batch.lidar is not None and batch.lidar.numel() > 0 and lidar_dim > 0:
            dist_norm = batch.lidar[..., 2::3]
            lidar_min = min(lidar_min, float(dist_norm.min().item()))
            lidar_max = max(lidar_max, float(dist_norm.max().item()))
            if torch.logical_or(dist_norm < -1e-3, dist_norm > 1.0 + 1e-3).any():
                lidar_ok = False

    return dict(
        reward_per_step=total_reward / max(1, total_steps),
        waypoints=total_waypoints,
        collisions=total_collisions,
        episodes=total_done,
        lidar_min=lidar_min,
        lidar_max=lidar_max,
        lidar_ok=lidar_ok,
    )


@torch.no_grad()
def run_eval_episode(
    agent: ActorCriticAgent,
    env: SE2WaypointObstacleEnv,
    *,
    history_window: int,
    device: str,
    lidar_dim: int,
    seed: int,
    obs_normalizer: Optional[ObservationNormalizer] = None,
    waypoint_normalizer: Optional[ObservationNormalizer] = None,
    lidar_normalizer: Optional[ObservationNormalizer] = None,
) -> EvalEpisodeLog:
    obs, info = env.reset(seed=seed)
    dtype = torch.float32
    state = torch.as_tensor(obs[: env.nx], dtype=dtype, device=device)
    history = torch.zeros(history_window, env.nx, dtype=dtype, device=device)
    history[-1] = state
    waypoint = torch.as_tensor(info["target_waypoint"], dtype=dtype, device=device).view(1, -1)
    lidar = info.get("lidar")
    memories = agent.init_state(batch_size=1)

    positions: List[np.ndarray] = [state[:2].cpu().numpy()]
    headings: List[float] = [float(state[2].cpu().item())]
    target_hist: List[np.ndarray] = [info["target_waypoint"].copy()]
    visited: List[np.ndarray] = []
    rewards: List[float] = []
    actions: List[np.ndarray] = []
    distances: List[float] = []
    obstacle_distances: List[float] = []
    lidar_min = float("inf")
    lidar_max = 0.0
    collision_step: Optional[int] = None
    prev_wp_count = int(info.get("waypoints_reached", 0))
    current_target = info["target_waypoint"].copy()
    q_history: List[np.ndarray] = []
    r_history: List[np.ndarray] = []
    last_q_final: Optional[np.ndarray] = None
    last_r_final: Optional[np.ndarray] = None

    for step in range(env.episode_len):
        lidar_tensor = None
        if lidar is not None:
            lidar_tensor = torch.as_tensor(lidar, dtype=dtype, device=device).view(1, -1)
            if lidar_tensor.numel() != lidar_dim:
                raise RuntimeError(
                    f"Eval lidar dim mismatch: env returned {lidar_tensor.numel()} expected {lidar_dim}."
                )
        norm_history = history.unsqueeze(0)
        norm_state = state.unsqueeze(0)
        norm_waypoint = waypoint.unsqueeze(0)
        norm_lidar = lidar_tensor
        if obs_normalizer is not None:
            obs_normalizer.to(device)
            norm_history = obs_normalizer.normalize(norm_history)
            norm_state = obs_normalizer.normalize(norm_state)
        if waypoint_normalizer is not None:
            waypoint_normalizer.to(device)
            norm_waypoint = waypoint_normalizer.normalize(norm_waypoint)
        if norm_lidar is not None and lidar_normalizer is not None:
            lidar_normalizer.to(device)
            norm_lidar = lidar_normalizer.normalize(norm_lidar)

        action, memories, _ = agent.act(
            norm_history,
            state=norm_state,
            raw_state=state.unsqueeze(0),
            waypoint_seq=norm_waypoint,
            raw_waypoint_seq=waypoint.unsqueeze(0),
            lidar=norm_lidar,
            stochastic=False,
            return_plan=False,
        )
        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        rewards.append(float(reward))
        actions.append(action.squeeze(0).cpu().numpy())
        obstacle_distances.append(float(info.get("min_obstacle_distance", float("inf"))))

        if info.get("collision") and collision_step is None:
            collision_step = step

        new_wp_count = int(info.get("waypoints_reached", prev_wp_count))
        if new_wp_count > prev_wp_count:
            visited.append(current_target.copy())
        prev_wp_count = new_wp_count

        state = torch.as_tensor(obs[: env.nx], dtype=dtype, device=device)
        history = torch.roll(history, shifts=-1, dims=0)
        history[-1] = state
        current_target = np.asarray(info["target_waypoint"], dtype=np.float32).copy()
        waypoint = torch.as_tensor(current_target, dtype=dtype, device=device).view(1, -1)
        lidar = info.get("lidar")

        distances.append(float(np.linalg.norm(state[:2].cpu().numpy() - current_target)))
        positions.append(state[:2].cpu().numpy())
        headings.append(float(state[2].cpu().item()))
        target_hist.append(current_target.copy())

        # Extract MPC cost diagonals to track how the actor adjusts costs.
        q_diag, r_diag, q_final, r_final = extract_cost_diagonals(agent)
        q_history.append(q_diag.astype(np.float32))
        r_history.append(r_diag.astype(np.float32))
        last_q_final = q_final.astype(np.float32)
        last_r_final = r_final.astype(np.float32)

        if lidar is not None and len(lidar) == lidar_dim:
            dist_norm = np.asarray(lidar[2::3], dtype=np.float32)
            lidar_min = min(lidar_min, float(dist_norm.min()))
            lidar_max = max(lidar_max, float(dist_norm.max()))

        if terminated or truncated:
            break

    positions_arr = np.stack(positions)
    headings_arr = np.asarray(headings, dtype=np.float32)
    targets_arr = np.stack(target_hist)
    visited_arr = np.stack(visited) if visited else np.zeros((0, 2), dtype=np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32) if actions else np.zeros((0, env.nu), dtype=np.float32)
    distances_arr = np.asarray(distances, dtype=np.float32) if distances else np.zeros((0,), dtype=np.float32)
    obstacle_dist_arr = (
        np.asarray(obstacle_distances, dtype=np.float32) if obstacle_distances else np.zeros((0,), dtype=np.float32)
    )
    obstacles = [(obs.center.copy(), float(obs.radius)) for obs in env.obstacles]
    if q_history:
        q_hist = np.stack(q_history, axis=0).astype(np.float32)
    else:
        q_hist = np.zeros((0, agent.config.actor.mpc.state_dim), dtype=np.float32)
    if r_history:
        r_hist = np.stack(r_history, axis=0).astype(np.float32)
    else:
        r_hist = np.zeros((0, agent.config.actor.mpc.action_dim), dtype=np.float32)
    q_term = (
        last_q_final.astype(np.float32)
        if last_q_final is not None
        else np.zeros(agent.config.actor.mpc.state_dim, dtype=np.float32)
    )
    r_term = (
        last_r_final.astype(np.float32)
        if last_r_final is not None
        else np.zeros(agent.config.actor.mpc.action_dim, dtype=np.float32)
    )

    return EvalEpisodeLog(
        positions=positions_arr,
        headings=headings_arr,
        target_history=targets_arr,
        visited_waypoints=visited_arr,
        rewards=rewards_arr,
        actions=actions_arr,
        distances=distances_arr,
        obstacle_distances=obstacle_dist_arr,
        q_history=q_hist,
        r_history=r_hist,
        q_terminal=q_term,
        r_terminal=r_term,
        dt=float(env.dt),
        goal_radius=float(env.goal_radius),
        obstacles=obstacles,
        total_reward=float(rewards_arr.sum()),
        steps=len(rewards_arr),
        waypoints_reached=prev_wp_count,
        collision=collision_step is not None,
        collision_step=collision_step,
        lidar_min=lidar_min,
        lidar_max=lidar_max,
        arena_radius=env.arena_radius,
    )


def save_eval_episode_plot(log: EvalEpisodeLog, path: Path) -> None:
    """Save a 3x2 PNG mirroring tracking eval plots with obstacles, waypoints and Q/R."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    dt = log.dt
    time_axis = np.arange(log.positions.shape[0]) * dt
    action_time_axis = np.arange(log.actions.shape[0]) * dt

    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle(
        f"SE2 Obstacles — Episode\nReward: {log.total_reward:.2f} | Waypoints: {log.waypoints_reached}",
        fontsize=14,
        fontweight="bold",
    )

    # 1) Trajectory with obstacles and numbered waypoints
    ax = axes[0, 0]
    ax.plot(log.positions[:, 0], log.positions[:, 1], "g-", linewidth=2.0, label="Path", zorder=3)
    ax.scatter(log.positions[0, 0], log.positions[0, 1], color="blue", marker="o", s=80, label="Start", zorder=4)
    ax.scatter(log.positions[-1, 0], log.positions[-1, 1], color="red", marker="x", s=100, label="End", zorder=4)
    for idx, wp in enumerate(log.visited_waypoints):
        ax.scatter(wp[0], wp[1], color="green", s=120, edgecolors="black", linewidth=1.5, zorder=5)
        ax.text(
            wp[0],
            wp[1],
            str(idx + 1),
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=10,
            zorder=6,
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="green", edgecolor="black", linewidth=1),
        )
    if log.target_history.size > 0:
        tx, ty = log.target_history[-1]
        ax.scatter(tx, ty, color="orange", s=130, edgecolors="black", linewidth=1.2, label="Current target", zorder=4)
    arena_circle = plt.Circle((0.0, 0.0), log.arena_radius, fill=False, edgecolor="black", linewidth=1.2)
    ax.add_patch(arena_circle)
    for center, radius in log.obstacles:
        circle = plt.Circle(center, radius, facecolor="grey", alpha=0.55, edgecolor="black", linewidth=1.0)
        ax.add_patch(circle)
    extent = log.arena_radius
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Trajectory & Waypoints", fontweight="bold")

    # 2) Control commands
    ax = axes[0, 1]
    if log.actions.size > 0:
        ax.plot(action_time_axis, log.actions[:, 0], label="v_s", color="tab:blue")
        ax.plot(action_time_axis, log.actions[:, 1], label="v_l", color="tab:orange")
        ax.plot(action_time_axis, log.actions[:, 2], label="ω", color="tab:green")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("command")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Control Commands", fontweight="bold")

    # 3) Rewards (instant + cumulative)
    ax = axes[1, 0]
    if log.rewards.size > 0:
        ax.plot(action_time_axis, log.rewards, color="tab:orange", label="instant")
        ax.plot(action_time_axis, np.cumsum(log.rewards), color="tab:red", label="cumulative")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Reward Evolution", fontweight="bold")

    # 4) Distance to target (and obstacle distance)
    ax = axes[1, 1]
    if log.distances.size > 0:
        ax.plot(time_axis[: log.distances.size], log.distances, color="tab:purple", label="dist→target")
        ax.axhline(
            y=log.goal_radius,
            color="green",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"goal radius {log.goal_radius:.2f} m",
        )
    if log.obstacle_distances.size > 0:
        ax.plot(action_time_axis, log.obstacle_distances, color="tab:gray", linestyle="--", label="dist→obstacle")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Distance to Target / Obstacles", fontweight="bold")

    # 5) State components over time (x, y, heading rad wrapped)
    ax = axes[2, 0]
    ax.plot(time_axis, log.positions[:, 0], label="x", color="tab:blue")
    ax.plot(time_axis, log.positions[:, 1], label="y", color="tab:orange")
    theta_wrapped = ((log.headings + np.pi) % (2.0 * np.pi)) - np.pi
    ax.plot(time_axis, theta_wrapped, label="theta [rad]", color="tab:green")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("state")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("State Components", fontweight="bold")

    # 6) Control norm and obstacle distance
    ax = axes[2, 1]
    if log.actions.size > 0:
        ctrl_norm = np.linalg.norm(log.actions, axis=1)
        ax.plot(action_time_axis, ctrl_norm, color="tab:blue", label="‖u‖")
    if log.obstacle_distances.size > 0:
        ax.plot(action_time_axis, log.obstacle_distances, color="tab:gray", linestyle="--", label="dist→obstacle")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("norm / distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Control Norm & Obstacle Dist", fontweight="bold")

    # 7) Q history (log scale)
    ax = axes[3, 0]
    if log.q_history.size > 0:
        t_q = np.arange(log.q_history.shape[0]) * log.dt
        for idx in range(log.q_history.shape[1]):
            ax.plot(t_q, log.q_history[:, idx], label=f"Q[{idx}]")
        ax.set_yscale("log")
        if log.q_terminal.size > 0:
            final_q_text = ", ".join(f"{val:.2e}" for val in log.q_terminal)
            ax.text(
                0.02,
                0.95,
                f"Final Q diag: [{final_q_text}]",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
    else:
        ax.text(0.5, 0.5, "Q history unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Q diag")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Q Matrix Evolution", fontweight="bold")

    # 8) R history (log scale)
    ax = axes[3, 1]
    if log.r_history.size > 0:
        t_r = np.arange(log.r_history.shape[0]) * log.dt
        for idx in range(log.r_history.shape[1]):
            ax.plot(t_r, log.r_history[:, idx], label=f"R[{idx}]")
        ax.set_yscale("log")
        if log.r_terminal.size > 0:
            final_r_text = ", ".join(f"{val:.2e}" for val in log.r_terminal)
            ax.text(
                0.02,
                0.95,
                f"Final R diag: [{final_r_text}]",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
    else:
        ax.text(0.5, 0.5, "R history unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("R diag")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("R Matrix Evolution", fontweight="bold")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_agent(
    agent: ActorCriticAgent,
    env_kwargs: Dict,
    *,
    episodes: int,
    history_window: int,
    device: str,
    lidar_dim: int,
    obs_normalizer: Optional[ObservationNormalizer] = None,
    waypoint_normalizer: Optional[ObservationNormalizer] = None,
    lidar_normalizer: Optional[ObservationNormalizer] = None,
    save_dir: Optional[Path] = None,
    save_maps: bool = False,
    seed_offset: int = 0,
) -> Dict[str, float]:
    """Greedy evaluation rollouts against the obstacle environment."""
    env = SE2WaypointObstacleEnv(**env_kwargs)
    totals = {
        "reward": 0.0,
        "steps": 0,
        "collisions": 0,
        "waypoints": 0,
        "episodes": 0,
        "lidar_min": float("inf"),
        "lidar_max": 0.0,
    }

    save_dir = save_dir if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        for ep in range(episodes):
            log = run_eval_episode(
                agent,
                env,
                history_window=history_window,
                device=device,
                lidar_dim=lidar_dim,
                seed=seed_offset + ep,
                obs_normalizer=obs_normalizer,
                waypoint_normalizer=waypoint_normalizer,
                lidar_normalizer=lidar_normalizer,
            )
            totals["reward"] += log.total_reward
            totals["steps"] += log.steps
            totals["collisions"] += int(log.collision)
            totals["waypoints"] += log.waypoints_reached
            totals["episodes"] += 1
            totals["lidar_min"] = min(totals["lidar_min"], log.lidar_min)
            totals["lidar_max"] = max(totals["lidar_max"], log.lidar_max)

            if save_dir is not None and save_maps:
                save_eval_episode_plot(log, save_dir / f"episode_{ep:02d}.png")
    finally:
        env.close()

    episodes = max(1, totals["episodes"])
    return {
        "reward_per_episode": totals["reward"] / episodes,
        "steps_per_episode": totals["steps"] / episodes,
        "waypoints_per_episode": totals["waypoints"] / episodes,
        "collision_rate": totals["collisions"] / episodes,
        "lidar_min": totals["lidar_min"],
        "lidar_max": totals["lidar_max"],
    }


def load_best_or_latest_checkpoint(
    agent: ActorCriticAgent,
    training_cfg,
) -> tuple[Optional[Path], Optional[ObservationNormalizer], Optional[ObservationNormalizer], Optional[ObservationNormalizer]]:
    """Load the best (or latest) checkpoint into the provided agent and return normalizers."""
    if training_cfg.checkpoint_dir is None:
        return None, None, None, None
    ckpt_dir = Path(training_cfg.checkpoint_dir).expanduser()
    cfg = CheckpointConfig(
        directory=ckpt_dir,
        metric=training_cfg.checkpoint_metric,
        mode=training_cfg.checkpoint_mode,
        keep_last=max(1, training_cfg.checkpoint_keep_last),
    )
    manager = CheckpointManager(cfg)
    target = manager.best_checkpoint() or manager.latest_checkpoint()
    if target is None:
        return None, None, None, None
    payload = manager.load(path=target)
    state = payload.get("agent")
    if state is None:
        raise KeyError(f"Checkpoint {target} missing 'agent' weights.")
    # Filter out x_ref and u_ref parameters that may have incompatible batch sizes
    filtered_state = {
        key: tensor
        for key, tensor in state.items()
        if ".cost_module.x_ref" not in key and ".cost_module.u_ref" not in key
    }
    dropped = set(state.keys()) - set(filtered_state.keys())
    if dropped:
        print(f"Warning: Dropped incompatible parameters from checkpoint: {sorted(dropped)}")
    incompatible = agent.load_state_dict(filtered_state, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing or unexpected:
        print(f"Warning: Checkpoint loading issues (missing={missing}, unexpected={unexpected})")
    obs_norm = waypoint_norm = lidar_norm = None
    if payload.get("observation_normalizer") is not None:
        obs_norm = ObservationNormalizer(name="observation")
        obs_norm.load_state_dict(payload["observation_normalizer"])
    if payload.get("waypoint_normalizer") is not None:
        waypoint_norm = ObservationNormalizer(name="waypoint")
        waypoint_norm.load_state_dict(payload["waypoint_normalizer"])
    if payload.get("lidar_normalizer") is not None:
        lidar_norm = ObservationNormalizer(name="lidar")
        lidar_norm.load_state_dict(payload["lidar_normalizer"])
    return Path(target), obs_norm, waypoint_norm, lidar_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to experiment YAML/JSON.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="CLI override tokens (section.field=value). Can be provided multiple times.",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu/cuda/auto (default: auto).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for envs and torch.")
    parser.add_argument("--total-iters", type=int, default=50, help="Number of PPO update cycles.")
    parser.add_argument(
        "--rollouts-per-iter",
        type=int,
        default=1,
        help="How many rollout batches feed each PPO update.",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="Override sampler.num_envs.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Override sampler.rollout_steps.")
    parser.add_argument("--episode-len", type=int, default=None, help="Override sampler.episode_len.")
    parser.add_argument("--history-window", type=int, default=None, help="Clamp model.history_window.")
    parser.add_argument("--mpc-horizon", type=int, default=None, help="Override actor.mpc.horizon.")
    parser.add_argument("--dt", type=float, default=0.05, help="Environment integration timestep.")
    parser.add_argument("--waypoint-range", type=float, default=4.0, help="Waypoints sampled in [-range, range].")
    parser.add_argument("--goal-radius", type=float, default=0.15, help="Distance threshold for waypoint success.")
    parser.add_argument(
        "--min-start-radius",
        type=float,
        default=0.2,
        help="Minimum distance from origin when resetting the environment.",
    )
    parser.add_argument("--max-speed", type=float, default=1.0, help="Velocity / yaw-rate command saturation.")
    parser.add_argument(
        "--progress-gain",
        type=float,
        default=1.0,
        help="Reward gain for waypoint progress.",
    )
    parser.add_argument(
        "--action-penalty",
        type=float,
        default=0.01,
        help="Penalty coefficient on control magnitude (||u||); if > 0 overrides control-gain.",
    )
    parser.add_argument("--living-penalty", type=float, default=0.01, help="Constant reward penalty per step.")
    parser.add_argument(
        "--control-gain",
        type=float,
        default=0.0,
        help="Positive reward per unit control magnitude (||u||) when action-penalty == 0.",
    )
    parser.add_argument("--goal-bonus", type=float, default=15.0, help="Reward added when a waypoint is reached.")

    # Obstacle / lidar specific flags
    parser.add_argument("--num-obstacles", type=int, default=3, help="Number of circular obstacles.")
    parser.add_argument(
        "--obstacle-radius-min",
        type=float,
        default=0.3,
        help="Minimum obstacle radius.",
    )
    parser.add_argument(
        "--obstacle-radius-max",
        type=float,
        default=0.7,
        help="Maximum obstacle radius.",
    )
    parser.add_argument(
        "--arena-radius",
        type=float,
        default=4.0,
        help="Radius of the arena used for obstacle placement.",
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=25.0,
        help="Reward penalty applied when a collision occurs.",
    )
    parser.add_argument(
        "--proximity-gain",
        type=float,
        default=15.0,
        help="Penalty applied when near obstacles (quadratic inside proximity-margin).",
    )
    parser.add_argument(
        "--proximity-margin",
        type=float,
        default=0.75,
        help="Distance (m) within which the obstacle proximity penalty ramps in.",
    )
    parser.add_argument("--lidar-beams", type=int, default=36, help="Number of lidar beams.")
    parser.add_argument("--lidar-fov", type=float, default=360.0, help="Lidar field of view in degrees.")
    parser.add_argument("--lidar-range", type=float, default=4.0, help="Maximum lidar range in metres.")

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory where training checkpoints will be stored.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="How often (in PPO iters) checkpoints are written (0=never). Default 1 to mirror other examples.",
    )
    parser.add_argument(
        "--checkpoint-keep-last",
        type=int,
        default=3,
        help="How many recent checkpoints to retain when checkpointing is enabled.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Run greedy evaluation for this many episodes after training (0=skip).",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=REPO_ROOT / "eval_outputs" / "se2_kinematic_obstacles",
        help="Directory where evaluation maps are saved when --eval-episodes > 0.",
    )
    parser.add_argument(
        "--eval-save-maps",
        action="store_true",
        help="Save obstacle map PNGs for evaluation episodes.",
    )

    return parser.parse_args()


def build_env_kwargs(args: argparse.Namespace, *, episode_len: int) -> Dict:
    return dict(
        dt=args.dt,
        episode_len=episode_len,
        waypoint_range=args.waypoint_range,
        goal_radius=args.goal_radius,
        min_start_radius=args.min_start_radius,
        max_speed=args.max_speed,
        progress_gain=args.progress_gain,
        action_penalty=args.action_penalty,
        living_penalty=args.living_penalty,
        goal_bonus=args.goal_bonus,
        control_gain=args.control_gain,
        enable_obstacles=True,
        num_obstacles=args.num_obstacles,
        obstacle_radius_min=args.obstacle_radius_min,
        obstacle_radius_max=args.obstacle_radius_max,
        arena_radius=args.arena_radius,
        proximity_gain=args.proximity_gain,
        proximity_margin=args.proximity_margin,
        enable_lidar=True,
        lidar_num_beams=args.lidar_beams,
        lidar_fov=np.deg2rad(args.lidar_fov),
        lidar_max_range=args.lidar_range,
        collision_penalty=args.collision_penalty,
    )


def resolve_device(cli_device: str, config_device: str) -> str:
    if cli_device == "auto":
        if config_device in {"cpu", "cuda"}:
            target = config_device
        else:
            target = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if cli_device not in {"cpu", "cuda"}:
            raise ValueError("device must be one of: cpu, cuda, auto.")
        target = cli_device
    if target == "cuda" and not torch.cuda.is_available():
        print("CUDA non disponibile, uso CPU.")
        return "cpu"
    if cli_device in {"cpu", "cuda"}:
        return target
    return target


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.eval_episodes == 0 and os.getenv("PYCHARM_HOSTED"):
        # Integrazione PyCharm: esegui automaticamente una breve eval dopo il training.
        args.eval_episodes = 10
        args.eval_save_maps = True
        print("[pycharm] Abilito evaluation automatica (10 episodi) con salvataggio mappe.")
    overrides = args.override if args.override else None
    config = load_experiment_config(args.config, overrides=overrides)

    episode_len = args.episode_len if args.episode_len is not None else config.sampler.episode_len
    env_kwargs = build_env_kwargs(args, episode_len=episode_len)

    if args.num_envs is not None:
        config.sampler.num_envs = max(1, args.num_envs)
    if args.rollout_steps is not None:
        config.sampler.rollout_steps = max(1, args.rollout_steps)
    config.sampler.episode_len = int(env_kwargs["episode_len"])
    if args.history_window is not None:
        config.model.history_window = max(1, args.history_window)
    if args.mpc_horizon is not None:
        config.model.actor.mpc.horizon = max(1, args.mpc_horizon)
    if args.checkpoint_interval is not None:
        config.training.checkpoint_interval = max(0, args.checkpoint_interval)
        config.training.checkpoint_dir = str(args.checkpoint_dir)
        config.training.checkpoint_keep_last = max(1, args.checkpoint_keep_last)
    checkpoint_dir = Path(config.training.checkpoint_dir or DEFAULT_CHECKPOINT_DIR).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.training.checkpoint_dir = str(checkpoint_dir)
    # Allinea agli altri esempi: checkpoint sempre abilitati (>=1).
    config.training.checkpoint_interval = max(1, config.training.checkpoint_interval)
    config.training.checkpoint_metric = "episode_reward"
    config.training.checkpoint_mode = "max"

    device = resolve_device(args.device, config.device)
    config.device = device

    set_global_seed(args.seed)

    dims = probe_dimensions(dict(env_kwargs, env_id=0))
    rollout_len = config.sampler.rollout_steps
    lidar_dim = int(3 * env_kwargs["lidar_num_beams"])
    prepare_config_with_lidar(
        config,
        dims=dims,
        history_window=config.model.history_window,
        rollout_len=rollout_len,
        mpc_horizon=config.model.actor.mpc.horizon,
        dt=env_kwargs["dt"],
        action_limit=env_kwargs["max_speed"],
        device=device,
        lidar_dim=lidar_dim,
    )

    # Build agent + dynamics
    agent_cfg = config.model.build_agent_config()
    agent_cfg.device = device
    dynamics_fn, dynamics_jac = build_se2_kinematic_dynamics(
        max_speed=env_kwargs["max_speed"],
    )
    agent = ActorCriticAgent(
        agent_cfg,
        dynamics_fn=dynamics_fn,
        dynamics_jacobian_fn=dynamics_jac,
    )

    loop = TrainingLoop(agent, config.training)

    print("=== SE(2) Kinematic Obstacles + Lidar ACMPC Training ===")
    print(f"Device: {device} | Seed: {config.seed}")
    print(
        "Vector envs: {envs} | Rollout steps: {rollout} | Episode len: {episode}".format(
            envs=config.sampler.num_envs,
            rollout=config.sampler.rollout_steps,
            episode=env_kwargs["episode_len"],
        )
    )
    print(
        "MPC horizon: {horizon} | History window: {history} | Max speed: {speed} m/s".format(
            horizon=config.model.actor.mpc.horizon,
            history=config.model.history_window,
            speed=env_kwargs["max_speed"],
        )
    )
    print(
        "Obstacles: {num} | Lidar beams: {beams} | Lidar range: {rng} m".format(
            num=env_kwargs["num_obstacles"],
            beams=env_kwargs["lidar_num_beams"],
            rng=env_kwargs["lidar_max_range"],
        )
    )
    print(
        f"Checkpoints every {config.training.checkpoint_interval} iters -> {checkpoint_dir} "
        f"(keep_last={config.training.checkpoint_keep_last}, metric={config.training.checkpoint_metric})"
    )

    env_manager = None
    try:
        env_manager = build_env_manager(
            num_envs=config.sampler.num_envs,
            env_kwargs=env_kwargs,
            seed=config.seed + config.sampler.seed_offset,
            device=device,
        )
        collector = RolloutCollector(
            agent=agent,
            env_manager=env_manager,
            history_window=config.model.history_window,
            horizon=config.model.actor.mpc.horizon,
            device=torch.device(device),
            observation_normalizer=loop.observation_normalizer,
            waypoint_normalizer=loop.waypoint_normalizer,
            lidar_normalizer=loop.lidar_normalizer,
        )

        total_iters = max(1, args.total_iters)
        for iteration in range(1, total_iters + 1):
            batches: List[RolloutBatch] = [
                collector.collect(horizon=config.sampler.rollout_steps)
                for _ in range(args.rollouts_per_iter)
            ]
            rollout_stats = _rollout_stats(batches, lidar_dim=lidar_dim)
            metrics = loop.run(batches)
            episode_reward = rollout_stats["reward_per_step"] * config.sampler.rollout_steps
            score = (
                episode_reward
                + 20.0 * rollout_stats["waypoints"]
                - 50.0 * rollout_stats["collisions"]
            )

            progress = iteration / float(total_iters)
            bar_length = 25
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)

            summary = (
                f"\rIter {iteration}/{total_iters} "
                f"[{progress * 100:5.1f}%] {bar} "
                f"policy={metrics.policy_loss:.4f} "
                f"value={metrics.value_loss:.4f} "
                f"entropy={metrics.entropy:.4f} "
                f"kl={metrics.approx_kl:.5f}"
            )
            if metrics.actor_grad_norm is not None:
                summary += f" |grad_actor|={metrics.actor_grad_norm:.3f}"
            if metrics.critic_grad_norm is not None:
                summary += f" |grad_critic|={metrics.critic_grad_norm:.3f}"
            summary += (
                f" wp={rollout_stats['waypoints']} coll={rollout_stats['collisions']}"
                f" r/step={rollout_stats['reward_per_step']:.3f}"
            )
            if rollout_stats["lidar_min"] < float("inf"):
                summary += (
                    f" lidar_norm=[{rollout_stats['lidar_min']:.2f},{rollout_stats['lidar_max']:.2f}]"
                )
                if not rollout_stats["lidar_ok"]:
                    summary += " LIDAR_RANGE_ERR"
            end_char = "\n" if iteration == total_iters else ""
            print(summary, end=end_char, flush=True)

            if (
                loop.checkpoint_manager is not None
                and iteration % config.training.checkpoint_interval == 0
            ):
                loop.checkpoint_manager.save(
                    step=iteration,
                    metrics={
                        "episode_reward": episode_reward,
                        "score": score,
                        "policy_loss": metrics.policy_loss,
                        "value_loss": metrics.value_loss,
                    },
                    agent=agent,
                    actor_opt=loop.actor_opt,
                    critic_opt=loop.critic_opt,
                    grad_manager=loop.grad_manager,
                    reward_normalizer=loop.reward_normalizer,
                    observation_normalizer=loop.observation_normalizer,
                    waypoint_normalizer=loop.waypoint_normalizer,
                    lidar_normalizer=loop.lidar_normalizer,
                    actor_scheduler=loop.actor_scheduler,
                    critic_scheduler=loop.critic_scheduler,
                )

        if args.eval_episodes > 0:
            # Usa il best (o latest) checkpoint salvato.
            ckpt_path, obs_norm_ckpt, wp_norm_ckpt, lidar_norm_ckpt = load_best_or_latest_checkpoint(agent, config.training)
            if ckpt_path is not None:
                agent.to(device)
                print(f"Evaluating using checkpoint: {ckpt_path}")
            else:
                print("No checkpoint found; evaluating model in RAM.")

            eval_maps_dir = args.eval_output_dir if args.eval_save_maps else None
            eval_stats = evaluate_agent(
                agent,
                env_kwargs,
                episodes=max(1, args.eval_episodes),
                history_window=config.model.history_window,
                device=device,
                lidar_dim=lidar_dim,
                obs_normalizer=obs_norm_ckpt,
                waypoint_normalizer=wp_norm_ckpt,
                lidar_normalizer=lidar_norm_ckpt,
                save_dir=eval_maps_dir,
                save_maps=args.eval_save_maps,
                seed_offset=args.seed + 1000,
            )
            print(
                "Eval (greedy): "
                f"reward/ep={eval_stats['reward_per_episode']:.3f} "
                f"waypoints/ep={eval_stats['waypoints_per_episode']:.2f} "
                f"coll_rate={eval_stats['collision_rate']:.2f} "
                f"steps/ep={eval_stats['steps_per_episode']:.1f}"
            )
            if eval_maps_dir is not None:
                print(f"Saved eval maps to {Path(eval_maps_dir).expanduser()}")

    except KeyboardInterrupt:
        print("Training interrupted by user; closing environments.")
    finally:
        if env_manager is not None:
            env_manager.close()


if __name__ == "__main__":
    main()
