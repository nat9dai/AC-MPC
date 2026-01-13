"""Advanced evaluation and plotting utilities for AC-MPC waypoint tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .double_integrator_AC_DRL import EvalEpisodeLog  # noqa: F401


@dataclass
class EpisodeStats:
    total_reward: float
    waypoints: int
    final_distance: float
    cumulative_reward: np.ndarray


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_individual_run_plot(
    log: "EvalEpisodeLog",
    path: Path,
    *,
    title_prefix: str = "AC-MPC Waypoint Task",
) -> None:
    """Create a rich 3x2 plot summarising a single evaluation episode."""

    _ensure_parent(path)
    dt = log.dt
    time_axis = np.arange(log.positions.shape[0]) * dt
    action_time_axis = np.arange(log.actions.shape[0]) * dt

    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    fig.suptitle(
        f"{title_prefix} — Episode\nReward: {log.total_reward:.2f} | Waypoints: {log.waypoints_reached}",
        fontsize=14,
        fontweight="bold",
    )

    # Trajectory
    ax = axes[0, 0]
    traj = log.positions
    ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=2.0, label="Trajectory", zorder=5)
    ax.scatter(traj[0, 0], traj[0, 1], color="green", marker="o", s=80, label="Start", zorder=6)
    ax.scatter(traj[-1, 0], traj[-1, 1], color="red", marker="x", s=100, label="End", zorder=7)
    for idx, wp in enumerate(log.visited_waypoints):
        ax.scatter(wp[0], wp[1], color="green", s=120, edgecolors="black", linewidth=1.5, zorder=8)
        # Add a green background circle for better text visibility
        ax.text(wp[0], wp[1], str(idx + 1), color="white", ha="center", va="center", 
               fontweight="bold", fontsize=10, zorder=10,
               bbox=dict(boxstyle="circle,pad=0.2", facecolor="green", edgecolor="black", linewidth=1))
    if log.target_history.size > 0:
        tx, ty = log.target_history[-1]
        ax.scatter(tx, ty, color="yellow", s=130, edgecolors="black", linewidth=1.2, label="Current target")
    extent = max(1.5, np.abs(traj).max() + 0.5)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("2D Trajectory & Waypoints", fontweight="bold")

    # Control commands
    ax = axes[0, 1]
    if log.actions.size > 0:
        actions = log.actions
        act_dim = actions.shape[1] if actions.ndim == 2 else 0
        if act_dim == 2:
            ax.plot(action_time_axis, actions[:, 0], label="vx cmd", color="tab:blue")
            ax.plot(action_time_axis, actions[:, 1], label="vy cmd", color="tab:red")
        elif act_dim == 3:
            ax.plot(action_time_axis, actions[:, 0], label="Vs", color="tab:blue")
            ax.plot(action_time_axis, actions[:, 1], label="Vl", color="tab:red")
            ax.plot(action_time_axis, actions[:, 2], label="w", color="tab:green")
        else:
            for idx in range(act_dim):
                ax.plot(action_time_axis, actions[:, idx], label=f"u[{idx}]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("command")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Control Commands", fontweight="bold")

    # Rewards
    ax = axes[1, 0]
    if log.rewards.size > 0:
        ax.plot(action_time_axis, log.rewards, color="tab:orange", label="instant reward")
        ax.plot(action_time_axis, np.cumsum(log.rewards), color="tab:red", label="cumulative reward")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Reward Evolution", fontweight="bold")

    # Distance to waypoint
    ax = axes[1, 1]
    if log.distances.size > 0:
        ax.plot(time_axis, log.distances, color="tab:purple", label="distance")
        ax.axhline(
            y=log.goal_radius,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"goal radius ({log.goal_radius:.2f} m)",
        )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Distance to Target", fontweight="bold")

    # Q history
    ax = axes[2, 0]
    if log.q_history.size > 0:
        t_q = np.arange(log.q_history.shape[0]) * log.dt
        for idx in range(log.q_history.shape[1]):
            ax.plot(t_q, log.q_history[:, idx], label=f"Q[{idx}]")
        ax.set_yscale("log")
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

    # R history
    ax = axes[2, 1]
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

    for row in axes:
        for ax in row:
            if ax not in (axes[2, 0], axes[2, 1]):
                ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_multi_run_plot(
    logs: Sequence["EvalEpisodeLog"],
    path: Path,
    *,
    title_prefix: str = "AC-MPC Waypoint Task",
) -> None:
    """Aggregate multiple evaluation runs into a single diagnostic plot."""

    if not logs:
        return
    _ensure_parent(path)

    dt = logs[0].dt
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    rewards = np.array([log.total_reward for log in logs], dtype=np.float32)
    waypoints = np.array([log.waypoints_reached for log in logs], dtype=np.float32)
    mean_reward = float(rewards.mean())
    std_reward = float(rewards.std())
    mean_wp = float(waypoints.mean())
    std_wp = float(waypoints.std())
    fig.suptitle(
        f"{title_prefix} — {len(logs)} Evaluation Runs\n"
        f"Reward: {mean_reward:.1f} ± {std_reward:.1f} | Waypoints: {mean_wp:.1f} ± {std_wp:.1f}",
        fontsize=16,
        fontweight="bold",
    )

    # Trajectories
    ax = axes[0, 0]
    for idx, log in enumerate(logs):
        traj = log.positions
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.4 if idx else 0.9, linewidth=1.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Trajectories", fontweight="bold")

    # Commands (last run)
    last = logs[-1]
    time_axis = np.arange(last.actions.shape[0]) * dt
    ax = axes[0, 1]
    if last.actions.size > 0:
        actions = last.actions
        act_dim = actions.shape[1] if actions.ndim == 2 else 0
        if act_dim == 2:
            ax.plot(time_axis, actions[:, 0], label="vx cmd")
            ax.plot(time_axis, actions[:, 1], label="vy cmd")
        elif act_dim == 3:
            ax.plot(time_axis, actions[:, 0], label="Vs")
            ax.plot(time_axis, actions[:, 1], label="Vl")
            ax.plot(time_axis, actions[:, 2], label="w")
        else:
            for idx in range(act_dim):
                ax.plot(time_axis, actions[:, idx], label=f"u[{idx}]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("command [m/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Commands (last run)", fontweight="bold")

    # Velocity + distance (last run)
    ax = axes[1, 0]
    t_full = np.arange(last.velocities.shape[0]) * dt
    ax2 = ax.twinx()
    ax.plot(t_full, last.velocities[:, 0], label="vx", color="tab:blue")
    ax.plot(t_full, last.velocities[:, 1], label="vy", color="tab:green")
    ax2.plot(t_full, last.distances, label="distance", color="tab:purple")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity [m/s]")
    ax2.set_ylabel("distance [m]")
    ax.grid(True, alpha=0.3)
    lines = ax.lines + ax2.lines
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right")
    ax.set_title("Velocity & Distance (last run)", fontweight="bold")

    # Reward histories
    ax = axes[1, 1]
    min_len = min((log.rewards.size for log in logs if log.rewards.size > 0), default=0)
    for idx, log in enumerate(logs):
        if log.rewards.size == 0:
            continue
        t = np.arange(log.rewards.size) * log.dt
        ax.plot(t, log.rewards, alpha=0.4 if idx else 0.9)
    if min_len > 0 and len(logs) > 1:
        stacked = np.stack([log.rewards[:min_len] for log in logs if log.rewards.size >= min_len], axis=0)
        mean_curve = stacked.mean(axis=0)
        ax.plot(np.arange(min_len) * dt, mean_curve, color="black", linewidth=2.5, label="mean")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.set_title("Reward Evolution", fontweight="bold")

    # Bar chart stats
    metrics = ["Reward", "Waypoints", "Final distance"]
    final_dist = np.array([log.distances[-1] if log.distances.size > 0 else np.nan for log in logs])
    means = [mean_reward, mean_wp, float(np.nanmean(final_dist))]
    stds = [std_reward, std_wp, float(np.nanstd(final_dist))]
    ax = axes[2, 0]
    x = np.arange(len(metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=["tab:orange", "tab:green", "tab:purple"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(True, alpha=0.3)
    ax.set_title("Performance Summary", fontweight="bold")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std * 0.5, f"{mean:.1f}±{std:.1f}", ha="center")

    # Cumulative reward
    ax = axes[2, 1]
    for log in logs:
        if log.rewards.size == 0:
            continue
        cumulative = np.cumsum(log.rewards)
        t = np.arange(cumulative.size) * log.dt
        ax.plot(t, cumulative, alpha=0.4)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cumulative reward")
    ax.grid(True, alpha=0.3)
    ax.set_title("Cumulative Reward", fontweight="bold")

    # Q/R multi-run (use last log for temporal profile)
    last_log = logs[-1]
    ax_q = axes[3, 0]
    if last_log.q_history.size > 0:
        t_q = np.arange(last_log.q_history.shape[0]) * last_log.dt
        for idx in range(last_log.q_history.shape[1]):
            ax_q.plot(t_q, last_log.q_history[:, idx], label=f"Q[{idx}]")
        ax_q.set_yscale("log")
    else:
        ax_q.text(0.5, 0.5, "Q history unavailable", ha="center", va="center", transform=ax_q.transAxes)
    ax_q.set_xlabel("time [s]")
    ax_q.set_ylabel("Q diag")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=8, loc="upper right")
    ax_q.set_title("Q Matrix Evolution (last run)", fontweight="bold")

    ax_r = axes[3, 1]
    if last_log.r_history.size > 0:
        t_r = np.arange(last_log.r_history.shape[0]) * last_log.dt
        for idx in range(last_log.r_history.shape[1]):
            ax_r.plot(t_r, last_log.r_history[:, idx], label=f"R[{idx}]")
        ax_r.set_yscale("log")
    else:
        ax_r.text(0.5, 0.5, "R history unavailable", ha="center", va="center", transform=ax_r.transAxes)
    ax_r.set_xlabel("time [s]")
    ax_r.set_ylabel("R diag")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=8, loc="upper right")
    ax_r.set_title("R Matrix Evolution (last run)", fontweight="bold")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
