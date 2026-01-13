"""Train and evaluate ACMPC on the SE(2) kinematic waypoint task."""

from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ACMPC import (  # noqa: E402
    ActorCriticAgent,
    ActorCriticState,
    DiagnosticsOptions,
    TrainingLoop,
    load_experiment_config,
)
from ACMPC.training.normalization import ObservationNormalizer  # noqa: E402
from ACMPC.envs import SE2WaypointEnv, build_se2_kinematic_dynamics  # noqa: E402
from ACMPC.experiment_config import ExperimentConfig  # noqa: E402
from ACMPC.sampling import RolloutBatch, RolloutCollector  # noqa: E402
from ACMPC.training.checkpoint import CheckpointConfig, CheckpointManager  # noqa: E402
from examples.se2_waypoint_common import (  # noqa: E402
    build_env_manager,
    prepare_config,
    probe_dimensions,
)
from examples.double_integrator_eval_viz import (  # noqa: E402
    save_individual_run_plot,
    save_multi_run_plot,
)

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "se2_kinematic_fixed.yaml"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "se2_kinematic"
DEFAULT_EVAL_OUTPUT_DIR = REPO_ROOT / "eval_outputs" / "se2_kinematic"


@dataclass
class EvalEpisodeLog:
    positions: np.ndarray
    velocities: np.ndarray
    target_history: np.ndarray
    visited_waypoints: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    distances: np.ndarray
    q_history: np.ndarray
    r_history: np.ndarray
    q_terminal: np.ndarray
    r_terminal: np.ndarray
    total_reward: float
    steps: int
    waypoints_reached: int
    dt: float
    goal_radius: float


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
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for envs and torch.")
    parser.add_argument("--total-iters", type=int, default=500, help="Number of PPO update cycles.")
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
    # Task: waypoints in a square and a fixed success radius.
    parser.add_argument("--waypoint-range", type=float, default=2.0, help="Waypoints sampled in [-range, range].")
    parser.add_argument("--goal-radius", type=float, default=0.15, help="Distance threshold for waypoint success.")
    parser.add_argument(
        "--min-start-radius",
        type=float,
        default=0.2,
        help="Minimum distance from origin when resetting the environment.",
    )
    # Limite sui comandi di velocità nel sistema cinematico SE(2).
    parser.add_argument("--max-speed", type=float, default=4.0, help="Velocity / yaw-rate command saturation.")
    # Reward stile AC-MPC (paper): progresso verso il waypoint meno una piccola
    # penalità sull'uso del controllo, più bonus quando il waypoint è raggiunto.
    # Qui usiamo un guadagno di progresso e un bonus di goal più alti, e
    # premiamo leggermente l'uso di controlli di ampiezza maggiore.
    parser.add_argument("--progress-gain", type=float, default=2.0, help="Reward gain for waypoint progress.")
    parser.add_argument(
        "--action-penalty",
        type=float,
        default=0.0,
        help="Penalty coefficient on control magnitude (||u||), analogous to b in the paper.",
    )
    parser.add_argument("--living-penalty", type=float, default=0.0, help="Constant reward penalty per step.")
    parser.add_argument(
        "--control-gain",
        type=float,
        default=0.01,
        help="Positive reward per unit control magnitude (used only when action-penalty == 0).",
    )
    parser.add_argument("--goal-bonus", type=float, default=20.0, help="Reward added when a waypoint is reached.")
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
        help="How often (in PPO updates) to save checkpoints (>=1).",
    )
    parser.add_argument("--eval-runs", type=int, default=20, help="Number of evaluation episodes to run.")
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=None,
        help="Maximum steps per evaluation episode (defaults to episode length).",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=DEFAULT_EVAL_OUTPUT_DIR,
        help="Directory for evaluation plots and summaries.",
    )
    parser.add_argument("--eval-seed", type=int, default=None, help="Base random seed for evaluation rollouts.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluation stage even if checkpoints exist.")
    parser.add_argument(
        "--resume",
        choices=["auto", "best", "latest", "none"],
        default="auto",
        help="Checkpoint resume strategy for training (default: auto).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Salta completamente il training e lancia solo l'evaluation sul checkpoint disponibile.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: Optional[str], config_default: str) -> str:
    if requested is None:
        candidate = config_default
    else:
        token = requested.strip().lower()
        if token == "auto":
            candidate = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            candidate = requested
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA non disponibile, uso CPU.")
        return "cpu"
    return candidate


def build_env_kwargs(args: argparse.Namespace, *, episode_len: int) -> Dict[str, float | int]:
    return {
        "dt": float(args.dt),
        "episode_len": int(episode_len),
        "waypoint_range": float(args.waypoint_range),
        "goal_radius": float(args.goal_radius),
        "min_start_radius": float(args.min_start_radius),
        "max_speed": float(args.max_speed),
        "progress_gain": float(args.progress_gain),
        "action_penalty": float(args.action_penalty),
        "living_penalty": float(args.living_penalty),
        "goal_bonus": float(args.goal_bonus),
        "control_gain": float(args.control_gain),
    }


def apply_cli_overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
    *,
    device: str,
    env_kwargs: Dict[str, float | int],
    checkpoint_dir: Path,
    resume_path: Optional[Path],
) -> None:
    config.seed = args.seed
    config.device = device
    config.training.device = device
    config.model.actor.mpc.device = device
    if args.num_envs is not None:
        config.sampler.num_envs = max(1, args.num_envs)
    if args.rollout_steps is not None:
        config.sampler.rollout_steps = max(1, args.rollout_steps)
    else:
        config.sampler.rollout_steps = int(env_kwargs["episode_len"])
    config.sampler.episode_len = int(env_kwargs["episode_len"])
    if args.history_window is not None:
        config.model.history_window = max(1, args.history_window)
    if args.mpc_horizon is not None:
        config.model.actor.mpc.horizon = max(1, args.mpc_horizon)
    config.training.checkpoint_dir = str(checkpoint_dir)
    config.training.checkpoint_interval = max(1, args.checkpoint_interval)
    config.training.resume_path = str(resume_path) if resume_path else None
    config.training.resume_best = False
    config.training.resume_latest = False
    config.synchronise()


def create_diagnostics_options(config: ExperimentConfig) -> DiagnosticsOptions:
    return DiagnosticsOptions(
        enable_cost_monitor=config.diagnostics.enable_cost_monitor,
        enable_reward_stats=config.diagnostics.enable_reward_stats,
        enable_plan_drift_checker=config.diagnostics.enable_plan_drift_checker,
        enable_safety_hooks=config.diagnostics.enable_safety_hooks,
        enable_mpve_metrics=config.training.mpve.enabled,
    )


def print_run_header(config: ExperimentConfig, env_kwargs: Dict[str, float | int], device: str, checkpoint_dir: Path) -> None:
    print("=== SE(2) Kinematic ACMPC Training ===")
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
    print(f"Checkpoint dir: {checkpoint_dir}")
    print()


def extract_cost_diagonals(agent: ActorCriticAgent) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return running and terminal Q/R diagonals from the actor's MPC head."""

    head = agent.actor.mpc_head
    module = head._controller.cost_module  # type: ignore[attr-defined]
    nx = agent.actor.state_dim
    nu = agent.actor.action_dim

    def _select_stage(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4:
            return tensor[0, 0]
        if tensor.dim() == 3:
            return tensor[0]
        return tensor

    running = _select_stage(module.C.detach().cpu())
    q_running = torch.diagonal(running[:nx, :nx]).cpu().numpy()
    r_running = torch.diagonal(running[nx:, nx:]).cpu().numpy() if nu > 0 else np.zeros(0, dtype=np.float32)

    final_mat = module.C_final.detach().cpu()
    if final_mat.dim() == 3:
        final_mat = final_mat[0]
    q_final = torch.diagonal(final_mat[:nx, :nx]).cpu().numpy()
    r_final = (
        torch.diagonal(final_mat[nx:, nx:]).cpu().numpy()
        if final_mat.shape[0] >= nx + nu and nu > 0
        else np.zeros(0, dtype=np.float32)
    )
    return q_running, r_running, q_final, r_final


def discover_checkpoints(
    training_cfg,
    checkpoint_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    cfg = CheckpointConfig(
        directory=checkpoint_dir,
        metric=training_cfg.checkpoint_metric or "value_loss",
        mode=training_cfg.checkpoint_mode or "min",
        keep_last=max(1, training_cfg.checkpoint_keep_last or 1),
    )
    manager = CheckpointManager(cfg)
    return manager.best_checkpoint(), manager.latest_checkpoint()


def select_resume_path(
    mode: str,
    *,
    best: Optional[Path],
    latest: Optional[Path],
) -> Optional[Path]:
    if mode == "none":
        return None
    prefer_best = mode in {"auto", "best"}
    prefer_latest = mode == "latest"
    if prefer_best:
        return best or latest
    if prefer_latest:
        return latest or best
    return best or latest


def compute_mean_waypoints(batches: Sequence[RolloutBatch]) -> Optional[float]:
    waypoint_counts: List[int] = []
    for batch in batches:
        info_seq = batch.info  # Stored as [time][env]
        num_envs = batch.num_envs
        horizon = batch.horizon
        for env_idx in range(num_envs):
            max_count: Optional[int] = None
            for t in range(horizon):
                info = None
                if info_seq and t < len(info_seq) and env_idx < len(info_seq[t]):
                    info = info_seq[t][env_idx]
                if info and "waypoints_reached" in info:
                    count = int(info["waypoints_reached"])
                    max_count = count if max_count is None else max(max_count, count)
            if max_count is not None:
                waypoint_counts.append(max_count)
    if not waypoint_counts:
        return None
    return float(np.mean(waypoint_counts))


def compute_mean_episode_reward(batches: Sequence[RolloutBatch]) -> Optional[float]:
    """Compute mean episodic return over all completed (or truncated) episodes."""

    episode_returns: List[float] = []
    for batch in batches:
        rewards = batch.reward.cpu().numpy()
        dones = batch.done.cpu().numpy().astype(bool)
        mask = batch.mask.cpu().numpy()
        num_envs, horizon = rewards.shape
        for env_idx in range(num_envs):
            ret = 0.0
            episode_open = False
            for t in range(horizon):
                if mask[env_idx, t] <= 0.0:
                    continue
                episode_open = True
                ret += float(rewards[env_idx, t])
                if dones[env_idx, t]:
                    episode_returns.append(ret)
                    ret = 0.0
                    episode_open = False
            if episode_open:
                episode_returns.append(ret)
    if not episode_returns:
        return None
    return float(np.mean(episode_returns))


def run_eval_episode(
    agent: ActorCriticAgent,
    env: SE2WaypointEnv,
    *,
    history_window: int,
    max_steps: int,
    device: torch.device,
    seed: Optional[int],
    obs_normalizer: Optional[ObservationNormalizer] = None,
    waypoint_normalizer: Optional[ObservationNormalizer] = None,
    lidar_normalizer: Optional[ObservationNormalizer] = None,
) -> EvalEpisodeLog:
    obs, info = env.reset(seed=seed)
    state_dim = agent.config.actor.mpc.state_dim
    action_dim = agent.config.actor.mpc.action_dim
    horizon = agent.config.actor.mpc.horizon

    state = obs[:state_dim].astype(np.float32)
    waypoint = info["target_waypoint"].astype(np.float32)
    waypoints_reached = int(info.get("waypoints_reached", 0))

    history = torch.zeros(history_window, state_dim, device=device)
    history[-1] = torch.from_numpy(state).to(device)
    memories: Optional[ActorCriticState] = agent.init_state(batch_size=1)
    warm_start = torch.zeros(1, horizon, action_dim, device=device)

    positions = [state[:2].copy()]
    # For the SE(2) kinematic model we log world-frame velocities [vx, vy].
    velocities = [np.zeros(2, dtype=np.float32)]
    target_history = [waypoint.copy()]
    visited_waypoints: List[np.ndarray] = []
    actions = []
    rewards = []
    distances = [np.linalg.norm(state[:2] - waypoint)]
    q_history: List[np.ndarray] = []
    r_history: List[np.ndarray] = []
    last_q_final: Optional[np.ndarray] = None
    last_r_final: Optional[np.ndarray] = None

    agent.eval()
    total_reward = 0.0
    steps = 0

    while steps < max_steps:
        history_batch = history.unsqueeze(0)
        state_tensor = torch.from_numpy(state).to(device).unsqueeze(0)
        waypoint_tensor = torch.from_numpy(waypoint).to(device).view(1, 1, -1)

        norm_history = history_batch
        norm_state = state_tensor
        norm_waypoint = waypoint_tensor
        if obs_normalizer is not None:
            obs_normalizer.to(device)
            norm_history = obs_normalizer.normalize(history_batch)
            norm_state = obs_normalizer.normalize(state_tensor)
        if waypoint_normalizer is not None:
            waypoint_normalizer.to(device)
            norm_waypoint = waypoint_normalizer.normalize(waypoint_tensor)

        with torch.no_grad():
            action_tensor, memories, plan = agent.act(
                norm_history,
                state=norm_state,
                raw_state=state_tensor,
                memories=memories,
                waypoint_seq=norm_waypoint,
                raw_waypoint_seq=waypoint_tensor,
                warm_start=warm_start,
                return_plan=True,
            )
        action = action_tensor.squeeze(0).detach().cpu().numpy()
        if plan is not None:
            warm_start = plan[1].detach()
        else:
            warm_start.zero_()

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)

        next_state = next_obs[:state_dim].astype(np.float32)
        next_waypoint = next_info["target_waypoint"].astype(np.float32)
        new_waypoints = int(next_info.get("waypoints_reached", waypoints_reached))

        actions.append(action.copy())
        rewards.append(float(reward))
        positions.append(next_state[:2].copy())
        # Approximate executed world-frame velocities from finite differences.
        vx = (next_state[0] - state[0]) / float(getattr(env, "dt", 0.05))
        vy = (next_state[1] - state[1]) / float(getattr(env, "dt", 0.05))
        velocities.append(np.array([vx, vy], dtype=np.float32))
        target_history.append(next_waypoint.copy())
        distances.append(np.linalg.norm(next_state[:2] - next_waypoint))

        q_diag, r_diag, q_final, r_final = extract_cost_diagonals(agent)
        q_history.append(q_diag.astype(np.float32))
        r_history.append(r_diag.astype(np.float32))
        last_q_final = q_final.astype(np.float32)
        last_r_final = r_final.astype(np.float32)

        if new_waypoints > waypoints_reached:
            visited_waypoints.append(waypoint.copy())
        waypoints_reached = new_waypoints

        state = next_state
        waypoint = next_waypoint
        history = torch.roll(history, shifts=-1, dims=0)
        history[-1] = torch.from_numpy(state).to(device)
        steps += 1

        if done:
            break

    visited_array = (
        np.stack(visited_waypoints, axis=0).astype(np.float32) if visited_waypoints else np.zeros((0, 2), dtype=np.float32)
    )
    actions_array = np.stack(actions, axis=0).astype(np.float32) if actions else np.zeros((0, action_dim), dtype=np.float32)
    rewards_array = np.asarray(rewards, dtype=np.float32) if rewards else np.zeros((0,), dtype=np.float32)
    velocity_array = np.stack(velocities, axis=0).astype(np.float32)
    distance_array = np.asarray(distances, dtype=np.float32)
    q_array = np.stack(q_history, axis=0).astype(np.float32) if q_history else np.zeros((0, agent.actor.state_dim), dtype=np.float32)
    r_array = np.stack(r_history, axis=0).astype(np.float32) if r_history else np.zeros((0, agent.actor.action_dim), dtype=np.float32)
    q_final_array = last_q_final if last_q_final is not None else np.zeros(agent.actor.state_dim, dtype=np.float32)
    r_final_array = last_r_final if last_r_final is not None else np.zeros(agent.actor.action_dim, dtype=np.float32)
    goal_radius = float(getattr(env, "goal_radius", 0.15))

    return EvalEpisodeLog(
        positions=np.stack(positions, axis=0).astype(np.float32),
        velocities=velocity_array,
        target_history=np.stack(target_history, axis=0).astype(np.float32),
        visited_waypoints=visited_array,
        actions=actions_array,
        rewards=rewards_array,
        distances=distance_array,
        q_history=q_array,
        r_history=r_array,
        q_terminal=q_final_array,
        r_terminal=r_final_array,
        total_reward=float(total_reward),
        steps=steps,
        waypoints_reached=waypoints_reached,
        dt=float(getattr(env, "dt", 0.05)),
        goal_radius=goal_radius,
    )


def evaluate_checkpoint(
    agent_cfg,
    checkpoint_path: Path,
    *,
    dynamics_fn,
    dynamics_jacobian_fn,
    env_kwargs: Dict[str, float | int],
    history_window: int,
    eval_runs: int,
    eval_max_steps: int,
    output_dir: Path,
    device: str,
    seed: int,
) -> None:
    if checkpoint_path is None or not checkpoint_path.exists():
        print("No checkpoint available: skipping evaluation.")
        return

    print(f"\n=== Evaluating checkpoint {checkpoint_path.name} ===")
    agent_cfg_eval = copy.deepcopy(agent_cfg)
    agent_cfg_eval.device = device
    eval_agent = ActorCriticAgent(
        agent_cfg_eval,
        dynamics_fn=dynamics_fn,
        dynamics_jacobian_fn=dynamics_jacobian_fn,
    )
    payload = torch.load(checkpoint_path, map_location=device)
    if "agent" not in payload:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'agent' weights.")
    state_dict = payload["agent"]
    filtered_state_dict = {
        key: tensor
        for key, tensor in state_dict.items()
        if ".cost_module.x_ref" not in key and ".cost_module.u_ref" not in key
    }
    dropped = set(state_dict.keys()) - set(filtered_state_dict.keys())
    incompatible = eval_agent.load_state_dict(filtered_state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if dropped or missing or unexpected:
        print(
            "Warning: some tensors were ignored during checkpoint loading "
            f"(dropped={sorted(dropped)}, missing={missing}, unexpected={unexpected})."
        )
    eval_agent.to(device)

    obs_norm_state = payload.get("observation_normalizer")
    wp_norm_state = payload.get("waypoint_normalizer")
    lidar_norm_state = payload.get("lidar_normalizer")
    obs_normalizer = None
    waypoint_normalizer = None
    lidar_normalizer = None
    if obs_norm_state is not None:
        obs_normalizer = ObservationNormalizer(name="observation")
        obs_normalizer.load_state_dict(obs_norm_state)
        obs_normalizer.to(torch.device(device))
    if wp_norm_state is not None:
        waypoint_normalizer = ObservationNormalizer(name="waypoint")
        waypoint_normalizer.load_state_dict(wp_norm_state)
        waypoint_normalizer.to(torch.device(device))
    if lidar_norm_state is not None:
        lidar_normalizer = ObservationNormalizer(name="lidar")
        lidar_normalizer.load_state_dict(lidar_norm_state)
        lidar_normalizer.to(torch.device(device))

    logs: List[EvalEpisodeLog] = []
    env = SE2WaypointEnv(**env_kwargs)
    try:
        for episode_idx in range(eval_runs):
            episode_seed = seed + episode_idx
            log = run_eval_episode(
                eval_agent,
                env,
                history_window=history_window,
                max_steps=eval_max_steps,
                device=torch.device(device),
                seed=episode_seed,
                obs_normalizer=obs_normalizer,
                waypoint_normalizer=waypoint_normalizer,
                lidar_normalizer=lidar_normalizer,
            )
            logs.append(log)
            plot_path = output_dir / f"episode_{episode_idx:02d}.png"
            save_individual_run_plot(log, plot_path, title_prefix="Economic AC-MPC (SE2 Kinematic)")
            print(
                f"Ep {episode_idx:02d}: steps={log.steps:03d} | reward={log.total_reward:.2f} | waypoints={log.waypoints_reached}"
            )
    finally:
        env.close()

    if not logs:
        print("No evaluation episodes executed.")
        return

    avg_waypoints = float(np.mean([log.waypoints_reached for log in logs]))
    avg_reward = float(np.mean([log.total_reward for log in logs]))
    summary_path = output_dir / "summary.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Checkpoint: {checkpoint_path}\n")
        handle.write(f"Episodes: {len(logs)}\n")
        handle.write(f"Average waypoints: {avg_waypoints:.3f}\n")
        handle.write(f"Average reward: {avg_reward:.3f}\n")
        if any(log.q_terminal.size > 0 for log in logs):
            q_term = np.stack([log.q_terminal for log in logs if log.q_terminal.size > 0], axis=0)
            handle.write(f"Final Q diag (mean): {np.mean(q_term, axis=0)}\n")
        if any(log.r_terminal.size > 0 for log in logs):
            r_term = np.stack([log.r_terminal for log in logs if log.r_terminal.size > 0], axis=0)
            handle.write(f"Final R diag (mean): {np.mean(r_term, axis=0)}\n")
    save_multi_run_plot(logs, output_dir / "multi_run_summary.png", title_prefix="Economic AC-MPC (SE2 Kinematic)")
    print(f"Evaluation completed. Average waypoints={avg_waypoints:.2f} | Average reward={avg_reward:.2f}")
    print(f"Plots saved in: {output_dir}")


def main() -> None:
    args = parse_args()
    overrides = args.override if args.override else None
    config = load_experiment_config(args.config, overrides=overrides)

    episode_len = args.episode_len if args.episode_len is not None else config.sampler.episode_len
    env_kwargs = build_env_kwargs(args, episode_len=episode_len)

    checkpoint_dir = args.checkpoint_dir.expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = args.eval_output_dir.expanduser()
    best_ckpt, latest_ckpt = discover_checkpoints(config.training, checkpoint_dir)
    resume_target = select_resume_path(args.resume, best=best_ckpt, latest=latest_ckpt)

    device = resolve_device(args.device, config.device)
    if args.rollouts_per_iter < 1:
        raise ValueError("rollouts-per-iter must be >= 1.")
    if not args.eval_only and args.total_iters < 1:
        raise ValueError("total-iters must be >= 1 se il training è abilitato.")

    set_global_seed(args.seed)
    apply_cli_overrides(
        config,
        args,
        device=device,
        env_kwargs=env_kwargs,
        checkpoint_dir=checkpoint_dir,
        resume_path=resume_target,
    )

    dims = probe_dimensions(dict(env_kwargs, env_id=0))
    rollout_len = config.sampler.rollout_steps
    prepare_config(
        config,
        dims=dims,
        history_window=config.model.history_window,
        rollout_len=rollout_len,
        mpc_horizon=config.model.actor.mpc.horizon,
        dt=env_kwargs["dt"],
        action_limit=env_kwargs["max_speed"],
        device=device,
    )

    agent_cfg = config.model.build_agent_config()
    agent_cfg.device = device
    agent_cfg_eval_template = copy.deepcopy(agent_cfg)
    dynamics_fn, dynamics_jac = build_se2_kinematic_dynamics(
        max_speed=env_kwargs["max_speed"],
    )
    agent = ActorCriticAgent(
        agent_cfg,
        dynamics_fn=dynamics_fn,
        dynamics_jacobian_fn=dynamics_jac,
    )
    with torch.no_grad():
        # Start with a low but sane log_std so the policy is not numerically singular.
        agent.actor.log_std.fill_(-3.0)

    diagnostics = create_diagnostics_options(config)
    loop = TrainingLoop(agent, config.training, diagnostics=diagnostics)
    print_run_header(config, env_kwargs, device, checkpoint_dir)

    # Se esiste già un checkpoint e non è stato richiesto esplicitamente
    # eval-only, skip training and proceed directly to evaluation.
    perform_training = not args.eval_only and args.total_iters > 0 and resume_target is None
    checkpoint_path: Optional[Path] = resume_target

    if perform_training:
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

            for iteration in range(1, args.total_iters + 1):
                batches = [
                    collector.collect(horizon=config.sampler.rollout_steps)
                    for _ in range(args.rollouts_per_iter)
                ]
                metrics = loop.run(batches)
                avg_waypoints = compute_mean_waypoints(batches)
                avg_reward = compute_mean_episode_reward(batches)

                progress = iteration / float(args.total_iters)
                bar_length = 25
                filled = int(bar_length * progress)
                bar = "█" * filled + "░" * (bar_length - filled)

                summary = (
                    f"\rIter {iteration}/{args.total_iters} "
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
                if avg_reward is not None:
                    summary += f" reward={avg_reward:.2f}"
                if avg_waypoints is not None:
                    summary += f" waypoints={avg_waypoints:.2f}"
                end_char = "\n" if iteration == args.total_iters else ""
                print(summary, end=end_char, flush=True)

        except KeyboardInterrupt:
            print("Interrupted by user; closing environments.")
        finally:
            if env_manager is not None:
                env_manager.close()

        if loop.checkpoint_manager is not None:
            checkpoint_path = loop.checkpoint_manager.best_checkpoint()
            if checkpoint_path is None:
                checkpoint_path = loop.checkpoint_manager.latest_checkpoint()
        if checkpoint_path is None:
            print("Warning: no checkpoint saved; skipping evaluation.")
    else:
        print("Training disabled: proceeding directly to evaluation.")

    if not args.skip_eval and args.eval_runs > 0:
        eval_seed = args.eval_seed if args.eval_seed is not None else (args.seed + 1337)
        eval_max_steps = args.eval_max_steps if args.eval_max_steps is not None else env_kwargs["episode_len"]
        if checkpoint_path is None:
            print("No checkpoint available: cannot execute evaluation.")
        else:
            evaluate_checkpoint(
                agent_cfg_eval_template,
                checkpoint_path,
                dynamics_fn=dynamics_fn,
                dynamics_jacobian_fn=dynamics_jac,
                env_kwargs=env_kwargs,
                history_window=config.model.history_window,
                eval_runs=args.eval_runs,
                eval_max_steps=int(eval_max_steps),
                output_dir=eval_output_dir,
                device=device,
                seed=eval_seed,
            )
    else:
        print("Evaluation disabled or number of runs set to 0.")


if __name__ == "__main__":
    main()
