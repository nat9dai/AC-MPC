#!/usr/bin/env python3
"""Dummy training script used by tests/examples/test_run_dummy_training.py.

It runs a tiny PPO training loop for the SE(2) waypoint environment using the
`se2_kinematic_fixed.yaml` defaults, with logging options optionally overridden
by a minimal config (e.g. configs/minimal.yaml).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ACMPC import ActorCriticAgent, RolloutCollector, TrainingLoop, load_experiment_config  # type: ignore  # noqa: E402
from examples.se2_waypoint_common import (  # type: ignore  # noqa: E402
    build_env_manager,
    prepare_config,
    probe_dimensions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--tensorboard", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    base_cfg = load_experiment_config(ROOT / "configs" / "se2_kinematic_fixed.yaml")
    override_cfg = load_experiment_config(args.config)

    # Merge training/logging fields from the override config.
    base_cfg.training.log_interval = override_cfg.training.log_interval
    base_cfg.training.log_to_stdout = override_cfg.training.log_to_stdout
    base_cfg.training.checkpoint_interval = override_cfg.training.checkpoint_interval
    base_cfg.training.log_jsonl_path = str(Path(args.log_dir) / "training.jsonl")
    base_cfg.training.tensorboard_dir = args.tensorboard
    base_cfg.training.device = "cpu"
    base_cfg.device = "cpu"

    device = "cpu"

    env_kwargs = {
        "dt": 0.05,
        "episode_len": 400,
        "waypoint_range": 2.0,
        "goal_radius": 0.15,
        "min_start_radius": 0.2,
        "max_speed": 4.0,
        "progress_gain": 2.0,
        "action_penalty": 0.0,
        "living_penalty": 0.0,
        "goal_bonus": 20.0,
        "control_gain": 0.01,
    }

    dims = probe_dimensions(dict(env_kwargs, env_id=0))
    prepare_config(
        base_cfg,
        dims=dims,
        history_window=base_cfg.model.history_window,
        rollout_len=base_cfg.sampler.rollout_steps,
        mpc_horizon=base_cfg.model.actor.mpc.horizon,
        dt=env_kwargs["dt"],
        action_limit=env_kwargs["max_speed"],
        device=device,
    )

    env_manager = build_env_manager(
        num_envs=base_cfg.sampler.num_envs,
        env_kwargs=env_kwargs,
        seed=base_cfg.seed,
        device=device,
    )

    from ACMPC.envs import build_se2_kinematic_dynamics  # type: ignore  # noqa: E402

    dynamics_fn, dynamics_jac = build_se2_kinematic_dynamics(max_speed=env_kwargs["max_speed"])
    agent_cfg = base_cfg.model.build_agent_config()
    agent_cfg.device = device
    agent = ActorCriticAgent(agent_cfg, dynamics_fn=dynamics_fn, dynamics_jacobian_fn=dynamics_jac)

    collector = RolloutCollector(
        agent=agent,
        env_manager=env_manager,
        history_window=base_cfg.model.history_window,
        horizon=base_cfg.model.actor.mpc.horizon,
        device=torch.device(device),
        collect_plan_rewards=False,
        collect_plan_observations=False,
    )

    loop = TrainingLoop(agent, base_cfg.training)

    for _ in range(max(args.steps, 1)):
        rollout = collector.collect(horizon=base_cfg.sampler.rollout_steps)
        loop.run([rollout])

    print("Aggiornamenti PPO")


if __name__ == "__main__":
    main()
