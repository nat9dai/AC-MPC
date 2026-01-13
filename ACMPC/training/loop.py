"""Proximal policy optimisation utilities for ACMPC."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ..agent import ActorCriticAgent
from .checkpoint import CheckpointConfig, CheckpointManager
from .diagnostics import DiagnosticsManager, DiagnosticsOptions
from .gradients import GradientManager, GradientManagerConfig
from .logger import LoggerConfig, TrainingLogger
from .normalization import ObservationNormalizer, RewardNormalizer
from ..sampling.rollout import RolloutBatch


@dataclass
class TrainingBatch:
    """Flattened batch of PPO training samples.

    The optional ``mask`` flags padded rollouts so downstream steps can drop
    invalid samples before computing losses.
    """

    obs_seq: Tensor  # [N, T, obs_dim]
    state: Tensor  # [N, state_dim]
    action: Tensor  # [N, action_dim]
    old_log_prob: Tensor  # [N]
    returns: Tensor  # [N]
    advantages: Tensor  # [N]
    raw_state: Optional[Tensor] = None  # [N, state_dim]
    waypoint_seq: Optional[Tensor] = None  # [N, W, waypoint_dim]
    raw_waypoint_seq: Optional[Tensor] = None  # [N, W, waypoint_dim]
    warm_start: Optional[Tensor] = None  # [N, horizon, action_dim]
    mask: Optional[Tensor] = None  # [N]
    old_value: Optional[Tensor] = None  # [N]
    mpve_target: Optional[Tensor] = None  # [N]
    lidar_seq: Optional[Tensor] = None  # [N, lidar_dim]


@dataclass
@dataclass
class MPVEConfig:
    """Configuration for Model-Predictive Value Expansion."""

    enabled: bool = False
    td_k: int = 1
    horizon: Optional[int] = None
    loss_weight: float = 1.0

    def validate(self) -> None:
        if self.td_k < 1:
            raise ValueError("mpve.td_k must be >= 1.")
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("mpve.horizon must be >= 1 when provided.")
        if self.loss_weight <= 0.0:
            raise ValueError("mpve.loss_weight must be > 0.")


@dataclass
class TrainingConfig:
    """Hyper-parameters controlling PPO updates."""

    device: str = "cpu"
    actor_lr: float = 2e-4
    critic_lr: float = 1e-4
    weight_decay: float = 0.0
    clip_param: float = 0.12
    entropy_coeff: float = 0.04
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    mini_batch_size: int = 128
    advantage_normalization: bool = True
    use_amp: bool = False
    target_kl: Optional[float] = 0.02
    actor_scheduler_T0: int = 20
    critic_scheduler_patience: int = 10
    gamma: float = 0.98
    gae_lambda: float = 0.90
    mpve: MPVEConfig = field(default_factory=MPVEConfig)
    grad_norm_type: float = 2.0
    gradient_accumulation_steps: int = 1
    log_grad_norm: bool = False
    normalize_returns: bool = False
    return_clip_value: Optional[float] = None
    reward_norm_epsilon: float = 1e-8
    value_clip_range: Optional[float] = None
    log_interval: int = 10
    log_to_stdout: bool = False
    log_jsonl_path: Optional[str] = None
    log_append: bool = True
    tensorboard_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 0
    checkpoint_metric: str = "value_loss"
    checkpoint_mode: str = "min"
    checkpoint_keep_last: int = 3
    resume_path: Optional[str] = None
    resume_best: bool = False
    resume_latest: bool = False
    # Alignment auxiliary loss for cost map regularization (double integrator specific)
    alignment_loss_coeff: float = 0.0
    normalize_observations: bool = False
    normalize_state: Optional[bool] = None
    normalize_waypoint: Optional[bool] = None
    normalize_lidar: Optional[bool] = None
    observation_norm_epsilon: float = 1e-8
    lr_schedule: str = "none"  # options: none, linear
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_final_factor: float = 0.1

    def resolve_normalization(self) -> None:
        if self.normalize_state is None:
            self.normalize_state = self.normalize_observations
        if self.normalize_waypoint is None:
            self.normalize_waypoint = self.normalize_observations
        if self.normalize_lidar is None:
            self.normalize_lidar = self.normalize_observations

    def __post_init__(self) -> None:
        self.mpve.validate()
        resume_flags = sum(
            1 for flag in (self.resume_path, self.resume_best, self.resume_latest) if bool(flag)
        )
        if resume_flags > 1:
            raise ValueError("Specify at most one of resume_path/resume_best/resume_latest.")
        if self.lr_schedule not in {"none", "linear"}:
            raise ValueError("lr_schedule must be one of: none, linear.")
        if self.lr_warmup_steps < 0 or self.lr_decay_steps < 0:
            raise ValueError("lr_warmup_steps and lr_decay_steps must be >= 0.")
        if self.lr_final_factor <= 0:
            raise ValueError("lr_final_factor must be > 0.")
        self.resolve_normalization()


@dataclass
class TrainingMetrics:
    """Aggregated metrics recorded during PPO updates."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    updates: int
    actor_grad_norm: Optional[float] = None
    critic_grad_norm: Optional[float] = None
    alignment_loss: Optional[float] = None


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Compute Generalised Advantage Estimation for batched rollouts."""

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [batch, time]")
    batch, time = rewards.shape
    if values.shape != (batch, time + 1):
        raise ValueError("values must have shape [batch, time+1]")
    if dones.shape != (batch, time):
        raise ValueError("dones must have shape [batch, time]")

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(time)):
        mask = 1.0 - dones[:, t].float()
        delta = rewards[:, t] + gamma * values[:, t + 1] * mask - values[:, t]
        gae = delta + gamma * lam * mask * gae
        advantages[:, t] = gae

    returns = advantages + values[:, :-1]
    return advantages, returns


def rollout_to_training_batch(
    agent: ActorCriticAgent,
    rollout: RolloutBatch,
    config: TrainingConfig,
    *,
    reward_normalizer: Optional[RewardNormalizer] = None,
    obs_normalizer: Optional[ObservationNormalizer] = None,
    waypoint_normalizer: Optional[ObservationNormalizer] = None,
    lidar_normalizer: Optional[ObservationNormalizer] = None,
) -> TrainingBatch:
    """Convert a rollout batch into a :class:`TrainingBatch` with GAE targets."""

    if config.mpve.enabled:
        raise NotImplementedError("MPVE support is not implemented yet in the refactored trainer.")
    if config.normalize_returns and reward_normalizer is None:
        raise ValueError("Reward normalizer required when normalize_returns is enabled.")

    device = torch.device(config.device)

    history = rollout.history.to(device=device)
    reward = rollout.reward.to(device=device)
    done = rollout.done.to(device=device)
    mask = rollout.mask.to(device=device)
    state = rollout.state.to(device=device)
    action = rollout.action.to(device=device)
    log_prob = rollout.log_prob.to(device=device)
    waypoint_seq = rollout.waypoint_seq.to(device=device)
    next_history = rollout.next_history.to(device=device)
    next_state = rollout.next_state.to(device=device)
    next_waypoint_seq = rollout.next_waypoint_seq.to(device=device)
    plan_actions = rollout.plan_actions.to(device=device)
    old_value = getattr(rollout, "old_value", None)
    old_value = old_value.to(device=device) if old_value is not None else None

    num_envs, horizon, history_len, obs_dim = history.shape
    history_flat = history.view(num_envs * horizon, history_len, obs_dim).contiguous()
    state_flat = state.view(num_envs * horizon, state.size(-1)).to(device=device)
    mask_flat = mask.view(num_envs * horizon)
    old_value_flat = old_value.view(num_envs * horizon).to(device=device) if old_value is not None else None

    waypoint_flat: Optional[Tensor]
    if waypoint_seq.numel() == 0 or waypoint_seq.size(2) == 0:
        waypoint_flat = None
        next_waypoint = None
    else:
        waypoint_flat = waypoint_seq.view(
            num_envs * horizon,
            waypoint_seq.size(2),
            waypoint_seq.size(3),
        ).contiguous()
        next_waypoint = next_waypoint_seq
    # Optional lidar sequence: [env, time, lidar_dim]
    lidar_seq = getattr(rollout, "lidar", None)
    next_lidar = getattr(rollout, "next_lidar", None)
    lidar_flat: Optional[Tensor] = None
    if lidar_seq is not None and lidar_seq.numel() > 0:
        lidar_seq = lidar_seq.to(device=device)
        if lidar_seq.dim() != 3:
            raise ValueError("lidar must have shape [env, time, lidar_dim].")
        _, _, lidar_dim = lidar_seq.shape
        lidar_flat = lidar_seq.view(num_envs * horizon, lidar_dim).contiguous()
        if next_lidar is not None and next_lidar.numel() > 0:
            next_lidar = next_lidar.to(device=device)
        else:
            next_lidar = torch.zeros(num_envs, lidar_dim, device=device, dtype=history.dtype)
    else:
        next_lidar = None

    raw_state_flat = state_flat
    raw_next_state = next_state
    raw_waypoint_flat = waypoint_flat
    raw_next_waypoint = next_waypoint
    raw_lidar_flat = lidar_flat
    raw_next_lidar = next_lidar

    history_for_backbone = history_flat
    state_for_backbone = state_flat
    waypoint_for_backbone = waypoint_flat
    next_waypoint_for_backbone = next_waypoint
    lidar_for_backbone = lidar_flat
    next_lidar_for_backbone = next_lidar

    local_obs_norm = obs_normalizer
    local_wp_norm = waypoint_normalizer
    local_lidar_norm = lidar_normalizer

    normalize_state = bool(config.normalize_state)
    normalize_waypoint = bool(config.normalize_waypoint)
    normalize_lidar = bool(config.normalize_lidar)

    if normalize_state:
        if local_obs_norm is None:
            local_obs_norm = ObservationNormalizer(
                epsilon=config.observation_norm_epsilon,
                name="observation",
            )
        local_obs_norm.to(device)
        local_obs_norm.update(raw_state_flat, mask=mask_flat)
        state_for_backbone = local_obs_norm.normalize(raw_state_flat, mask=mask_flat)
        history_for_backbone = local_obs_norm.normalize(history_flat)
        next_history = local_obs_norm.normalize(next_history)

    if normalize_waypoint:
        waypoint_mask: Optional[Tensor] = None
        if waypoint_flat is not None:
            if local_wp_norm is None:
                local_wp_norm = ObservationNormalizer(
                    epsilon=config.observation_norm_epsilon,
                    name="waypoint",
                )
            local_wp_norm.to(device)
            waypoint_mask = mask_flat.view(-1, 1, 1)
            local_wp_norm.update(waypoint_flat, mask=waypoint_mask)
            waypoint_for_backbone = local_wp_norm.normalize(waypoint_flat, mask=waypoint_mask)
            if next_waypoint is not None:
                next_waypoint_for_backbone = local_wp_norm.normalize(next_waypoint)

    if normalize_lidar and lidar_flat is not None:
        if local_lidar_norm is None:
            local_lidar_norm = ObservationNormalizer(
                epsilon=config.observation_norm_epsilon,
                name="lidar",
            )
        local_lidar_norm.to(device)
        local_lidar_norm.update(lidar_flat, mask=mask_flat)
        lidar_for_backbone = local_lidar_norm.normalize(lidar_flat, mask=mask_flat)
        if next_lidar is not None:
            next_lidar_for_backbone = local_lidar_norm.normalize(next_lidar)

    previous_mode = agent.training
    agent.eval()
    # Determine whether the critic is configured to consume lidar features.
    critic_cfg = getattr(getattr(agent, "config", None), "critic", None)
    critic_uses_lidar = bool(getattr(critic_cfg, "include_lidar", False)) if critic_cfg is not None else False

    with torch.no_grad():
        critic_kwargs: Dict[str, Tensor] = {}
        if waypoint_for_backbone is not None:
            critic_kwargs["waypoint_seq"] = waypoint_for_backbone
        if raw_waypoint_flat is not None:
            critic_kwargs["raw_waypoint_seq"] = raw_waypoint_flat
        # FIX: Pass raw_state for SE2 geometric transforms (matches actor)
        critic_kwargs["raw_state"] = raw_state_flat
        if critic_uses_lidar and lidar_for_backbone is not None:
            critic_kwargs["lidar"] = lidar_for_backbone
        value_flat = agent.value(history_for_backbone, **critic_kwargs).value.view(num_envs, horizon)

        next_kwargs: Dict[str, Tensor] = {}
        if next_waypoint_for_backbone is not None:
            next_kwargs["waypoint_seq"] = next_waypoint_for_backbone
        if raw_next_waypoint is not None:
            next_kwargs["raw_waypoint_seq"] = raw_next_waypoint
        # FIX: Pass raw_state for bootstrap values (use next_state as raw_state)
        next_kwargs["raw_state"] = raw_next_state
        if critic_uses_lidar and next_lidar_for_backbone is not None:
            next_kwargs["lidar"] = next_lidar_for_backbone
        bootstrap_values = agent.value(next_history, **next_kwargs).value  # [E]
    agent.train(previous_mode)

    # Build value tensor with bootstrap column.
    value_tensor = torch.cat([value_flat, bootstrap_values.unsqueeze(1)], dim=1)

    mask_bool = mask > 0.0
    reward = reward * mask
    done = torch.where(
        mask_bool,
        done,
        torch.ones_like(done, dtype=done.dtype),
    )

    advantages, returns = compute_gae(
        rewards=reward,
        values=value_tensor,
        dones=done,
        gamma=config.gamma,
        lam=config.gae_lambda,
    )
    advantages = advantages * mask
    returns = returns * mask

    if config.normalize_returns and reward_normalizer is not None:
        returns = reward_normalizer.normalize(returns, mask=mask)

    obs_seq = history_for_backbone
    action_flat = action.view(num_envs * horizon, action.size(-1)).to(device=device)
    log_prob_flat = log_prob.view(num_envs * horizon).to(device=device)
    warm_start = (
        plan_actions.view(num_envs * horizon, plan_actions.size(2), plan_actions.size(3)).to(device=device)
        if plan_actions.numel() > 0
        else None
    )
    mpve_target_flat: Optional[Tensor] = None

    if config.mpve.enabled:
        if rollout.plan_rewards is None:
            raise ValueError("MPVE enabled but rollout is missing predicted plan rewards.")
        plan_rewards = rollout.plan_rewards.to(device=device)
        plan_states = rollout.plan_states.to(device=device)
        plan_observations = rollout.plan_observations.to(device=device) if rollout.plan_observations is not None else None
        if plan_rewards.dim() != 3:
            raise ValueError("plan_rewards must have shape [env, time, horizon].")
        mpve_horizon = plan_rewards.size(2)
        if config.mpve.horizon is not None:
            mpve_horizon = min(mpve_horizon, config.mpve.horizon)
        if mpve_horizon < 1:
            raise ValueError("MPVE horizon must be >= 1 when MPVE is enabled.")
        td_k = min(config.mpve.td_k, mpve_horizon)
        obs_dim = history.shape[-1]
        if plan_observations is not None:
            if plan_observations.dim() != 4:
                raise ValueError("plan_observations must have shape [env, time, horizon+1, obs_dim].")
            if plan_observations.shape[0] != num_envs or plan_observations.shape[1] != horizon:
                raise ValueError("plan_observations env/time dimensions must match rollout.")
            if plan_observations.shape[-1] != obs_dim:
                raise ValueError("plan_observations last dimension must match observation dimension.")
            if plan_observations.shape[2] < mpve_horizon + 1:
                raise ValueError("plan_observations horizon insufficient for configured MPVE horizon.")
        else:
            state_dim = plan_states.shape[-1]
            if state_dim != obs_dim:
                raise NotImplementedError(
                    "MPVE requires plan observations or state dimensionality matching observations. "
                    "Provide an observation prediction function or disable MPVE."
                )
        dtype = history.dtype
        mpve_targets = torch.zeros(num_envs, horizon, device=device, dtype=dtype)

        trimmed_states = (
            plan_observations[:, :, : mpve_horizon + 1, :]
            if plan_observations is not None
            else plan_states[:, :, : mpve_horizon + 1, :obs_dim]
        )
        trimmed_rewards = plan_rewards[:, :, :mpve_horizon]

        previous_mode = agent.training
        agent.eval()
        with torch.no_grad():
            for env_idx in range(num_envs):
                for step_idx in range(horizon):
                    hist_seq = history[env_idx, step_idx].clone()
                    reward_seq = trimmed_rewards[env_idx, step_idx]
                    obs_seq = trimmed_states[env_idx, step_idx]
                    waypoint = (
                        waypoint_seq[env_idx, step_idx]
                        if waypoint_seq.numel() > 0
                        else None
                    )

                    target_accum = torch.zeros((), device=device, dtype=dtype)
                    for k in range(1, td_k + 1):
                        hist_seq = torch.roll(hist_seq, shifts=-1, dims=0)
                        hist_seq[-1] = obs_seq[k]

                        critic_kwargs = {}
                        if waypoint is not None:
                            critic_kwargs["waypoint_seq"] = waypoint.unsqueeze(0)
                            critic_kwargs["raw_waypoint_seq"] = waypoint.unsqueeze(0)
                        # FIX: Pass raw_state for SE2 transforms
                        critic_kwargs["raw_state"] = obs_seq[k].unsqueeze(0)
                        future_value = agent.value(hist_seq.unsqueeze(0), **critic_kwargs).value.squeeze(0)

                        discount = torch.ones((), device=device, dtype=dtype)
                        reward_sum = torch.zeros((), device=device, dtype=dtype)
                        for j in range(k):
                            reward_sum = reward_sum + discount * reward_seq[j]
                            discount = discount * config.gamma
                        target_k = reward_sum + discount * future_value
                        target_accum = target_accum + target_k

                    mpve_targets[env_idx, step_idx] = target_accum / float(td_k)
        agent.train(previous_mode)

        mpve_target_flat = mpve_targets.reshape(-1)

    return TrainingBatch(
        obs_seq=obs_seq,
        state=state_for_backbone,
        raw_state=raw_state_flat,
        action=action_flat,
        old_log_prob=log_prob_flat,
        returns=returns.reshape(-1),
        advantages=advantages.reshape(-1),
        waypoint_seq=waypoint_for_backbone,
        raw_waypoint_seq=raw_waypoint_flat,
        warm_start=warm_start,
        mask=mask_flat,
        mpve_target=mpve_target_flat,
        lidar_seq=lidar_for_backbone,
        old_value=old_value_flat,
    )


class TrainingLoop:
    """PPO training loop for an :class:`ActorCriticAgent`."""

    def __init__(
        self,
        agent: ActorCriticAgent,
        config: TrainingConfig,
        diagnostics: Optional[DiagnosticsOptions] = None,
    ) -> None:
        self.agent = agent
        self.config = config
        self.device = torch.device(config.device)
        self.agent.to(self.device)
        if config.log_interval < 1:
            raise ValueError("log_interval must be >= 1.")
        if config.checkpoint_mode not in {"min", "max"}:
            raise ValueError("checkpoint_mode must be 'min' or 'max'.")
        if config.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be >= 0.")
        if config.checkpoint_dir is not None and config.checkpoint_keep_last < 1:
            raise ValueError("checkpoint_keep_last must be >= 1 when checkpoint_dir is set.")

        self.actor_opt = torch.optim.Adam(
            self.agent.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.weight_decay,
        )
        self.critic_opt = torch.optim.Adam(
            self.agent.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )

        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.actor_opt,
            T_0=max(1, config.actor_scheduler_T0),
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_opt,
            patience=max(1, config.critic_scheduler_patience),
            factor=0.5,
        )

        gradient_cfg = GradientManagerConfig(
            max_norm=config.max_grad_norm,
            norm_type=config.grad_norm_type,
            accumulation_steps=max(1, config.gradient_accumulation_steps),
            use_amp=config.use_amp,
            log_norm=config.log_grad_norm,
        )
        self.grad_manager = GradientManager(
            actor_opt=self.actor_opt,
            critic_opt=self.critic_opt,
            actor_params=list(self.agent.actor.parameters()),
            critic_params=list(self.agent.critic.parameters()),
            device=self.device,
            config=gradient_cfg,
        )
        self._base_actor_lr = config.actor_lr
        self._base_critic_lr = config.critic_lr
        self.reward_normalizer = (
            RewardNormalizer(
                epsilon=config.reward_norm_epsilon,
                clip_value=config.return_clip_value,
            )
            if config.normalize_returns
            else None
        )
        self.observation_normalizer = (
            ObservationNormalizer(epsilon=config.observation_norm_epsilon, name="observation")
            if config.normalize_state
            else None
        )
        self.waypoint_normalizer = (
            ObservationNormalizer(epsilon=config.observation_norm_epsilon, name="waypoint")
            if config.normalize_waypoint
            else None
        )
        self.lidar_normalizer = (
            ObservationNormalizer(epsilon=config.observation_norm_epsilon, name="lidar")
            if config.normalize_lidar
            else None
        )
        logger_jsonl = Path(config.log_jsonl_path).expanduser() if config.log_jsonl_path else None
        tensorboard_dir = Path(config.tensorboard_dir).expanduser() if config.tensorboard_dir else None
        if config.log_to_stdout or logger_jsonl is not None or tensorboard_dir is not None or config.wandb_project:
            logger_cfg = LoggerConfig(
                log_interval=config.log_interval,
                log_to_stdout=config.log_to_stdout,
                jsonl_path=logger_jsonl,
                append_jsonl=config.log_append,
                tensorboard_dir=tensorboard_dir,
                wandb_project=config.wandb_project,
                wandb_entity=config.wandb_entity,
                wandb_run_name=config.wandb_run_name,
                wandb_mode=config.wandb_mode,
            )
            self.logger: Optional[TrainingLogger] = TrainingLogger(logger_cfg)
        else:
            self.logger = None

        if config.checkpoint_dir is not None and config.checkpoint_interval > 0:
            checkpoint_cfg = CheckpointConfig(
                directory=Path(config.checkpoint_dir).expanduser(),
                metric=config.checkpoint_metric,
                mode=config.checkpoint_mode,
                keep_last=config.checkpoint_keep_last,
            )
            self.checkpoint_manager: Optional[CheckpointManager] = CheckpointManager(checkpoint_cfg)
        else:
            self.checkpoint_manager = None
        self._checkpoint_interval = config.checkpoint_interval
        self._global_updates = 0
        self.diagnostics_manager = DiagnosticsManager(diagnostics) if diagnostics else None
        self._last_checkpoint_metrics: Optional[Dict[str, float]] = None

        if self.config.resume_path or self.config.resume_best or self.config.resume_latest:
            self.resume_from_checkpoint(
                path=self.config.resume_path,
                best=self.config.resume_best,
                latest=self.config.resume_latest,
            )

    def run(self, batches: Iterable[TrainingBatch | RolloutBatch]) -> TrainingMetrics:
        batches = list(batches) if not isinstance(batches, Sequence) else batches
        if len(batches) == 0:
            raise ValueError("At least one training batch is required for PPO updates.")

        first = batches[0]
        diag_metrics: Optional[Dict[str, float]] = None
        if isinstance(first, RolloutBatch):
            training_batches = [
                rollout_to_training_batch(
                    self.agent,
                    batch,
                    self.config,
                    reward_normalizer=self.reward_normalizer,
                    obs_normalizer=self.observation_normalizer,
                    waypoint_normalizer=self.waypoint_normalizer,
                    lidar_normalizer=self.lidar_normalizer,
                )
                for batch in batches  # type: ignore[arg-type]
            ]
            if self.diagnostics_manager is not None:
                diag_metrics = self.diagnostics_manager.process_rollouts(batches)
        elif isinstance(first, TrainingBatch):
            training_batches = batches  # type: ignore[assignment]
        else:
            raise TypeError("Unsupported batch type provided to TrainingLoop.run.")

        stacked = self._stack_batches(training_batches)
        return self._run_epochs(stacked, extra_metrics=diag_metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _stack_batches(self, batches: Sequence[TrainingBatch]) -> Dict[str, Tensor | None]:
        keys = [
            "obs_seq",
            "state",
            "raw_state",
            "action",
            "old_log_prob",
            "returns",
            "advantages",
            "waypoint_seq",
            "raw_waypoint_seq",
            "lidar_seq",
            "old_value",
        ]
        stacked: Dict[str, Tensor | None] = {}
        for key in keys:
            tensors = [getattr(batch, key) for batch in batches]
            if tensors and tensors[0] is not None:
                stacked[key] = torch.cat(tensors, dim=0).to(self.device)
            else:
                stacked[key] = None

        warm_tensors = [batch.warm_start for batch in batches if batch.warm_start is not None]
        stacked["warm_start"] = (
            torch.cat(warm_tensors, dim=0).to(self.device) if warm_tensors else None
        )
        mask_tensors = [batch.mask for batch in batches if batch.mask is not None]
        stacked["mask"] = (
            torch.cat(mask_tensors, dim=0).to(self.device) if mask_tensors else None
        )
        mpve_tensors = [batch.mpve_target for batch in batches if batch.mpve_target is not None]
        stacked["mpve_target"] = (
            torch.cat(mpve_tensors, dim=0).to(self.device) if mpve_tensors else None
        )
        return stacked

    def _run_epochs(
        self,
        stacked: Dict[str, Tensor | None],
        *,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> TrainingMetrics:
        num_samples = stacked["action"].shape[0]  # type: ignore[index]
        mini_batch_size = min(self.config.mini_batch_size, num_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        updates = 0
        grad_actor_sum = 0.0
        grad_critic_sum = 0.0
        grad_norm_updates = 0

        metric_keys = [
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "clip_fraction",
            "returns_mean",
            "returns_std",
            "returns_min",
            "returns_max",
            "advantages_mean",
            "advantages_std",
            "advantages_min",
            "advantages_max",
            "log_prob_mean",
            "log_prob_std",
            "log_prob_min",
            "log_prob_max",
            "old_log_prob_mean",
            "old_log_prob_std",
            "old_log_prob_min",
            "old_log_prob_max",
            "value_mean",
            "value_std",
            "value_min",
            "value_max",
            "value_error_mean",
            "value_error_abs_mean",
            "log_std_min",
            "log_std_max",
            "std_min",
            "std_max",
            "mask_valid_fraction",
        ]

        # We also track the maximum per-update KL to trigger early stopping if needed.
        max_kl_this_iter = 0.0

        def flush_pending(
            pending: Dict[str, float],
            pending_count: int,
            actor_norm: Optional[float],
            critic_norm: Optional[float],
            epoch_value_losses: list[float],
        ) -> int:
            nonlocal total_policy_loss, total_value_loss, total_entropy, total_kl, updates
            nonlocal grad_actor_sum, grad_critic_sum, grad_norm_updates
            if pending_count == 0:
                return 0
            mean_metrics = {key: pending[key] / pending_count for key in metric_keys}
            total_policy_loss += mean_metrics["policy_loss"]
            total_value_loss += mean_metrics["value_loss"]
            total_entropy += mean_metrics["entropy"]
            total_kl += mean_metrics["approx_kl"]
            nonlocal max_kl_this_iter
            max_kl_this_iter = max(max_kl_this_iter, float(mean_metrics["approx_kl"]))
            epoch_value_losses.append(mean_metrics["value_loss"])
            updates += 1
            if actor_norm is not None:
                grad_actor_sum += actor_norm
            if critic_norm is not None:
                grad_critic_sum += critic_norm
            if actor_norm is not None or critic_norm is not None:
                grad_norm_updates += 1
            self._global_updates += 1
            log_entry: Dict[str, float] = dict(mean_metrics)
            if actor_norm is not None:
                log_entry["actor_grad_norm"] = float(actor_norm)
            if critic_norm is not None:
                log_entry["critic_grad_norm"] = float(critic_norm)
            if self.config.max_grad_norm is not None:
                if actor_norm is not None:
                    clip_ratio = min(1.0, self.config.max_grad_norm / (actor_norm + 1e-8))
                    log_entry["actor_clip_ratio"] = float(clip_ratio)
                if critic_norm is not None:
                    clip_ratio = min(1.0, self.config.max_grad_norm / (critic_norm + 1e-8))
                    log_entry["critic_clip_ratio"] = float(clip_ratio)
            log_entry["max_kl_this_iter"] = float(max_kl_this_iter)
            if extra_metrics:
                log_entry.update(extra_metrics)
            if self.logger is not None:
                self.logger.log(self._global_updates, log_entry)
            if (
                self.checkpoint_manager is not None
                and self._checkpoint_interval > 0
                and self._global_updates % self._checkpoint_interval == 0
            ):
                self.checkpoint_manager.save(
                    step=self._global_updates,
                    metrics=log_entry,
                    agent=self.agent,
                    actor_opt=self.actor_opt,
                    critic_opt=self.critic_opt,
                    grad_manager=self.grad_manager,
                    reward_normalizer=self.reward_normalizer,
                    observation_normalizer=self.observation_normalizer,
                    waypoint_normalizer=self.waypoint_normalizer,
                    lidar_normalizer=self.lidar_normalizer,
                    actor_scheduler=self.actor_scheduler,
                    critic_scheduler=self.critic_scheduler,
                )
            for key in metric_keys:
                pending[key] = 0.0
            return 0

        for epoch in range(self.config.ppo_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            epoch_value_losses: list[float] = []
            pending = {key: 0.0 for key in metric_keys}
            pending_count = 0

            for start in range(0, num_samples, mini_batch_size):
                idx = permutation[start : start + mini_batch_size]
                mini_batch = self._slice_batch(stacked, idx)
                self._apply_lr_schedule(self._global_updates + 1)

                self.grad_manager.prepare_microbatch()
                with self.grad_manager.autocast():
                    loss, metrics = self._compute_losses(mini_batch)

                pending_count += 1
                for key in metric_keys:
                    pending[key] += metrics[key]

                stepped, actor_norm, critic_norm = self.grad_manager.backward(loss)
                if stepped:
                    pending_count = flush_pending(pending, pending_count, actor_norm, critic_norm, epoch_value_losses)

                if (
                    self.config.target_kl is not None
                    and metrics["approx_kl"] > self.config.target_kl
                ):
                    break  # Stop this epoch early if KL for the mini-batch is too large.

            stepped, actor_norm, critic_norm = self.grad_manager.finalize()
            if stepped or pending_count > 0:
                pending_count = flush_pending(pending, pending_count, actor_norm, critic_norm, epoch_value_losses)

            self.actor_scheduler.step(epoch + 1)
            if epoch_value_losses:
                mean_value_loss = sum(epoch_value_losses) / len(epoch_value_losses)
                self.critic_scheduler.step(mean_value_loss)

            # Global early-stop for this PPO iteration if KL has grown too much.
            if (
                self.config.target_kl is not None
                and max_kl_this_iter > 5.0 * self.config.target_kl
            ):
                # Avoid further epochs on this batch; PPO has moved far enough.
                break

        if updates == 0:
            raise RuntimeError("No PPO updates were performed; check batch configuration.")

        actor_grad_norm = (
            grad_actor_sum / grad_norm_updates if grad_norm_updates > 0 else None
        )
        critic_grad_norm = (
            grad_critic_sum / grad_norm_updates if grad_norm_updates > 0 else None
        )

        metrics = TrainingMetrics(
            policy_loss=total_policy_loss / updates,
            value_loss=total_value_loss / updates,
            entropy=total_entropy / updates,
            approx_kl=total_kl / updates,
            updates=updates,
            actor_grad_norm=actor_grad_norm,
            critic_grad_norm=critic_grad_norm,
        )
        if self.logger is not None:
            self.logger.close()
        return metrics

    def resume_from_checkpoint(
        self,
        path: Optional[str] = None,
        *,
        best: bool = False,
        latest: bool = False,
    ) -> Dict[str, Any]:
        """Load training state from a saved checkpoint."""

        if path is not None:
            checkpoint_path = Path(path).expanduser()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
            payload = torch.load(checkpoint_path, map_location="cpu")
            payload["_checkpoint_path"] = checkpoint_path
        else:
            if self.checkpoint_manager is None:
                raise RuntimeError("Checkpoint manager not configured; provide explicit path to resume.")
            payload = self.checkpoint_manager.load(best=best, latest=latest)
            checkpoint_path = Path(payload["_checkpoint_path"])

        agent_state = payload["agent"]
        controller = getattr(getattr(self.agent.actor, "mpc_head", None), "_controller", None)
        if controller is not None and hasattr(controller, "cost_module"):
            x_ref_key = "actor.mpc_head._controller.cost_module.x_ref"
            u_ref_key = "actor.mpc_head._controller.cost_module.u_ref"
            if x_ref_key in agent_state and u_ref_key in agent_state:
                with torch.no_grad():
                    controller.cost_module.set_reference(
                        agent_state[x_ref_key].to(self.device),
                        agent_state[u_ref_key].to(self.device),
                    )

        self.agent.load_state_dict(agent_state)
        self.agent.to(self.device)
        self.actor_opt.load_state_dict(payload["actor_optimizer"])
        self.critic_opt.load_state_dict(payload["critic_optimizer"])
        self.grad_manager.load_state_dict(payload["grad_manager"])

        payload_reward_norm = payload.get("reward_normalizer")
        if payload_reward_norm is not None:
            if self.reward_normalizer is None:
                raise RuntimeError(
                    "Checkpoint contains reward normalizer state but current config has normalize_returns disabled."
                )
            self.reward_normalizer.load_state_dict(payload_reward_norm)

        scheduler_state = payload.get("actor_scheduler")
        if scheduler_state is not None and self.actor_scheduler is not None:
            self.actor_scheduler.load_state_dict(scheduler_state)
            if hasattr(self.actor_scheduler, "_last_lr"):
                self.actor_scheduler._last_lr = [group["lr"] for group in self.actor_opt.param_groups]

        scheduler_state = payload.get("critic_scheduler")
        if scheduler_state is not None and self.critic_scheduler is not None:
            self.critic_scheduler.load_state_dict(scheduler_state)
            if hasattr(self.critic_scheduler, "_last_lr"):
                self.critic_scheduler._last_lr = [group["lr"] for group in self.critic_opt.param_groups]

        obs_norm_state = payload.get("observation_normalizer")
        if obs_norm_state is not None:
            if self.observation_normalizer is None:
                raise RuntimeError(
                    "Checkpoint contains observation normalizer state but current config has normalize_state disabled."
                )
            self.observation_normalizer.load_state_dict(obs_norm_state)
        wp_norm_state = payload.get("waypoint_normalizer")
        if wp_norm_state is not None:
            if self.waypoint_normalizer is None:
                raise RuntimeError(
                    "Checkpoint contains waypoint normalizer state but current config has normalize_waypoint disabled."
                )
            self.waypoint_normalizer.load_state_dict(wp_norm_state)
        lidar_norm_state = payload.get("lidar_normalizer")
        if lidar_norm_state is not None:
            if self.lidar_normalizer is None:
                raise RuntimeError(
                    "Checkpoint contains lidar normalizer state but current config has normalize_lidar disabled."
                )
            self.lidar_normalizer.load_state_dict(lidar_norm_state)

        self._global_updates = int(payload.get("step", 0))
        self._last_checkpoint_metrics = payload.get("metrics")
        payload["_loaded_to_device"] = self.device.type
        return payload

    def _slice_batch(self, stacked: Dict[str, Tensor | None], indices: Tensor) -> Dict[str, Tensor | None]:
        sliced: Dict[str, Tensor | None] = {}
        for key, value in stacked.items():
            if value is None:
                sliced[key] = None
            else:
                sliced[key] = value.index_select(0, indices)
        return sliced

    def _compute_losses(self, mini_batch: Dict[str, Tensor | None]) -> tuple[Tensor, Dict[str, float]]:
        obs_seq = mini_batch["obs_seq"]  # type: ignore[index]
        state = mini_batch["state"]  # type: ignore[index]
        action = mini_batch["action"]  # type: ignore[index]
        old_log_prob = mini_batch["old_log_prob"]  # type: ignore[index]
        returns = mini_batch["returns"]  # type: ignore[index]
        advantages = mini_batch["advantages"]  # type: ignore[index]
        waypoint_seq = mini_batch.get("waypoint_seq")
        raw_state = mini_batch.get("raw_state")
        raw_waypoint_seq = mini_batch.get("raw_waypoint_seq")
        warm_start = mini_batch["warm_start"]
        mask = mini_batch.get("mask")
        mpve_target = mini_batch.get("mpve_target")
        lidar_seq = mini_batch.get("lidar_seq")
        old_value = mini_batch.get("old_value")

        valid_fraction = 1.0
        if mask is not None:
            if mask.shape != returns.shape:
                raise ValueError("mask must share the flattened shape of the training samples.")
            valid = mask > 0.0 if mask.dtype != torch.bool else mask
            if not bool(valid.any()):
                raise ValueError("No valid samples remain after applying the rollout mask.")
            valid_fraction = float(valid.float().mean().item())

            obs_seq = obs_seq[valid]
            state = state[valid]
            action = action[valid]
            old_log_prob = old_log_prob[valid]
            returns = returns[valid]
            advantages = advantages[valid]
            if waypoint_seq is not None:
                waypoint_seq = waypoint_seq[valid]
            if raw_state is not None:
                raw_state = raw_state[valid]
            if raw_waypoint_seq is not None:
                raw_waypoint_seq = raw_waypoint_seq[valid]
            if warm_start is not None:
                warm_start = warm_start[valid]
            if mpve_target is not None:
                mpve_target = mpve_target[valid]
            if lidar_seq is not None:
                lidar_seq = lidar_seq[valid]
            if old_value is not None:
                old_value = old_value[valid]

        if self.config.advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        log_prob, entropy, values = self.agent.evaluate_actions(
            obs_seq,
            state=state,
            actions=action,
            warm_start=warm_start,
            waypoint_seq=waypoint_seq,
            raw_state=raw_state,
            raw_waypoint_seq=raw_waypoint_seq,
            lidar=lidar_seq,
        )

        ratio = torch.exp(log_prob - old_log_prob)
        clip_low = 1.0 - self.config.clip_param
        clip_high = 1.0 + self.config.clip_param
        clip_fraction = ((ratio < clip_low) | (ratio > clip_high)).float().mean()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coeff * entropy.mean()

        if self.config.value_clip_range is not None and old_value is not None:
            clip_range = float(self.config.value_clip_range)
            value_pred_clipped = old_value + torch.clamp(values - old_value, -clip_range, clip_range)
            value_losses = torch.stack(
                [
                    F.mse_loss(values, returns, reduction="none"),
                    F.mse_loss(value_pred_clipped, returns, reduction="none"),
                ],
                dim=0,
            )
            value_loss = value_losses.max(0).values.mean()
        else:
            value_loss = F.mse_loss(values, returns)
        if mpve_target is not None:
            mpve_loss = F.mse_loss(values, mpve_target)
            value_loss = value_loss + self.config.mpve.loss_weight * mpve_loss
        else:
            mpve_loss = None
        approx_kl = (old_log_prob - log_prob).mean()
        value_error = returns - values

        def _stats(tensor: Tensor) -> tuple[float, float, float, float]:
            tensor = tensor.detach()
            return (
                float(tensor.mean().item()),
                float(tensor.std(unbiased=False).item()),
                float(tensor.min().item()),
                float(tensor.max().item()),
            )

        ratio_mean, ratio_std, ratio_min, ratio_max = _stats(ratio)
        returns_mean, returns_std, returns_min, returns_max = _stats(returns)
        advantages_mean, advantages_std, advantages_min, advantages_max = _stats(advantages)
        log_prob_mean, log_prob_std, log_prob_min, log_prob_max = _stats(log_prob)
        old_log_prob_mean, old_log_prob_std, old_log_prob_min, old_log_prob_max = _stats(old_log_prob)
        value_mean, value_std, value_min, value_max = _stats(values)
        value_error_mean = float(value_error.detach().mean().item())
        value_error_abs_mean = float(value_error.detach().abs().mean().item())

        log_std_min = float("nan")
        log_std_max = float("nan")
        std_min = float("nan")
        std_max = float("nan")
        actor = getattr(self.agent, "actor", None)
        if actor is not None and hasattr(actor, "log_std"):
            log_std_param = actor.log_std.detach()
            if hasattr(actor, "_log_std_min") and hasattr(actor, "_log_std_max"):
                log_std_param = log_std_param.clamp(actor._log_std_min, actor._log_std_max)
            log_std_min = float(log_std_param.min().item())
            log_std_max = float(log_std_param.max().item())
            std = torch.exp(log_std_param)
            std_min = float(std.min().item())
            std_max = float(std.max().item())
        
        # Auxiliary alignment loss (if enabled)
        alignment_loss = None
        if self.config.alignment_loss_coeff > 0.0:
            alignment_loss = self._compute_alignment_loss(obs_seq, state, waypoint_seq, warm_start)
        
        total_loss = policy_loss + self.config.value_loss_coeff * value_loss
        if alignment_loss is not None:
            total_loss = total_loss + self.config.alignment_loss_coeff * alignment_loss

        metrics = {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.mean().detach().cpu()),
            "approx_kl": float(approx_kl.detach().cpu()),
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            "clip_fraction": float(clip_fraction.detach().cpu()),
            "returns_mean": returns_mean,
            "returns_std": returns_std,
            "returns_min": returns_min,
            "returns_max": returns_max,
            "advantages_mean": advantages_mean,
            "advantages_std": advantages_std,
            "advantages_min": advantages_min,
            "advantages_max": advantages_max,
            "log_prob_mean": log_prob_mean,
            "log_prob_std": log_prob_std,
            "log_prob_min": log_prob_min,
            "log_prob_max": log_prob_max,
            "old_log_prob_mean": old_log_prob_mean,
            "old_log_prob_std": old_log_prob_std,
            "old_log_prob_min": old_log_prob_min,
            "old_log_prob_max": old_log_prob_max,
            "value_mean": value_mean,
            "value_std": value_std,
            "value_min": value_min,
            "value_max": value_max,
            "value_error_mean": value_error_mean,
            "value_error_abs_mean": value_error_abs_mean,
            "log_std_min": log_std_min,
            "log_std_max": log_std_max,
            "std_min": std_min,
            "std_max": std_max,
            "mask_valid_fraction": float(valid_fraction),
        }
        if mpve_loss is not None:
            metrics["mpve_value_loss"] = float(mpve_loss.detach().cpu())
        if alignment_loss is not None:
            metrics["alignment_loss"] = float(alignment_loss.detach().cpu())
        return total_loss, metrics

    def _compute_alignment_loss(
        self, 
        obs_seq: Tensor, 
        state: Tensor, 
        waypoint_seq: Optional[Tensor], 
        warm_start: Optional[Tensor]
    ) -> Tensor:
        """Compute alignment loss between MPC action and optimal direction to waypoint.
        
        This auxiliary loss encourages the cost map to generate parameters that lead
        to MPC actions aligned with the reward structure (movement toward waypoints).
        
        Args:
            obs_seq: Observation sequence [N, T, obs_dim]
            state: Current states [N, state_dim] 
            waypoint_seq: Target waypoints [N, W, waypoint_dim]
            warm_start: Optional warm start actions [N, horizon, action_dim]
            
        Returns:
            Alignment loss tensor (scalar)
        """
        if waypoint_seq is None or waypoint_seq.numel() == 0:
            # No waypoints available, return zero loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get deterministic action from actor (mean of Normal distribution)
        # Note: We need to enable gradients for the alignment loss computation
        actor_output = self.agent.actor(
            obs_seq, 
            state=state, 
            warm_start=warm_start, 
            waypoint_seq=waypoint_seq,
            stochastic=False  # Get deterministic mean
        )
        mpc_action = actor_output.action  # [N, action_dim]
        
        # Extract position from state and waypoint target
        current_pos = state[:, :2]  # [N, 2] - (x, y) position
        target_waypoint = waypoint_seq[:, 0, :]  # [N, 2] - first waypoint (x_target, y_target)
        
        # Optimal direction: vector pointing toward waypoint
        optimal_direction = target_waypoint - current_pos  # [N, 2]
        
        # Normalize directions with epsilon for stability
        eps = 1e-8
        mpc_action_norm = torch.norm(mpc_action, dim=1, keepdim=True) + eps  # [N, 1]
        optimal_direction_norm = torch.norm(optimal_direction, dim=1, keepdim=True) + eps  # [N, 1]
        
        mpc_action_unit = mpc_action / mpc_action_norm  # [N, 2]
        optimal_direction_unit = optimal_direction / optimal_direction_norm  # [N, 2]
        
        # Cosine similarity between MPC action and optimal direction
        cos_sim = torch.sum(mpc_action_unit * optimal_direction_unit, dim=1)  # [N]
        
        # Weight by action magnitude to avoid good alignment with near-zero actions
        action_magnitude = torch.norm(mpc_action, dim=1)  # [N]
        magnitude_weight = torch.sigmoid(action_magnitude - 0.1)  # Sigmoid centered at 0.1
        
        weighted_cos_sim = cos_sim * magnitude_weight  # [N]
        
        # Alignment loss: negative mean cosine similarity (to minimize)
        # Higher cosine similarity = better alignment = lower loss
        alignment_loss = -weighted_cos_sim.mean()
        
        return alignment_loss

    def _compute_lr_multiplier(self, step: int) -> float:
        """Compute LR scaling factor based on schedule and global step."""
        if self.config.lr_schedule == "none":
            return 1.0
        warmup = max(0, self.config.lr_warmup_steps)
        decay = max(0, self.config.lr_decay_steps)
        if warmup == 0 and decay == 0:
            return 1.0
        if step <= warmup and warmup > 0:
            return step / float(warmup)
        if decay == 0:
            return 1.0
        decay_step = step - warmup
        if decay_step >= decay:
            return self.config.lr_final_factor
        frac = 1.0 - decay_step / float(decay)
        return self.config.lr_final_factor + frac * (1.0 - self.config.lr_final_factor)

    def _apply_lr_schedule(self, step: int) -> None:
        """Update optimizer learning rates according to schedule."""
        factor = self._compute_lr_multiplier(step)
        for group in self.actor_opt.param_groups:
            group["lr"] = self._base_actor_lr * factor
        for group in self.critic_opt.param_groups:
            group["lr"] = self._base_critic_lr * factor
