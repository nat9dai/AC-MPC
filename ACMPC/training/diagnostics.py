"""Diagnostics helpers for ACMPC PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor

from ..sampling.rollout import RolloutBatch


@dataclass
class DiagnosticsOptions:
    enable_cost_monitor: bool = False
    enable_reward_stats: bool = True
    enable_plan_drift_checker: bool = False
    enable_safety_hooks: bool = True
    enable_mpve_metrics: bool = False

    def merge(self, other: Optional["DiagnosticsOptions"]) -> "DiagnosticsOptions":
        if other is None:
            return self
        return DiagnosticsOptions(
            enable_cost_monitor=other.enable_cost_monitor or self.enable_cost_monitor,
            enable_reward_stats=other.enable_reward_stats or self.enable_reward_stats,
            enable_plan_drift_checker=other.enable_plan_drift_checker or self.enable_plan_drift_checker,
            enable_safety_hooks=other.enable_safety_hooks or self.enable_safety_hooks,
            enable_mpve_metrics=other.enable_mpve_metrics or self.enable_mpve_metrics,
        )


class DiagnosticsManager:
    """Computes optional diagnostic metrics from rollout batches."""

    def __init__(self, options: DiagnosticsOptions) -> None:
        self.options = options
        self._monitors: List[BaseMonitor] = []
        if options.enable_reward_stats:
            reward_monitor = RewardStatsMonitor()
            self._monitors.append(reward_monitor)
            # Episode returns and cost estimates share reward tensors.
            self._monitors.append(EpisodeReturnMonitor())
            if options.enable_cost_monitor:
                self._monitors.append(CostMonitor(reward_monitor))
        elif options.enable_cost_monitor:
            # Cost monitor requires reward data; add standalone instance.
            self._monitors.append(CostMonitor())
        if options.enable_plan_drift_checker:
            self._monitors.append(PlanDriftMonitor())
        if options.enable_safety_hooks:
            self._monitors.append(WarmStartMonitor())
        if options.enable_mpve_metrics:
            self._monitors.append(MPVERewardMonitor())

    def process_rollouts(self, batches: Iterable[RolloutBatch]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for batch in batches:
            for monitor in self._monitors:
                monitor.update(batch)

        for monitor in self._monitors:
            metrics.update(monitor.metrics())
        return metrics


class BaseMonitor:
    """Abstract base class for diagnostics monitors."""

    def update(self, batch: RolloutBatch) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def metrics(self) -> Dict[str, float]:  # pragma: no cover - interface
        raise NotImplementedError


class RewardStatsMonitor(BaseMonitor):
    """Tracks distribution of realised rewards."""

    def __init__(self) -> None:
        self._rewards: List[Tensor] = []

    def update(self, batch: RolloutBatch) -> None:
        self._rewards.append(batch.reward)

    def metrics(self) -> Dict[str, float]:
        if not self._rewards:
            return {}
        stacked = torch.cat([r.reshape(-1) for r in self._rewards], dim=0)
        return {
            "reward_mean": float(stacked.mean().item()),
            "reward_std": float(stacked.std(unbiased=False).item()),
            "reward_min": float(stacked.min().item()),
            "reward_max": float(stacked.max().item()),
        }


class EpisodeReturnMonitor(BaseMonitor):
    """Approximates mean episodic return from rollout batches."""

    def __init__(self) -> None:
        self._rewards: List[Tensor] = []
        self._dones: List[Tensor] = []
        self._masks: List[Tensor] = []

    def update(self, batch: RolloutBatch) -> None:
        self._rewards.append(batch.reward)
        self._dones.append(batch.done)
        self._masks.append(batch.mask)

    def metrics(self) -> Dict[str, float]:
        if not self._rewards:
            return {}

        episode_returns: List[float] = []
        for rewards, dones, masks in zip(self._rewards, self._dones, self._masks):
            rewards_np = rewards.cpu().numpy()
            dones_np = dones.cpu().numpy().astype(bool)
            masks_np = masks.cpu().numpy()
            num_envs, horizon = rewards_np.shape
            for env_idx in range(num_envs):
                ret = 0.0
                episode_open = False
                for t in range(horizon):
                    if masks_np[env_idx, t] <= 0.0:
                        continue
                    episode_open = True
                    ret += float(rewards_np[env_idx, t])
                    if dones_np[env_idx, t]:
                        episode_returns.append(ret)
                        ret = 0.0
                        episode_open = False
                if episode_open:
                    episode_returns.append(ret)

        if not episode_returns:
            return {}
        return {"episode_reward": float(sum(episode_returns) / len(episode_returns))}


class CostMonitor(BaseMonitor):
    """Estimates mean cost from realised rewards (negative sign)."""

    def __init__(self, reward_monitor: RewardStatsMonitor | None = None) -> None:
        self._rewards: List[Tensor] = []
        self._shared_monitor = reward_monitor

    def update(self, batch: RolloutBatch) -> None:
        if self._shared_monitor is not None:
            # Reward monitor already tracks reward tensors; reuse.
            return
        self._rewards.append(batch.reward)

    def metrics(self) -> Dict[str, float]:
        if self._shared_monitor is not None:
            reward_tensors = self._shared_monitor._rewards  # noqa: SLF001 - shared cache
        else:
            reward_tensors = self._rewards
        if not reward_tensors:
            return {}
        stacked = torch.cat([r.reshape(-1) for r in reward_tensors], dim=0)
        return {"cost_mean": float((-stacked).mean().item())}


class PlanDriftMonitor(BaseMonitor):
    """Measures deviation between executed actions and MPC first action."""

    def __init__(self) -> None:
        self._actions: List[Tensor] = []
        self._mpc_actions: List[Tensor] = []

    def update(self, batch: RolloutBatch) -> None:
        if batch.plan_actions.numel() == 0:
            return
        self._actions.append(batch.action)
        self._mpc_actions.append(batch.plan_actions[:, :, 0])

    def metrics(self) -> Dict[str, float]:
        if not self._actions or not self._mpc_actions:
            return {}
        act = torch.cat([a.reshape(-1, a.shape[-1]) for a in self._actions], dim=0)
        plan = torch.cat([p.reshape(-1, p.shape[-1]) for p in self._mpc_actions], dim=0)
        drift = torch.norm(act - plan, dim=-1)
        return {"plan_drift_l2": float(drift.mean().item())}


class WarmStartMonitor(BaseMonitor):
    """Tracks warm-start resets to assess MPC cache stability."""

    def __init__(self) -> None:
        self._sources: List[str] = []

    def update(self, batch: RolloutBatch) -> None:
        for env_sources in batch.warm_start_source:
            self._sources.extend(env_sources)

    def metrics(self) -> Dict[str, float]:
        if not self._sources:
            return {}
        resets = sum(1 for src in self._sources if src != "cache")
        return {"warm_start_reset_rate": resets / max(len(self._sources), 1)}


class MPVERewardMonitor(BaseMonitor):
    """Aggregates statistics for predicted MPC rewards."""

    def __init__(self) -> None:
        self._predicted: List[Tensor] = []

    def update(self, batch: RolloutBatch) -> None:
        if batch.plan_rewards is None:
            return
        self._predicted.append(batch.plan_rewards)

    def metrics(self) -> Dict[str, float]:
        if not self._predicted:
            return {}
        stacked = torch.cat([r.reshape(-1) for r in self._predicted], dim=0)
        return {
            "mpve_reward_mean": float(stacked.mean().item()),
            "mpve_reward_std": float(stacked.std(unbiased=False).item()),
        }
