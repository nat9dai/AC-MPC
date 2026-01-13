"""Checkpoint manager for ACMPC PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class CheckpointConfig:
    directory: Path
    metric: str = "value_loss"
    mode: str = "min"  # or "max"
    keep_last: int = 5

    def validate(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("checkpoint mode must be 'min' or 'max'.")
        if self.keep_last < 1:
            raise ValueError("checkpoint keep_last must be >= 1.")


class CheckpointManager:
    """Persist training state snapshots with simple retention policy."""

    def __init__(self, config: CheckpointConfig) -> None:
        config.validate()
        self.config = config
        self.dir = config.directory
        self.dir.mkdir(parents=True, exist_ok=True)

        self._saved: List[Path] = []
        if config.mode == "min":
            self._best_value: float = float("inf")
        else:
            self._best_value = -float("inf")
        self._best_path: Optional[Path] = None
        self._scan_existing()

    def save(
        self,
        *,
        step: int,
        metrics: Dict[str, float],
        agent,
        actor_opt,
        critic_opt,
        grad_manager,
        reward_normalizer=None,
        observation_normalizer=None,
        waypoint_normalizer=None,
        lidar_normalizer=None,
        actor_scheduler=None,
        critic_scheduler=None,
    ) -> Optional[Path]:
        metric_value = metrics.get(self.config.metric)
        if metric_value is None or not torch.isfinite(torch.tensor(metric_value)):
            return None

        is_better = (
            metric_value < self._best_value
            if self.config.mode == "min"
            else metric_value > self._best_value
        )
        checkpoint_path = self.dir / f"checkpoint_step{step:07d}.pt"

        payload = {
            "step": step,
            "metrics": metrics,
            "agent": agent.state_dict(),
            "actor_optimizer": actor_opt.state_dict(),
            "critic_optimizer": critic_opt.state_dict(),
            "grad_manager": grad_manager.state_dict(),
            "reward_normalizer": None if reward_normalizer is None else reward_normalizer.state_dict(),
            "observation_normalizer": None
            if observation_normalizer is None
            else observation_normalizer.state_dict(),
            "waypoint_normalizer": None
            if waypoint_normalizer is None
            else waypoint_normalizer.state_dict(),
            "lidar_normalizer": None if lidar_normalizer is None else lidar_normalizer.state_dict(),
            "actor_scheduler": None if actor_scheduler is None else actor_scheduler.state_dict(),
            "critic_scheduler": None if critic_scheduler is None else critic_scheduler.state_dict(),
        }

        torch.save(payload, checkpoint_path)
        self._saved.append(checkpoint_path)

        if is_better:
            self._best_value = float(metric_value)
            self._best_path = checkpoint_path

        self._prune()
        return checkpoint_path

    def latest_checkpoint(self) -> Optional[Path]:
        return self._saved[-1] if self._saved else None

    def best_checkpoint(self) -> Optional[Path]:
        return self._best_path

    def load(self, path: Optional[Path] = None, *, best: bool = False, latest: bool = False) -> Dict[str, Any]:
        if path is None:
            if best:
                path = self.best_checkpoint()
            elif latest:
                path = self.latest_checkpoint()
        if path is None:
            raise FileNotFoundError("No checkpoint available to load.")
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
        payload = torch.load(checkpoint_path, map_location="cpu")
        payload["_checkpoint_path"] = checkpoint_path
        return payload

    def _prune(self) -> None:
        while len(self._saved) > self.config.keep_last:
            path = self._saved.pop(0)
            if path == self._best_path:
                # Keep the best checkpoint; move it to the end of the queue.
                self._saved.append(path)
                # If there is only the best checkpoint left we can stop pruning.
                if len(self._saved) <= self.config.keep_last:
                    break
                continue
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def _scan_existing(self) -> None:
        existing = sorted(self.dir.glob("checkpoint_step*.pt"))
        for path in existing:
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception:
                continue
            metrics = payload.get("metrics", {})
            value = metrics.get(self.config.metric)
            if value is None:
                continue
            self._saved.append(path)
            is_better = (
                value < self._best_value if self.config.mode == "min" else value > self._best_value
            )
            if is_better:
                self._best_value = float(value)
                self._best_path = path
