"""Lightweight training logger for PPO loop."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class LoggerConfig:
    log_interval: int
    log_to_stdout: bool = False
    jsonl_path: Optional[Path] = None
    append_jsonl: bool = True
    tensorboard_dir: Optional[Path] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    wandb_resume: Optional[str] = None
    wandb_id: Optional[str] = None

    def validate(self) -> None:
        if self.log_interval < 1:
            raise ValueError("log_interval must be >= 1.")


class TrainingLogger:
    """Handles periodic metric logging to stdout and JSONL files."""

    def __init__(self, config: LoggerConfig) -> None:
        config.validate()
        self.config = config
        self.jsonl_path = config.jsonl_path
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout = sys.stdout
        self._tb_writer = None
        self._wandb_run = None

        if config.tensorboard_dir is not None:
            config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                self._tb_writer = SummaryWriter(log_dir=str(config.tensorboard_dir))
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[logger] TensorBoard non disponibile: {exc}", file=sys.stderr)
                self._tb_writer = None

        if config.wandb_project is not None:
            try:
                import wandb  # type: ignore

                init_kwargs = {
                    "project": config.wandb_project,
                    "entity": config.wandb_entity,
                    "name": config.wandb_run_name,
                    "mode": config.wandb_mode,
                    "id": config.wandb_id,
                    "resume": config.wandb_resume,
                }
                # Remove None values
                init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
                self._wandb_run = wandb.init(**init_kwargs)
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[logger] Impossibile inizializzare wandb: {exc}", file=sys.stderr)
                self._wandb_run = None

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        if step % self.config.log_interval != 0:
            return
        payload = {
            "timestamp": time.time(),
            "step": step,
            "metrics": metrics,
        }
        if self.config.log_to_stdout:
            print(f"[update {step}] " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()), file=self._stdout)
        if self.jsonl_path is not None:
            line = json.dumps(payload, sort_keys=True)
            mode = "a" if self.config.append_jsonl else "w"
            with self.jsonl_path.open(mode, encoding="utf-8") as f:
                f.write(line + "\n")
            # After first write switch to append to avoid truncation on subsequent calls when append_jsonl is False
            self.config.append_jsonl = True
        if self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, global_step=step)
        if self._wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.log(metrics, step=step)
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[logger] wandb.log fallito: {exc}", file=sys.stderr)

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None
        if self._wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.finish()
            except Exception:
                pass
            self._wandb_run = None
