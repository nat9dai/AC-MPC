"""Gradient management utilities for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn import utils as nn_utils


def _grad_norm(parameters: Iterable[Tensor], norm_type: float = 2.0) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        param_norm = grad.norm(norm_type)
        total += float(param_norm.item() ** norm_type)
    return float(total ** (1.0 / norm_type)) if total > 0.0 else 0.0


@dataclass
class GradientManagerConfig:
    """Configuration controlling gradient updates."""

    max_norm: Optional[float]
    norm_type: float = 2.0
    accumulation_steps: int = 1
    use_amp: bool = True
    log_norm: bool = False

    def validate(self) -> None:
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1.")
        if self.max_norm is not None and self.max_norm <= 0:
            raise ValueError("max_norm must be positive when specified.")
        if self.norm_type <= 0:
            raise ValueError("norm_type must be positive.")


class GradientManager:
    """Handles AMP contexts, gradient accumulation, and clipping."""

    def __init__(
        self,
        *,
        actor_opt: torch.optim.Optimizer,
        critic_opt: torch.optim.Optimizer,
        actor_params: Iterable[Tensor],
        critic_params: Iterable[Tensor],
        device: torch.device,
        config: GradientManagerConfig,
    ) -> None:
        config.validate()
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.actor_params = list(actor_params)
        self.critic_params = list(critic_params)
        self.device = device
        self.config = config

        amp_enabled = config.use_amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=amp_enabled)

        self._accumulated = 0
        self._require_zero = True
        self.last_actor_grad_norm: Optional[float] = None
        self.last_critic_grad_norm: Optional[float] = None

    def autocast(self):
        return autocast(device_type=self.device.type, enabled=self.scaler.is_enabled())

    def prepare_microbatch(self) -> None:
        if self._require_zero:
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            self._require_zero = False

    def backward(self, loss: Tensor) -> Tuple[bool, Optional[float], Optional[float]]:
        # Skip updates if loss is non-finite.
        if not torch.isfinite(loss):
            return False, None, None

        scaled_loss = loss / self.config.accumulation_steps
        if self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._accumulated += 1
        if self._accumulated < self.config.accumulation_steps:
            return False, None, None

        actor_norm, critic_norm = self._apply_step()
        self._accumulated = 0
        self._require_zero = True
        self.last_actor_grad_norm = actor_norm
        self.last_critic_grad_norm = critic_norm
        return True, actor_norm, critic_norm

    def finalize(self) -> Tuple[bool, Optional[float], Optional[float]]:
        if self._accumulated == 0:
            return False, None, None
        actor_norm, critic_norm = self._apply_step()
        self._accumulated = 0
        self._require_zero = True
        self.last_actor_grad_norm = actor_norm
        self.last_critic_grad_norm = critic_norm
        return True, actor_norm, critic_norm

    def _apply_step(self) -> Tuple[Optional[float], Optional[float]]:
        actor_norm: Optional[float] = None
        critic_norm: Optional[float] = None

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.actor_opt)
            self.scaler.unscale_(self.critic_opt)

        if self.config.max_norm is not None:
            actor_norm = float(
                nn_utils.clip_grad_norm_(
                    self.actor_params,
                    self.config.max_norm,
                    norm_type=self.config.norm_type,
                    error_if_nonfinite=False,
                ).item()
            )
            critic_norm = float(
                nn_utils.clip_grad_norm_(
                    self.critic_params,
                    self.config.max_norm,
                    norm_type=self.config.norm_type,
                    error_if_nonfinite=False,
                ).item()
            )
        elif self.config.log_norm:
            actor_norm = _grad_norm(self.actor_params, self.config.norm_type)
            critic_norm = _grad_norm(self.critic_params, self.config.norm_type)

        # If any gradient becomes non-finite, abort this step to avoid corrupting parameters.
        def _has_nonfinite_grad(params: Iterable[Tensor]) -> bool:
            for p in params:
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    return True
            return False

        if _has_nonfinite_grad(self.actor_params) or _has_nonfinite_grad(self.critic_params):
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            return actor_norm, critic_norm

        if self.scaler.is_enabled():
            self.scaler.step(self.actor_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            self.actor_opt.step()
            self.critic_opt.step()

        return actor_norm, critic_norm

    def state_dict(self) -> dict:
        state = {
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "accumulated": self._accumulated,
        }
        return state

    def load_state_dict(self, state: dict) -> None:
        scaler_state = state.get("scaler")
        if scaler_state is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_state)
        self._accumulated = int(state.get("accumulated", 0))
        self._require_zero = self._accumulated == 0
