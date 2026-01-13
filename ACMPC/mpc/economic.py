"""Economic MPC heads with reusable differentiable controllers and warm-start logic."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import warnings

from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost
from DifferentialMPC.controller import GradMethod

if TYPE_CHECKING:  # pragma: no cover
    from ..models.cost_map import CostMapParameters


State = Tensor
Action = Tensor


def _as_tensor(
    value: Optional[Sequence[float] | float | Tensor],
    *,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    if value is None:
        return None
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.numel() == 1:
        tensor = tensor.expand(dim)
    if tensor.shape != (dim,):
        raise ValueError(f"Expected tensor of shape {(dim,)}, received {tuple(tensor.shape)}")
    return tensor


@dataclass
class EconomicMPCConfig:
    """Configuration for the economic MPC head."""

    horizon: int
    state_dim: int
    action_dim: int
    dt: float
    latent_dim: int
    device: str = "cpu"
    max_iter: int = 20
    tolerance: Optional[float] = None  # Backwards compatibility with legacy configs
    tol_x: float = 1e-4
    tol_u: float = 1e-4
    reg_eps: float = 1e-6
    delta_u: Optional[float] = None
    state_cost: Union[float, Sequence[float], Tensor] = 1.0
    action_cost: Union[float, Sequence[float], Tensor] = 0.1
    terminal_state_cost: Union[float, Sequence[float], Tensor] = 5.0
    u_min: Optional[Sequence[float] | float | Tensor] = None
    u_max: Optional[Sequence[float] | float | Tensor] = None
    require_analytic_jacobian: bool = False
    enable_autodiff_fallback: bool = True
    jacobian_cache_size: int = 1024
    jacobian_cache_max_age: int = 128
    warm_start_cache_max_size: int = 1
    warm_start_drift_tol: float = 1e-3

    def __post_init__(self) -> None:
        if self.tolerance is not None:
            self.tol_x = self.tolerance
            self.tol_u = self.tolerance
            self.tolerance = None
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if self.jacobian_cache_size < 0:
            raise ValueError("jacobian_cache_size must be >= 0")
        if self.jacobian_cache_max_age < 0:
            raise ValueError("jacobian_cache_max_age must be >= 0")
        if self.warm_start_cache_max_size < 0:
            raise ValueError("warm_start_cache_max_size must be >= 0")
        if self.warm_start_drift_tol < 0:
            raise ValueError("warm_start_drift_tol must be >= 0")


class JacobianCache:
    """Simple LRU cache for analytic Jacobians."""

    def __init__(self, max_size: int, max_age: int, precision: float = 1e-6) -> None:
        self.enabled = max_size > 0 and max_age > 0
        self.max_size = max_size
        self.max_age = max_age
        self.precision = precision
        self._entries: OrderedDict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[Tensor, Tensor, int]] = OrderedDict()

    def _quantise(self, tensor: Tensor) -> Tuple[int, ...]:
        scale = 1.0 / self.precision
        quantised = torch.round(tensor.detach().cpu() * scale).to(torch.int64)
        return tuple(quantised.view(-1).tolist())

    def _key(self, x: Tensor, u: Tensor) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return self._quantise(x), self._quantise(u)

    def step(self) -> None:
        if not self.enabled:
            return
        to_delete = []
        for key, (A, B, age) in self._entries.items():
            new_age = age + 1
            if new_age > self.max_age:
                to_delete.append(key)
            else:
                self._entries[key] = (A, B, new_age)
        for key in to_delete:
            self._entries.pop(key, None)

    def get(self, x: Tensor, u: Tensor) -> Optional[Tuple[Tensor, Tensor]]:
        if not self.enabled:
            return None
        key = self._key(x, u)
        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        A, B, _ = entry
        self._entries[key] = (A, B, 0)
        return A, B

    def add(self, x: Tensor, u: Tensor, jac: Tuple[Tensor, Tensor]) -> None:
        if not self.enabled:
            return
        key = self._key(x, u)
        self._entries[key] = (jac[0].detach(), jac[1].detach(), 0)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)


class WarmStartManager:
    """Maintains per-batch warm-start candidates with drift-based invalidation."""

    def __init__(self, horizon: int, action_dim: int, drift_tol: float, enabled: bool) -> None:
        self.enabled = enabled and horizon > 0
        self.horizon = horizon
        self.action_dim = action_dim
        self.drift_tol = drift_tol
        self._cached_state: Optional[Tensor] = None
        self._cached_signature: Optional[Tensor] = None
        self._cached_controls: Optional[Tensor] = None
        self._last_source: str = "zeros"

    @property
    def last_source(self) -> str:
        return self._last_source

    def invalidate(self) -> None:
        self._cached_state = None
        self._cached_signature = None
        self._cached_controls = None
        self._last_source = "zeros"

    def prepare(
        self,
        state: Tensor,
        signature: Optional[Tensor],
        provided: Optional[Tensor],
    ) -> Tensor:
        batch, horizon, action_dim = state.size(0), self.horizon, self.action_dim
        device, dtype = state.device, state.dtype

        if not self.enabled:
            self._last_source = "zeros"
            return torch.zeros(batch, horizon, action_dim, device=device, dtype=dtype)

        if provided is not None:
            warm = self._validate(provided, batch, device, dtype)
            self._last_source = "provided"
            return warm

        warm = self._fetch_from_cache(state, signature)
        if warm is not None:
            self._last_source = "cache"
            return warm

        self._last_source = "zeros"
        return torch.zeros(batch, horizon, action_dim, device=device, dtype=dtype)

    def update(
        self,
        state: Tensor,
        signature: Optional[Tensor],
        controls: Tensor,
        converged: bool,
    ) -> None:
        if not self.enabled:
            return
        if not converged:
            self.invalidate()
            return
        self._cached_state = state.detach().clone()
        self._cached_controls = controls.detach().clone()
        self._cached_signature = None if signature is None else signature.detach().clone()

    # Internal helpers -------------------------------------------------
    def _validate(
        self,
        warm_start: Tensor,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        warm = warm_start.to(device=device, dtype=dtype)
        if warm.dim() == 2:
            warm = warm.unsqueeze(0)
        expected_shape = (batch, self.horizon, self.action_dim)
        if warm.shape != expected_shape:
            raise ValueError(f"warm_start has shape {warm.shape}, expected {expected_shape}")
        if torch.isnan(warm).any():
            raise ValueError("warm_start contains NaNs")
        return warm

    def _fetch_from_cache(self, state: Tensor, signature: Optional[Tensor]) -> Optional[Tensor]:
        if self._cached_controls is None:
            return None
        if state.shape != self._cached_state.shape:
            return None
        drift = torch.norm(state - self._cached_state, dim=-1)
        if torch.any(drift > self.drift_tol):
            return None
        if signature is not None:
            if self._cached_signature is None or self._cached_signature.shape != signature.shape:
                return None
            if not torch.allclose(signature, self._cached_signature, atol=self.drift_tol, rtol=0.0):
                return None
        return self._cached_controls.clone()


class BaseMPCHead(nn.Module):
    """Base class implementing reusable MPC controller logic."""

    def __init__(
        self,
        config: EconomicMPCConfig,
        *,
        dynamics_fn: Callable[[State, Action, float], State],
        dynamics_jacobian_fn: Optional[Callable[[Tensor, Tensor, float], Tuple[Tensor, Tensor]]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.dynamics_fn = dynamics_fn
        self._dynamics_jacobian_fn = dynamics_jacobian_fn
        self.device = torch.device(config.device)
        self.dtype = torch.float32

        self._jacobian_cache = JacobianCache(
            config.jacobian_cache_size,
            config.jacobian_cache_max_age,
        ) if dynamics_jacobian_fn is not None else None
        self._grad_method = self._determine_initial_grad_method()
        self._controller = self._build_controller(self._grad_method)

        self._warm_start_manager = WarmStartManager(
            horizon=config.horizon,
            action_dim=config.action_dim,
            drift_tol=config.warm_start_drift_tol,
            enabled=config.warm_start_cache_max_size > 0,
        )

    # Controller lifecycle --------------------------------------------
    def _determine_initial_grad_method(self) -> GradMethod:
        if self._dynamics_jacobian_fn is None:
            if self.config.require_analytic_jacobian:
                raise ValueError("Analytic Jacobian required but no function provided.")
            if self.config.enable_autodiff_fallback:
                warnings.warn("Falling back to automatic differentiation for MPC Jacobians.")
            return GradMethod.AUTO_DIFF
        return GradMethod.ANALYTIC

    def _build_controller(self, grad_method: GradMethod) -> DifferentiableMPCController:
        cost_module = self._initial_cost_module(self.device, self.dtype)

        u_min = _as_tensor(self.config.u_min, dim=self.config.action_dim, device=self.device, dtype=self.dtype)
        u_max = _as_tensor(self.config.u_max, dim=self.config.action_dim, device=self.device, dtype=self.dtype)

        def _f_dyn(x: Tensor, u: Tensor, dt: float) -> Tensor:
            return self.dynamics_fn(x, u, dt)

        f_dyn_jac = None
        if grad_method is GradMethod.ANALYTIC and self._dynamics_jacobian_fn is not None:
            f_dyn_jac = self._wrap_jacobian(self._dynamics_jacobian_fn)

        controller = DifferentiableMPCController(
            f_dyn=_f_dyn,
            total_time=self.config.horizon * self.config.dt,
            step_size=self.config.dt,
            horizon=self.config.horizon,
            cost_module=cost_module,
            device=str(self.device),
            max_iter=self.config.max_iter,
            tol_x=self.config.tol_x,
            tol_u=self.config.tol_u,
            reg_eps=self.config.reg_eps,
            delta_u=self.config.delta_u,
            detach_unconverged=False,
            u_min=u_min,
            u_max=u_max,
            grad_method=grad_method,
            f_dyn_jac=f_dyn_jac,
        )
        return controller

    def _wrap_jacobian(
        self,
        jac_fn: Callable[[Tensor, Tensor, float], Tuple[Tensor, Tensor]],
    ) -> Callable[[Tensor, Tensor, float], Tuple[Tensor, Tensor]]:
        def _cached_jac(x: Tensor, u: Tensor, dt: float) -> Tuple[Tensor, Tensor]:
            if self._jacobian_cache is not None:
                cached = self._jacobian_cache.get(x, u)
                if cached is not None:
                    return cached
            jac = jac_fn(x, u, dt)
            if self._jacobian_cache is not None:
                self._jacobian_cache.add(x, u, jac)
            return jac

        return _cached_jac

    def _switch_to_autodiff(self, reason: Exception) -> None:
        if self._grad_method is GradMethod.AUTO_DIFF or not self.config.enable_autodiff_fallback:
            raise reason
        warnings.warn(
            "Disabling analytic MPC Jacobian due to runtime failure; falling back to auto-diff.",
            RuntimeWarning,
        )
        self._grad_method = GradMethod.AUTO_DIFF
        self._dynamics_jacobian_fn = None
        self._jacobian_cache = None
        self._controller = self._build_controller(self._grad_method)
        self._warm_start_manager.invalidate()

    # Interface for subclasses ----------------------------------------
    def _initial_cost_module(self, device: torch.device, dtype: torch.dtype) -> GeneralQuadCost:
        raise NotImplementedError

    def _set_cost(self, *args, **kwargs) -> None:
        raise NotImplementedError

    # Shared solve utilities ------------------------------------------
    def _solve(
        self,
        state: Tensor,
        *,
        warm_start: Optional[Tensor],
        signature: Optional[Tensor],
        return_plan: bool,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        state = state.to(device=self.device, dtype=self.dtype)
        batch = state.size(0)
        self._jacobian_cache.step() if self._jacobian_cache is not None else None

        warm = self._warm_start_manager.prepare(state, signature, warm_start)

        try:
            X, U = self._controller(state, U_init=warm)
        except RuntimeError as exc:
            self._switch_to_autodiff(exc)
            warm = torch.zeros(batch, self.config.horizon, self.config.action_dim, device=self.device, dtype=self.dtype)
            X, U = self._controller(state, U_init=warm)

        converged = bool(getattr(self._controller, "converged", True))
        self._warm_start_manager.update(state, signature, U, converged)

        action = U[:, 0]
        plan = (X, U) if return_plan else None
        return action, plan


class EconomicMPCHead(BaseMPCHead):
    """Transforms latent policy features into MPC actions with reusable controller."""

    def __init__(
        self,
        config: EconomicMPCConfig,
        *,
        dynamics_fn: Callable[[State, Action, float], State],
        dynamics_jacobian_fn: Optional[Callable[[Tensor, Tensor, float], Tuple[Tensor, Tensor]]] = None,
    ) -> None:
        super().__init__(config, dynamics_fn=dynamics_fn, dynamics_jacobian_fn=dynamics_jacobian_fn)
        self.latent_to_ref = nn.Sequential(
            nn.Linear(config.latent_dim, config.state_dim + config.action_dim),
            nn.Tanh(),
        )

    def _expand_weight(
        self,
        weight: Union[float, Sequence[float], Tensor],
        dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        tensor = torch.as_tensor(weight, device=device, dtype=dtype)
        if tensor.ndim == 0:
            return torch.eye(dim, device=device, dtype=dtype) * tensor
        if tensor.shape == (dim,):
            return torch.diag(tensor)
        if tensor.shape == (dim, dim):
            return tensor
        raise ValueError(f"Weight specification must be scalar, length-{dim} vector, or {dim}x{dim} matrix.")

    def _initial_cost_module(self, device: torch.device, dtype: torch.dtype) -> GeneralQuadCost:
        horizon = self.config.horizon
        nx, nu = self.config.state_dim, self.config.action_dim
        C = torch.zeros(horizon, nx + nu, nx + nu, device=device, dtype=dtype)
        c = torch.zeros(horizon, nx + nu, device=device, dtype=dtype)

        q_matrix = self._expand_weight(self.config.state_cost, nx, device=device, dtype=dtype)
        r_matrix = self._expand_weight(self.config.action_cost, nu, device=device, dtype=dtype)

        for t in range(horizon):
            C[t, :nx, :nx] = q_matrix
            C[t, nx:, nx:] = r_matrix

        C_final = torch.zeros(nx + nu, nx + nu, device=device, dtype=dtype)
        C_final[:nx, :nx] = self._expand_weight(self.config.terminal_state_cost, nx, device=device, dtype=dtype)
        c_final = torch.zeros(nx + nu, device=device, dtype=dtype)

        cost = GeneralQuadCost(nx=nx, nu=nu, C=C, c=c, C_final=C_final, c_final=c_final, device=device)
        x_ref = torch.zeros(1, horizon + 1, nx, device=device, dtype=dtype)
        u_ref = torch.zeros(1, horizon, nu, device=device, dtype=dtype)
        cost.set_reference(x_ref, u_ref)
        return cost

    def _set_cost(self, x_ref: Tensor, u_ref: Tensor) -> None:
        self._controller.cost_module.set_reference(x_ref, u_ref)

    def _compute_references(self, latent: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        params = self.latent_to_ref(latent)
        dx, du = torch.split(params, [self.config.state_dim, self.config.action_dim], dim=-1)

        x_target = state + dx
        x_ref = x_target.unsqueeze(1).repeat(1, self.config.horizon + 1, 1).contiguous()
        u_ref = du.unsqueeze(1).repeat(1, self.config.horizon, 1).contiguous()
        return x_ref, u_ref

    def _prepare_inputs(self, latent: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if latent.shape[0] != state.shape[0]:
            raise ValueError("Latent and state batch sizes must match.")
        if latent.shape[1] != self.config.latent_dim:
            raise ValueError(f"Expected latent dimension {self.config.latent_dim}, received {latent.shape[1]}")
        if state.shape[1] != self.config.state_dim:
            raise ValueError(f"Expected state dimension {self.config.state_dim}, received {state.shape[1]}")
        return latent.to(self.device, self.dtype), state.to(self.device, self.dtype)

    def forward(
        self,
        latent: Tensor,
        state: State,
        *,
        warm_start: Optional[Tensor] = None,
        return_plan: bool = False,
    ) -> Action | Tuple[Action, Tuple[Tensor, Tensor]]:
        latent, state = self._prepare_inputs(latent, state)
        x_ref, u_ref = self._compute_references(latent, state)
        self._set_cost(x_ref, u_ref)

        signature = torch.cat([x_ref[:, -1], u_ref[:, 0]], dim=-1)
        action, plan = self._solve(state, warm_start=warm_start, signature=signature, return_plan=return_plan)
        if return_plan:
            assert plan is not None
            return action, plan
        return action


class CostMapMPCHead(BaseMPCHead):
    """MPC head that consumes pre-parameterised quadratic costs from the actor."""

    def __init__(
        self,
        config: EconomicMPCConfig,
        *,
        dynamics_fn: Callable[[State, Action, float], State],
        dynamics_jacobian_fn: Optional[Callable[[Tensor, Tensor, float], Tuple[Tensor, Tensor]]] = None,
    ) -> None:
        super().__init__(config, dynamics_fn=dynamics_fn, dynamics_jacobian_fn=dynamics_jacobian_fn)
        self._base_costs_cache: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None

    def _expand_weight(
        self,
        weight: Union[float, Sequence[float], Tensor],
        dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        tensor = torch.as_tensor(weight, device=device, dtype=dtype)
        if tensor.ndim == 0:
            return torch.eye(dim, device=device, dtype=dtype) * tensor
        if tensor.shape == (dim,):
            return torch.diag(tensor)
        if tensor.shape == (dim, dim):
            return tensor
        raise ValueError(f"Weight specification must be scalar, length-{dim} vector, or {dim}x{dim} matrix.")

    def _get_base_costs(self, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if self._base_costs_cache is not None:
            # Check if cached tensors are on the correct device
            if self._base_costs_cache[0].device == device:
                return self._base_costs_cache

        nx, nu = self.config.state_dim, self.config.action_dim
        horizon = self.config.horizon

        # Build base matrices from config
        q_matrix = self._expand_weight(self.config.state_cost, nx, device=device, dtype=dtype)
        r_matrix = self._expand_weight(self.config.action_cost, nu, device=device, dtype=dtype)

        # Running C (Base)
        C_step = torch.zeros(nx + nu, nx + nu, device=device, dtype=dtype)
        C_step[:nx, :nx] = q_matrix
        C_step[nx:, nx:] = r_matrix
        # Expand to horizon: [H, nx+nu, nx+nu]
        base_running_C = C_step.unsqueeze(0).repeat(horizon, 1, 1)

        # Running c (Base) - usually zero unless linear terms configured, assuming zero for base
        base_running_c = torch.zeros(horizon, nx + nu, device=device, dtype=dtype)

        # Terminal C (Base)
        C_final = torch.zeros(nx + nu, nx + nu, device=device, dtype=dtype)
        tq_matrix = self._expand_weight(self.config.terminal_state_cost, nx, device=device, dtype=dtype)
        C_final[:nx, :nx] = tq_matrix
        base_terminal_C = C_final

        # Terminal c (Base)
        base_terminal_c = torch.zeros(nx + nu, device=device, dtype=dtype)

        self._base_costs_cache = (base_running_C, base_running_c, base_terminal_C, base_terminal_c)
        return self._base_costs_cache

    def _initial_cost_module(self, device: torch.device, dtype: torch.dtype) -> GeneralQuadCost:
        horizon = self.config.horizon
        nx, nu = self.config.state_dim, self.config.action_dim
        cost = GeneralQuadCost(
            nx=nx,
            nu=nu,
            C=torch.zeros(horizon, nx + nu, nx + nu, device=device, dtype=dtype),
            c=torch.zeros(horizon, nx + nu, device=device, dtype=dtype),
            C_final=torch.zeros(nx + nu, nx + nu, device=device, dtype=dtype),
            c_final=torch.zeros(nx + nu, device=device, dtype=dtype),
            device=device,
        )
        x_ref = torch.zeros(1, horizon + 1, nx, device=device, dtype=dtype)
        u_ref = torch.zeros(1, horizon, nu, device=device, dtype=dtype)
        cost.set_reference(x_ref, u_ref)
        return cost

    def _set_cost(self, cost: "CostMapParameters", x_ref: Tensor, u_ref: Tensor) -> None:
        device, dtype = self.device, self.dtype
        base_running_C, base_running_c, base_terminal_C, base_terminal_c = self._get_base_costs(device, dtype)

        # RESIDUAL COST LEARNING: Add Base Costs to Network Predictions
        # Network predicts deviation/refinement, Config provides stable priors.
        self._controller.cost_module.C = cost.running_C.to(device=device, dtype=dtype) + base_running_C
        self._controller.cost_module.c = cost.running_c.to(device=device, dtype=dtype) + base_running_c
        self._controller.cost_module.C_final = cost.terminal_C.to(device=device, dtype=dtype) + base_terminal_C
        self._controller.cost_module.c_final = cost.terminal_c.to(device=device, dtype=dtype) + base_terminal_c

        self._controller.cost_module.set_reference(x_ref, u_ref)

    def _validate_state(self, state: Tensor) -> Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() != 2:
            raise ValueError("State tensor must be rank 1 or 2.")
        if state.shape[1] != self.config.state_dim:
            raise ValueError(
                f"Expected state dimension {self.config.state_dim}, received {state.shape[1]}"
            )
        return state.to(self.device, self.dtype)

    def _prepare_references(
        self,
        batch: int,
        x_ref: Optional[Tensor],
        u_ref: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        nx, nu = self.config.state_dim, self.config.action_dim
        horizon = self.config.horizon

        # State references
        if x_ref is None:
            x_ref_tensor = torch.zeros(batch, horizon + 1, nx, device=self.device, dtype=self.dtype)
        else:
            x_ref_tensor = x_ref.to(self.device, self.dtype)
            if x_ref_tensor.dim() == 2:
                if x_ref_tensor.shape != (horizon + 1, nx):
                    raise ValueError(
                        f"x_ref with shape {x_ref_tensor.shape} is incompatible with horizon {horizon} and state_dim {nx}."
                    )
                x_ref_tensor = x_ref_tensor.unsqueeze(0)
            if x_ref_tensor.dim() != 3 or x_ref_tensor.shape[1:] != (horizon + 1, nx):
                raise ValueError(
                    f"x_ref must have shape [batch, horizon+1, state_dim] or [horizon+1, state_dim], got {x_ref_tensor.shape}."
                )
            if x_ref_tensor.shape[0] == 1 and batch > 1:
                x_ref_tensor = x_ref_tensor.expand(batch, -1, -1)
            if x_ref_tensor.shape[0] != batch:
                raise ValueError(
                    f"x_ref batch dimension {x_ref_tensor.shape[0]} does not match state batch size {batch}."
                )

        # Action references
        if u_ref is None:
            u_ref_tensor = torch.zeros(batch, horizon, nu, device=self.device, dtype=self.dtype)
        else:
            u_ref_tensor = u_ref.to(self.device, self.dtype)
            if u_ref_tensor.dim() == 2:
                if u_ref_tensor.shape != (horizon, nu):
                    raise ValueError(
                        f"u_ref with shape {u_ref_tensor.shape} is incompatible with horizon {horizon} and action_dim {nu}."
                    )
                u_ref_tensor = u_ref_tensor.unsqueeze(0)
            if u_ref_tensor.dim() != 3 or u_ref_tensor.shape[1:] != (horizon, nu):
                raise ValueError(
                    f"u_ref must have shape [batch, horizon, action_dim] or [horizon, action_dim], got {u_ref_tensor.shape}."
                )
            if u_ref_tensor.shape[0] == 1 and batch > 1:
                u_ref_tensor = u_ref_tensor.expand(batch, -1, -1)
            if u_ref_tensor.shape[0] != batch:
                raise ValueError(
                    f"u_ref batch dimension {u_ref_tensor.shape[0]} does not match state batch size {batch}."
                )

        return x_ref_tensor, u_ref_tensor

    def forward(
        self,
        *,
        state: Tensor,
        cost: "CostMapParameters",
        x_ref: Optional[Tensor] = None,
        u_ref: Optional[Tensor] = None,
        warm_start: Optional[Tensor] = None,
        return_plan: bool = False,
    ) -> Action | Tuple[Action, Tuple[Tensor, Tensor]]:
        state = self._validate_state(state)
        batch = state.size(0)
        x_ref_tensor, u_ref_tensor = self._prepare_references(batch, x_ref, u_ref)
        self._set_cost(cost, x_ref_tensor, u_ref_tensor)

        # When explicit references are provided, include them in the warm-start signature
        # so that cached controls are invalidated if the tracking target changes.
        signature: Optional[Tensor]
        if x_ref is None and u_ref is None:
            signature = None
        else:
            signature = torch.cat([x_ref_tensor[:, -1], u_ref_tensor[:, 0]], dim=-1)

        action, plan = self._solve(state, warm_start=warm_start, signature=signature, return_plan=return_plan)
        if return_plan:
            assert plan is not None
            return action, plan
        return action


__all__ = [
    "EconomicMPCConfig",
    "EconomicMPCHead",
    "CostMapMPCHead",
]
