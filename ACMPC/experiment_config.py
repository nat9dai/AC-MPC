"""Unified configuration definitions for the redesigned ACMPC stack.

The module implements a single authoritative configuration surface for the
complete PPO training pipeline.  All components – models, differentiable MPC
head, environment sampler, optimiser loop, diagnostics, and logging – derive
their parameters from :class:`ExperimentConfig`.  Configuration documents are
loaded from YAML/JSON with strict validation and can be overridden via CLI
tokens (``section.subsection.field=value``) to ensure reproducible, research
grade experiments.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Type, get_args, get_origin, get_type_hints

try:  # Optional dependency; YAML support is enabled when available.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - handled gracefully at runtime.
    yaml = None  # type: ignore

from .model_config import ActorConfig, AgentConfig, CriticConfig, TransformerConfig
from .mpc.economic import EconomicMPCConfig
from .training.loop import TrainingConfig as PPOLoopConfig

class ExperimentConfigError(RuntimeError):
    """Base class for configuration related errors."""


class ConfigLoaderError(ExperimentConfigError):
    """Raised when a configuration document cannot be parsed."""


class ConfigValidationError(ExperimentConfigError):
    """Raised when configuration values violate expected constraints."""


class OverrideParseError(ExperimentConfigError):
    """Raised when CLI override tokens are malformed or invalid."""


def _copy_default_mpc_controller() -> EconomicMPCConfig:
    """Return a detached copy of the canonical EconomicMPCConfig."""

    actor_cfg = ActorConfig()
    return replace(actor_cfg.mpc)


def _ensure_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigValidationError(f"{name} must be strictly positive (received {value}).")


def _ensure_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ConfigValidationError(f"{name} must be non-negative (received {value}).")


@dataclass
class ModelSection:
    """Configuration container for model-related parameters."""

    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    history_window: int = 128
    segment_len: int = 32
    mem_len: int = 96
    max_history_tokens: int = 512
    kv_cache_max_tokens: int = 63
    waypoint_sequence_len: int = 8
    waypoint_dim: int = 4
    include_prev_action: bool = True
    prev_action_dim: int = 4
    include_lidar: bool = False
    lidar_dim: int = 0
    enable_distributional_critic: bool = False
    distributional_num_quantiles: int = 32
    distributional_embedding_dim: int = 128
    distributional_kappa: float = 1.0

    def synchronise(self) -> None:
        """Propagate shared Transformer parameters across actor and critic."""

        self.transformer.mem_len = self.mem_len
        self.actor.transformer.mem_len = self.mem_len
        self.critic.transformer.mem_len = self.mem_len

    def validate(self) -> None:
        """Validate Transformer / embedding constraints."""

        _ensure_positive("model.history_window", self.history_window)
        _ensure_positive("model.segment_len", self.segment_len)
        _ensure_positive("model.mem_len", self.mem_len)
        _ensure_positive("model.max_history_tokens", self.max_history_tokens)
        _ensure_positive("model.kv_cache_max_tokens", self.kv_cache_max_tokens)
        if self.segment_len > self.history_window:
            raise ConfigValidationError(
                "model.segment_len must not exceed model.history_window."
            )
        if self.mem_len < self.segment_len:
            raise ConfigValidationError("model.mem_len must be >= model.segment_len.")
        if self.max_history_tokens < self.history_window:
            raise ConfigValidationError(
                "model.max_history_tokens must be >= model.history_window."
            )
        if self.kv_cache_max_tokens < self.segment_len:
            raise ConfigValidationError(
                "model.kv_cache_max_tokens must be >= model.segment_len."
            )
        _ensure_positive("model.waypoint_sequence_len", self.waypoint_sequence_len)
        _ensure_positive("model.waypoint_dim", self.waypoint_dim)
        if self.include_prev_action:
            _ensure_positive("model.prev_action_dim", self.prev_action_dim)
        if self.include_lidar:
            _ensure_positive("model.lidar_dim", self.lidar_dim)
        if self.distributional_num_quantiles <= 0:
            raise ConfigValidationError(
                "model.distributional_num_quantiles must be positive when distributional critic is enabled."
            )
        if self.enable_distributional_critic:
            _ensure_positive("model.distributional_embedding_dim", self.distributional_embedding_dim)

    def build_agent_config(self) -> AgentConfig:
        """Return an AgentConfig derived from the model section."""

        self.synchronise()
        return AgentConfig(actor=self.actor, critic=self.critic)


@dataclass
class MPCSection:
    """Configuration for the economic MPC head and controller."""

    controller: EconomicMPCConfig = field(default_factory=_copy_default_mpc_controller)
    max_iter: int = 20
    tol_x: float = 1e-4
    tol_u: float = 1e-4
    reg_eps: float = 1e-6
    delta_u: Optional[float] = None  # Trust-region radius; None disables it.
    require_analytic_jacobian: bool = True
    enable_autodiff_fallback: bool = True
    jacobian_cache_size: int = 512
    jacobian_cache_max_age: int = 1000
    warm_start_cache_size: int = 64
    warm_start_drift_tol: float = 1e-3
    cost_diag_min: float = 0.1
    cost_diag_max: float = 1e4
    linear_term_bound: float = 1e3

    def validate(self) -> None:
        _ensure_positive("mpc.max_iter", self.max_iter)
        _ensure_positive("mpc.tol_x", self.tol_x)
        _ensure_positive("mpc.tol_u", self.tol_u)
        _ensure_positive("mpc.reg_eps", self.reg_eps)
        if self.delta_u is not None:
            _ensure_positive("mpc.delta_u", self.delta_u)
        _ensure_positive("mpc.jacobian_cache_size", self.jacobian_cache_size)
        _ensure_positive("mpc.jacobian_cache_max_age", self.jacobian_cache_max_age)
        _ensure_positive("mpc.warm_start_cache_size", self.warm_start_cache_size)
        _ensure_positive("mpc.warm_start_drift_tol", self.warm_start_drift_tol)
        if self.cost_diag_min <= 0:
            raise ConfigValidationError("mpc.cost_diag_min must be > 0.")
        if self.cost_diag_max <= self.cost_diag_min:
            raise ConfigValidationError("mpc.cost_diag_max must exceed cost_diag_min.")
        _ensure_positive("mpc.linear_term_bound", self.linear_term_bound)


@dataclass
class SamplerSection:
    """Parameters governing rollout collection."""

    num_envs: int = 12
    num_eval_envs: int = 0
    rollout_steps: int = 128
    episode_len: int = 500
    history_window: Optional[int] = None
    seed_offset: int = 0
    env_backend: str = "multiprocessing"
    deterministic_eval: bool = True
    max_concurrent_resets: int = 4
    reset_on_crash: bool = True

    def validate(self) -> None:
        _ensure_positive("sampler.num_envs", self.num_envs)
        _ensure_non_negative("sampler.num_eval_envs", self.num_eval_envs)
        _ensure_positive("sampler.rollout_steps", self.rollout_steps)
        _ensure_positive("sampler.episode_len", self.episode_len)
        if self.history_window is not None:
            _ensure_positive("sampler.history_window", self.history_window)
        _ensure_non_negative("sampler.seed_offset", self.seed_offset)
        _ensure_positive("sampler.max_concurrent_resets", self.max_concurrent_resets)


@dataclass
class DiagnosticsSection:
    """Toggles and intervals for diagnostics and logging."""

    enable_cost_monitor: bool = False
    enable_reward_stats: bool = True
    enable_plan_drift_checker: bool = False
    enable_safety_hooks: bool = True
    log_interval: int = 10
    checkpoint_interval: int = 50
    metric_flush_interval: int = 100

    def validate(self) -> None:
        _ensure_positive("diagnostics.log_interval", self.log_interval)
        _ensure_positive("diagnostics.checkpoint_interval", self.checkpoint_interval)
        _ensure_positive("diagnostics.metric_flush_interval", self.metric_flush_interval)


@dataclass
class LoggingSection:
    """Backends and paths for experiment logging."""

    experiment_root: Optional[str] = None
    tensorboard_dir: Optional[str] = None
    jsonl_path: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_level: str = "INFO"

    def validate(self) -> None:
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigValidationError(
                "logging.log_level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL."
            )


@dataclass
class ExperimentConfig:
    """Top-level configuration covering every subsystem."""

    seed: int = 0
    device: str = "cuda"
    model: ModelSection = field(default_factory=ModelSection)
    mpc: MPCSection = field(default_factory=MPCSection)
    sampler: SamplerSection = field(default_factory=SamplerSection)
    training: PPOLoopConfig = field(default_factory=PPOLoopConfig)
    diagnostics: DiagnosticsSection = field(default_factory=DiagnosticsSection)
    logging: LoggingSection = field(default_factory=LoggingSection)

    def __post_init__(self) -> None:
        self.synchronise()

    def synchronise(self) -> None:
        """Synchronise shared parameters across sections."""

        if self.seed < 0:
            raise ConfigValidationError("Experiment seed must be non-negative.")
        self.model.synchronise()
        self.sampler.history_window = self.model.history_window
        self.training.resolve_normalization()

    def validate(self) -> None:
        self.model.validate()
        self.mpc.validate()
        self.sampler.validate()
        self.training.mpve.validate()
        self.diagnostics.validate()
        self.logging.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the configuration."""

        return {
            "seed": self.seed,
            "device": self.device,
            "model": _dataclass_to_dict(self.model),
            "mpc": _dataclass_to_dict(self.mpc),
            "sampler": _dataclass_to_dict(self.sampler),
            "training": _dataclass_to_dict(self.training),
            "diagnostics": _dataclass_to_dict(self.diagnostics),
            "logging": _dataclass_to_dict(self.logging),
        }

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> ExperimentConfig:
        """Build an ExperimentConfig from a nested dictionary."""

        if not isinstance(config_dict, Mapping):
            raise ConfigLoaderError("Experiment configuration must be a mapping.")

        config = cls()
        _update_dataclass(config, config_dict)
        config.synchronise()
        config.validate()
        return config


def load_experiment_config(
    path: str | Path | None = None,
    *,
    overrides: Optional[Sequence[str]] = None,
    base: Optional[ExperimentConfig] = None,
) -> ExperimentConfig:
    """Load an ExperimentConfig from disk and apply CLI overrides.

    Parameters
    ----------
    path:
        Optional path to a YAML/JSON document.  When ``None`` a fresh default
        configuration is used.
    overrides:
        Optional sequence of CLI-style overrides (``section.field=value``).  The
        parser supports dotted paths across nested sections and performs type
        coercion according to the dataclass annotations.
    base:
        Optional configuration instance providing defaults before file/override
        application.  The object is deep-copied to avoid in-place mutation.
    """

    config = ExperimentConfig.from_dict(base.to_dict()) if base is not None else ExperimentConfig()

    if path is not None:
        document = _load_config_document(Path(path))
        _update_dataclass(config, document)

    if overrides:
        _apply_overrides(config, overrides)

    config.synchronise()
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Dataclass helpers
# ---------------------------------------------------------------------------


def _dataclass_to_dict(instance: Any) -> Dict[str, Any]:
    if not is_dataclass(instance):
        raise TypeError("_dataclass_to_dict expects a dataclass instance.")
    result: Dict[str, Any] = {}
    for field_info in fields(instance):
        value = getattr(instance, field_info.name)
        if is_dataclass(value):
            result[field_info.name] = _dataclass_to_dict(value)
        else:
            result[field_info.name] = value
    return result


def _is_optional(annotation: Any) -> tuple[Any, bool]:
    args = get_args(annotation)
    if not args:
        return annotation, False
    non_none = [arg for arg in args if arg is not type(None)]  # noqa: E721
    if len(non_none) == len(args):
        return annotation, False
    if len(non_none) != 1:
        return annotation, False
    return non_none[0], True


def _update_dataclass(instance: Any, values: Mapping[str, Any]) -> None:
    if not is_dataclass(instance):
        raise ConfigLoaderError("Attempted to update a non-dataclass instance.")
    field_map = {field_info.name: field_info for field_info in fields(instance)}
    for key, value in values.items():
        if key not in field_map:
            raise ConfigLoaderError(
                f"Unknown configuration field '{key}' for {type(instance).__name__}."
            )
        field_info = field_map[key]
        current_value = getattr(instance, key)
        field_type = _resolve_field_type(instance, field_info)
        new_value = _coerce_assignment(field_type, current_value, value, path=key)
        setattr(instance, key, new_value)


def _coerce_assignment(annotation: Any, current_value: Any, new_value: Any, *, path: str) -> Any:
    base_type, is_optional = _is_optional(annotation)
    if new_value is None:
        if is_optional:
            return None
        raise ConfigLoaderError(f"Field '{path}' cannot be set to null.")

    if is_dataclass_type(base_type):
        if not isinstance(new_value, Mapping):
            raise ConfigLoaderError(f"Field '{path}' expects a mapping for nested dataclass updates.")
        target = current_value if is_dataclass(current_value) else base_type()
        _update_dataclass(target, new_value)
        return target

    origin = get_origin(base_type)
    if origin in {list, Sequence}:  # type: ignore[arg-type]
        return _coerce_list(new_value, base_type, path)
    if origin is tuple:
        return _coerce_tuple(new_value, base_type, path)
    if origin in {dict, Mapping, MutableMapping}:  # type: ignore[arg-type]
        return _coerce_mapping(new_value, base_type, path)

    return _coerce_primitive(base_type, new_value, path)


def _coerce_list(value: Any, annotation: Any, path: str) -> list[Any]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ConfigLoaderError(f"Field '{path}' expects a list value.")
    args = get_args(annotation)
    item_type = args[0] if args else Any
    return [_coerce_primitive(item_type, item, f"{path}[]") for item in value]


def _coerce_tuple(value: Any, annotation: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ConfigLoaderError(f"Field '{path}' expects a tuple value.")
    args = get_args(annotation)
    items = list(value)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_coerce_primitive(args[0], item, f"{path}[]") for item in items)
    if len(args) != len(items):
        raise ConfigLoaderError(
            f"Field '{path}' expects {len(args)} items, received {len(items)}."
        )
    return tuple(_coerce_primitive(arg, item, f"{path}[{idx}]") for idx, (arg, item) in enumerate(zip(args, items)))


def _coerce_mapping(value: Any, annotation: Any, path: str) -> dict[Any, Any]:
    if not isinstance(value, Mapping):
        raise ConfigLoaderError(f"Field '{path}' expects a mapping value.")
    key_type, value_type = (get_args(annotation) + (Any, Any))[:2]
    return {
        _coerce_primitive(key_type, k, f"{path}.key"): _coerce_primitive(value_type, v, f"{path}.value")
        for k, v in value.items()
    }


def _coerce_primitive(expected_type: Any, value: Any, path: str) -> Any:
    if expected_type in {Any, MISSING}:
        return value
    if expected_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1"}:
                return True
            if lowered in {"false", "0"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        raise ConfigLoaderError(f"Field '{path}' expects a boolean value.")
    if expected_type is int:
        if isinstance(value, bool):
            raise ConfigLoaderError(f"Field '{path}' expects an integer value.")
        if isinstance(value, (int,)):
            return int(value)
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError as exc:
                raise ConfigLoaderError(f"Field '{path}' expects an integer value.") from exc
        raise ConfigLoaderError(f"Field '{path}' expects an integer value.")
    if expected_type is float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as exc:
                raise ConfigLoaderError(f"Field '{path}' expects a float value.") from exc
        raise ConfigLoaderError(f"Field '{path}' expects a float value.")
    if expected_type is str:
        if isinstance(value, (str, Path)):
            return str(value)
        return str(value)
    if expected_type in {list, tuple, dict}:
        return value
    if is_dataclass_type(expected_type):
        if isinstance(value, expected_type):
            return value
        if isinstance(value, Mapping):
            instance = expected_type()
            _update_dataclass(instance, value)
            return instance
    return value


def is_dataclass_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and is_dataclass(annotation)


# ---------------------------------------------------------------------------
# Override helpers
# ---------------------------------------------------------------------------


def _apply_overrides(config: ExperimentConfig, overrides: Sequence[str]) -> None:
    for token in overrides:
        if "=" not in token:
            raise OverrideParseError(f"Invalid override '{token}'; expected 'path=value'.")
        key, raw_value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise OverrideParseError(f"Invalid override '{token}'; empty field path.")
        value = _interpret_literal(raw_value.strip())
        _assign_override(config, key.split("."), value, path=key)


def _interpret_literal(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value == "":
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _assign_override(target: Any, path_tokens: Sequence[str], value: Any, *, path: str) -> None:
    if not path_tokens:
        raise OverrideParseError("Override path cannot be empty.")
    current = target
    for token in path_tokens[:-1]:
        if not is_dataclass(current):
            raise OverrideParseError(
                f"Cannot descend into '{token}' on non-dataclass instance while processing '{path}'."
            )
        field_info = _get_field_info(type(current), token)
        field_type = _resolve_field_type(current, field_info)
        next_value = getattr(current, token)
        base_type, is_optional = _is_optional(field_type)
        if next_value is None:
            if not is_dataclass_type(base_type):
                raise OverrideParseError(
                    f"Field '{token}' is None and cannot be expanded; expected nested dataclass for '{path}'."
                )
            next_value = base_type()
            setattr(current, token, next_value)
        current = next_value

    final_token = path_tokens[-1]
    if not is_dataclass(current):
        raise OverrideParseError(
            f"Cannot set field '{final_token}' on non-dataclass instance while processing '{path}'."
        )
    field_info = _get_field_info(type(current), final_token)
    field_type = _resolve_field_type(current, field_info)
    current_value = getattr(current, final_token)
    coerced = _coerce_assignment(field_type, current_value, value, path=path)
    setattr(current, final_token, coerced)


def _get_field_info(cls: Type[Any], field_name: str):
    for field_info in fields(cls):
        if field_info.name == field_name:
            return field_info
    raise OverrideParseError(f"Unknown field '{field_name}' for {cls.__name__}.")


@lru_cache(maxsize=None)
def _type_hints_cache(cls: Type[Any]) -> Dict[str, Any]:
    return get_type_hints(cls)


def _resolve_field_type(owner: Any | Type[Any], field_info) -> Any:
    cls = owner if isinstance(owner, type) else type(owner)
    hints = _type_hints_cache(cls)
    return hints.get(field_info.name, field_info.type)


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------


def _load_config_document(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ConfigLoaderError(f"Configuration file '{path}' does not exist.")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ConfigLoaderError(
                "PyYAML is required to load YAML configuration files but is not installed."
            )
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        raise ConfigLoaderError(
            f"Unsupported configuration extension '{suffix}'. Use .json, .yaml, or .yml."
        )
    if not isinstance(data, Mapping):
        raise ConfigLoaderError("Top-level configuration document must be a mapping.")
    return data


__all__ = [
    "ConfigLoaderError",
    "ConfigValidationError",
    "DiagnosticsSection",
    "ExperimentConfig",
    "ExperimentConfigError",
    "LoggingSection",
    "ModelSection",
    "MPCSection",
    "OverrideParseError",
    "SamplerSection",
    "load_experiment_config",
]
