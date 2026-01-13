"""Configuration dataclasses for the refactored ACMPC package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .mpc import EconomicMPCConfig


@dataclass
class CostMapBounds:
    """Bounds applied to the diagonal and linear cost coefficients."""

    q_min: float = 0.1
    q_max: float = 1e4
    r_min: float = 0.1
    r_max: float = 1e4
    # FIX: Reduced from 1e4 to 100 to prevent linear terms from dominating
    linear_state_bound: float = 100.0
    linear_action_bound: float = 100.0


@dataclass
class CostMapConfig:
    """Configuration for the neural cost map head."""

    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.0
    bounds: CostMapBounds = field(default_factory=CostMapBounds)
    # Optional training-time exploration noise on cost parameters.
    # When > 0, small Gaussian noise is injected into Q/R (diag) and/or
    # linear terms during training to encourage exploration in cost space.
    noise_scale_diag: float = 0.0
    noise_scale_linear: float = 0.0


@dataclass
class TransformerConfig:
    """Hyper-parameters for the Transformer-XL backbone."""

    d_model: int = 256
    n_heads: int = 8
    d_inner: int = 1024
    dropout: float = 0.1
    mem_len: int = 64
    n_layers: int = 6


@dataclass
class MLPConfig:
    """Hyper-parameters for the MLP backbone."""

    hidden_dim: int = 512
    output_dim: int = 512
    num_layers: int = 2
    activation: str = "relu"  # options: relu, gelu
    dropout: float = 0.0


def _default_mpc_config() -> EconomicMPCConfig:
    d_model = TransformerConfig().d_model
    return EconomicMPCConfig(
        horizon=16,
        state_dim=16,
        action_dim=4,
        dt=0.02,
        latent_dim=d_model,
    )


@dataclass
class ActorConfig:
    """Configuration for the actor model."""

    input_dim: int = 0
    policy_head: str = "mpc"  # options: mpc, direct
    backbone_type: str = "transformer"  # options: transformer, mlp
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    mpc: EconomicMPCConfig = field(default_factory=_default_mpc_config)
    cost_map: CostMapConfig | None = field(default_factory=CostMapConfig)
    include_prev_action: bool = False
    prev_action_dim: int = 0
    include_lidar: bool = False
    lidar_dim: int = 0
    waypoint_dim: int = 0
    waypoint_sequence_len: int = 0
    tanh_rescale_actions: bool = False
    # When True and waypoints are available, the MPC head treats
    # the current waypoint as an external tracking reference.
    use_waypoint_as_ref: bool = False
    kv_cache_max_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if self.policy_head not in {"mpc", "direct"}:
            raise ValueError("actor.policy_head must be one of {'mpc', 'direct'}.")
        if self.backbone_type not in {"transformer", "mlp"}:
            raise ValueError("actor.backbone_type must be one of {'transformer', 'mlp'}.")
        if self.input_dim <= 0:
            self.input_dim = self.mpc.state_dim
        if self.backbone_type == "transformer":
            if getattr(self.mpc, "latent_dim", None) is None or self.mpc.latent_dim <= 0:
                self.mpc.latent_dim = self.transformer.d_model
            if self.transformer.d_model != self.mpc.latent_dim:
                raise ValueError("Transformer d_model must match MPC latent_dim.")
        elif self.backbone_type == "mlp":
            if getattr(self.mpc, "latent_dim", None) is None or self.mpc.latent_dim <= 0:
                self.mpc.latent_dim = self.mlp.output_dim
            if self.mlp.output_dim != self.mpc.latent_dim:
                raise ValueError("MLP output_dim must match MPC latent_dim.")
        if self.include_prev_action and self.prev_action_dim <= 0:
            self.prev_action_dim = self.mpc.action_dim
        if not self.include_prev_action:
            self.prev_action_dim = 0
        if not self.include_lidar:
            self.lidar_dim = 0
        if self.waypoint_dim < 0:
            raise ValueError("waypoint_dim must be >= 0.")
        if self.waypoint_sequence_len < 0:
            raise ValueError("waypoint_sequence_len must be >= 0.")


@dataclass
class CriticConfig:
    """Configuration for the critic model."""

    input_dim: int = 0
    backbone_type: str = "transformer"  # options: transformer, mlp
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    value_head_hidden: int = 256
    include_prev_action: bool = False
    prev_action_dim: int = 0
    include_lidar: bool = False
    lidar_dim: int = 0
    waypoint_dim: int = 0
    waypoint_sequence_len: int = 0
    kv_cache_max_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if self.backbone_type not in {"transformer", "mlp"}:
            raise ValueError("critic.backbone_type must be one of {'transformer', 'mlp'}.")
        if self.input_dim <= 0:
            if self.backbone_type == "transformer":
                self.input_dim = self.transformer.d_model
            else:
                self.input_dim = self.mlp.output_dim
        if not self.include_prev_action:
            self.prev_action_dim = 0
        if not self.include_lidar:
            self.lidar_dim = 0
        if self.waypoint_dim < 0:
            raise ValueError("waypoint_dim must be >= 0.")
        if self.waypoint_sequence_len < 0:
            raise ValueError("waypoint_sequence_len must be >= 0.")


@dataclass
class AgentConfig:
    """Configuration bundling actor, critic, and optimisation settings."""

    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    device: str = "cpu"

    def __post_init__(self) -> None:
        # Sync backbone type if not explicitly set
        if self.critic.backbone_type == "transformer" and self.actor.backbone_type == "mlp":
            # Default to matching actor backbone
            self.critic.backbone_type = "mlp"
        elif self.critic.backbone_type == "mlp" and self.actor.backbone_type == "transformer":
            # Default to matching actor backbone
            self.critic.backbone_type = "transformer"
        if self.critic.input_dim <= 0:
            self.critic.input_dim = self.actor.input_dim
        if self.critic.include_prev_action and self.critic.prev_action_dim <= 0:
            self.critic.prev_action_dim = self.actor.prev_action_dim
        if self.critic.include_lidar and self.critic.lidar_dim <= 0:
            self.critic.lidar_dim = self.actor.lidar_dim
        if self.critic.waypoint_dim <= 0 and self.actor.waypoint_dim > 0:
            self.critic.waypoint_dim = self.actor.waypoint_dim
        if self.critic.waypoint_sequence_len <= 0 and self.actor.waypoint_sequence_len > 0:
            self.critic.waypoint_sequence_len = self.actor.waypoint_sequence_len
        if self.critic.kv_cache_max_tokens is None:
            self.critic.kv_cache_max_tokens = self.actor.kv_cache_max_tokens
