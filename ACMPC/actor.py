"""Public actor interface preserving legacy import paths."""

from __future__ import annotations

from .models.actor import ActorOutput, TransformerActor

__all__ = ["TransformerActor", "ActorOutput"]
