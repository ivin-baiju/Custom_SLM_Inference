"""
Architecture registry — maps architecture identifier strings to model classes.

Provides a decorator-based registration pattern so each architecture module
self-registers on import.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Type

logger = logging.getLogger(__name__)

# The global registry: architecture name → model class.
ARCHITECTURE_REGISTRY: dict[str, Type[Any]] = {}


def register_architecture(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Class decorator that registers a model class under *name*.

    Usage::

        @register_architecture("LlamaForCausalLM")
        class LlamaModel(BaseTransformerModel):
            ...

    Parameters
    ----------
    name:
        Canonical architecture string (e.g. ``"LlamaForCausalLM"``).

    Returns
    -------
    Callable
        The unmodified class, now also stored in :data:`ARCHITECTURE_REGISTRY`.
    """

    def _decorator(cls: Type[Any]) -> Type[Any]:
        if name in ARCHITECTURE_REGISTRY:
            logger.warning(
                "Architecture '%s' already registered (%s); overwriting with %s",
                name,
                ARCHITECTURE_REGISTRY[name].__name__,
                cls.__name__,
            )
        ARCHITECTURE_REGISTRY[name] = cls
        logger.debug("Registered architecture '%s' → %s", name, cls.__name__)
        return cls

    return _decorator


def get_architecture(name: str) -> Type[Any]:
    """Look up a registered model class by architecture name.

    Parameters
    ----------
    name:
        Canonical architecture string.

    Returns
    -------
    Type
        The registered model class.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in ARCHITECTURE_REGISTRY:
        raise KeyError(
            f"Architecture '{name}' not found. "
            f"Available: {sorted(ARCHITECTURE_REGISTRY.keys())}"
        )
    return ARCHITECTURE_REGISTRY[name]


def list_architectures() -> list[str]:
    """Return a sorted list of all registered architecture names."""
    return sorted(ARCHITECTURE_REGISTRY.keys())
