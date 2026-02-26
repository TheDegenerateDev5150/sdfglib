from typing import Callable, Optional
from docc.sdfg import StructuredSDFG

TargetScheduleFn = Callable[[StructuredSDFG, str], None]

_target_registry: dict[str, TargetScheduleFn] = {}


def register_target(name: str, schedule_fn: TargetScheduleFn) -> None:
    """Register a custom target scheduler.

    The schedule function will be called with:
    - sdfg: The StructuredSDFG to schedule (has _native_ptr for native access)
    - category: The target category (e.g., "desktop", "server")

    Args:
        name: Target name (e.g., "openmp")
        schedule_fn: Function that performs scheduling transformations
    """
    _target_registry[name] = schedule_fn


def unregister_target(name: str) -> None:
    """Unregister a custom target scheduler."""
    _target_registry.pop(name, None)


def get_target(name: str) -> Optional[TargetScheduleFn]:
    """Get a registered target scheduler, or None if not found."""
    return _target_registry.get(name)


def is_custom_target(name: str) -> bool:
    """Check if a target name has a custom scheduler registered."""
    return name in _target_registry
