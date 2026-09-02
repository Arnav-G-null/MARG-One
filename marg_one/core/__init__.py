"""
MARG-One Core Subsystem Interface Definitions.
Provides base interfaces for state management, pipeline coordination, and event buses.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModuleState:
    """Represents standard health and telemetry state for MARG-One subsystems."""
    name: str
    is_active: bool
    fps: float = 0.0
    last_update_timestamp: float = 0.0
    payload: Optional[Dict[str, Any]] = None
