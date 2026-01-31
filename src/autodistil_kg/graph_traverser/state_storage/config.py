"""
State Storage Configuration Module.

This module defines configuration classes for state storage providers.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class StateStorageConfig:
    """Configuration for state storage."""
    provider: str  # "redis", etc.
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    key_prefix: str = "graph_traverser:"
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.provider:
            raise ValueError("State storage provider must be specified")
