"""
Graph Database Configuration Module.

This module defines configuration classes for graph database providers.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class GraphDatabaseConfig:
    """Configuration for graph database connection."""
    provider: str  # "neo4j", etc.
    uri: str
    user: str
    password: str
    database: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.provider:
            raise ValueError("Graph database provider must be specified")
        if not self.uri:
            raise ValueError("Graph database URI must be specified")
        if not self.user:
            raise ValueError("Graph database user must be specified")
        if not self.password:
            raise ValueError("Graph database password must be specified")
