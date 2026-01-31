"""
State Storage Interface Module.

This module defines the abstract interface for state storage providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class NodeState(str, Enum):
    """State of a node in the traversal."""
    VISITED = "visited"
    NOT_VISITED = "not_visited"
    IN_PROGRESS = "in_progress"
    SKIPPED = "skipped"


@dataclass
class NodeMetadata:
    """Metadata associated with a node during traversal."""
    node_id: str
    state: NodeState
    visit_count: int
    last_visited: Optional[float]  # timestamp
    metadata: Dict[str, Any]


class StateStorage(ABC):
    """Abstract interface for state storage providers."""
    
    @abstractmethod
    def get_node_state(self, node_id: str) -> Optional[NodeMetadata]:
        """Get the state of a node."""
        pass
    
    @abstractmethod
    def set_node_state(self, node_id: str, metadata: NodeMetadata) -> None:
        """Set the state of a node."""
        pass
    
    @abstractmethod
    def mark_visited(self, node_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark a node as visited."""
        pass
    
    @abstractmethod
    def mark_not_visited(self, node_id: str) -> None:
        """Mark a node as not visited."""
        pass
    
    @abstractmethod
    def get_visited_nodes(self) -> List[str]:
        """Get all visited node IDs."""
        pass
    
    @abstractmethod
    def get_unvisited_nodes(self, all_node_ids: List[str]) -> List[str]:
        """Get all unvisited node IDs from a list."""
        pass
    
    @abstractmethod
    def clear_state(self) -> None:
        """Clear all state."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the traversal state."""
        pass
