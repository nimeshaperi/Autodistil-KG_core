"""
Graph Database Interface Module.

This module defines the abstract interface for graph database providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol


class GraphNode(Protocol):
    """Protocol for graph nodes."""
    id: str
    labels: List[str]
    properties: Dict[str, Any]


class GraphEdge(Protocol):
    """Protocol for graph edges."""
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]


class GraphDatabase(ABC):
    """Abstract interface for graph database providers."""
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the graph database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the graph database."""
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID."""
        pass
    
    @abstractmethod
    def get_neighbors(
        self, 
        node_id: str, 
        relationship_types: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes of a given node."""
        pass
    
    @abstractmethod
    def get_node_properties(self, node_id: str) -> Dict[str, Any]:
        """Get all properties of a node."""
        pass
    
    @abstractmethod
    def get_relationships(
        self, 
        node_id: str, 
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all relationships of a node."""
        pass
    
    @abstractmethod
    def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a custom query on the graph database."""
        pass
    
    @abstractmethod
    def get_node_count(self) -> int:
        """Get total number of nodes in the graph."""
        pass
