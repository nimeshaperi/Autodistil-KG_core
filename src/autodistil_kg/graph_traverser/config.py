"""
Graph Traverser Configuration Module.

This module defines high-level configuration classes that compose
the individual module configurations.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

# Import module configs
from .graph_db.config import GraphDatabaseConfig
from .llm.config import LLMConfig
from .state_storage.config import StateStorageConfig


class TraversalStrategy(str, Enum):
    """Enumeration of traversal strategies."""
    BFS = "bfs"  # Breadth-First Search
    DFS = "dfs"  # Depth-First Search
    RANDOM = "random"
    SEMANTIC = "semantic"  # Semantic-aware traversal using LLM
    REASONING = "reasoning"  # Deep multi-hop reasoning with subgraph exploration


@dataclass
class TraversalConfig:
    """Configuration for graph traversal."""
    strategy: TraversalStrategy
    max_nodes: Optional[int] = None  # Maximum number of nodes to traverse
    max_depth: Optional[int] = None  # Maximum depth for traversal
    relationship_types: Optional[List[str]] = None  # Filter by relationship types
    node_labels: Optional[List[str]] = None  # Filter by node labels
    seed_node_ids: Optional[List[str]] = None  # Starting nodes
    reasoning_depth: int = 2  # Subgraph depth for REASONING strategy
    max_paths_per_node: int = 15  # Max paths to reason over per node
    additional_params: dict = field(default_factory=dict)


@dataclass
class DatasetGenerationConfig:
    """Configuration for dataset generation."""
    seed_prompts: List[str] = field(default_factory=list)  # Seed prompt templates
    system_message: Optional[str] = None
    prompt_template: Optional[str] = None
    include_metadata: bool = True
    output_format: str = "jsonl"  # "jsonl" or "json"
    output_path: Optional[str] = None
    additional_params: dict = field(default_factory=dict)


@dataclass
class GraphTraverserAgentConfig:
    """Complete configuration for GraphTraverserAgent."""
    graph_db: GraphDatabaseConfig
    llm: LLMConfig
    state_storage: StateStorageConfig
    traversal: TraversalConfig
    dataset: DatasetGenerationConfig
    
    def validate(self) -> None:
        """Validate the configuration."""
        self.graph_db.validate()
        self.llm.validate()
        self.state_storage.validate()
        if not self.traversal.strategy:
            raise ValueError("Traversal strategy must be specified")
