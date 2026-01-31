"""
Graph Traverser Agent - A semantic-aware agent that traverses a Knowledge Graph
and creates CHATML compliant datasets.
"""
import logging
import random
from typing import List, Dict, Any, Optional, Deque
from collections import deque
import time

# Import from modular interfaces
from .graph_db import GraphDatabase
from .llm import LLMClient, LLMMessage
from .state_storage import StateStorage, NodeMetadata, NodeState
from .chatml import ChatMLDataset, ChatMLFormatter, ChatMLConversation
from .config import (
    GraphTraverserAgentConfig,
    TraversalStrategy
)
from .prompts import (
    format_semantic_selection_prompt,
    format_node_context
)

logger = logging.getLogger(__name__)


def _short_id(node_id: str, max_len: int = 24) -> str:
    """Shorten node ID for logging."""
    s = str(node_id)
    return f"{s[:max_len]}..." if len(s) > max_len else s


class GraphTraverserAgent:
    """
    Semantic-aware agent that traverses a Knowledge Graph and creates
    CHATML compliant datasets.
    """
    
    def __init__(
        self,
        graph_db: GraphDatabase,
        llm_client: LLMClient,
        state_storage: StateStorage,
        config: GraphTraverserAgentConfig
    ):
        """
        Initialize the Graph Traverser Agent.
        
        Args:
            graph_db: Graph database instance
            llm_client: LLM client instance
            state_storage: State storage instance
            config: Configuration object
        """
        self.graph_db = graph_db
        self.llm_client = llm_client
        self.state_storage = state_storage
        self.config = config
        
        self.dataset = ChatMLDataset()
        self.visited_count = 0
        self.current_depth = 0
    
    def traverse(self) -> ChatMLDataset:
        """
        Start the graph traversal and dataset generation process.
        
        Returns:
            ChatMLDataset containing all generated conversations
        """
        logger.info("Starting graph traversal...")
        
        # Connect to all services
        self._connect_services()
        
        try:
            # Get starting nodes
            start_nodes = self._get_start_nodes()
            
            if not start_nodes:
                logger.warning("No starting nodes found. Cannot traverse.")
                return self.dataset
            
            logger.info(
                "Starting traversal from %d nodes (strategy=%s, max_nodes=%s, max_depth=%s)",
                len(start_nodes),
                self.config.traversal.strategy.value,
                self.config.traversal.max_nodes,
                self.config.traversal.max_depth,
            )

            # Traverse based on strategy
            if self.config.traversal.strategy == TraversalStrategy.BFS:
                self._traverse_bfs(start_nodes)
            elif self.config.traversal.strategy == TraversalStrategy.DFS:
                self._traverse_dfs(start_nodes)
            elif self.config.traversal.strategy == TraversalStrategy.RANDOM:
                self._traverse_random(start_nodes)
            elif self.config.traversal.strategy == TraversalStrategy.SEMANTIC:
                self._traverse_semantic(start_nodes)
            else:
                raise ValueError(f"Unknown traversal strategy: {self.config.traversal.strategy}")
            
            logger.info(f"Traversal complete. Generated {len(self.dataset)} conversations.")
            
            # Save dataset if output path is specified
            if self.config.dataset.output_path:
                self._save_dataset()
            
            return self.dataset
            
        finally:
            self._disconnect_services()
    
    def _connect_services(self) -> None:
        """Connect to all required services."""
        logger.info("Connecting to Neo4j...")
        if hasattr(self.graph_db, 'connect'):
            self.graph_db.connect()
        logger.info("Connecting to Redis state storage...")
        if hasattr(self.state_storage, 'connect'):
            self.state_storage.connect()
        logger.info("Services connected")

    def _disconnect_services(self) -> None:
        """Disconnect from all required services."""
        logger.debug("Disconnecting services...")
        if hasattr(self.graph_db, 'disconnect'):
            self.graph_db.disconnect()
        if hasattr(self.state_storage, 'disconnect'):
            self.state_storage.disconnect()
    
    def _get_start_nodes(self) -> List[str]:
        """Get starting nodes for traversal."""
        if self.config.traversal.seed_node_ids:
            return self.config.traversal.seed_node_ids
        
        # Query for nodes matching criteria
        query = "MATCH (n)"
        conditions = []
        params = {}
        
        if self.config.traversal.node_labels:
            labels = ":".join(self.config.traversal.node_labels)
            query = f"MATCH (n:{labels})"
        
        query += " RETURN elementId(n) as node_id LIMIT 100"
        
        try:
            results = self.graph_db.query(query, params)
            return [str(r["node_id"]) for r in results]
        except Exception as e:
            logger.error(f"Error getting start nodes: {e}")
            return []
    
    def _traverse_bfs(self, start_nodes: List[str]) -> None:
        """Breadth-First Search traversal."""
        queue: Deque[tuple[str, int]] = deque()
        visited_set = set()

        for node_id in start_nodes:
            if node_id not in visited_set:
                queue.append((node_id, 0))
                visited_set.add(node_id)

        logger.info("BFS initialized: queue=%d nodes", len(queue))

        while queue:
            if self._should_stop():
                logger.info("Stopping: reached max_nodes=%d", self.config.traversal.max_nodes)
                break

            node_id, depth = queue.popleft()

            if self.config.traversal.max_depth and depth > self.config.traversal.max_depth:
                continue

            self.current_depth = depth
            self._process_node(node_id)

            neighbors = self.graph_db.get_neighbors(
                node_id,
                relationship_types=self.config.traversal.relationship_types
            )
            new_count = 0
            for neighbor in neighbors:
                neighbor_id = neighbor["id"]
                if neighbor_id not in visited_set:
                    visited_set.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
                    new_count += 1

            logger.debug(
                "BFS: processed %s (depth=%d), +%d neighbors, queue=%d, processed=%d",
                _short_id(node_id), depth, new_count, len(queue), self.visited_count,
            )
    
    def _traverse_dfs(self, start_nodes: List[str]) -> None:
        """Depth-First Search traversal."""
        stack: List[tuple[str, int]] = []
        visited_set = set()

        for node_id in reversed(start_nodes):
            if node_id not in visited_set:
                stack.append((node_id, 0))
                visited_set.add(node_id)

        logger.info("DFS initialized: stack=%d nodes", len(stack))

        while stack:
            if self._should_stop():
                logger.info("Stopping: reached max_nodes=%d", self.config.traversal.max_nodes)
                break

            node_id, depth = stack.pop()

            if self.config.traversal.max_depth and depth > self.config.traversal.max_depth:
                continue

            self.current_depth = depth
            self._process_node(node_id)

            neighbors = self.graph_db.get_neighbors(
                node_id,
                relationship_types=self.config.traversal.relationship_types
            )
            new_count = 0
            for neighbor in neighbors:
                neighbor_id = neighbor["id"]
                if neighbor_id not in visited_set:
                    visited_set.add(neighbor_id)
                    stack.append((neighbor_id, depth + 1))
                    new_count += 1

            logger.debug(
                "DFS: processed %s (depth=%d), +%d neighbors, stack=%d, processed=%d",
                _short_id(node_id), depth, new_count, len(stack), self.visited_count,
            )
    
    def _traverse_random(self, start_nodes: List[str]) -> None:
        """Random traversal."""
        visited_set = set()
        unvisited = list(start_nodes)
        random.shuffle(unvisited)

        logger.info("Random traversal initialized: %d nodes to process", len(unvisited))

        while unvisited:
            if self._should_stop():
                break

            node_id = unvisited.pop()
            visited_set.add(node_id)

            self._process_node(node_id)

            neighbors = self.graph_db.get_neighbors(
                node_id,
                relationship_types=self.config.traversal.relationship_types
            )
            for neighbor in neighbors:
                neighbor_id = neighbor["id"]
                if neighbor_id not in visited_set and neighbor_id not in unvisited:
                    unvisited.append(neighbor_id)

            random.shuffle(unvisited)
            logger.debug("Random: processed %s, remaining=%d", _short_id(node_id), len(unvisited))
    
    def _traverse_semantic(self, start_nodes: List[str]) -> None:
        """
        Semantic-aware traversal using LLM to decide which nodes to visit next.
        """
        visited_set = set()
        candidates = list(start_nodes)

        logger.info("Semantic traversal initialized: %d candidates", len(candidates))

        while candidates:
            if self._should_stop():
                break

            if len(candidates) > 1:
                selected_node = self._select_semantic_node(candidates, visited_set)
            else:
                selected_node = candidates[0]

            candidates.remove(selected_node)
            visited_set.add(selected_node)

            self._process_node(selected_node)

            neighbors = self.graph_db.get_neighbors(
                selected_node,
                relationship_types=self.config.traversal.relationship_types
            )
            for neighbor in neighbors:
                neighbor_id = neighbor["id"]
                if neighbor_id not in visited_set and neighbor_id not in candidates:
                    candidates.append(neighbor_id)

            logger.debug("Semantic: processed %s, candidates=%d", _short_id(selected_node), len(candidates))
    
    def _select_semantic_node(
        self,
        candidates: List[str],
        visited: set
    ) -> str:
        """
        Use LLM to select the most semantically relevant node from candidates.
        """
        # Get information about candidate nodes
        candidate_info = []
        for node_id in candidates[:10]:  # Limit to 10 for LLM context
            node = self.graph_db.get_node(node_id)
            if node:
                candidate_info.append({
                    "id": node_id,
                    "labels": node.get("labels", []),
                    "properties": node.get("properties", {})
                })
        
        # Create prompt for LLM using versioned prompt
        prompt = format_semantic_selection_prompt(candidate_info, version="current")
        
        messages = [
            LLMMessage(role="user", content=prompt)
        ]
        
        try:
            response = self.llm_client.generate(messages, temperature=0.3, max_tokens=10)
            # Parse response to get node index
            try:
                index = int(response.strip()) - 1
                if 0 <= index < len(candidate_info):
                    return candidate_info[index]["id"]
            except ValueError:
                pass
        except Exception as e:
            logger.warning(f"Error in semantic selection: {e}")
        
        # Fallback to first candidate
        return candidates[0]
    
    def _process_node(self, node_id: str) -> None:
        """
        Process a single node: generate prompt/response pair and add to dataset.
        """
        existing_state = self.state_storage.get_node_state(node_id)
        if existing_state and existing_state.state == NodeState.VISITED:
            logger.debug("Node %s already visited (Redis), skipping", _short_id(node_id))
            return

        logger.info(
            "[%d/%s] Processing node %s (depth=%d)...",
            self.visited_count + 1,
            self.config.traversal.max_nodes or "∞",
            _short_id(node_id),
            self.current_depth,
        )

        self._mark_in_progress(node_id)

        node = self.graph_db.get_node(node_id)
        if not node:
            logger.warning("Node %s not found in graph, marking skipped", _short_id(node_id))
            self._mark_skipped(node_id)
            return
        
        # Get neighbors for context
        neighbors = self.graph_db.get_neighbors(node_id, limit=5)
        
        # Generate prompt using seed prompts or default
        prompt = self._generate_prompt(node, neighbors)
        
        logger.debug("Calling LLM for response...")
        response = self._generate_response(node, neighbors, prompt)
        logger.debug("LLM response received")
        
        # Create CHATML conversation
        metadata = {
            "node_id": node_id,
            "labels": node.get("labels", []),
            "depth": self.current_depth,
            "timestamp": time.time()
        }
        
        conversation = ChatMLFormatter.create_conversation_pair(
            prompt=prompt,
            response=response,
            system_message=self.config.dataset.system_message,
            metadata=metadata if self.config.dataset.include_metadata else None
        )
        
        # Add to dataset
        self.dataset.add_conversation(conversation)
        
        try:
            self.state_storage.mark_visited(node_id, metadata={"processed": True})
        except Exception as e:
            logger.warning("Failed to persist visited state for %s: %s", _short_id(node_id), e)
        self.visited_count += 1

        logger.info(
            "  ✓ Node %s done → Redis (visited=%d, labels=%s)",
            _short_id(node_id),
            self.visited_count,
            node.get("labels", []),
        )
    
    def _generate_prompt(
        self,
        node: Dict[str, Any],
        neighbors: List[Dict[str, Any]]
    ) -> str:
        """Generate a prompt for the node."""
        # Use seed prompts if available
        if self.config.dataset.seed_prompts:
            template = random.choice(self.config.dataset.seed_prompts)
            try:
                return template.format(
                    labels=node.get("labels", []),
                    properties=node.get("properties", {}),
                    neighbors=neighbors
                )
            except KeyError:
                pass
        
        # Use custom template if provided
        if self.config.dataset.prompt_template:
            try:
                return self.config.dataset.prompt_template.format(
                    labels=node.get("labels", []),
                    properties=node.get("properties", {}),
                    neighbors=neighbors
                )
            except KeyError:
                pass
        
        # Default prompt
        return ChatMLFormatter.format_node_prompt(node)
    
    def _generate_response(
        self,
        node: Dict[str, Any],
        neighbors: List[Dict[str, Any]],
        prompt: str
    ) -> str:
        """Generate a response using the LLM."""
        messages = []
        
        if self.config.dataset.system_message:
            messages.append(LLMMessage(role="system", content=self.config.dataset.system_message))
        
        # Add context about the node using versioned template
        context = format_node_context(
            labels=node.get('labels', []),
            properties=node.get('properties', {}),
            neighbors_count=len(neighbors) if neighbors else 0,
            version="current"
        )
        
        messages.append(LLMMessage(role="user", content=f"{context}\n\n{prompt}"))
        
        try:
            response = self.llm_client.generate(
                messages,
                temperature=0.7,
                max_tokens=500
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response for node: {e}"
    
    def _mark_in_progress(self, node_id: str) -> None:
        """Persist node to Redis as in-progress before expensive LLM call."""
        try:
            existing = self.state_storage.get_node_state(node_id)
            metadata = NodeMetadata(
                node_id=node_id,
                state=NodeState.IN_PROGRESS,
                visit_count=existing.visit_count if existing else 0,
                last_visited=time.time(),
                metadata={"processing": True},
            )
            self.state_storage.set_node_state(node_id, metadata)
        except Exception as e:
            logger.debug(f"Could not persist in-progress state for {node_id}: {e}")

    def _mark_skipped(self, node_id: str) -> None:
        """Persist node as skipped (e.g. not found) to Redis."""
        try:
            existing = self.state_storage.get_node_state(node_id)
            metadata = NodeMetadata(
                node_id=node_id,
                state=NodeState.SKIPPED,
                visit_count=existing.visit_count if existing else 0,
                last_visited=time.time(),
                metadata={"reason": "not_found"},
            )
            self.state_storage.set_node_state(node_id, metadata)
        except Exception as e:
            logger.debug(f"Could not persist skipped state for {node_id}: {e}")

    def _should_stop(self) -> bool:
        """Check if traversal should stop."""
        if self.config.traversal.max_nodes:
            return self.visited_count >= self.config.traversal.max_nodes
        return False
    
    def _save_dataset(self) -> None:
        """Save the dataset to file."""
        path = self.config.dataset.output_path
        if self.config.dataset.output_format == "jsonl":
            self.dataset.save_jsonl(path)
        else:
            self.dataset.save_json(path)
        logger.info("Dataset saved: %s (%d conversations)", path, len(self.dataset))
