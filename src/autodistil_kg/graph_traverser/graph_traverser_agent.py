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
    format_node_context,
    format_path_description,
    format_path_reasoning_prompt,
    format_subgraph_synthesis_prompt,
    format_reasoning_qa_prompt,
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
            elif self.config.traversal.strategy == TraversalStrategy.REASONING:
                self._traverse_reasoning(start_nodes)
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
        
        # Use the graph_db's compatible ID expression
        id_expr = getattr(self.graph_db, '_node_id_expr', lambda v: f"toString(id({v}))")("n")
        query += f" RETURN {id_expr} as node_id LIMIT 100"
        
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
    
    def _traverse_reasoning(self, start_nodes: List[str]) -> None:
        """
        Deep reasoning traversal: for each node, extract a depth-N subgraph,
        reason through each path, synthesize understanding, then generate
        rich QA pairs for SLM distillation.
        """
        visited_set = set()
        queue: Deque[tuple[str, int]] = deque()

        for node_id in start_nodes:
            if node_id not in visited_set:
                queue.append((node_id, 0))
                visited_set.add(node_id)

        reasoning_depth = self.config.traversal.reasoning_depth
        max_paths = self.config.traversal.max_paths_per_node

        logger.info(
            "Reasoning traversal initialized: %d seed nodes, subgraph_depth=%d, max_paths=%d",
            len(queue), reasoning_depth, max_paths,
        )

        while queue:
            if self._should_stop():
                logger.info("Stopping: reached max_nodes=%d", self.config.traversal.max_nodes)
                break

            node_id, depth = queue.popleft()

            if self.config.traversal.max_depth and depth > self.config.traversal.max_depth:
                continue

            self.current_depth = depth
            self._process_node_reasoning(node_id, reasoning_depth, max_paths)

            # After processing, add unvisited subgraph nodes to the queue
            # so the traversal expands outward through the graph
            subgraph = self.graph_db.get_subgraph(
                node_id,
                depth=1,
                relationship_types=self.config.traversal.relationship_types,
            )
            for nid in subgraph.get("nodes", {}):
                if nid not in visited_set:
                    visited_set.add(nid)
                    queue.append((nid, depth + 1))

            logger.debug(
                "Reasoning: processed %s (depth=%d), queue=%d, processed=%d",
                _short_id(node_id), depth, len(queue), self.visited_count,
            )

    def _process_node_reasoning(
        self,
        node_id: str,
        reasoning_depth: int,
        max_paths: int,
    ) -> None:
        """
        Deep reasoning processing for a single node:
        1. Query depth-N subgraph
        2. Reason through each path
        3. Synthesize all path reasonings
        4. Generate distillation-ready QA pair
        """
        existing_state = self.state_storage.get_node_state(node_id)
        if existing_state and existing_state.state == NodeState.VISITED:
            logger.debug("Node %s already visited, skipping", _short_id(node_id))
            return

        logger.info(
            "[%d/%s] Deep reasoning on node %s (depth=%d, subgraph_depth=%d)...",
            self.visited_count + 1,
            self.config.traversal.max_nodes or "∞",
            _short_id(node_id),
            self.current_depth,
            reasoning_depth,
        )

        self._mark_in_progress(node_id)

        # Step 1: Get the full subgraph around this node
        subgraph = self.graph_db.get_subgraph(
            node_id,
            depth=reasoning_depth,
            relationship_types=self.config.traversal.relationship_types,
        )

        center = subgraph["center"]
        paths = subgraph["paths"]
        nodes_map = subgraph["nodes"]
        edges = subgraph["edges"]

        if not center or (not center.get("labels") and not center.get("properties")):
            logger.warning("Node %s has no data, marking skipped", _short_id(node_id))
            self._mark_skipped(node_id)
            return

        logger.info(
            "  Subgraph: %d nodes, %d edges, %d paths",
            len(nodes_map), len(edges), len(paths),
        )

        # Deduplicate paths by their string representation to avoid redundant reasoning
        unique_paths = []
        seen_path_keys = set()
        for path in paths:
            key = format_path_description(path)
            if key not in seen_path_keys:
                seen_path_keys.add(key)
                unique_paths.append(path)

        # Limit paths
        if len(unique_paths) > max_paths:
            # Prefer longer paths (more multi-hop reasoning potential)
            unique_paths.sort(key=lambda p: len(p), reverse=True)
            unique_paths = unique_paths[:max_paths]

        logger.info("  Reasoning over %d unique paths...", len(unique_paths))

        # Step 2: Reason through each path
        path_analyses = []
        for i, path in enumerate(unique_paths):
            logger.debug(
                "  Path %d/%d: %s",
                i + 1, len(unique_paths), format_path_description(path)[:100],
            )
            analysis = self._reason_through_path(center, path)
            if analysis:
                path_analyses.append(analysis)

        if not path_analyses:
            logger.warning("  No path analyses produced for %s, skipping", _short_id(node_id))
            self._mark_skipped(node_id)
            return

        # Step 3: Synthesize all path reasonings
        logger.info("  Synthesizing %d path analyses...", len(path_analyses))
        synthesis = self._synthesize_subgraph(
            center, path_analyses, len(nodes_map), len(edges)
        )

        # Step 4: Generate distillation QA pair
        logger.info("  Generating QA pair for distillation...")
        qa_pair = self._generate_reasoning_qa(center, synthesis)

        # Create CHATML conversations
        # Conv 1: The full reasoning chain (system + synthesis as teaching material)
        metadata = {
            "node_id": node_id,
            "labels": center.get("labels", []),
            "depth": self.current_depth,
            "subgraph_nodes": len(nodes_map),
            "subgraph_edges": len(edges),
            "paths_analyzed": len(path_analyses),
            "strategy": "reasoning",
            "timestamp": time.time(),
        }

        # Create the main distillation conversation from the QA pair
        conversation = ChatMLFormatter.create_conversation_pair(
            prompt=qa_pair["question"],
            response=qa_pair["answer"],
            system_message=self.config.dataset.system_message,
            metadata=metadata if self.config.dataset.include_metadata else None,
        )
        self.dataset.add_conversation(conversation)

        # Also add the synthesis as a second training example (teach the model the deep reasoning)
        synthesis_prompt = (
            f"Explain everything you know about {center.get('properties', {}).get('name', center.get('labels', ['this entity'])[0] if center.get('labels') else 'this entity')} "
            f"and its relationships in the knowledge graph, including multi-hop inferences."
        )
        synthesis_conv = ChatMLFormatter.create_conversation_pair(
            prompt=synthesis_prompt,
            response=synthesis,
            system_message=self.config.dataset.system_message,
            metadata={**metadata, "type": "synthesis"} if self.config.dataset.include_metadata else None,
        )
        self.dataset.add_conversation(synthesis_conv)

        try:
            self.state_storage.mark_visited(node_id, metadata={"processed": True, "strategy": "reasoning"})
        except Exception as e:
            logger.warning("Failed to persist visited state for %s: %s", _short_id(node_id), e)
        self.visited_count += 1

        logger.info(
            "  ✓ Node %s done → 2 conversations (visited=%d)",
            _short_id(node_id), self.visited_count,
        )

    def _reason_through_path(
        self,
        center_node: Dict[str, Any],
        path: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Use LLM to reason through a single path from the subgraph.
        Returns the LLM's step-by-step reasoning analysis.
        """
        prompt = format_path_reasoning_prompt(center_node, path)
        messages = []

        if self.config.dataset.system_message:
            messages.append(LLMMessage(
                role="system",
                content="You are a knowledge graph reasoning engine. Analyze paths deeply and extract multi-step knowledge.",
            ))

        messages.append(LLMMessage(role="user", content=prompt))

        try:
            response = self.llm_client.generate(
                messages, temperature=0.4, max_tokens=800
            )
            return response.strip()
        except Exception as e:
            logger.warning("Error reasoning through path: %s", e)
            return None

    def _synthesize_subgraph(
        self,
        center_node: Dict[str, Any],
        path_analyses: List[str],
        num_nodes: int,
        num_edges: int,
    ) -> str:
        """
        Synthesize multiple path-level analyses into a comprehensive understanding.
        """
        prompt = format_subgraph_synthesis_prompt(
            center_node, path_analyses, num_nodes, num_edges
        )
        messages = []

        if self.config.dataset.system_message:
            messages.append(LLMMessage(
                role="system",
                content="You are a knowledge synthesis engine. Combine multiple analyses into comprehensive, educational summaries.",
            ))

        messages.append(LLMMessage(role="user", content=prompt))

        try:
            response = self.llm_client.generate(
                messages, temperature=0.5, max_tokens=1500
            )
            return response.strip()
        except Exception as e:
            logger.error("Error synthesizing subgraph: %s", e)
            return "Error during synthesis: " + str(e)

    def _generate_reasoning_qa(
        self,
        center_node: Dict[str, Any],
        synthesis: str,
    ) -> Dict[str, str]:
        """
        Generate a high-quality QA pair from the synthesis, suitable for SLM distillation.
        Returns dict with 'question' and 'answer' keys.
        """
        prompt = format_reasoning_qa_prompt(center_node, synthesis)
        messages = []

        if self.config.dataset.system_message:
            messages.append(LLMMessage(
                role="system",
                content="You are a training data generator. Create high-quality question-answer pairs for language model training.",
            ))

        messages.append(LLMMessage(role="user", content=prompt))

        try:
            response = self.llm_client.generate(
                messages, temperature=0.6, max_tokens=1200
            )

            # Parse the QA pair from the response
            question = ""
            answer = ""

            if "**Question:**" in response and "**Answer:**" in response:
                parts = response.split("**Answer:**", 1)
                question = parts[0].replace("**Question:**", "").strip()
                answer = parts[1].strip()
            elif "Question:" in response and "Answer:" in response:
                parts = response.split("Answer:", 1)
                question = parts[0].replace("Question:", "").strip()
                answer = parts[1].strip()
            else:
                # Fallback: use the synthesis as answer and generate a simple question
                name = (
                    center_node.get("properties", {}).get("name")
                    or center_node.get("properties", {}).get("title")
                    or ", ".join(center_node.get("labels", ["this entity"]))
                )
                question = f"What can you tell me about {name} and its relationships, including any multi-step inferences?"
                answer = response

            return {"question": question, "answer": answer}

        except Exception as e:
            logger.error("Error generating QA pair: %s", e)
            name = center_node.get("properties", {}).get("name", "this entity")
            return {
                "question": f"What do you know about {name}?",
                "answer": synthesis,
            }

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
