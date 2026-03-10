"""
Prompts used by the Graph Traverser Agent.

This module contains all prompts used in the agent, versioned for easy editing
and tracking. Each prompt is versioned (V1, V2, etc.) to allow for evolution
while maintaining backward compatibility.
"""

# ============================================================================
# Semantic Node Selection Prompts
# ============================================================================

SEMANTIC_NODE_SELECTION_PROMPT_V1 = """Given the following candidate nodes from a knowledge graph, select the most semantically relevant one to explore next. Consider the context and relationships.

Candidates:
{candidates_list}

Respond with just the number (1-{max_candidates}) of the selected node."""


SEMANTIC_NODE_SELECTION_PROMPT_V2 = """You are analyzing a knowledge graph traversal. Given the following candidate nodes, select the most semantically relevant and informative node to explore next. Consider:
- The semantic relationships between nodes
- The information value of each node
- The context of the current traversal path

Candidates:
{candidates_list}

Provide only the number (1-{max_candidates}) of your selected node."""


# Current version (default)
SEMANTIC_NODE_SELECTION_PROMPT = SEMANTIC_NODE_SELECTION_PROMPT_V1


# ============================================================================
# Node Context Building Templates
# ============================================================================

NODE_CONTEXT_TEMPLATE_V1 = """Node Information:
Labels: {labels}
Properties: {properties}
{neighbors_info}"""


NODE_CONTEXT_TEMPLATE_V2 = """## Node Details

**Labels:** {labels}

**Properties:**
{properties_formatted}

{neighbors_info}"""


# Current version (default)
NODE_CONTEXT_TEMPLATE = NODE_CONTEXT_TEMPLATE_V1


# ============================================================================
# Deep Reasoning Prompts (REASONING strategy)
# ============================================================================

PATH_REASONING_PROMPT_V1 = """You are a knowledge graph reasoning engine. Analyze this path from a knowledge graph and extract deep, structured reasoning.

## Center Entity
{center_entity}

## Path to Analyze
{path_description}

## Instructions
Think step-by-step through this path:
1. **Entity Analysis**: What is each entity in this path? What are its key properties?
2. **Relationship Reasoning**: What does each relationship tell us? Why do these entities connect this way?
3. **Multi-hop Inference**: What knowledge can ONLY be derived by following this full chain of connections? What implicit facts emerge?
4. **Contextual Understanding**: How does this path contribute to understanding the center entity in a broader context?

Provide your reasoning as a structured, detailed analysis. Focus on extractable knowledge that would help a smaller language model understand these concepts and their connections."""


SUBGRAPH_SYNTHESIS_PROMPT_V1 = """You are synthesizing knowledge from a knowledge graph exploration. You have analyzed multiple paths radiating from a center entity. Now combine these path-level insights into a comprehensive understanding.

## Center Entity
{center_entity}

## Path Analyses
{path_analyses}

## Subgraph Statistics
- Total unique entities: {num_nodes}
- Total relationships: {num_edges}
- Total paths analyzed: {num_paths}

## Instructions
Synthesize all path analyses into a comprehensive knowledge summary:
1. **Core Concepts**: What are the fundamental concepts and how do they relate to the center entity?
2. **Key Relationships**: What are the most important relationship patterns? Are there recurring themes?
3. **Multi-hop Knowledge**: What complex, multi-step facts emerge from combining paths? What can we infer that isn't directly stated?
4. **Teaching Summary**: Write a clear, detailed explanation that would teach someone (or a smaller AI model) everything important about this entity and its connected knowledge. Be thorough — include specific facts, relationships, and inferred knowledge.

Write in a clear, educational style suitable for training a language model."""


REASONING_QA_GENERATION_PROMPT_V1 = """Based on the following deep knowledge graph analysis, generate a high-quality question-answer pair suitable for training a smaller language model.

## Knowledge Analysis
{synthesis}

## Center Entity
{center_entity}

## Instructions
Generate a question that:
- Requires multi-step reasoning or connecting multiple pieces of knowledge
- Cannot be answered with a single fact lookup
- Tests understanding of relationships and inferences
- Is natural and educational

Then provide a comprehensive, well-structured answer that:
- Walks through the reasoning step by step
- References specific entities and relationships
- Includes both directly stated and inferred knowledge
- Is detailed enough to teach the concepts

Format your response as:
**Question:** [Your question here]

**Answer:** [Your detailed answer here]"""


# Current versions
PATH_REASONING_PROMPT = PATH_REASONING_PROMPT_V1
SUBGRAPH_SYNTHESIS_PROMPT = SUBGRAPH_SYNTHESIS_PROMPT_V1
REASONING_QA_GENERATION_PROMPT = REASONING_QA_GENERATION_PROMPT_V1


# ============================================================================
# Helper Functions
# ============================================================================

def format_path_description(path: list) -> str:
    """
    Format a subgraph path into a human-readable chain description.

    Args:
        path: Alternating list of node dicts and edge dicts.
              Nodes have 'id', 'labels', 'properties'.
              Edges have 'source_id', 'target_id', 'type', 'properties'.

    Returns:
        A readable string like:
        [Person: Alice] --(KNOWS)--> [Person: Bob] --(WORKS_AT)--> [Company: Acme]
    """
    parts = []
    for i, element in enumerate(path):
        if i % 2 == 0:
            # Node
            labels = ", ".join(element.get("labels", []))
            name = (
                element.get("properties", {}).get("name")
                or element.get("properties", {}).get("title")
                or element.get("properties", {}).get("id")
                or element.get("id", "?")
            )
            props = element.get("properties", {})
            prop_summary = ", ".join(
                f"{k}: {v}" for k, v in list(props.items())[:5]
            )
            parts.append(f"[{labels}: {name}]({prop_summary})")
        else:
            # Edge
            rel_type = element.get("type", "RELATED_TO")
            rel_props = element.get("properties", {})
            if rel_props:
                prop_str = " {" + ", ".join(f"{k}: {v}" for k, v in list(rel_props.items())[:3]) + "}"
            else:
                prop_str = ""
            parts.append(f"--({rel_type}{prop_str})-->")
    return " ".join(parts)


def format_center_entity(node: dict) -> str:
    """Format a center node for inclusion in prompts."""
    labels = ", ".join(node.get("labels", []))
    props = node.get("properties", {})
    prop_lines = "\n".join(f"  - {k}: {v}" for k, v in props.items())
    name = props.get("name") or props.get("title") or node.get("id", "Unknown")
    return f"**{name}** (Labels: {labels})\nProperties:\n{prop_lines}"


def format_path_reasoning_prompt(
    center_node: dict,
    path: list,
    version: str = "V1",
) -> str:
    """
    Build a prompt asking the LLM to reason through a single path.

    Args:
        center_node: The center entity dict.
        path: Alternating node/edge list.
        version: Prompt version.

    Returns:
        Formatted prompt string.
    """
    template = PATH_REASONING_PROMPT_V1
    return template.format(
        center_entity=format_center_entity(center_node),
        path_description=format_path_description(path),
    )


def format_subgraph_synthesis_prompt(
    center_node: dict,
    path_analyses: list,
    num_nodes: int,
    num_edges: int,
    version: str = "V1",
) -> str:
    """
    Build a prompt to synthesize multiple path analyses.

    Args:
        center_node: Center entity dict.
        path_analyses: List of strings, each a path-level reasoning.
        num_nodes: Total unique nodes in the subgraph.
        num_edges: Total edges in the subgraph.
        version: Prompt version.

    Returns:
        Formatted prompt string.
    """
    analyses_text = ""
    for i, analysis in enumerate(path_analyses, 1):
        analyses_text += f"\n### Path {i}\n{analysis}\n"

    template = SUBGRAPH_SYNTHESIS_PROMPT_V1
    return template.format(
        center_entity=format_center_entity(center_node),
        path_analyses=analyses_text,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_paths=len(path_analyses),
    )


def format_reasoning_qa_prompt(
    center_node: dict,
    synthesis: str,
    version: str = "V1",
) -> str:
    """
    Build a prompt to generate a distillation-ready QA pair from synthesis.

    Args:
        center_node: Center entity dict.
        synthesis: The synthesized knowledge text.
        version: Prompt version.

    Returns:
        Formatted prompt string.
    """
    template = REASONING_QA_GENERATION_PROMPT_V1
    return template.format(
        center_entity=format_center_entity(center_node),
        synthesis=synthesis,
    )


def format_semantic_selection_prompt(
    candidate_info: list,
    version: str = "V1"
) -> str:
    """
    Format the semantic node selection prompt with candidate information.
    
    Args:
        candidate_info: List of dicts with 'id', 'labels', 'properties'
        version: Prompt version to use ('V1', 'V2', or 'current')
    
    Returns:
        Formatted prompt string
    """
    if version == "current" or version.upper() == "V1":
        prompt_template = SEMANTIC_NODE_SELECTION_PROMPT_V1
    elif version.upper() == "V2":
        prompt_template = SEMANTIC_NODE_SELECTION_PROMPT_V2
    else:
        prompt_template = SEMANTIC_NODE_SELECTION_PROMPT
    
    # Format candidates list
    candidates_list = []
    for i, info in enumerate(candidate_info, start=1):
        labels_str = ", ".join(info.get("labels", []))
        props_str = str(info.get("properties", {}))
        candidates_list.append(
            f"{i}. Node {info['id']}: {labels_str} - {props_str}"
        )
    
    candidates_text = "\n".join(candidates_list)
    max_candidates = len(candidate_info)
    
    return prompt_template.format(
        candidates_list=candidates_text,
        max_candidates=max_candidates
    )


def format_node_context(
    labels: list,
    properties: dict,
    neighbors_count: int = 0,
    version: str = "V1"
) -> str:
    """
    Format node context information for LLM prompts.
    
    Args:
        labels: List of node labels
        properties: Dictionary of node properties
        neighbors_count: Number of neighboring nodes
        version: Template version to use ('V1', 'V2', or 'current')
    
    Returns:
        Formatted context string
    """
    if version == "current" or version.upper() == "V1":
        template = NODE_CONTEXT_TEMPLATE_V1
        labels_str = ", ".join(labels) if labels else "None"
        props_str = str(properties)
        neighbors_info = f"\nRelated Nodes: {neighbors_count} neighbors" if neighbors_count > 0 else ""
        
        return template.format(
            labels=labels_str,
            properties=props_str,
            neighbors_info=neighbors_info
        )
    elif version.upper() == "V2":
        template = NODE_CONTEXT_TEMPLATE_V2
        labels_str = ", ".join(labels) if labels else "None"
        
        # Format properties as key-value pairs
        props_lines = []
        for key, value in properties.items():
            props_lines.append(f"  - {key}: {value}")
        props_formatted = "\n".join(props_lines) if props_lines else "  (no properties)"
        
        neighbors_info = f"\n**Related Nodes:** {neighbors_count} neighbors" if neighbors_count > 0 else ""
        
        return template.format(
            labels=labels_str,
            properties_formatted=props_formatted,
            neighbors_info=neighbors_info
        )
    else:
        # Default to V1
        return format_node_context(labels, properties, neighbors_count, "V1")
