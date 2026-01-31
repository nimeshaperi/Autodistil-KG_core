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
# Helper Functions
# ============================================================================

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
