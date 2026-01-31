"""Re-export LLM interface from top-level autodistil_kg.llm."""
from autodistil_kg.llm.interface import LLMClient, LLMMessage

__all__ = ["LLMClient", "LLMMessage"]
