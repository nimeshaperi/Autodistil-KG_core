"""Re-export LLM factory from top-level autodistil_kg.llm."""
from autodistil_kg.llm.factory import create_llm_client

__all__ = ["create_llm_client"]
