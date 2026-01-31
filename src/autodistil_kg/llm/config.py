"""
LLM Configuration Module.

This module defines configuration classes for LLM providers.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    OLLAMA = "ollama"
    VLLM = "vllm"


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: str  # "openai", "gemini", "claude", "ollama", "vllm"
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    # Gemini-specific
    project_id: Optional[str] = None
    location: Optional[str] = None
    credentials_path: Optional[str] = None
    # Additional parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.provider:
            raise ValueError("LLM provider must be specified")
        
        provider_lower = self.provider.lower()
        
        # Validate provider-specific requirements
        if provider_lower == LLMProvider.OPENAI.value:
            if not self.api_key:
                raise ValueError("OpenAI requires api_key")
        elif provider_lower == LLMProvider.GEMINI.value:
            if not self.project_id:
                raise ValueError("Gemini requires project_id")
        elif provider_lower == LLMProvider.CLAUDE.value:
            if not self.api_key:
                raise ValueError("Claude requires api_key")
