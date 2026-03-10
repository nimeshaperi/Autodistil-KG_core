"""
Predictors: generate answers from different systems for evaluation.

Each predictor wraps a specific inference backend (LLM API, finetuned model,
or Graph RAG) behind a common interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Predictor(ABC):
    """Abstract base for a system that generates answers."""

    @abstractmethod
    def predict(self, system_prompt: str, user_question: str) -> str:
        """Return a generated answer given a system prompt and user question."""

    def close(self) -> None:
        """Release resources (optional override)."""


class BaseLLMPredictor(Predictor):
    """Generates predictions via the LLM factory (OpenAI, Ollama, vLLM, etc.)."""

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        from ..llm.config import LLMConfig
        from ..llm.factory import create_llm_client

        extra = extra or {}
        llm_config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            project_id=extra.get("project_id"),
            location=extra.get("location"),
            credentials_path=extra.get("credentials_path"),
            additional_params=extra,
        )
        self._client = create_llm_client(llm_config)

    def predict(self, system_prompt: str, user_question: str) -> str:
        from ..llm.interface import LLMMessage

        messages: List[LLMMessage] = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=user_question))

        return self._client.generate(messages, temperature=0.1)


class FinetunedModelPredictor(Predictor):
    """Loads a LoRA-adapted model via Unsloth and runs local inference."""

    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 2048,
        max_new_tokens: int = 512,
    ) -> None:
        from unsloth import FastLanguageModel

        logger.info("Loading finetuned model from %s", model_path)
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
        )
        FastLanguageModel.for_inference(self._model)
        self._max_new_tokens = max_new_tokens

    def predict(self, system_prompt: str, user_question: str) -> str:
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_question})

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens.
        new_tokens = outputs[0][input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


class GraphRAGPredictor(Predictor):
    """Wraps the GraphRAGEngine from the graphrag submodule."""

    def __init__(self, rag_config: Dict[str, Any]) -> None:
        from autodistil_kg_graphrag.config import (
            GraphRAGConfig,
            Neo4jConfig,
            LLMConfig as GraphRAGLLMConfig,
            EmbeddingConfig,
            RetrieverConfig,
            RetrieverType,
        )
        from autodistil_kg_graphrag.query_engine import GraphRAGEngine

        neo4j_cfg = Neo4jConfig(
            uri=rag_config.get("neo4j_uri", "bolt://localhost:7687"),
            user=rag_config.get("neo4j_user", "neo4j"),
            password=rag_config.get("neo4j_password", "password"),
            database=rag_config.get("neo4j_database", "neo4j"),
        )
        llm_cfg = GraphRAGLLMConfig(
            api_key=rag_config.get("llm_api_key", ""),
            model=rag_config.get("llm_model", "gpt-4"),
        )
        embed_cfg = EmbeddingConfig(
            api_key=rag_config.get("embedding_api_key", rag_config.get("llm_api_key", "")),
            model=rag_config.get("embedding_model", "text-embedding-3-small"),
            dimensions=int(rag_config.get("embedding_dimensions", 1536)),
        )

        enabled_retrievers = rag_config.get("retrievers", ["vector", "cypher", "synonym"])
        retriever_cfg = RetrieverConfig(
            enabled=[RetrieverType(r) for r in enabled_retrievers],
        )

        config = GraphRAGConfig(
            neo4j=neo4j_cfg,
            llm=llm_cfg,
            embedding=embed_cfg,
            retriever=retriever_cfg,
        )

        self._engine = GraphRAGEngine(config)
        logger.info("Initialising GraphRAGEngine for evaluation")
        self._engine.initialise()

    def predict(self, system_prompt: str, user_question: str) -> str:
        response = self._engine.query(user_question)
        return response.answer


def build_predictor(system_config) -> Predictor:
    """Build the appropriate Predictor from an EvalSystemConfig."""
    kind = system_config.kind

    if kind == "distilled":
        if not system_config.model_path:
            raise ValueError("Distilled system requires model_path")
        extra = system_config.extra or {}
        return FinetunedModelPredictor(
            model_path=system_config.model_path,
            max_seq_length=int(extra.get("max_seq_length", 2048)),
            max_new_tokens=int(extra.get("max_new_tokens", 512)),
        )

    if kind == "base":
        if not system_config.provider:
            raise ValueError("Base system requires provider")
        extra = system_config.extra or {}
        return BaseLLMPredictor(
            provider=system_config.provider,
            model=system_config.model,
            api_key=extra.get("api_key"),
            base_url=extra.get("base_url"),
            extra=extra,
        )

    if kind == "external":
        if not system_config.provider:
            raise ValueError("External system requires provider")
        extra = system_config.extra or {}
        return BaseLLMPredictor(
            provider=system_config.provider,
            model=system_config.model,
            api_key=extra.get("api_key"),
            base_url=extra.get("base_url"),
            extra=extra,
        )

    if kind == "graph_rag":
        if not system_config.rag_config:
            raise ValueError("Graph RAG system requires rag_config")
        return GraphRAGPredictor(rag_config=system_config.rag_config)

    raise ValueError(f"Unknown system kind: {kind}")
