"""
Pipeline Module.

Orchestrates configurable stages: Graph Traverser -> ChatML Converter -> FineTuner -> Evaluator.
Each stage can be run independently or as part of the full pipeline.
"""
from .interfaces import Stage, StageResult, PipelineContext
from .config import (
    PipelineConfig,
    GraphTraverserStageConfig,
    ChatMLConverterStageConfig,
    FineTunerStageConfig,
    EvaluatorStageConfig,
)
from .pipeline import Pipeline
from .stages.graph_traverser_stage import GraphTraverserStage
from .stages.chatml_converter_stage import ChatMLConverterStage
from .stages.finetuner_stage import FineTunerStage
from .stages.evaluator_stage import EvaluatorStage

__all__ = [
    # Interfaces
    "Stage",
    "StageResult",
    "PipelineContext",
    # Config
    "PipelineConfig",
    "GraphTraverserStageConfig",
    "ChatMLConverterStageConfig",
    "FineTunerStageConfig",
    "EvaluatorStageConfig",
    # Runner
    "Pipeline",
    # Stages
    "GraphTraverserStage",
    "ChatMLConverterStage",
    "FineTunerStage",
    "EvaluatorStage",
]
