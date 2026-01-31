"""
autodistil_kg: Graph Traversing, ChatML Dataset Creation, FineTuning, Evaluation.

Pipeline: Graph Traverser -> ChatML Converter -> FineTuner -> Evaluator.
Each stage is modular and can be run separately or as part of the pipeline.

Subpackages: chatml, graph_traverser, pipeline, finetuner.
Import explicitly to avoid pulling optional deps (e.g. graph_traverser requires neo4j).
"""
from . import chatml
from . import pipeline
from . import finetuner
# graph_traverser optional: requires neo4j; use "from autodistil_kg.graph_traverser import ..."

__all__ = [
    "chatml",
    "pipeline",
    "finetuner",
]
