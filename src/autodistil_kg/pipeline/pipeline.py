"""
Pipeline runner: run all stages or individual stages with linked I/O.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .interfaces import PipelineContext, StageResult
from .config import PipelineConfig
from .stages.graph_traverser_stage import GraphTraverserStage
from .stages.chatml_converter_stage import ChatMLConverterStage
from .stages.finetuner_stage import FineTunerStage
from .stages.evaluator_stage import EvaluatorStage

logger = logging.getLogger(__name__)

STAGE_NAMES = ("graph_traverser", "chatml_converter", "finetuner", "evaluator")


class Pipeline:
    """
    Orchestrates stages: Graph Traverser -> ChatML Converter -> FineTuner -> Evaluator.
    Each stage can be run separately (standalone) or as part of the full pipeline.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._stages: Dict[str, object] = {}
        self._build_stages()

    def _build_stages(self) -> None:
        cfg = self.config
        if cfg.graph_traverser is not None:
            self._stages["graph_traverser"] = GraphTraverserStage(cfg.graph_traverser)
        if cfg.chatml_converter is not None:
            self._stages["chatml_converter"] = ChatMLConverterStage(cfg.chatml_converter)
        if cfg.finetuner is not None:
            self._stages["finetuner"] = FineTunerStage(cfg.finetuner)
        if cfg.evaluator is not None:
            self._stages["evaluator"] = EvaluatorStage(cfg.evaluator)

    def _resolve_output_dir(self, stage_name: str) -> Optional[str]:
        base = self.config.output_dir
        if not base:
            return None
        return str(Path(base) / stage_name)

    def run(
        self,
        stages: Optional[Sequence[str]] = None,
        context: Optional[PipelineContext] = None,
    ) -> Tuple[PipelineContext, List[StageResult]]:
        """
        Run the pipeline or a subset of stages.
        Stages receive and update the same context (paths and in-memory data).
        """
        context = context or PipelineContext()
        run_order = stages or self.config.run_stages or list(self._stages.keys())
        # Preserve logical order
        ordered = [s for s in STAGE_NAMES if s in run_order and s in self._stages]
        results: List[StageResult] = []
        for name in ordered:
            stage = self._stages[name]
            logger.info("Running stage: %s", name)
            result = stage.run(context)
            results.append(result)
            if not result.success:
                logger.error("Stage %s failed: %s", name, result.error)
                break
        return context, results

    def run_stage(
        self,
        stage_name: str,
        context: Optional[PipelineContext] = None,
    ) -> StageResult:
        """Run a single stage by name. Use for standalone execution (e.g. only graph traverser)."""
        if stage_name not in self._stages:
            raise ValueError(f"Unknown or unconfigured stage: {stage_name}. Available: {list(self._stages.keys())}")
        context = context or PipelineContext()
        return self._stages[stage_name].run(context)

    @property
    def available_stages(self) -> List[str]:
        """Return list of configured stage names."""
        return list(self._stages.keys())
