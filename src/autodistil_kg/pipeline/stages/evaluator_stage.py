"""
Evaluator stage: stub for future evaluation (metrics on finetuned model).
Runnable standalone or after finetuner.
"""
import logging
from pathlib import Path

from ..interfaces import Stage, StageResult, PipelineContext
from ..config import EvaluatorStageConfig

logger = logging.getLogger(__name__)


class EvaluatorStage(Stage):
    """Stage that evaluates a finetuned model. Stub implementation."""

    name = "evaluator"

    def __init__(self, config: EvaluatorStageConfig):
        self.config = config

    def run(self, context: PipelineContext) -> StageResult:
        model_path = self.config.model_path or context.model_output_path
        eval_path = self.config.eval_dataset_path or context.prepared_dataset_path or context.chatml_dataset_path
        if not model_path or not eval_path:
            return StageResult(
                success=False,
                error="Evaluator requires model_path and eval_dataset_path (or set in context).",
            )
        if not Path(model_path).exists():
            return StageResult(success=False, error=f"Model path does not exist: {model_path}")
        if not Path(eval_path).exists():
            return StageResult(success=False, error=f"Eval dataset path does not exist: {eval_path}")

        # Stub: no real evaluation yet
        report = {"model_path": model_path, "eval_dataset_path": eval_path, "metrics": {}}
        output_report = self.config.output_report_path or context.eval_report_path
        if output_report:
            Path(output_report).parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(output_report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            context.eval_report_path = output_report

        return StageResult(success=True, output=report, metadata=report)
