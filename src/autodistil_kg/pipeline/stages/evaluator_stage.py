"""
Evaluator stage: integrates EvalG-based evaluation over one or more systems.

This stage is intended to:
- Take the fine-tuned model (and optionally other systems: base models, external
  APIs, graph-RAG pipelines) and an evaluation dataset.
- Delegate metric computation to EvalG via the evalg_adapter.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..interfaces import Stage, StageResult, PipelineContext
from ..config import EvaluatorStageConfig
from ...eval.evalg_adapter import EvalSystemConfig, run_evalg

logger = logging.getLogger(__name__)


class EvaluatorStage(Stage):
    """Stage that evaluates one or more systems using EvalG."""

    name = "evaluator"

    def __init__(self, config: EvaluatorStageConfig):
        self.config = config

    def _build_systems(self, model_path: str) -> List[EvalSystemConfig]:
        """
        Build the list of systems to pass into EvalG.

        Always includes the distilled (finetuned) model.
        Auto-creates base model and Graph RAG systems from first-class config
        fields when present. Additional systems can still be supplied via
        ``config.additional_params["systems"]``.
        """
        systems: List[EvalSystemConfig] = []

        # 1. Always include the distilled (finetuned) model.
        systems.append(
            EvalSystemConfig(
                id="distilled",
                label="Finetuned model",
                kind="distilled",
                model_path=model_path,
            )
        )

        # 2. Auto-create base model system from config fields.
        if self.config.base_model_provider:
            systems.append(
                EvalSystemConfig(
                    id="base",
                    label=f"Base model ({self.config.base_model_name or 'default'} via {self.config.base_model_provider})",
                    kind="base",
                    provider=self.config.base_model_provider,
                    model=self.config.base_model_name,
                    extra={
                        "api_key": self.config.base_model_api_key,
                        "base_url": self.config.base_model_base_url,
                    },
                )
            )

        # 3. Auto-create Graph RAG system from config fields.
        if self.config.graph_rag_config:
            systems.append(
                EvalSystemConfig(
                    id="graph_rag",
                    label="Graph RAG",
                    kind="graph_rag",
                    rag_config=self.config.graph_rag_config,
                )
            )

        # 4. Additional systems from additional_params (backward-compatible).
        extra = self.config.additional_params or {}
        raw_systems = extra.get("systems", [])
        if isinstance(raw_systems, list):
            for idx, s in enumerate(raw_systems):
                if not isinstance(s, dict):
                    logger.warning("Ignoring non-dict system config at index %s", idx)
                    continue
                try:
                    systems.append(
                        EvalSystemConfig(
                            id=str(s.get("id") or f"system_{idx}"),
                            label=s.get("label"),
                            kind=s.get("kind", "external"),
                            model_path=s.get("model_path"),
                            provider=s.get("provider"),
                            model=s.get("model"),
                            rag_config=s.get("rag_config"),
                            predictions_path=s.get("predictions_path"),
                            extra=s.get("extra") or {},
                        )
                    )
                except Exception as e:
                    logger.warning("Failed to parse system config at index %s: %s", idx, e)
        elif raw_systems:
            logger.warning("EvaluatorStage.additional_params.systems is not a list; ignoring.")

        return systems

    def run(self, context: PipelineContext) -> StageResult:
        model_path = self.config.model_path or context.model_output_path
        eval_path = (
            self.config.eval_dataset_path
            or context.prepared_dataset_path
            or context.chatml_dataset_path
        )
        if not model_path or not eval_path:
            return StageResult(
                success=False,
                error="Evaluator requires model_path and eval_dataset_path (or set in context).",
            )
        if not Path(model_path).exists():
            return StageResult(success=False, error=f"Model path does not exist: {model_path}")
        if not Path(eval_path).exists():
            return StageResult(success=False, error=f"Eval dataset path does not exist: {eval_path}")

        output_report = self.config.output_report_path or context.eval_report_path
        if not output_report:
            # Default under the model directory.
            base_dir = Path(model_path)
            if base_dir.is_file():
                base_dir = base_dir.parent
            output_report = str(base_dir / "eval_report.json")

        systems = self._build_systems(model_path)
        extra: Dict[str, Any] = self.config.additional_params or {}

        evalg_mode = self.config.evalg_mode or extra.get("evalg_mode", "internal")

        evalg_command = extra.get("evalg_command")
        if isinstance(evalg_command, list):
            evalg_cmd_list = [str(x) for x in evalg_command]
        elif isinstance(evalg_command, str):
            evalg_cmd_list = [evalg_command]
        else:
            evalg_cmd_list = None

        # Build evalg_extra_args from first-class config fields.
        evalg_extra_args: Dict[str, Any] = extra.get("evalg_extra_args") or {}
        if self.config.metrics:
            evalg_extra_args.setdefault("metrics", self.config.metrics)
        if self.config.max_eval_samples:
            evalg_extra_args.setdefault("max_samples", self.config.max_eval_samples)
        if self.config.judge_provider:
            evalg_extra_args.setdefault("judge_config", {
                "provider": self.config.judge_provider,
                "model": self.config.judge_model,
                "api_key": self.config.judge_api_key,
            })

        logger.info(
            "Running evaluator with EvalG (mode=%s) on dataset=%s, model=%s, systems=%s",
            evalg_mode,
            eval_path,
            model_path,
            [s.id for s in systems],
        )

        try:
            report = run_evalg(
                eval_dataset_path=str(eval_path),
                systems=systems,
                output_report_path=str(output_report),
                evalg_mode=str(evalg_mode),
                evalg_command=evalg_cmd_list,
                evalg_extra_args=evalg_extra_args,
            )
        except Exception as e:
            logger.exception("EvalG evaluation failed")
            return StageResult(success=False, error=str(e))

        context.eval_report_path = str(output_report)

        metadata: Dict[str, Any] = {
            "model_path": str(model_path),
            "eval_dataset_path": str(eval_path),
            "eval_report_path": str(output_report),
        }
        if isinstance(report, dict):
            metadata["metrics"] = report.get("metrics", {})
            metadata["raw_report"] = report

        return StageResult(success=True, output=report, metadata=metadata)

