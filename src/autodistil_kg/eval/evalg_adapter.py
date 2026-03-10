"""
EvalG adapter and interfaces.

This module intentionally keeps the integration with EvalG light-weight and
config-driven so the concrete EvalG library or CLI can be wired in without
changing the rest of the pipeline.

High-level design:
- EvaluatorStage will call `run_evalg` with:
  - path to an evaluation dataset (JSONL)
  - a list of "systems" (distilled model, base model, external models, graph-RAG)
  - output path for the EvalG report
- `run_evalg` is responsible for invoking EvalG (library or CLI) and writing a
  JSON report with per-system metrics.

Because EvalG can be used in different ways (Python API, CLI, Docker, etc.),
this module focuses on the interface and leaves the concrete invocation hooks
in a single place.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalSystemConfig:
    """
    Description of a system to be evaluated by EvalG.

    Examples:
    - kind = "distilled", model_path points to the fine-tuned model
    - kind = "base", provider/model identify a base LLM
    - kind = "external", provider/model point to OpenAI / Anthropic, etc.
    - kind = "graph_rag", rag_config points to a graph-RAG pipeline config
    """

    id: str
    label: Optional[str] = None
    kind: str = "distilled"  # distilled | base | external | graph_rag

    # Generic fields; EvalG integration can interpret these as needed
    model_path: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    rag_config: Optional[Dict[str, Any]] = None

    # Optional path to pre-computed predictions for this system.
    # When set, EvalG can consume predictions directly instead of generating them.
    predictions_path: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_evalg(
    eval_dataset_path: str,
    systems: List[EvalSystemConfig],
    output_report_path: str,
    evalg_mode: str = "cli",
    evalg_command: Optional[List[str]] = None,
    evalg_extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run EvalG given an evaluation dataset and a list of systems.

    Parameters
    ----------
    eval_dataset_path:
        Path to the evaluation dataset JSONL (questions, references, graph context, etc.).
    systems:
        Systems to compare (distilled/base/external/graph_rag). The concrete EvalG
        integration decides how to interpret these.
    output_report_path:
        Where the EvalG JSON report should be written.
    evalg_mode:
        How to invoke EvalG. Currently supports:
        - "cli": call an external CLI command provided via `evalg_command`.
        - "noop": do not call EvalG; instead, emit a minimal structured stub.
    evalg_command:
        When `evalg_mode="cli"`, the base command to execute, e.g.:
        ["python", "scripts/run_evalg.py"] or ["evalg", "run"].
    evalg_extra_args:
        Optional extra configuration forwarded to EvalG (e.g. metric set, judge settings).

    Returns
    -------
    Parsed JSON report (dict) that was written to `output_report_path`.

    Notes
    -----
    - This function is intentionally conservative: if the CLI invocation fails or
      returns invalid JSON, it logs an error and falls back to a stub report so
      the pipeline can still complete.
    - You are expected to adapt `evalg_mode == "cli"` to your specific EvalG setup.
    """

    eval_dataset_path = str(Path(eval_dataset_path).resolve())
    output_report_path = str(Path(output_report_path).resolve())
    Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)

    systems_payload = [s.to_dict() for s in systems]
    evalg_extra_args = evalg_extra_args or {}

    if evalg_mode == "internal":
        from .internal_evaluator import InternalEvaluator

        evaluator = InternalEvaluator(
            eval_dataset_path=eval_dataset_path,
            systems=systems,
            output_report_path=output_report_path,
            metrics=evalg_extra_args.get("metrics", ["rouge"]),
            judge_config=evalg_extra_args.get("judge_config"),
            max_samples=evalg_extra_args.get("max_samples"),
        )
        return evaluator.run()

    if evalg_mode == "noop":
        logger.warning("EvalG mode is 'noop' – emitting stub metrics only.")
        report = {
            "evalg_mode": "noop",
            "eval_dataset_path": eval_dataset_path,
            "systems": systems_payload,
            "metrics": {},  # user can post-process and fill in later
        }
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    if evalg_mode == "cli":
        if not evalg_command:
            logger.warning(
                "EvalG mode 'cli' requested but no evalg_command provided; "
                "falling back to stub report."
            )
            return run_evalg(
                eval_dataset_path=eval_dataset_path,
                systems=systems,
                output_report_path=output_report_path,
                evalg_mode="noop",
                evalg_extra_args=evalg_extra_args,
            )

        payload = {
            "eval_dataset_path": eval_dataset_path,
            "systems": systems_payload,
            "output_report_path": output_report_path,
            "extra": evalg_extra_args,
        }

        try:
            proc = subprocess.run(
                evalg_command,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception as e:
            logger.exception("Failed to invoke EvalG CLI: %s", e)
            return run_evalg(
                eval_dataset_path=eval_dataset_path,
                systems=systems,
                output_report_path=output_report_path,
                evalg_mode="noop",
                evalg_extra_args=evalg_extra_args,
            )

        if proc.returncode != 0:
            logger.error("EvalG CLI exited with code %s: %s", proc.returncode, proc.stderr)
            return run_evalg(
                eval_dataset_path=eval_dataset_path,
                systems=systems,
                output_report_path=output_report_path,
                evalg_mode="noop",
                evalg_extra_args=evalg_extra_args,
            )

        # EvalG is expected to write the report to output_report_path. Attempt to load it.
        try:
            with open(output_report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("EvalG CLI completed but report could not be read: %s", e)
            return run_evalg(
                eval_dataset_path=eval_dataset_path,
                systems=systems,
                output_report_path=output_report_path,
                evalg_mode="noop",
                evalg_extra_args=evalg_extra_args,
            )

    logger.warning("Unknown EvalG mode '%s'; using stub report.", evalg_mode)
    return run_evalg(
        eval_dataset_path=eval_dataset_path,
        systems=systems,
        output_report_path=output_report_path,
        evalg_mode="noop",
        evalg_extra_args=evalg_extra_args,
    )

