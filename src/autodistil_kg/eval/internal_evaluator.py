"""
Internal evaluator: orchestrates prediction generation and metric scoring
for comparing base model, finetuned model, and Graph RAG systems.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evalg_adapter import EvalSystemConfig
from .predictors import Predictor, build_predictor
from .scorers import Scorer, build_scorers

logger = logging.getLogger(__name__)


def _extract_eval_sample(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract system_prompt, question, and reference from a CHATML record.

    Expected format:
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    The system message is optional.
    """
    messages = record.get("messages")
    if not messages or not isinstance(messages, list):
        return None

    system_prompt = ""
    question = ""
    reference = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            question = content
        elif role == "assistant":
            reference = content

    if not question:
        return None

    return {
        "system_prompt": system_prompt,
        "question": question,
        "reference": reference,
    }


class InternalEvaluator:
    """Runs in-process evaluation across multiple systems.

    1. Loads the eval dataset (JSONL)
    2. Builds a predictor per system
    3. Generates predictions for every (system, sample) pair
    4. Scores predictions with configured metrics
    5. Writes a structured JSON comparison report
    """

    def __init__(
        self,
        eval_dataset_path: str,
        systems: List[EvalSystemConfig],
        output_report_path: str,
        metrics: Optional[List[str]] = None,
        judge_config: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.eval_dataset_path = eval_dataset_path
        self.systems = systems
        self.output_report_path = output_report_path
        self.metrics = metrics or ["rouge"]
        self.judge_config = judge_config
        self.max_samples = max_samples

    def _load_dataset(self) -> List[Dict[str, str]]:
        """Load and parse the eval JSONL file."""
        samples: List[Dict[str, str]] = []
        path = Path(self.eval_dataset_path)
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at line %d", line_no)
                    continue

                sample = _extract_eval_sample(record)
                if sample is None:
                    logger.warning("Skipping record at line %d: missing question", line_no)
                    continue
                samples.append(sample)

                if self.max_samples and len(samples) >= self.max_samples:
                    break

        logger.info("Loaded %d evaluation samples from %s", len(samples), path)
        return samples

    def _build_predictors(self) -> Dict[str, Predictor]:
        """Build a predictor for each system, skipping those that fail to initialise."""
        predictors: Dict[str, Predictor] = {}
        for sys_cfg in self.systems:
            try:
                predictors[sys_cfg.id] = build_predictor(sys_cfg)
                logger.info("Built predictor for system %r (kind=%s)", sys_cfg.id, sys_cfg.kind)
            except Exception as e:
                logger.error(
                    "Failed to build predictor for system %r: %s — skipping", sys_cfg.id, e
                )
        return predictors

    def _generate_predictions(
        self,
        samples: List[Dict[str, str]],
        predictors: Dict[str, Predictor],
    ) -> List[Dict[str, str]]:
        """Generate predictions for every (system, sample) pair.

        Returns a list (one per sample) of dicts mapping system_id -> prediction.
        """
        all_predictions: List[Dict[str, str]] = []

        for idx, sample in enumerate(samples):
            preds: Dict[str, str] = {}
            for sys_id, predictor in predictors.items():
                try:
                    pred = predictor.predict(sample["system_prompt"], sample["question"])
                    preds[sys_id] = pred
                except Exception as e:
                    logger.error(
                        "Prediction failed for system %r on sample %d: %s", sys_id, idx, e
                    )
                    preds[sys_id] = f"[ERROR: {e}]"
            all_predictions.append(preds)

            if (idx + 1) % 10 == 0:
                logger.info("Generated predictions for %d / %d samples", idx + 1, len(samples))

        return all_predictions

    def _score_predictions(
        self,
        samples: List[Dict[str, str]],
        all_predictions: List[Dict[str, str]],
        scorers: List[Scorer],
    ) -> List[Dict[str, Dict[str, float]]]:
        """Score every (system, sample) prediction.

        Returns a list (one per sample) of dicts mapping system_id -> merged scores.
        """
        all_scores: List[Dict[str, Dict[str, float]]] = []

        for idx, (sample, preds) in enumerate(zip(samples, all_predictions)):
            sample_scores: Dict[str, Dict[str, float]] = {}
            for sys_id, prediction in preds.items():
                merged: Dict[str, float] = {}
                for scorer in scorers:
                    try:
                        result = scorer.score(
                            prediction=prediction,
                            reference=sample["reference"],
                            question=sample["question"],
                        )
                        merged.update(result)
                    except Exception as e:
                        logger.error(
                            "Scorer %r failed for system %r on sample %d: %s",
                            scorer.name, sys_id, idx, e,
                        )
                sample_scores[sys_id] = merged
            all_scores.append(sample_scores)

            if (idx + 1) % 10 == 0:
                logger.info("Scored %d / %d samples", idx + 1, len(samples))

        return all_scores

    @staticmethod
    def _aggregate_metrics(
        all_scores: List[Dict[str, Dict[str, float]]],
        system_ids: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-system average metrics."""
        sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for sample_scores in all_scores:
            for sys_id in system_ids:
                scores = sample_scores.get(sys_id, {})
                for metric_name, value in scores.items():
                    sums[sys_id][metric_name] += value
                    counts[sys_id][metric_name] += 1

        aggregated: Dict[str, Dict[str, float]] = {}
        for sys_id in system_ids:
            aggregated[sys_id] = {}
            for metric_name in sums[sys_id]:
                n = counts[sys_id][metric_name]
                aggregated[sys_id][metric_name] = round(sums[sys_id][metric_name] / n, 4) if n else 0.0

        return aggregated

    def _build_report(
        self,
        samples: List[Dict[str, str]],
        all_predictions: List[Dict[str, str]],
        all_scores: List[Dict[str, Dict[str, float]]],
        aggregate: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Assemble the final evaluation report."""
        systems_section: Dict[str, Any] = {}
        for sys_cfg in self.systems:
            if sys_cfg.id in aggregate:
                systems_section[sys_cfg.id] = {
                    "label": sys_cfg.label or sys_cfg.id,
                    "kind": sys_cfg.kind,
                    "aggregate_metrics": aggregate[sys_cfg.id],
                }

        per_question: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            per_question.append({
                "index": idx,
                "question": sample["question"],
                "reference": sample["reference"],
                "predictions": all_predictions[idx],
                "scores": all_scores[idx],
            })

        return {
            "evalg_mode": "internal",
            "eval_dataset_path": str(Path(self.eval_dataset_path).resolve()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(samples),
            "metrics_used": self.metrics,
            "systems": systems_section,
            "per_question": per_question,
        }

    def run(self) -> Dict[str, Any]:
        """Execute the full evaluation pipeline."""
        logger.info("Starting internal evaluation")

        # 1. Load dataset
        samples = self._load_dataset()
        if not samples:
            logger.warning("No evaluation samples found")
            report = {
                "evalg_mode": "internal",
                "eval_dataset_path": str(Path(self.eval_dataset_path).resolve()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "num_samples": 0,
                "metrics_used": self.metrics,
                "systems": {},
                "per_question": [],
            }
            Path(self.output_report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            return report

        # 2. Build predictors
        predictors = self._build_predictors()
        if not predictors:
            raise RuntimeError("No predictors could be initialised; aborting evaluation")

        # 3. Build scorers
        scorers = build_scorers(self.metrics, judge_config=self.judge_config)
        if not scorers:
            logger.warning("No scorers configured; predictions will be generated but not scored")

        # 4. Generate predictions
        logger.info("Generating predictions for %d systems across %d samples", len(predictors), len(samples))
        all_predictions = self._generate_predictions(samples, predictors)

        # 5. Score predictions
        logger.info("Scoring predictions with %d scorer(s)", len(scorers))
        all_scores = self._score_predictions(samples, all_predictions, scorers)

        # 6. Aggregate
        system_ids = [s.id for s in self.systems if s.id in predictors]
        aggregate = self._aggregate_metrics(all_scores, system_ids)

        # 7. Build and write report
        report = self._build_report(samples, all_predictions, all_scores, aggregate)

        Path(self.output_report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Evaluation report written to %s", self.output_report_path)

        # Cleanup
        for predictor in predictors.values():
            predictor.close()

        return report
