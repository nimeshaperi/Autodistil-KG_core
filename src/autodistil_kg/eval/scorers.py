"""
Scorers: compute metrics comparing predictions against reference answers.

Uses DeepEval for robust, production-grade evaluation metrics including
answer relevancy, faithfulness, correctness (G-Eval), and hallucination.

Each scorer returns a dict of named float scores so multiple sub-metrics
can be reported from a single scorer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Scorer(ABC):
    """Abstract base for a metric scorer."""

    name: str = "base"

    @abstractmethod
    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, float]:
        """Score a single prediction against a reference answer."""


class AnswerRelevancyScorer(Scorer):
    """Measures how relevant the generated answer is to the question using DeepEval."""

    name = "answer_relevancy"

    def __init__(self, model: Optional[str] = None) -> None:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        self._metric_cls = AnswerRelevancyMetric
        self._test_case_cls = LLMTestCase
        self._model = model

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, float]:
        try:
            metric = self._metric_cls(
                threshold=0.5,
                **({"model": self._model} if self._model else {}),
            )
            test_case = self._test_case_cls(
                input=question,
                actual_output=prediction,
                expected_output=reference,
            )
            metric.measure(test_case)
            return {"answer_relevancy": round(metric.score, 4)}
        except Exception as e:
            logger.warning("AnswerRelevancyScorer failed: %s", e)
            return {"answer_relevancy": 0.0}


class CorrectnessScorer(Scorer):
    """G-Eval based correctness scoring using DeepEval.

    Evaluates accuracy, completeness, and relevance on a 1-5 scale.
    This replaces the custom LLM judge with DeepEval's G-Eval framework.
    """

    name = "correctness"

    def __init__(self, model: Optional[str] = None) -> None:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        self._geval_cls = GEval
        self._test_case_cls = LLMTestCase
        self._params = LLMTestCaseParams
        self._model = model

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        test_case = self._test_case_cls(
            input=question,
            actual_output=prediction,
            expected_output=reference,
        )

        criteria = {
            "accuracy": "Determine whether the prediction contains correct information consistent with the reference answer. Score 0 if completely wrong, 1 if perfectly accurate.",
            "completeness": "Determine whether the prediction covers all key points present in the reference answer. Score 0 if nothing is covered, 1 if fully complete.",
            "relevance": "Determine whether the prediction is focused on answering the question asked. Score 0 if completely off-topic, 1 if perfectly relevant.",
        }

        for dim, criterion in criteria.items():
            try:
                metric = self._geval_cls(
                    name=dim,
                    criteria=criterion,
                    evaluation_params=[
                        self._params.INPUT,
                        self._params.ACTUAL_OUTPUT,
                        self._params.EXPECTED_OUTPUT,
                    ],
                    threshold=0.5,
                    **({"model": self._model} if self._model else {}),
                )
                metric.measure(test_case)
                results[f"judge_{dim}"] = round(metric.score, 4)
            except Exception as e:
                logger.warning("GEval %s failed: %s", dim, e)
                results[f"judge_{dim}"] = 0.0

        if results:
            vals = [v for v in results.values() if v > 0]
            results["judge_avg"] = round(sum(vals) / len(vals), 4) if vals else 0.0

        return results


class FaithfulnessScorer(Scorer):
    """Measures factual consistency of the prediction against the reference."""

    name = "faithfulness"

    def __init__(self, model: Optional[str] = None) -> None:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        self._metric_cls = FaithfulnessMetric
        self._test_case_cls = LLMTestCase
        self._model = model

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, float]:
        try:
            metric = self._metric_cls(
                threshold=0.5,
                **({"model": self._model} if self._model else {}),
            )
            test_case = self._test_case_cls(
                input=question,
                actual_output=prediction,
                retrieval_context=[reference],
            )
            metric.measure(test_case)
            return {"faithfulness": round(metric.score, 4)}
        except Exception as e:
            logger.warning("FaithfulnessScorer failed: %s", e)
            return {"faithfulness": 0.0}


class HallucinationScorer(Scorer):
    """Detects hallucinated content in predictions using DeepEval."""

    name = "hallucination"

    def __init__(self, model: Optional[str] = None) -> None:
        from deepeval.metrics import HallucinationMetric
        from deepeval.test_case import LLMTestCase

        self._metric_cls = HallucinationMetric
        self._test_case_cls = LLMTestCase
        self._model = model

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, float]:
        try:
            metric = self._metric_cls(
                threshold=0.5,
                **({"model": self._model} if self._model else {}),
            )
            test_case = self._test_case_cls(
                input=question,
                actual_output=prediction,
                context=[reference],
            )
            metric.measure(test_case)
            # HallucinationMetric returns score where higher = more hallucination
            # Invert so higher = better (less hallucination)
            return {"hallucination": round(1.0 - metric.score, 4)}
        except Exception as e:
            logger.warning("HallucinationScorer failed: %s", e)
            return {"hallucination": 0.0}


def build_scorers(
    metrics: List[str],
    judge_config: Optional[Dict[str, Any]] = None,
) -> List[Scorer]:
    """Instantiate scorers from a list of metric names.

    Available metrics:
      - "answer_relevancy": DeepEval AnswerRelevancyMetric
      - "correctness": DeepEval GEval for accuracy/completeness/relevance (replaces llm_judge)
      - "faithfulness": DeepEval FaithfulnessMetric
      - "hallucination": DeepEval HallucinationMetric

    Backward-compatible aliases:
      - "llm_judge" -> "correctness"
      - "rouge" -> "answer_relevancy" (closest semantic equivalent)
    """
    # Resolve the judge model from config (e.g. "gpt-4o", "claude-3-5-sonnet")
    judge_model = None
    if judge_config:
        # DeepEval model string format: provider/model or just model
        provider = judge_config.get("provider", "")
        model = judge_config.get("model", "")
        if provider and model:
            judge_model = model  # DeepEval handles provider via env vars
        elif model:
            judge_model = model

    # Metric name aliases for backward compatibility
    ALIASES: Dict[str, str] = {
        "rouge": "answer_relevancy",
        "llm_judge": "correctness",
    }

    scorers: List[Scorer] = []
    seen: set[str] = set()
    for name in metrics:
        resolved = ALIASES.get(name, name)
        if resolved in seen:
            continue
        seen.add(resolved)

        if resolved == "answer_relevancy":
            scorers.append(AnswerRelevancyScorer(model=judge_model))
        elif resolved == "correctness":
            scorers.append(CorrectnessScorer(model=judge_model))
        elif resolved == "faithfulness":
            scorers.append(FaithfulnessScorer(model=judge_model))
        elif resolved == "hallucination":
            scorers.append(HallucinationScorer(model=judge_model))
        else:
            logger.warning("Unknown metric %r; skipping", name)
    return scorers
