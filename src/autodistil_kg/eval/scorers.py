"""
Scorers: compute metrics comparing predictions against reference answers.

Each scorer returns a dict of named float scores so multiple sub-metrics
(e.g. rouge-1, rouge-2, rouge-L) can be reported from a single scorer.
"""

from __future__ import annotations

import json
import logging
import re
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


class RougeScorer(Scorer):
    """ROUGE-1 / ROUGE-2 / ROUGE-L F1 scores via the ``rouge-score`` library."""

    name = "rouge"

    def __init__(self) -> None:
        from rouge_score import rouge_scorer as _rs

        self._scorer = _rs.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, float]:
        scores = self._scorer.score(reference, prediction)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }


class LLMJudgeScorer(Scorer):
    """Uses an LLM to rate a prediction on accuracy, completeness, and relevance."""

    name = "llm_judge"

    _PROMPT_TEMPLATE = """\
You are an expert evaluator. Given a question, a reference (gold) answer, and a \
candidate prediction, rate the prediction on the following dimensions using a \
1-5 integer scale (1 = very poor, 5 = excellent):

- **accuracy**: Does the prediction contain correct information consistent with the reference?
- **completeness**: Does the prediction cover all key points in the reference?
- **relevance**: Is the prediction focused on answering the question?

Return ONLY a JSON object with these three keys and integer values. Example:
{{"accuracy": 4, "completeness": 3, "relevance": 5}}

Question:
{question}

Reference answer:
{reference}

Candidate prediction:
{prediction}

JSON rating:"""

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        from ..llm.config import LLMConfig
        from ..llm.factory import create_llm_client

        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self._client = create_llm_client(config)

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, float]:
        from ..llm.interface import LLMMessage

        prompt = self._PROMPT_TEMPLATE.format(
            question=question,
            reference=reference,
            prediction=prediction,
        )
        messages = [LLMMessage(role="user", content=prompt)]

        for attempt in range(3):
            try:
                raw = self._client.generate(messages, temperature=0.0, max_tokens=128)
                ratings = self._parse_ratings(raw)
                avg = sum(ratings.values()) / len(ratings) if ratings else 0.0
                return {
                    "judge_accuracy": ratings.get("accuracy", 0.0),
                    "judge_completeness": ratings.get("completeness", 0.0),
                    "judge_relevance": ratings.get("relevance", 0.0),
                    "judge_avg": round(avg, 2),
                }
            except Exception:
                if attempt == 2:
                    logger.warning("LLM judge failed after 3 attempts; returning zeros")
                    return {
                        "judge_accuracy": 0.0,
                        "judge_completeness": 0.0,
                        "judge_relevance": 0.0,
                        "judge_avg": 0.0,
                    }

        # Unreachable but keeps the type checker happy.
        return {"judge_accuracy": 0.0, "judge_completeness": 0.0, "judge_relevance": 0.0, "judge_avg": 0.0}

    @staticmethod
    def _parse_ratings(raw: str) -> Dict[str, float]:
        """Extract the JSON ratings object from raw LLM output."""
        # Try to find a JSON block in the response.
        match = re.search(r"\{[^}]+\}", raw)
        if not match:
            raise ValueError(f"No JSON object found in LLM judge response: {raw!r}")
        data = json.loads(match.group())
        return {
            "accuracy": float(data["accuracy"]),
            "completeness": float(data["completeness"]),
            "relevance": float(data["relevance"]),
        }


def build_scorers(
    metrics: List[str],
    judge_config: Optional[Dict[str, Any]] = None,
) -> List[Scorer]:
    """Instantiate scorers from a list of metric names."""
    scorers: List[Scorer] = []
    for name in metrics:
        if name == "rouge":
            scorers.append(RougeScorer())
        elif name == "llm_judge":
            if not judge_config:
                logger.warning("llm_judge metric requested but no judge_config provided; skipping")
                continue
            scorers.append(
                LLMJudgeScorer(
                    provider=judge_config["provider"],
                    model=judge_config.get("model"),
                    api_key=judge_config.get("api_key"),
                    base_url=judge_config.get("base_url"),
                )
            )
        else:
            logger.warning("Unknown metric %r; skipping", name)
    return scorers
