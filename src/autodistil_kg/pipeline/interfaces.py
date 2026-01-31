"""
Pipeline interfaces: stage contract and context passed between stages.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StageResult:
    """Result of running a single stage."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """
    Context passed between pipeline stages.
    Holds paths and optional in-memory objects so stages can be run in any order
    when used standalone (e.g. load from path) or chained (use previous output).
    """
    # Paths (set by stages or user for standalone runs)
    chatml_dataset_path: Optional[str] = None  # JSONL from graph traverser or converter
    prepared_dataset_path: Optional[str] = None  # Tokenized/prepared for finetuning
    model_output_path: Optional[str] = None  # Saved model/adapters path
    eval_report_path: Optional[str] = None  # Evaluator output path

    # In-memory (optional; used when chaining without writing to disk)
    chatml_dataset: Any = None  # ChatMLDataset instance
    prepared_dataset: Any = None  # HuggingFace Dataset or None

    # Arbitrary extras for stage-specific data
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging or persistence."""
        return {
            "chatml_dataset_path": self.chatml_dataset_path,
            "prepared_dataset_path": self.prepared_dataset_path,
            "model_output_path": self.model_output_path,
            "eval_report_path": self.eval_report_path,
            "extra": self.extra,
        }


class Stage(ABC):
    """Abstract base for a pipeline stage. Each stage is configurable and runnable independently."""

    name: str = "base"

    @abstractmethod
    def run(self, context: PipelineContext) -> StageResult:
        """
        Execute the stage. Read inputs from context, write outputs back to context.
        Returns StageResult with success flag and optional output.
        """
        pass

    def validate_config(self) -> None:
        """Override to validate stage-specific config. Raise ValueError if invalid."""
        pass
