"""
ChatML Converter stage: normalize / prepare CHATML for finetuning.
Loads from context or path; outputs prepared JSONL or in-memory dataset.
Runnable standalone (e.g. load from file) or after graph traverser.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..interfaces import Stage, StageResult, PipelineContext
from ..config import ChatMLConverterStageConfig
from ...chatml import ChatMLDataset, ChatMLConversation, ChatMLMessage

logger = logging.getLogger(__name__)


class ChatMLConverterStage(Stage):
    """Stage that converts/normalizes CHATML and optionally prepares for finetuning."""

    name = "chatml_converter"

    def __init__(self, config: ChatMLConverterStageConfig):
        self.config = config

    def _load_dataset(self, context: PipelineContext) -> Optional[ChatMLDataset]:
        if context.chatml_dataset is not None:
            return context.chatml_dataset
        path = self.config.input_path or context.chatml_dataset_path
        if not path or not Path(path).exists():
            return None
        dataset = ChatMLDataset()
        dataset.load_jsonl(path)
        return dataset

    def _conversation_to_messages_dict(self, conv: ChatMLConversation) -> Dict[str, Any]:
        return {
            "messages": [{"role": m.role, "content": m.content} for m in conv.messages],
            **({"metadata": conv.metadata} if conv.metadata else {}),
        }

    def run(self, context: PipelineContext) -> StageResult:
        try:
            dataset = self._load_dataset(context)
            if dataset is None or len(dataset) == 0:
                return StageResult(
                    success=False,
                    error="No CHATML dataset in context and no valid input_path.",
                )

            if not self.config.prepare_for_finetuning:
                output_path = self.config.output_path or context.prepared_dataset_path
                if output_path:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    dataset.save_jsonl(output_path)
                    context.prepared_dataset_path = output_path
                context.chatml_dataset = dataset
                return StageResult(
                    success=True,
                    output=dataset,
                    metadata={"conversations_count": len(dataset)},
                )

            # Prepare for finetuning: output list of dicts with "messages" key (and optionally "text" if tokenizer applied elsewhere)
            prepared: List[Dict[str, Any]] = []
            for conv in dataset.conversations:
                prepared.append(self._conversation_to_messages_dict(conv))

            output_path = self.config.output_path or context.prepared_dataset_path
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    for item in prepared:
                        f.write(json.dumps(item) + "\n")
                context.prepared_dataset_path = str(path)
                logger.info("Saved prepared dataset to %s", path)

            context.extra["prepared_messages"] = prepared
            return StageResult(
                success=True,
                output=prepared,
                metadata={"conversations_count": len(prepared)},
            )
        except Exception as e:
            logger.exception("ChatML converter stage failed")
            return StageResult(success=False, error=str(e))
