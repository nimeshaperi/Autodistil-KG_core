"""
Pipeline and per-stage configuration.
Each stage has its own config dataclass so it can be configured separately and linked.
Graph traverser types are imported lazily so pipeline can be used without neo4j when not running that stage.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph_traverser.config import GraphTraverserAgentConfig, TraversalConfig, DatasetGenerationConfig
    from ..graph_traverser.graph_db.config import GraphDatabaseConfig
    from ..graph_traverser.state_storage.config import StateStorageConfig


@dataclass
class GraphTraverserStageConfig:
    """Configuration for the Graph Traverser stage (context extraction + CHATML generation)."""
    # Full agent config; can be built from sub-configs or passed directly
    agent_config: Optional[Any] = None  # GraphTraverserAgentConfig when set
    # If agent_config is None, these can be set (and factory will build agent_config)
    graph_db: Optional[Any] = None  # GraphDatabaseConfig
    llm_config: Optional[Any] = None  # LLMConfig from graph_traverser.llm
    state_storage: Optional[Any] = None  # StateStorageConfig
    traversal: Optional[Any] = None  # TraversalConfig
    dataset: Optional[Any] = None  # DatasetGenerationConfig
    # Output path for this stage (JSONL). If None, only in-memory context is updated.
    output_path: Optional[str] = None

    def get_agent_config(self):
        from ..graph_traverser.config import GraphTraverserAgentConfig
        if self.agent_config is not None:
            return self.agent_config
        if all([self.graph_db, self.llm_config, self.state_storage, self.traversal, self.dataset]):
            return GraphTraverserAgentConfig(
                graph_db=self.graph_db,
                llm=self.llm_config,
                state_storage=self.state_storage,
                traversal=self.traversal,
                dataset=self.dataset,
            )
        raise ValueError(
            "GraphTraverserStageConfig: provide either agent_config or "
            "graph_db, llm_config, state_storage, traversal, dataset"
        )


@dataclass
class ChatMLConverterStageConfig:
    """Configuration for the ChatML Converter stage (normalize / prepare format)."""
    # Input: path to JSONL or use context.chatml_dataset_path / context.chatml_dataset
    input_path: Optional[str] = None
    # Output path for prepared JSONL (conversations with 'messages' or 'text' for finetuning)
    output_path: Optional[str] = None
    # If True, output format includes tokenized 'text' field for direct use by finetuner
    prepare_for_finetuning: bool = True
    # Chat template name for formatting (e.g. gemma3, chatml, llama3)
    chat_template: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FineTunerStageConfig:
    """Configuration for the FineTuner stage. Delegates to finetuner module config."""
    # Pass the finetuner module's config (UnslothFineTunerConfig)
    finetuner_config: Optional[Any] = None
    # Or set key options here and stage will build finetuner config
    model_name: Optional[str] = None
    model_type: Optional[str] = None  # gemma, llama, qwen, etc.
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    output_dir: Optional[str] = None
    max_seq_length: int = 2048
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-4
    # If None, pipeline context prepared_dataset_path / chatml_dataset_path will be used
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorStageConfig:
    """Configuration for the Evaluator stage (stub for future implementation)."""
    model_path: Optional[str] = None
    eval_dataset_path: Optional[str] = None
    output_report_path: Optional[str] = None
    metrics: Optional[list] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """
    Full pipeline configuration with optional config per stage.
    Stages can be enabled/disabled and configured independently.
    """
    graph_traverser: Optional[GraphTraverserStageConfig] = None
    chatml_converter: Optional[ChatMLConverterStageConfig] = None
    finetuner: Optional[FineTunerStageConfig] = None
    evaluator: Optional[EvaluatorStageConfig] = None
    # Global options
    output_dir: Optional[str] = None  # Base directory for stage outputs if not set per-stage
    run_stages: Optional[list] = None  # e.g. ["graph_traverser", "chatml_converter", "finetuner"]
