"""CHATML dataset generation and management."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChatMLMessage:
    """Represents a message in CHATML format."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatMLConversation:
    """Represents a complete conversation in CHATML format."""
    messages: List[ChatMLMessage]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in self.messages
            ]
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ChatMLDataset:
    """Manages a collection of CHATML conversations."""
    
    def __init__(self):
        """Initialize an empty dataset."""
        self.conversations: List[ChatMLConversation] = []
    
    def add_conversation(self, conversation: ChatMLConversation) -> None:
        """Add a conversation to the dataset."""
        self.conversations.append(conversation)
    
    def add_from_messages(
        self,
        messages: List[ChatMLMessage],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a conversation from a list of messages."""
        conversation = ChatMLConversation(messages=messages, metadata=metadata)
        self.add_conversation(conversation)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert dataset to list of dictionaries."""
        return [conv.to_dict() for conv in self.conversations]
    
    def to_jsonl(self) -> str:
        """Convert dataset to JSONL format (one conversation per line)."""
        lines = []
        for conv in self.conversations:
            lines.append(json.dumps(conv.to_dict()))
        return "\n".join(lines)
    
    def save_jsonl(self, filepath: str) -> None:
        """Save dataset to a JSONL file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_jsonl())
        logger.info(f"Saved {len(self.conversations)} conversations to {filepath}")
    
    def save_json(self, filepath: str) -> None:
        """Save dataset to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, indent=2)
        logger.info(f"Saved {len(self.conversations)} conversations to {filepath}")
    
    def load_jsonl(self, filepath: str) -> None:
        """Load dataset from a JSONL file."""
        self.conversations = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    messages = [
                        ChatMLMessage(role=msg["role"], content=msg["content"])
                        for msg in data["messages"]
                    ]
                    metadata = data.get("metadata")
                    self.add_conversation(
                        ChatMLConversation(messages=messages, metadata=metadata)
                    )
        logger.info(f"Loaded {len(self.conversations)} conversations from {filepath}")
    
    def __len__(self) -> int:
        """Get the number of conversations in the dataset."""
        return len(self.conversations)
    
    def __iter__(self):
        """Iterate over conversations."""
        return iter(self.conversations)
