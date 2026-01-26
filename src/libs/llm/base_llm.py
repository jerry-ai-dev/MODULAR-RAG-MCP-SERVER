"""Abstract base class for LLM providers.

This module defines the pluggable interface for Language Model providers,
enabling seamless switching between different backends (OpenAI, Azure, Ollama, etc.)
through configuration-driven instantiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """Represents a single message in a chat conversation.
    
    Attributes:
        role: The role of the message sender ('system', 'user', or 'assistant').
        content: The text content of the message.
    """
    role: str
    content: str


@dataclass
class ChatResponse:
    """Response from an LLM chat completion.
    
    Attributes:
        content: The generated text response.
        model: The model identifier that generated the response.
        usage: Optional token usage statistics (prompt_tokens, completion_tokens, total_tokens).
        raw_response: Optional raw response from the provider for debugging.
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class BaseLLM(ABC):
    """Abstract base class for LLM providers.
    
    All LLM implementations must inherit from this class and implement
    the chat() method. This ensures consistent interface across different
    providers (OpenAI, Azure, DeepSeek, Ollama, etc.).
    
    Design Principles Applied:
    - Pluggable: Subclasses can be swapped without changing upstream code.
    - Observable: Accepts optional TraceContext for observability integration.
    - Config-Driven: Instances are created via factory based on settings.
    """
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion response.
        
        Args:
            messages: List of conversation messages (role + content).
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.).
        
        Returns:
            ChatResponse containing the generated text and metadata.
        
        Raises:
            ValueError: If messages list is empty or malformed.
            RuntimeError: If the LLM provider call fails.
        """
        pass
    
    def validate_messages(self, messages: List[Message]) -> None:
        """Validate message list structure.
        
        Args:
            messages: List of messages to validate.
        
        Raises:
            ValueError: If messages list is empty or contains invalid roles.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        valid_roles = {"system", "user", "assistant"}
        for i, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise ValueError(f"Message at index {i} is not a Message instance")
            if msg.role not in valid_roles:
                raise ValueError(
                    f"Message at index {i} has invalid role '{msg.role}'. "
                    f"Must be one of: {valid_roles}"
                )
            if not msg.content or not msg.content.strip():
                raise ValueError(f"Message at index {i} has empty content")
