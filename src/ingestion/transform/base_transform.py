"""Base class for chunk transform operations."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """Abstract base class for chunk transformation operations.
    
    Transform operations process chunks to enhance their quality, add metadata,
    or prepare them for downstream processing (embedding, indexing).
    
    Design Principles:
        - Single Responsibility: Each transform does ONE type of enhancement
        - Atomic Operations: Failure in one chunk doesn't affect others
        - Observable: Records processing info in TraceContext
        - Graceful Degradation: Returns original chunk on unrecoverable errors
    """
    
    @abstractmethod
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Transform a list of chunks.
        
        Args:
            chunks: List of chunks to transform
            trace: Optional trace context for observability
            
        Returns:
            List of transformed chunks (same length as input)
            
        Raises:
            ValueError: If input validation fails
        """
        pass
