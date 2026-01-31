"""Trace context for observability across pipeline stages.

This is a minimal implementation for Phase C. Will be enhanced in Phase F.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class TraceContext:
    """Minimal trace context for recording pipeline stages.
    
    Attributes:
        trace_id: Unique identifier for this trace
        started_at: Timestamp when trace was created
        stages: Dictionary storing data from each pipeline stage
    """
    
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.now)
    stages: Dict[str, Any] = field(default_factory=dict)
    
    def record_stage(self, stage_name: str, data: Dict[str, Any]) -> None:
        """Record data from a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage (e.g., "chunk_refiner", "embedding")
            data: Stage-specific data to record
        """
        self.stages[stage_name] = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
    
    def get_stage_data(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve recorded data for a specific stage.
        
        Args:
            stage_name: Name of the stage to retrieve
            
        Returns:
            Stage data dict or None if stage not found
        """
        return self.stages.get(stage_name)
