"""Core data types and contracts for the entire pipeline.

This module defines the fundamental data structures used across all pipeline stages:
- ingestion (loaders, transforms, embedding, storage)
- retrieval (query engine, search, reranking)
- mcp_server (tools, response formatting)

Design Principles:
- Centralized contracts: All stages use these types to avoid coupling
- Serializable: All types support dict/JSON conversion
- Extensible metadata: Minimum required fields with flexible extension
- Type-safe: Full type hints for static analysis
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class Document:
    """Represents a raw document loaded from source.
    
    This is the output of Loaders (e.g., PDF Loader) before splitting.
    
    Attributes:
        id: Unique identifier for the document (e.g., file hash or path-based ID)
        text: Document content in standardized Markdown format.
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Document-level metadata including:
            - source_path (required): Original file path
            - doc_type: Document type (e.g., 'pdf', 'markdown')
            - title: Document title extracted or inferred
            - page_count: Total pages (if applicable)
            - images: List of image references (see Images Field Specification below)
            - Any other custom metadata
    
    Images Field Specification (metadata.images):
        Structure: List[{"id": str, "path": str, "page": int, "text_offset": int, 
                        "text_length": int, "position": dict}]
        Fields:
            - id: Unique image identifier (format: {doc_hash}_{page}_{seq})
            - path: Image file storage path (convention: data/images/{collection}/{image_id}.png)
            - page: Page number in original document (optional, for paginated docs like PDF)
            - text_offset: Starting character position of placeholder in Document.text (0-based)
            - text_length: Length of placeholder string (typically len("[IMAGE: {image_id}]"))
            - position: Physical position info in original doc (optional, e.g., PDF coords, pixel position)
        Note: text_offset and text_length enable precise placeholder location, 
              supporting scenarios where the same image appears multiple times
    
    Example:
        >>> doc = Document(
        ...     id="doc_abc123",
        ...     text="# Title\\n\\nContent...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "doc_type": "pdf",
        ...         "title": "Annual Report 2025"
        ...     }
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required metadata fields."""
        if "source_path" not in self.metadata:
            raise ValueError("Document metadata must contain 'source_path'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        return cls(**data)


@dataclass
class Chunk:
    """Represents a text chunk after splitting a Document.
    
    This is the output of Splitters and input to Transform pipeline.
    Each chunk maintains traceability to its source document.
    
    Attributes:
        id: Unique chunk identifier (e.g., hash-based or sequential)
        text: Chunk content (subset of original document text).
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Chunk-level metadata inherited and extended from Document:
            - source_path (required): Original file path
            - chunk_index: Sequential position in document (0-based)
            - start_offset: Character offset in original document (optional)
            - end_offset: Character offset in original document (optional)
            - source_ref: Reference to parent document ID (optional)
            - images: Subset of Document.images that fall within this chunk (optional)
            - Any document-level metadata propagated from Document
        start_offset: Starting character position in original document (optional)
        end_offset: Ending character position in original document (optional)
        source_ref: Reference to parent Document.id (optional)
    
    Note: If chunk contains image placeholders, metadata.images should contain
          only the image references relevant to this chunk's text range.
    
    Example:
        >>> chunk = Chunk(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "page": 1
        ...     },
        ...     start_offset=0,
        ...     end_offset=150
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    source_ref: Optional[str] = None
    
    def __post_init__(self):
        """Validate required metadata fields."""
        if "source_path" not in self.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary."""
        return cls(**data)


@dataclass
class ChunkRecord:
    """Represents a fully processed chunk ready for storage and retrieval.
    
    This is the output of the embedding pipeline and the data structure
    stored in vector databases. It extends Chunk with vector representations.
    
    Attributes:
        id: Unique chunk identifier (must be stable for idempotent upsert)
        text: Chunk content (same as Chunk.text).
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Extended metadata including:
            - source_path (required): Original file path
            - chunk_index: Sequential position
            - All metadata from Chunk
            - images: Image references from Chunk (see Document.images specification)
            - Any enrichment from Transform pipeline (title, summary, tags)
            - image_captions: Dict[image_id, caption_text] if multimodal enrichment applied
        dense_vector: Dense embedding vector (e.g., from OpenAI, BGE)
        sparse_vector: Sparse vector for BM25/keyword matching (optional)
    
    Note: Image captions generated by ImageCaptioner are stored in metadata.image_captions
          as a dictionary mapping image_id to generated caption text.
    
    Example:
        >>> record = ChunkRecord(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "title": "Introduction",
        ...         "summary": "Overview of project goals"
        ...     },
        ...     dense_vector=[0.1, 0.2, ..., 0.3],
        ...     sparse_vector={"word1": 0.5, "word2": 0.3}
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate required metadata fields."""
        if "source_path" not in self.metadata:
            raise ValueError("ChunkRecord metadata must contain 'source_path'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        """Create ChunkRecord from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_chunk(cls, chunk: Chunk, dense_vector: Optional[List[float]] = None,
                   sparse_vector: Optional[Dict[str, float]] = None) -> "ChunkRecord":
        """Create ChunkRecord from a Chunk with vectors.
        
        Args:
            chunk: Source Chunk object
            dense_vector: Dense embedding vector
            sparse_vector: Sparse vector representation
            
        Returns:
            ChunkRecord with all fields populated from chunk
        """
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )


# Type aliases for convenience
Metadata = Dict[str, Any]
Vector = List[float]
SparseVector = Dict[str, float]
