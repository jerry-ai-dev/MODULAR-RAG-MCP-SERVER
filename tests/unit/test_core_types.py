"""Unit tests for core data types (Document, Chunk, ChunkRecord).

Tests cover:
- Type instantiation
- Required field validation
- Serialization (to_dict/from_dict)
- Metadata conventions
- Helper methods
"""

import pytest
from src.core.types import Document, Chunk, ChunkRecord


class TestDocument:
    """Test Document data type."""
    
    def test_document_creation_valid(self):
        """Test creating a valid Document."""
        doc = Document(
            id="doc_123",
            text="# Title\n\nContent here",
            metadata={"source_path": "data/test.pdf"}
        )
        assert doc.id == "doc_123"
        assert doc.text == "# Title\n\nContent here"
        assert doc.metadata["source_path"] == "data/test.pdf"
    
    def test_document_requires_source_path(self):
        """Test that Document requires source_path in metadata."""
        with pytest.raises(ValueError, match="must contain 'source_path'"):
            Document(
                id="doc_123",
                text="Content",
                metadata={}
            )
    
    def test_document_optional_metadata_fields(self):
        """Test Document with extended metadata."""
        doc = Document(
            id="doc_123",
            text="Content",
            metadata={
                "source_path": "data/test.pdf",
                "doc_type": "pdf",
                "title": "Test Document",
                "page_count": 10,
                "images": ["img1.png", "img2.png"]
            }
        )
        assert doc.metadata["doc_type"] == "pdf"
        assert doc.metadata["title"] == "Test Document"
        assert doc.metadata["page_count"] == 10
        assert len(doc.metadata["images"]) == 2
    
    def test_document_serialization(self):
        """Test Document to_dict and from_dict."""
        original = Document(
            id="doc_123",
            text="Content",
            metadata={"source_path": "data/test.pdf", "title": "Test"}
        )
        
        # Serialize
        data = original.to_dict()
        assert data["id"] == "doc_123"
        assert data["text"] == "Content"
        assert data["metadata"]["source_path"] == "data/test.pdf"
        
        # Deserialize
        restored = Document.from_dict(data)
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata


class TestChunk:
    """Test Chunk data type."""
    
    def test_chunk_creation_valid(self):
        """Test creating a valid Chunk."""
        chunk = Chunk(
            id="chunk_123_001",
            text="## Section 1\n\nFirst paragraph",
            metadata={"source_path": "data/test.pdf", "chunk_index": 0}
        )
        assert chunk.id == "chunk_123_001"
        assert chunk.text == "## Section 1\n\nFirst paragraph"
        assert chunk.metadata["chunk_index"] == 0
    
    def test_chunk_requires_source_path(self):
        """Test that Chunk requires source_path in metadata."""
        with pytest.raises(ValueError, match="must contain 'source_path'"):
            Chunk(
                id="chunk_123",
                text="Content",
                metadata={"chunk_index": 0}
            )
    
    def test_chunk_with_offsets(self):
        """Test Chunk with start/end offsets."""
        chunk = Chunk(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf"},
            start_offset=0,
            end_offset=100
        )
        assert chunk.start_offset == 0
        assert chunk.end_offset == 100
    
    def test_chunk_with_source_ref(self):
        """Test Chunk with parent document reference."""
        chunk = Chunk(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf"},
            source_ref="doc_123"
        )
        assert chunk.source_ref == "doc_123"
    
    def test_chunk_serialization(self):
        """Test Chunk to_dict and from_dict."""
        original = Chunk(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=100,
            source_ref="doc_123"
        )
        
        # Serialize
        data = original.to_dict()
        assert data["id"] == "chunk_123_001"
        assert data["start_offset"] == 0
        assert data["end_offset"] == 100
        assert data["source_ref"] == "doc_123"
        
        # Deserialize
        restored = Chunk.from_dict(data)
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.start_offset == original.start_offset
        assert restored.end_offset == original.end_offset
        assert restored.source_ref == original.source_ref


class TestChunkRecord:
    """Test ChunkRecord data type."""
    
    def test_chunk_record_creation_valid(self):
        """Test creating a valid ChunkRecord."""
        record = ChunkRecord(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf", "chunk_index": 0},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"word1": 0.5, "word2": 0.3}
        )
        assert record.id == "chunk_123_001"
        assert len(record.dense_vector) == 3
        assert record.sparse_vector["word1"] == 0.5
    
    def test_chunk_record_requires_source_path(self):
        """Test that ChunkRecord requires source_path in metadata."""
        with pytest.raises(ValueError, match="must contain 'source_path'"):
            ChunkRecord(
                id="chunk_123",
                text="Content",
                metadata={}
            )
    
    def test_chunk_record_without_vectors(self):
        """Test ChunkRecord can be created without vectors (for intermediate stages)."""
        record = ChunkRecord(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf"}
        )
        assert record.dense_vector is None
        assert record.sparse_vector is None
    
    def test_chunk_record_serialization(self):
        """Test ChunkRecord to_dict and from_dict."""
        original = ChunkRecord(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf", "title": "Section 1"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"word": 0.5}
        )
        
        # Serialize
        data = original.to_dict()
        assert data["id"] == "chunk_123_001"
        assert data["dense_vector"] == [0.1, 0.2, 0.3]
        assert data["sparse_vector"] == {"word": 0.5}
        
        # Deserialize
        restored = ChunkRecord.from_dict(data)
        assert restored.id == original.id
        assert restored.dense_vector == original.dense_vector
        assert restored.sparse_vector == original.sparse_vector
    
    def test_chunk_record_from_chunk(self):
        """Test creating ChunkRecord from Chunk."""
        chunk = Chunk(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=100
        )
        
        dense_vec = [0.1, 0.2, 0.3]
        sparse_vec = {"word": 0.5}
        
        record = ChunkRecord.from_chunk(chunk, dense_vec, sparse_vec)
        
        assert record.id == chunk.id
        assert record.text == chunk.text
        assert record.metadata == chunk.metadata
        assert record.dense_vector == dense_vec
        assert record.sparse_vector == sparse_vec
    
    def test_chunk_record_metadata_isolation(self):
        """Test that metadata is copied not shared between Chunk and ChunkRecord."""
        chunk = Chunk(
            id="chunk_123",
            text="Content",
            metadata={"source_path": "data/test.pdf", "key": "original"}
        )
        
        record = ChunkRecord.from_chunk(chunk)
        record.metadata["key"] = "modified"
        
        # Original chunk metadata should be unchanged
        assert chunk.metadata["key"] == "original"
        assert record.metadata["key"] == "modified"


class TestMultimodalSupport:
    """Test multimodal image support according to C1 specification."""
    
    def test_document_with_image_placeholder(self):
        """Test Document with image placeholder in text."""
        doc = Document(
            id="doc_with_img",
            text="Here is some text.\n\n[IMAGE: abc123_1_0]\n\nMore text after image.",
            metadata={
                "source_path": "data/test.pdf",
                "images": [
                    {
                        "id": "abc123_1_0",
                        "path": "data/images/collection/abc123_1_0.png",
                        "page": 1,
                        "text_offset": 20,
                        "text_length": 21,
                        "position": {"x": 100, "y": 200, "width": 400, "height": 300}
                    }
                ]
            }
        )
        
        assert "[IMAGE: abc123_1_0]" in doc.text
        assert len(doc.metadata["images"]) == 1
        assert doc.metadata["images"][0]["id"] == "abc123_1_0"
        assert doc.metadata["images"][0]["text_offset"] == 20
        assert doc.metadata["images"][0]["text_length"] == 21
    
    def test_document_with_multiple_images(self):
        """Test Document with multiple image placeholders."""
        doc = Document(
            id="doc_multi_img",
            text="Text [IMAGE: img1] middle [IMAGE: img2] end",
            metadata={
                "source_path": "data/test.pdf",
                "images": [
                    {
                        "id": "img1",
                        "path": "data/images/collection/img1.png",
                        "page": 1,
                        "text_offset": 5,
                        "text_length": 14,
                        "position": {}
                    },
                    {
                        "id": "img2",
                        "path": "data/images/collection/img2.png",
                        "page": 2,
                        "text_offset": 27,
                        "text_length": 14,
                        "position": {}
                    }
                ]
            }
        )
        
        assert len(doc.metadata["images"]) == 2
        assert doc.text.count("[IMAGE:") == 2
    
    def test_chunk_with_image_reference(self):
        """Test Chunk containing image placeholder and relevant image metadata."""
        chunk = Chunk(
            id="chunk_with_img",
            text="Section content [IMAGE: abc123_1_0] continues here",
            metadata={
                "source_path": "data/test.pdf",
                "chunk_index": 0,
                "images": [
                    {
                        "id": "abc123_1_0",
                        "path": "data/images/collection/abc123_1_0.png",
                        "page": 1,
                        "text_offset": 16,
                        "text_length": 21,
                        "position": {}
                    }
                ]
            }
        )
        
        assert "[IMAGE: abc123_1_0]" in chunk.text
        assert "images" in chunk.metadata
        assert len(chunk.metadata["images"]) == 1
    
    def test_chunk_record_with_image_captions(self):
        """Test ChunkRecord with image captions from ImageCaptioner."""
        record = ChunkRecord(
            id="record_with_caption",
            text="Architecture diagram [IMAGE: diagram_001] shows the system",
            metadata={
                "source_path": "data/test.pdf",
                "chunk_index": 0,
                "images": [
                    {
                        "id": "diagram_001",
                        "path": "data/images/collection/diagram_001.png",
                        "page": 5,
                        "text_offset": 21,
                        "text_length": 21,
                        "position": {}
                    }
                ],
                "image_captions": {
                    "diagram_001": "System architecture showing three-tier design with load balancer"
                }
            },
            dense_vector=[0.1, 0.2, 0.3]
        )
        
        assert "image_captions" in record.metadata
        assert record.metadata["image_captions"]["diagram_001"]
        assert "architecture" in record.metadata["image_captions"]["diagram_001"].lower()
    
    def test_image_metadata_structure_validation(self):
        """Test that image metadata follows the C1 specification structure."""
        image_ref = {
            "id": "doc_hash_page_seq",
            "path": "data/images/collection/doc_hash_page_seq.png",
            "page": 1,
            "text_offset": 100,
            "text_length": 25,
            "position": {"x": 0, "y": 0, "width": 500, "height": 400}
        }
        
        # Verify all required fields are present
        assert "id" in image_ref
        assert "path" in image_ref
        assert "text_offset" in image_ref
        assert "text_length" in image_ref
        
        # Verify field types
        assert isinstance(image_ref["id"], str)
        assert isinstance(image_ref["path"], str)
        assert isinstance(image_ref["text_offset"], int)
        assert isinstance(image_ref["text_length"], int)
        assert isinstance(image_ref["position"], dict)
    
    def test_document_without_images(self):
        """Test Document without images (images field can be omitted or empty list)."""
        # Omit images field
        doc1 = Document(
            id="doc_no_img_1",
            text="Plain text document",
            metadata={"source_path": "data/test.txt"}
        )
        assert "images" not in doc1.metadata or doc1.metadata.get("images", []) == []
        
        # Explicit empty list
        doc2 = Document(
            id="doc_no_img_2",
            text="Plain text document",
            metadata={"source_path": "data/test.txt", "images": []}
        )
        assert doc2.metadata["images"] == []


class TestMetadataConventions:
    """Test metadata field conventions across types."""
    
    def test_source_path_required_everywhere(self):
        """Test that source_path is required in all types."""
        # Document
        with pytest.raises(ValueError):
            Document(id="d1", text="t", metadata={})
        
        # Chunk
        with pytest.raises(ValueError):
            Chunk(id="c1", text="t", metadata={})
        
        # ChunkRecord
        with pytest.raises(ValueError):
            ChunkRecord(id="r1", text="t", metadata={})
    
    def test_metadata_extensibility(self):
        """Test that metadata can be extended without breaking compatibility."""
        # Add arbitrary fields
        doc = Document(
            id="doc_123",
            text="Content",
            metadata={
                "source_path": "data/test.pdf",
                "custom_field_1": "value1",
                "custom_field_2": 123,
                "custom_field_3": ["list", "values"]
            }
        )
        
        # Should serialize and deserialize without issues
        data = doc.to_dict()
        restored = Document.from_dict(data)
        
        assert restored.metadata["custom_field_1"] == "value1"
        assert restored.metadata["custom_field_2"] == 123
        assert restored.metadata["custom_field_3"] == ["list", "values"]
    
    def test_metadata_propagation_pattern(self):
        """Test typical metadata propagation from Document -> Chunk -> ChunkRecord."""
        # Document level
        doc_metadata = {
            "source_path": "data/report.pdf",
            "doc_type": "pdf",
            "title": "Annual Report",
            "author": "John Doe"
        }
        
        doc = Document(id="doc_123", text="Full document text", metadata=doc_metadata.copy())
        
        # Chunk inherits and extends
        chunk_metadata = doc.metadata.copy()
        chunk_metadata.update({
            "chunk_index": 0,
            "page": 1
        })
        
        chunk = Chunk(
            id="chunk_123_001",
            text="First section",
            metadata=chunk_metadata,
            source_ref="doc_123"
        )
        
        # ChunkRecord inherits from chunk and adds enrichment
        record_metadata = chunk.metadata.copy()
        record_metadata.update({
            "summary": "Introduction section",
            "tags": ["intro", "overview"]
        })
        
        record = ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=record_metadata,
            dense_vector=[0.1, 0.2, 0.3]
        )
        
        # Verify propagation
        assert record.metadata["source_path"] == doc.metadata["source_path"]
        assert record.metadata["title"] == doc.metadata["title"]
        assert record.metadata["chunk_index"] == 0
        assert record.metadata["summary"] == "Introduction section"
