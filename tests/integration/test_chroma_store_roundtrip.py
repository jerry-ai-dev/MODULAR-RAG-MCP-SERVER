"""Integration tests for ChromaStore roundtrip operations.

This test suite validates the complete ChromaStore implementation through
real upsert→query roundtrip cycles, ensuring data persistence and retrieval
correctness.
"""

import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from src.core.settings import Settings
from src.libs.vector_store.chroma_store import ChromaStore


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_settings(temp_chroma_dir):
    """Create test settings with temporary ChromaDB directory."""
    # Create minimal settings object
    class VectorStoreConfig:
        provider = "chroma"
        collection_name = "test_collection"
        persist_directory = temp_chroma_dir
    
    class TestSettings:
        vector_store = VectorStoreConfig()
    
    return TestSettings()


@pytest.fixture
def chroma_store(test_settings):
    """Create a ChromaStore instance for testing."""
    store = ChromaStore(settings=test_settings)
    yield store
    # Cleanup: clear collection after each test
    try:
        store.clear()
    except Exception:
        pass


class TestChromaStoreBasicOperations:
    """Test basic CRUD operations on ChromaStore."""
    
    def test_upsert_single_record(self, chroma_store):
        """Test upserting a single record."""
        records = [
            {
                'id': 'doc1_chunk0',
                'vector': [0.1, 0.2, 0.3, 0.4, 0.5],
                'metadata': {'source': 'doc1.pdf', 'page': 1}
            }
        ]
        
        # Should not raise any exception
        chroma_store.upsert(records)
        
        # Verify record count
        stats = chroma_store.get_collection_stats()
        assert stats['count'] == 1
    
    def test_upsert_multiple_records(self, chroma_store):
        """Test upserting multiple records in batch."""
        records = [
            {
                'id': f'doc1_chunk{i}',
                'vector': [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i],
                'metadata': {'source': 'doc1.pdf', 'page': i}
            }
            for i in range(10)
        ]
        
        chroma_store.upsert(records)
        
        stats = chroma_store.get_collection_stats()
        assert stats['count'] == 10
    
    def test_upsert_idempotent(self, chroma_store):
        """Test that upserting same ID twice overwrites the first record."""
        # First upsert
        records_v1 = [
            {
                'id': 'doc1_chunk0',
                'vector': [0.1, 0.2, 0.3],
                'metadata': {'source': 'doc1.pdf', 'version': 1}
            }
        ]
        chroma_store.upsert(records_v1)
        
        # Second upsert with same ID but different data
        records_v2 = [
            {
                'id': 'doc1_chunk0',
                'vector': [0.5, 0.6, 0.7],
                'metadata': {'source': 'doc1.pdf', 'version': 2}
            }
        ]
        chroma_store.upsert(records_v2)
        
        # Should still have only 1 record (not 2)
        stats = chroma_store.get_collection_stats()
        assert stats['count'] == 1
        
        # Query should return the updated version
        results = chroma_store.query([0.5, 0.6, 0.7], top_k=1)
        assert len(results) == 1
        assert results[0]['metadata']['version'] == 2


class TestChromaStoreQueryOperations:
    """Test query and retrieval operations."""
    
    def test_query_empty_collection(self, chroma_store):
        """Test querying an empty collection returns empty list."""
        results = chroma_store.query([0.1, 0.2, 0.3], top_k=5)
        assert results == []
    
    def test_query_returns_top_k(self, chroma_store):
        """Test that query respects top_k parameter."""
        # Insert 10 records
        records = [
            {
                'id': f'doc1_chunk{i}',
                'vector': [0.1 * i, 0.2 * i, 0.3 * i],
                'metadata': {'source': 'doc1.pdf', 'chunk': i}
            }
            for i in range(10)
        ]
        chroma_store.upsert(records)
        
        # Query with top_k=5
        results = chroma_store.query([0.3, 0.6, 0.9], top_k=5)
        
        assert len(results) == 5
    
    def test_query_similarity_score(self, chroma_store):
        """Test that query returns results with similarity scores."""
        # Insert records
        records = [
            {
                'id': 'doc1_chunk0',
                'vector': [1.0, 0.0, 0.0],
                'metadata': {'source': 'doc1.pdf'}
            },
            {
                'id': 'doc1_chunk1',
                'vector': [0.0, 1.0, 0.0],
                'metadata': {'source': 'doc1.pdf'}
            },
        ]
        chroma_store.upsert(records)
        
        # Query with exact match to first vector
        results = chroma_store.query([1.0, 0.0, 0.0], top_k=2)
        
        # First result should have highest score (close to 1.0)
        assert len(results) == 2
        assert 'score' in results[0]
        assert results[0]['score'] > results[1]['score']
        assert results[0]['id'] == 'doc1_chunk0'
    
    def test_query_with_metadata_filters(self, chroma_store):
        """Test querying with metadata filters."""
        # Insert records from different sources
        records = [
            {
                'id': 'doc1_chunk0',
                'vector': [0.1, 0.2, 0.3],
                'metadata': {'source': 'doc1.pdf', 'page': 1}
            },
            {
                'id': 'doc2_chunk0',
                'vector': [0.1, 0.2, 0.3],
                'metadata': {'source': 'doc2.pdf', 'page': 1}
            },
            {
                'id': 'doc1_chunk1',
                'vector': [0.2, 0.3, 0.4],
                'metadata': {'source': 'doc1.pdf', 'page': 2}
            },
        ]
        chroma_store.upsert(records)
        
        # Query with filter for doc1.pdf only
        results = chroma_store.query(
            [0.1, 0.2, 0.3],
            top_k=10,
            filters={'source': 'doc1.pdf'}
        )
        
        # Should return only doc1 chunks
        assert len(results) == 2
        for result in results:
            assert result['metadata']['source'] == 'doc1.pdf'


class TestChromaStoreRoundtrip:
    """Test complete roundtrip: upsert → query → validate."""
    
    def test_roundtrip_preserves_metadata(self, chroma_store):
        """Test that metadata is preserved through upsert→query cycle."""
        records = [
            {
                'id': 'test_doc_chunk0',
                'vector': [0.5, 0.5, 0.5],
                'metadata': {
                    'source': 'test.pdf',
                    'page': 42,
                    'title': 'Test Document',
                    'tags': 'tag1,tag2,tag3',
                }
            }
        ]
        
        chroma_store.upsert(records)
        results = chroma_store.query([0.5, 0.5, 0.5], top_k=1)
        
        assert len(results) == 1
        assert results[0]['id'] == 'test_doc_chunk0'
        assert results[0]['metadata']['source'] == 'test.pdf'
        assert results[0]['metadata']['page'] == 42
        assert results[0]['metadata']['title'] == 'Test Document'
    
    def test_roundtrip_deterministic(self, chroma_store):
        """Test that same query returns same results deterministically."""
        # Insert records
        records = [
            {
                'id': f'chunk{i}',
                'vector': [float(i), float(i * 2), float(i * 3)],
                'metadata': {'index': i}
            }
            for i in range(5)
        ]
        chroma_store.upsert(records)
        
        # Query multiple times
        query_vector = [2.0, 4.0, 6.0]
        results1 = chroma_store.query(query_vector, top_k=3)
        results2 = chroma_store.query(query_vector, top_k=3)
        results3 = chroma_store.query(query_vector, top_k=3)
        
        # All results should be identical
        assert len(results1) == len(results2) == len(results3) == 3
        
        for i in range(3):
            assert results1[i]['id'] == results2[i]['id'] == results3[i]['id']
            assert abs(results1[i]['score'] - results2[i]['score']) < 1e-6
    
    def test_roundtrip_large_batch(self, chroma_store):
        """Test roundtrip with large batch (100+ records)."""
        # Insert 100 records
        records = [
            {
                'id': f'large_batch_chunk{i}',
                'vector': [float(i % 10), float((i * 2) % 10), float((i * 3) % 10)],
                'metadata': {'batch': 'large', 'index': i}
            }
            for i in range(100)
        ]
        chroma_store.upsert(records)
        
        # Verify count
        stats = chroma_store.get_collection_stats()
        assert stats['count'] == 100
        
        # Query should work normally
        results = chroma_store.query([5.0, 5.0, 5.0], top_k=10)
        assert len(results) == 10


class TestChromaStoreDeleteOperations:
    """Test delete and clear operations."""
    
    def test_delete_records(self, chroma_store):
        """Test deleting specific records by ID."""
        # Insert records
        records = [
            {'id': f'doc{i}', 'vector': [float(i)] * 3, 'metadata': {}}
            for i in range(5)
        ]
        chroma_store.upsert(records)
        
        # Delete 2 records
        chroma_store.delete(['doc1', 'doc3'])
        
        # Verify count
        stats = chroma_store.get_collection_stats()
        assert stats['count'] == 3
    
    def test_clear_collection(self, chroma_store):
        """Test clearing entire collection."""
        # Insert records
        records = [
            {'id': f'doc{i}', 'vector': [float(i)] * 3, 'metadata': {}}
            for i in range(10)
        ]
        chroma_store.upsert(records)
        
        # Clear collection
        chroma_store.clear()
        
        # Verify count is 0
        stats = chroma_store.get_collection_stats()
        assert stats['count'] == 0


class TestChromaStoreErrorHandling:
    """Test error handling and validation."""
    
    def test_upsert_empty_records_raises_error(self, chroma_store):
        """Test that upserting empty list raises ValueError."""
        with pytest.raises(ValueError, match="Records list cannot be empty"):
            chroma_store.upsert([])
    
    def test_upsert_missing_id_raises_error(self, chroma_store):
        """Test that record missing 'id' raises ValueError."""
        records = [
            {'vector': [0.1, 0.2, 0.3], 'metadata': {}}
        ]
        with pytest.raises(ValueError, match="missing required field: 'id'"):
            chroma_store.upsert(records)
    
    def test_upsert_missing_vector_raises_error(self, chroma_store):
        """Test that record missing 'vector' raises ValueError."""
        records = [
            {'id': 'test', 'metadata': {}}
        ]
        with pytest.raises(ValueError, match="missing required field: 'vector'"):
            chroma_store.upsert(records)
    
    def test_query_empty_vector_raises_error(self, chroma_store):
        """Test that querying with empty vector raises ValueError."""
        with pytest.raises(ValueError, match="Query vector cannot be empty"):
            chroma_store.query([], top_k=5)
    
    def test_query_invalid_top_k_raises_error(self, chroma_store):
        """Test that invalid top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            chroma_store.query([0.1, 0.2, 0.3], top_k=0)


class TestChromaStorePersistence:
    """Test data persistence across ChromaStore instances."""
    
    def test_data_persists_across_instances(self, test_settings):
        """Test that data persists when recreating ChromaStore instance."""
        # Create first instance and insert data
        store1 = ChromaStore(settings=test_settings)
        records = [
            {'id': 'persist_test', 'vector': [1.0, 2.0, 3.0], 'metadata': {'test': 'data'}}
        ]
        store1.upsert(records)
        
        # Create second instance (should load existing data)
        store2 = ChromaStore(settings=test_settings)
        
        # Verify data is accessible from second instance
        stats = store2.get_collection_stats()
        assert stats['count'] == 1
        
        results = store2.query([1.0, 2.0, 3.0], top_k=1)
        assert len(results) == 1
        assert results[0]['id'] == 'persist_test'
        
        # Cleanup
        store2.clear()


class TestChromaStoreMetadataSanitization:
    """Test metadata sanitization for ChromaDB compatibility."""
    
    def test_metadata_with_none_values(self, chroma_store):
        """Test that None values in metadata are handled correctly."""
        records = [
            {
                'id': 'test_none',
                'vector': [0.1, 0.2, 0.3],
                'metadata': {
                    'field1': 'value1',
                    'field2': None,  # Should be filtered out
                    'field3': 'value3'
                }
            }
        ]
        
        chroma_store.upsert(records)
        results = chroma_store.query([0.1, 0.2, 0.3], top_k=1)
        
        assert len(results) == 1
        assert 'field1' in results[0]['metadata']
        assert 'field3' in results[0]['metadata']
        # field2 (None) should be filtered out
        assert 'field2' not in results[0]['metadata']
    
    def test_metadata_with_list_values(self, chroma_store):
        """Test that list values are converted to strings."""
        records = [
            {
                'id': 'test_list',
                'vector': [0.1, 0.2, 0.3],
                'metadata': {
                    'tags': ['tag1', 'tag2', 'tag3']  # Should be joined as string
                }
            }
        ]
        
        chroma_store.upsert(records)
        results = chroma_store.query([0.1, 0.2, 0.3], top_k=1)
        
        assert len(results) == 1
        assert 'tags' in results[0]['metadata']
        # Should be comma-separated string
        assert results[0]['metadata']['tags'] == 'tag1,tag2,tag3'
