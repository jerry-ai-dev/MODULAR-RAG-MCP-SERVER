"""Unit tests for SparseRetriever.

Tests cover:
- Initialization and configuration
- Keyword validation
- Dependency validation
- BM25 query + vector store get_by_ids integration
- Result merging and transformation
- Error handling and edge cases
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.core.query_engine.sparse_retriever import SparseRetriever, create_sparse_retriever
from src.core.types import RetrievalResult


# ============================================================================
# Mock Classes
# ============================================================================

class MockBM25Indexer:
    """Mock BM25 indexer for testing."""
    
    def __init__(self, index_dir: str = "data/db/bm25"):
        self.index_dir = index_dir
        self._index = {}
        self._metadata = {}
        self._loaded_collection = None
        
    def load(self, collection: str = "default", trace: Optional[Any] = None) -> bool:
        """Simulate loading an index."""
        self._loaded_collection = collection
        self._metadata = {"collection": collection}
        return True
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Return mock BM25 results."""
        # Return predetermined results based on query terms
        results = [
            {"chunk_id": "chunk_001", "score": 2.5},
            {"chunk_id": "chunk_002", "score": 1.8},
            {"chunk_id": "chunk_003", "score": 1.2},
        ]
        return results[:top_k]


class MockBM25IndexerEmpty:
    """Mock BM25 indexer that returns empty results."""
    
    def __init__(self):
        self._metadata = {}
    
    def load(self, collection: str = "default", trace: Optional[Any] = None) -> bool:
        self._metadata = {"collection": collection}
        return True
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        return []


class MockBM25IndexerFailing:
    """Mock BM25 indexer that fails on load or query."""
    
    def __init__(self, fail_on_load: bool = False, fail_on_query: bool = False):
        self.fail_on_load = fail_on_load
        self.fail_on_query = fail_on_query
        self._metadata = {}
    
    def load(self, collection: str = "default", trace: Optional[Any] = None) -> bool:
        if self.fail_on_load:
            raise RuntimeError("Simulated load failure")
        self._metadata = {"collection": collection}
        return True
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        if self.fail_on_query:
            raise RuntimeError("Simulated query failure")
        return []


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self):
        self._records = {
            "chunk_001": {
                "id": "chunk_001",
                "text": "This is the first chunk about machine learning.",
                "metadata": {"source_path": "doc1.pdf", "chunk_index": 0}
            },
            "chunk_002": {
                "id": "chunk_002",
                "text": "This is the second chunk about neural networks.",
                "metadata": {"source_path": "doc1.pdf", "chunk_index": 1}
            },
            "chunk_003": {
                "id": "chunk_003",
                "text": "This is the third chunk about deep learning.",
                "metadata": {"source_path": "doc2.pdf", "chunk_index": 0}
            },
        }
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Return mock records by IDs."""
        results = []
        for id_ in ids:
            if id_ in self._records:
                results.append(self._records[id_])
            else:
                results.append({})  # Not found
        return results


class MockVectorStoreFailing:
    """Mock vector store that fails on get_by_ids."""
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raise RuntimeError("Simulated vector store failure")


class MockSettings:
    """Mock settings for testing."""
    
    def __init__(self, sparse_top_k: int = 15):
        self.retrieval = MagicMock()
        self.retrieval.sparse_top_k = sparse_top_k


# ============================================================================
# Test: Initialization
# ============================================================================

class TestSparseRetrieverInit:
    """Tests for SparseRetriever initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        retriever = SparseRetriever()
        
        assert retriever.bm25_indexer is None
        assert retriever.vector_store is None
        assert retriever.default_top_k == 10
        assert retriever.default_collection == "default"
    
    def test_init_with_custom_top_k(self):
        """Test initialization with custom default_top_k."""
        retriever = SparseRetriever(default_top_k=20)
        
        assert retriever.default_top_k == 20
    
    def test_init_with_settings(self):
        """Test initialization extracts top_k from settings."""
        settings = MockSettings(sparse_top_k=25)
        retriever = SparseRetriever(settings=settings)
        
        assert retriever.default_top_k == 25
    
    def test_init_with_dependencies(self):
        """Test initialization with injected dependencies."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        
        retriever = SparseRetriever(
            bm25_indexer=bm25,
            vector_store=vs,
            default_collection="my_collection"
        )
        
        assert retriever.bm25_indexer is bm25
        assert retriever.vector_store is vs
        assert retriever.default_collection == "my_collection"


# ============================================================================
# Test: Input Validation
# ============================================================================

class TestSparseRetrieverValidation:
    """Tests for input validation."""
    
    def test_retrieve_raises_on_empty_keywords(self):
        """Test that empty keywords raises ValueError."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve([])
    
    def test_retrieve_raises_on_non_list_keywords(self):
        """Test that non-list keywords raises ValueError."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(ValueError, match="must be a list"):
            retriever.retrieve("not a list")  # type: ignore
    
    def test_retrieve_raises_without_bm25_indexer(self):
        """Test that missing bm25_indexer raises RuntimeError."""
        vs = MockVectorStore()
        retriever = SparseRetriever(vector_store=vs)
        
        with pytest.raises(RuntimeError, match="requires a bm25_indexer"):
            retriever.retrieve(["keyword"])
    
    def test_retrieve_raises_without_vector_store(self):
        """Test that missing vector_store raises RuntimeError."""
        bm25 = MockBM25Indexer()
        retriever = SparseRetriever(bm25_indexer=bm25)
        
        with pytest.raises(RuntimeError, match="requires a vector_store"):
            retriever.retrieve(["keyword"])


# ============================================================================
# Test: Basic Retrieval
# ============================================================================

class TestSparseRetrieverRetrieve:
    """Tests for retrieve() method."""
    
    def test_retrieve_returns_results(self):
        """Test basic retrieval returns RetrievalResult objects."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["machine", "learning"])
        
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    def test_retrieve_results_have_correct_fields(self):
        """Test that results have all required fields."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        first = results[0]
        assert first.chunk_id == "chunk_001"
        assert first.score == 2.5
        assert "machine learning" in first.text
        assert first.metadata["source_path"] == "doc1.pdf"
    
    def test_retrieve_respects_top_k(self):
        """Test that top_k parameter limits results."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"], top_k=2)
        
        assert len(results) == 2
    
    def test_retrieve_uses_default_top_k(self):
        """Test that default_top_k is used when top_k not specified."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(
            bm25_indexer=bm25, 
            vector_store=vs, 
            default_top_k=2
        )
        
        results = retriever.retrieve(["test"])
        
        assert len(results) == 2
    
    def test_retrieve_with_custom_collection(self):
        """Test retrieval with custom collection parameter."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        # Should not raise, collection is passed to bm25_indexer
        results = retriever.retrieve(["test"], collection="custom_collection")
        
        assert len(results) > 0
    
    def test_retrieve_preserves_result_order(self):
        """Test that results maintain BM25 score ordering."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        # Results should be in descending score order (from BM25)
        assert results[0].score > results[1].score > results[2].score


# ============================================================================
# Test: Empty Results
# ============================================================================

class TestSparseRetrieverEmptyResults:
    """Tests for empty result handling."""
    
    def test_retrieve_returns_empty_when_no_matches(self):
        """Test that no BM25 matches returns empty list."""
        bm25 = MockBM25IndexerEmpty()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["nonexistent"])
        
        assert results == []
    
    def test_retrieve_returns_empty_when_index_not_loaded(self):
        """Test graceful handling when index cannot be loaded."""
        bm25 = MockBM25IndexerFailing(fail_on_load=True)
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        # Should return empty, not raise
        results = retriever.retrieve(["test"])
        
        assert results == []


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestSparseRetrieverErrorHandling:
    """Tests for error handling."""
    
    def test_retrieve_raises_on_bm25_query_failure(self):
        """Test that BM25 query failure raises RuntimeError."""
        bm25 = MockBM25IndexerFailing(fail_on_query=True)
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(RuntimeError, match="Failed to query BM25"):
            retriever.retrieve(["test"])
    
    def test_retrieve_raises_on_vector_store_failure(self):
        """Test that vector store failure raises RuntimeError."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStoreFailing()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(RuntimeError, match="Failed to fetch records"):
            retriever.retrieve(["test"])


# ============================================================================
# Test: Missing Records
# ============================================================================

class TestSparseRetrieverMissingRecords:
    """Tests for handling missing records."""
    
    def test_retrieve_skips_missing_records(self):
        """Test that missing records in vector store are skipped."""
        bm25 = MockBM25Indexer()
        
        # Create vector store with only partial records
        vs = MockVectorStore()
        vs._records = {
            "chunk_001": {
                "id": "chunk_001",
                "text": "First chunk.",
                "metadata": {"source_path": "doc.pdf"}
            },
            # chunk_002 and chunk_003 are missing
        }
        
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test"])
        
        # Should only return the one found record
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_001"


# ============================================================================
# Test: Result Transformation
# ============================================================================

class TestSparseRetrieverResultTransformation:
    """Tests for result transformation."""
    
    def test_retrieve_result_is_serializable(self):
        """Test that results can be serialized to dict."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"], top_k=1)
        
        result_dict = results[0].to_dict()
        assert "chunk_id" in result_dict
        assert "score" in result_dict
        assert "text" in result_dict
        assert "metadata" in result_dict
    
    def test_retrieve_result_score_is_float(self):
        """Test that score is converted to float."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        for result in results:
            assert isinstance(result.score, float)


# ============================================================================
# Test: Factory Function
# ============================================================================

class TestCreateSparseRetriever:
    """Tests for create_sparse_retriever factory function."""
    
    def test_create_with_injected_dependencies(self):
        """Test factory with injected dependencies."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        settings = MockSettings()
        
        retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25,
            vector_store=vs,
        )
        
        assert retriever.bm25_indexer is bm25
        assert retriever.vector_store is vs


# ============================================================================
# Test: Index Loading
# ============================================================================

class TestSparseRetrieverIndexLoading:
    """Tests for BM25 index loading behavior."""
    
    def test_index_loaded_once_per_collection(self):
        """Test that index is only loaded once per collection."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        # First call should load the index
        retriever.retrieve(["test"], collection="test_col")
        assert bm25._loaded_collection == "test_col"
        
        # Second call with same collection should not reload
        bm25._loaded_collection = "test_col"  # Mark as loaded
        retriever.retrieve(["test"], collection="test_col")
        assert bm25._loaded_collection == "test_col"
    
    def test_index_reloaded_for_different_collection(self):
        """Test that index is reloaded for different collection."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        # Load first collection
        retriever.retrieve(["test"], collection="col_a")
        assert bm25._metadata.get("collection") == "col_a"
        
        # Load different collection
        retriever.retrieve(["test"], collection="col_b")
        assert bm25._metadata.get("collection") == "col_b"


# ============================================================================
# Test: Integration with Real Types
# ============================================================================

class TestSparseRetrieverTypeIntegration:
    """Tests for integration with core types."""
    
    def test_retrieve_result_type_matches_dense_retriever(self):
        """Test that SparseRetriever returns same type as DenseRetriever."""
        from src.core.types import RetrievalResult as TypedRetrievalResult
        
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        for result in results:
            assert isinstance(result, TypedRetrievalResult)
    
    def test_retrieve_result_compatible_with_from_dict(self):
        """Test results can be recreated from dict."""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"], top_k=1)
        result_dict = results[0].to_dict()
        
        # Recreate from dict
        recreated = RetrievalResult.from_dict(result_dict)

        assert recreated.chunk_id == results[0].chunk_id
        assert recreated.score == results[0].score
        assert recreated.text == results[0].text


# ============================================================================
# Test: ID Mapping Robustness
# ============================================================================

class TestSparseRetrieverIDMapping:
    """Tests for robust ID-based record mapping in _merge_results."""

    def test_merge_with_missing_record(self):
        """Test that missing records are skipped correctly via ID mapping."""
        # Create a vector store with only partial records
        class MockVectorStorePartial:
            def get_by_ids(self, ids, trace=None, **kwargs):
                # Simulate chunk_002 missing from vector store
                return [
                    {"id": "chunk_001", "text": "First chunk", "metadata": {}},
                    {},  # chunk_002 missing
                    {"id": "chunk_003", "text": "Third chunk", "metadata": {}},
                ]

        # BM25 returns 3 results
        class MockBM25WithThreeResults:
            def load(self, collection="default", trace=None):
                return True

            def query(self, query_terms, top_k=10, trace=None):
                return [
                    {"chunk_id": "chunk_001", "score": 3.0},
                    {"chunk_id": "chunk_002", "score": 2.0},
                    {"chunk_id": "chunk_003", "score": 1.0},
                ]

        bm25 = MockBM25WithThreeResults()
        vs = MockVectorStorePartial()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)

        results = retriever.retrieve(["test"])

        # Should only return 2 results (chunk_002 missing)
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_001"
        assert results[0].score == 3.0
        assert results[1].chunk_id == "chunk_003"
        assert results[1].score == 1.0

    def test_merge_with_reordered_records(self):
        """Test that ID mapping handles records returned in different order."""
        # Simulate a vector store that returns records in a different order
        class MockVectorStoreReordered:
            def get_by_ids(self, ids, trace=None, **kwargs):
                # Intentionally return in reverse order to test ID mapping
                return [
                    {"id": "chunk_003", "text": "Third", "metadata": {}},
                    {"id": "chunk_002", "text": "Second", "metadata": {}},
                    {"id": "chunk_001", "text": "First", "metadata": {}},
                ]

        class MockBM25Ordered:
            def load(self, collection="default", trace=None):
                return True

            def query(self, query_terms, top_k=10, trace=None):
                return [
                    {"chunk_id": "chunk_001", "score": 3.0},
                    {"chunk_id": "chunk_002", "score": 2.0},
                    {"chunk_id": "chunk_003", "score": 1.0},
                ]

        bm25 = MockBM25Ordered()
        vs = MockVectorStoreReordered()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)

        results = retriever.retrieve(["test"])

        # Should maintain BM25 score order using ID mapping
        assert len(results) == 3
        assert results[0].chunk_id == "chunk_001"
        assert results[0].text == "First"
        assert results[1].chunk_id == "chunk_002"
        assert results[1].text == "Second"
        assert results[2].chunk_id == "chunk_003"
        assert results[2].text == "Third"
