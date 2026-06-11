"""Sparse Retriever for keyword-based search using BM25.

This module implements the SparseRetriever component that performs keyword-based
search using BM25 inverted indexes. It forms the Sparse route in the Hybrid
Search Engine, complementing the DenseRetriever's semantic search.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Sparse retriever using BM25 keyword-based search.
    
    This class performs keyword-based retrieval by:
    1. Querying the BM25 index with keywords to get matching chunk IDs and scores
    2. Fetching text and metadata from the vector store using get_by_ids()
    3. Returning normalized RetrievalResult objects
    
    Design Principles Applied:
    - Pluggable: Accepts bm25_indexer and vector_store via dependency injection.
    - Config-Driven: Default top_k and collection read from settings.
    - Observable: Accepts optional TraceContext for observability integration.
    - Fail-Fast: Validates inputs early with clear error messages.
    - Type-Safe: Returns standardized RetrievalResult objects (same as DenseRetriever).
    
    Attributes:
        bm25_indexer: The BM25 indexer for keyword search.
        vector_store: The vector store for fetching text and metadata.
        default_top_k: Default number of results to return.
        default_collection: Default BM25 index collection to query.
    
    Example:
        >>> from src.ingestion.storage.bm25_indexer import BM25Indexer
        >>> from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        >>> 
        >>> settings = Settings.load('config/settings.yaml')
        >>> bm25_indexer = BM25Indexer(index_dir="data/db/bm25")
        >>> bm25_indexer.load("default")
        >>> vector_store = VectorStoreFactory.create(settings)
        >>> 
        >>> retriever = SparseRetriever(
        ...     settings=settings,
        ...     bm25_indexer=bm25_indexer,
        ...     vector_store=vector_store
        ... )
        >>> results = retriever.retrieve(["RAG", "retrieval"], top_k=5)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 10,
        default_collection: str = "default",
    ) -> None:
        """Initialize SparseRetriever with dependencies.
        
        Args:
            settings: Application settings. Used to extract default_top_k if not provided.
            bm25_indexer: BM25 indexer for keyword search.
                          Required for actual retrieval operations.
            vector_store: Vector store for fetching text and metadata.
                          Required for actual retrieval operations.
            default_top_k: Default number of results to return (default: 10).
                           Can be overridden from settings.retrieval.sparse_top_k.
            default_collection: Default BM25 index collection name (default: "default").
        
        Note:
            Dependencies can be injected for testing (with mocks) or for
            production use (with real implementations from factories).
        """
        self.bm25_indexer = bm25_indexer
        self.vector_store = vector_store
        self.default_collection = default_collection
        
        # Extract default_top_k from settings if available
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, 'sparse_top_k', default_top_k
                )
        
        logger.info(
            f"SparseRetriever initialized with default_top_k={self.default_top_k}, "
            f"default_collection='{self.default_collection}'"
        )
    
    def retrieve(
        self,
        keywords: List[str],
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Retrieve chunks matching the given keywords using BM25.
        
        Args:
            keywords: List of keywords to search for (typically from QueryProcessor).
            top_k: Maximum number of results to return. If None, uses default_top_k.
            collection: BM25 index collection to query. If None, uses default_collection.
            trace: Optional TraceContext for observability (reserved for Stage F).
        
        Returns:
            List of RetrievalResult objects, sorted by BM25 score (descending).
            Each result contains chunk_id, score, text, and metadata.
        
        Raises:
            ValueError: If keywords list is empty.
            RuntimeError: If bm25_indexer or vector_store is not configured,
                          or if the retrieval operation fails.
        
        Example:
            >>> results = retriever.retrieve(["Azure", "OpenAI", "配置"])
            >>> for result in results:
            ...     print(f"[{result.score:.2f}] {result.chunk_id}: {result.text[:50]}...")
        """
        # Validate inputs
        self._validate_keywords(keywords)
        self._validate_dependencies()
        
        # Use defaults if not specified
        effective_top_k = top_k if top_k is not None else self.default_top_k
        effective_collection = collection if collection is not None else self.default_collection
        
        logger.debug(
            f"Retrieving for keywords={keywords[:5]}{'...' if len(keywords) > 5 else ''}, "
            f"top_k={effective_top_k}, collection='{effective_collection}'"
        )
        
        # Step 1: Ensure index is loaded
        if not self._ensure_index_loaded(effective_collection):
            logger.warning(
                f"BM25 index for collection '{effective_collection}' not available. "
                "Returning empty results."
            )
            return []
        
        # Step 2: Query BM25 index
        try:
            bm25_results = self.bm25_indexer.query(
                query_terms=keywords,
                top_k=effective_top_k,
                trace=trace,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query BM25 index: {e}. "
                "Check index availability and query terms."
            ) from e
        
        # Early return if no matches
        if not bm25_results:
            logger.debug("BM25 query returned no results")
            return []
        
        # Step 3: Fetch text and metadata from vector store
        chunk_ids = [r["chunk_id"] for r in bm25_results]
        try:
            records = self.vector_store.get_by_ids(chunk_ids, trace=trace)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch records from vector store: {e}. "
                "Check vector store configuration and data availability."
            ) from e
        
        # Step 4: Merge BM25 scores with text/metadata
        results = self._merge_results(bm25_results, records)
        
        logger.debug(f"Retrieved {len(results)} results for keywords")
        return results
    
    def _validate_keywords(self, keywords: List[str]) -> None:
        """Validate the keywords list.
        
        Args:
            keywords: Keywords list to validate.
        
        Raises:
            ValueError: If keywords is empty or not a list.
        """
        if not isinstance(keywords, list):
            raise ValueError(
                f"Keywords must be a list, got {type(keywords).__name__}"
            )
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        # Filter out empty strings but allow the call to proceed
        # (empty strings will simply not match anything)
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are configured.
        
        Raises:
            RuntimeError: If bm25_indexer or vector_store is None.
        """
        if self.bm25_indexer is None:
            raise RuntimeError(
                "SparseRetriever requires a bm25_indexer. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "SparseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )
    
    def _ensure_index_loaded(self, collection: str) -> bool:
        """Ensure the BM25 index is loaded for the given collection.
        
        Always reloads from disk because the index may have been updated
        by another process (e.g., dashboard ingestion).  The load is
        fast (a single JSON file read) compared to the overall query.
        
        Args:
            collection: The collection name to load.
        
        Returns:
            True if index is loaded and ready, False otherwise.
        """
        try:
            loaded = self.bm25_indexer.load(collection=collection)
            return loaded
        except Exception as e:
            logger.warning(f"Failed to load BM25 index for collection '{collection}': {e}")
            return False
    
    def _merge_results(
        self,
        bm25_results: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """Merge BM25 scores with text and metadata from vector store.

        Args:
            bm25_results: Results from BM25 query, each with 'chunk_id' and 'score'.
            records: Records from vector store, each with 'id', 'text', 'metadata'.
                     The records list should be in the same order as chunk_ids passed to get_by_ids.

        Returns:
            List of RetrievalResult objects with complete information.
        """
        results = []

        # Build ID-to-record mapping for explicit lookup
        # This is more robust than zip() when records may be missing or reordered
        record_map = {record.get('id'): record for record in records if record}

        for bm25_result in bm25_results:
            chunk_id = bm25_result["chunk_id"]
            score = bm25_result["score"]

            # Explicitly look up record by chunk_id
            record = record_map.get(chunk_id)

            # Handle case where record was not found
            if not record:
                logger.warning(
                    f"No record found in vector store for chunk_id='{chunk_id}'. "
                    "Skipping this result."
                )
                continue

            # Validate record has expected fields
            text = record.get('text', '')
            metadata = record.get('metadata', {})

            try:
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=str(text),
                    metadata=metadata,
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to create RetrievalResult for chunk_id='{chunk_id}': {e}. "
                    "Skipping this result."
                )
                continue

        return results


def create_sparse_retriever(
    settings: Settings,
    bm25_indexer: Optional[BM25Indexer] = None,
    vector_store: Optional[BaseVectorStore] = None,
    index_dir: str = "data/db/bm25",
) -> SparseRetriever:
    """Factory function to create a SparseRetriever with optional dependency injection.
    
    This function simplifies SparseRetriever creation by automatically creating
    dependencies from factories if not provided.
    
    Args:
        settings: Application settings.
        bm25_indexer: Optional pre-configured BM25 indexer.
                      If None, created with default index_dir.
        vector_store: Optional pre-configured vector store.
                      If None, created from VectorStoreFactory.
        index_dir: Directory for BM25 index files (default: "data/db/bm25").
    
    Returns:
        Configured SparseRetriever instance.
    
    Example:
        >>> settings = Settings.load('config/settings.yaml')
        >>> retriever = create_sparse_retriever(settings)
    """
    # Lazy import to avoid circular dependencies
    if bm25_indexer is None:
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        bm25_indexer = BM25Indexer(index_dir=index_dir)
    
    if vector_store is None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        vector_store = VectorStoreFactory.create(settings)
    
    return SparseRetriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
