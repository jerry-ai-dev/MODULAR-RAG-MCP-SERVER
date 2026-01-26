"""Smoke tests for package imports.

This module verifies that all key packages can be imported successfully.
It serves as a basic sanity check for the project structure.
"""

import pytest


@pytest.mark.unit
class TestSmokeImports:
    """Smoke tests to verify all key packages are importable."""

    def test_import_src_package(self) -> None:
        """Test that the src package can be imported."""
        import src
        assert src is not None

    def test_import_mcp_server(self) -> None:
        """Test that the mcp_server package can be imported."""
        from src import mcp_server
        assert mcp_server is not None

    def test_import_mcp_server_tools(self) -> None:
        """Test that the mcp_server.tools subpackage can be imported."""
        from src.mcp_server import tools
        assert tools is not None

    def test_import_core(self) -> None:
        """Test that the core package can be imported."""
        from src import core
        assert core is not None

    def test_import_core_query_engine(self) -> None:
        """Test that the core.query_engine subpackage can be imported."""
        from src.core import query_engine
        assert query_engine is not None

    def test_import_core_response(self) -> None:
        """Test that the core.response subpackage can be imported."""
        from src.core import response
        assert response is not None

    def test_import_core_trace(self) -> None:
        """Test that the core.trace subpackage can be imported."""
        from src.core import trace
        assert trace is not None

    def test_import_ingestion(self) -> None:
        """Test that the ingestion package can be imported."""
        from src import ingestion
        assert ingestion is not None

    def test_import_ingestion_embedding(self) -> None:
        """Test that the ingestion.embedding subpackage can be imported."""
        from src.ingestion import embedding
        assert embedding is not None

    def test_import_ingestion_storage(self) -> None:
        """Test that the ingestion.storage subpackage can be imported."""
        from src.ingestion import storage
        assert storage is not None

    def test_import_ingestion_transform(self) -> None:
        """Test that the ingestion.transform subpackage can be imported."""
        from src.ingestion import transform
        assert transform is not None

    def test_import_libs(self) -> None:
        """Test that the libs package can be imported."""
        from src import libs
        assert libs is not None

    def test_import_libs_embedding(self) -> None:
        """Test that the libs.embedding subpackage can be imported."""
        from src.libs import embedding
        assert embedding is not None

    def test_import_libs_evaluator(self) -> None:
        """Test that the libs.evaluator subpackage can be imported."""
        from src.libs import evaluator
        assert evaluator is not None

    def test_import_libs_llm(self) -> None:
        """Test that the libs.llm subpackage can be imported."""
        from src.libs import llm
        assert llm is not None

    def test_import_libs_loader(self) -> None:
        """Test that the libs.loader subpackage can be imported."""
        from src.libs import loader
        assert loader is not None

    def test_import_libs_reranker(self) -> None:
        """Test that the libs.reranker subpackage can be imported."""
        from src.libs import reranker
        assert reranker is not None

    def test_import_libs_splitter(self) -> None:
        """Test that the libs.splitter subpackage can be imported."""
        from src.libs import splitter
        assert splitter is not None

    def test_import_libs_vector_store(self) -> None:
        """Test that the libs.vector_store subpackage can be imported."""
        from src.libs import vector_store
        assert vector_store is not None

    def test_import_observability(self) -> None:
        """Test that the observability package can be imported."""
        from src import observability
        assert observability is not None

    def test_import_observability_dashboard(self) -> None:
        """Test that the observability.dashboard subpackage can be imported."""
        from src.observability import dashboard
        assert dashboard is not None

    def test_import_observability_evaluation(self) -> None:
        """Test that the observability.evaluation subpackage can be imported."""
        from src.observability import evaluation
        assert evaluation is not None
