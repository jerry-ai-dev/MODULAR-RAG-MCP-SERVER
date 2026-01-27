"""Unit tests for Splitter Factory and Base Splitter.

Test Coverage:
- Factory pattern: provider registration, creation, and routing
- Configuration-driven instantiation
- Error handling for unknown/missing providers
- Validation logic in BaseSplitter
"""

from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    """Fake splitter provider for testing.
    
    Splits text by whitespace for deterministic behavior.
    """
    
    def __init__(self, settings: Any = None, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs
        self.call_count = 0
    
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        self.validate_text(text)
        self.call_count += 1
        chunks = [chunk for chunk in text.split() if chunk]
        self.validate_chunks(chunks)
        return chunks


class TestBaseSplitter:
    """Tests for BaseSplitter validation helpers."""
    
    def test_validate_text_success(self):
        splitter = FakeSplitter()
        splitter.validate_text("hello world")
    
    def test_validate_text_empty(self):
        splitter = FakeSplitter()
        with pytest.raises(ValueError, match="cannot be empty"):
            splitter.validate_text("   ")
    
    def test_validate_text_non_string(self):
        splitter = FakeSplitter()
        with pytest.raises(ValueError, match="must be a string"):
            splitter.validate_text(123)  # type: ignore[arg-type]
    
    def test_validate_chunks_success(self):
        splitter = FakeSplitter()
        splitter.validate_chunks(["a", "b", "c"])
    
    def test_validate_chunks_empty_list(self):
        splitter = FakeSplitter()
        with pytest.raises(ValueError, match="cannot be empty"):
            splitter.validate_chunks([])
    
    def test_validate_chunks_non_string(self):
        splitter = FakeSplitter()
        with pytest.raises(ValueError, match="not a string"):
            splitter.validate_chunks(["ok", 1])  # type: ignore[list-item]
    
    def test_validate_chunks_empty_string(self):
        splitter = FakeSplitter()
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            splitter.validate_chunks(["ok", "   "])


class TestFakeSplitter:
    """Tests for FakeSplitter behavior."""
    
    def test_split_text_basic(self):
        splitter = FakeSplitter()
        result = splitter.split_text("hello world")
        assert result == ["hello", "world"]
    
    def test_split_text_increments_call_count(self):
        splitter = FakeSplitter()
        assert splitter.call_count == 0
        splitter.split_text("a b")
        assert splitter.call_count == 1
        splitter.split_text("c d e")
        assert splitter.call_count == 2
    
    def test_split_text_validates_input(self):
        splitter = FakeSplitter()
        with pytest.raises(ValueError, match="cannot be empty"):
            splitter.split_text("   ")


class TestSplitterFactory:
    """Tests for SplitterFactory."""
    
    def setup_method(self) -> None:
        SplitterFactory._PROVIDERS.clear()
    
    def test_register_provider_success(self):
        SplitterFactory.register_provider("fake", FakeSplitter)
        assert "fake" in SplitterFactory._PROVIDERS
        assert SplitterFactory._PROVIDERS["fake"] == FakeSplitter
    
    def test_register_provider_case_insensitive(self):
        SplitterFactory.register_provider("Recursive", FakeSplitter)
        assert "recursive" in SplitterFactory._PROVIDERS
    
    def test_register_provider_invalid_class(self):
        class NotASplitter:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseSplitter"):
            SplitterFactory.register_provider("invalid", NotASplitter)  # type: ignore[arg-type]
    
    def test_list_providers_empty(self):
        assert SplitterFactory.list_providers() == []
    
    def test_list_providers_sorted(self):
        SplitterFactory.register_provider("zebra", FakeSplitter)
        SplitterFactory.register_provider("alpha", FakeSplitter)
        SplitterFactory.register_provider("beta", FakeSplitter)
        assert SplitterFactory.list_providers() == ["alpha", "beta", "zebra"]
    
    def test_create_success(self):
        SplitterFactory.register_provider("fake", FakeSplitter)
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.splitter = "fake"
        
        splitter = SplitterFactory.create(settings)
        assert isinstance(splitter, FakeSplitter)
        assert splitter.settings == settings
    
    def test_create_case_insensitive(self):
        SplitterFactory.register_provider("fake", FakeSplitter)
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.splitter = "FAKE"
        
        splitter = SplitterFactory.create(settings)
        assert isinstance(splitter, FakeSplitter)
    
    def test_create_with_overrides(self):
        SplitterFactory.register_provider("fake", FakeSplitter)
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.splitter = "fake"
        
        splitter = SplitterFactory.create(settings, custom_param=123)
        assert splitter.kwargs["custom_param"] == 123
    
    def test_create_unknown_provider(self):
        SplitterFactory.register_provider("fake", FakeSplitter)
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.splitter = "unknown"
        
        with pytest.raises(ValueError) as exc_info:
            SplitterFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "Unsupported Splitter provider: 'unknown'" in error_message
        assert "Available providers: fake" in error_message
        assert "B7.5" in error_message
    
    def test_create_missing_provider_config(self):
        settings = MagicMock()
        settings.ingestion = None
        
        with pytest.raises(ValueError) as exc_info:
            SplitterFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "settings.ingestion.splitter" in error_message