"""Unit tests for ChunkRefiner transform."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.libs.llm.base_llm import BaseLLM


# Fixtures

@pytest.fixture
def noisy_chunks_data():
    """Load test data from fixtures."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "noisy_chunks.json"
    with open(fixture_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def mock_settings():
    """Create mock settings without LLM enabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': False}
    return settings


@pytest.fixture
def mock_settings_with_llm():
    """Create mock settings with LLM enabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': True}
    return settings


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    llm = Mock(spec=BaseLLM)
    llm.chat.return_value = "LLM refined text"
    return llm


@pytest.fixture
def sample_chunk():
    """Create a sample chunk."""
    return Chunk(
        id="test_chunk_001",
        text="Sample text with  extra   spaces\n\n\n\nand newlines",
        metadata={"source": "test.pdf", "source_path": "test.pdf"},
        source_ref="test_doc"
    )


# Test Rule-Based Refinement

class TestRuleBasedRefinement:
    """Test rule-based cleaning without LLM."""
    
    def test_remove_excessive_whitespace(self, mock_settings, noisy_chunks_data):
        """Test removal of excessive spaces and newlines."""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['excessive_whitespace']['input']
        expected = noisy_chunks_data['excessive_whitespace']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_remove_page_headers_footers(self, mock_settings, noisy_chunks_data):
        """Test removal of page headers and footers with separator lines."""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['page_header_footer']['input']
        expected = noisy_chunks_data['page_header_footer']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_remove_html_tags_and_comments(self, mock_settings, noisy_chunks_data):
        """Test removal of HTML tags and comments while preserving Markdown."""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['format_markers']['input']
        expected = noisy_chunks_data['format_markers']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_preserve_code_blocks(self, mock_settings, noisy_chunks_data):
        """Test that code blocks internal formatting is preserved."""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['code_blocks']['input']
        expected = noisy_chunks_data['code_blocks']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_not_overclean_good_text(self, mock_settings, noisy_chunks_data):
        """Test that clean text is not over-processed."""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['clean_text']['input']
        expected = noisy_chunks_data['clean_text']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_mixed_noise_scenario(self, mock_settings, noisy_chunks_data):
        """Test comprehensive noise removal with mixed issues."""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['mixed_noise']['input']
        expected = noisy_chunks_data['mixed_noise']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_empty_string_handling(self, mock_settings):
        """Test handling of empty strings."""
        refiner = ChunkRefiner(mock_settings)
        
        assert refiner._rule_based_refine("") == ""
        # Whitespace-only strings return empty after strip
        result = refiner._rule_based_refine("   \n\n  ")
        assert result.strip() == ""
    
    def test_none_handling(self, mock_settings):
        """Test handling of None input."""
        refiner = ChunkRefiner(mock_settings)
        result = refiner._rule_based_refine(None)
        assert result is None


# Test Transform Pipeline (Rule-Only Mode)

class TestTransformPipelineRuleOnly:
    """Test full transform pipeline without LLM."""
    
    def test_transform_single_chunk(self, mock_settings, sample_chunk):
        """Test transforming a single chunk."""
        refiner = ChunkRefiner(mock_settings)
        
        result = refiner.transform([sample_chunk])
        
        assert len(result) == 1
        assert result[0].id == sample_chunk.id
        assert result[0].text == "Sample text with extra spaces\n\nand newlines"
        assert result[0].metadata['refined_by'] == 'rule'
    
    def test_transform_multiple_chunks(self, mock_settings):
        """Test transforming multiple chunks."""
        refiner = ChunkRefiner(mock_settings)
        
        chunks = [
            Chunk(id="c1", text="Text  with   spaces", metadata={"source_path": "test1.pdf"}),
            Chunk(id="c2", text="Another\n\n\nchunk", metadata={"source_path": "test2.pdf"}),
            Chunk(id="c3", text="Clean text", metadata={"source_path": "test3.pdf"})
        ]
        
        result = refiner.transform(chunks)
        
        assert len(result) == 3
        assert result[0].text == "Text with spaces"
        assert result[1].text == "Another\n\nchunk"
        assert result[2].text == "Clean text"
        assert all(r.metadata['refined_by'] == 'rule' for r in result)
    
    def test_transform_empty_list(self, mock_settings):
        """Test transforming empty chunk list."""
        refiner = ChunkRefiner(mock_settings)
        result = refiner.transform([])
        assert result == []
    
    def test_metadata_preserved(self, mock_settings, sample_chunk):
        """Test that original metadata is preserved."""
        refiner = ChunkRefiner(mock_settings)
        
        result = refiner.transform([sample_chunk])
        
        assert result[0].metadata['source'] == 'test.pdf'
        assert 'refined_by' in result[0].metadata
    
    def test_trace_recording(self, mock_settings, sample_chunk):
        """Test that trace context records processing info."""
        refiner = ChunkRefiner(mock_settings)
        trace = TraceContext()
        
        refiner.transform([sample_chunk], trace=trace)
        
        stage_data = trace.get_stage_data('chunk_refiner')
        assert stage_data is not None
        assert stage_data['data']['total_chunks'] == 1
        assert stage_data['data']['success_count'] == 1
        assert stage_data['data']['use_llm'] is False


# Test LLM Enhancement Mode

class TestLLMEnhancement:
    """Test LLM-based refinement."""
    
    def test_llm_success_path(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """Test successful LLM refinement."""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.return_value = "Beautifully refined text by LLM"
        
        # Mock prompt loading
        refiner._prompt_template = "Refine this: {text}"
        
        result = refiner.transform([sample_chunk])
        
        assert len(result) == 1
        assert "Beautifully refined" in result[0].text
        assert result[0].metadata['refined_by'] == 'llm'
        mock_llm.chat.assert_called_once()
    
    def test_llm_fallback_on_failure(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """Test fallback to rule-based when LLM fails."""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.side_effect = Exception("LLM API error")
        
        result = refiner.transform([sample_chunk])
        
        assert len(result) == 1
        assert result[0].metadata['refined_by'] == 'rule'
        assert 'refine_fallback_reason' in result[0].metadata
        assert result[0].metadata['refine_fallback_reason'] == 'llm_failed'
    
    def test_llm_fallback_on_empty_response(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """Test fallback when LLM returns empty string."""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.return_value = ""
        refiner._prompt_template = "Refine: {text}"
        
        result = refiner.transform([sample_chunk])
        
        assert result[0].metadata['refined_by'] == 'rule'
        assert result[0].metadata['refine_fallback_reason'] == 'llm_failed'
    
    def test_llm_trace_recording(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """Test trace records LLM enhancement count."""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.return_value = "LLM result"
        refiner._prompt_template = "Refine: {text}"
        trace = TraceContext()
        
        refiner.transform([sample_chunk], trace=trace)
        
        stage_data = trace.get_stage_data('chunk_refiner')
        assert stage_data['data']['llm_enhanced_count'] == 1
        assert stage_data['data']['fallback_count'] == 0


# Test Prompt Loading

class TestPromptLoading:
    """Test prompt template loading."""
    
    def test_load_existing_prompt(self, mock_settings):
        """Test loading existing prompt file."""
        refiner = ChunkRefiner(mock_settings)
        
        # Use real prompt file
        prompt = refiner._load_prompt()
        
        assert prompt is not None
        assert '{text}' in prompt
    
    def test_prompt_cached_after_first_load(self, mock_settings):
        """Test that prompt is cached after first load."""
        refiner = ChunkRefiner(mock_settings)
        
        prompt1 = refiner._load_prompt()
        prompt2 = refiner._load_prompt()
        
        assert prompt1 is prompt2  # Same object reference
    
    def test_missing_prompt_file(self, mock_settings):
        """Test handling of missing prompt file."""
        refiner = ChunkRefiner(mock_settings, prompt_path="nonexistent.txt")
        
        prompt = refiner._load_prompt()
        
        assert prompt is None
    
    def test_prompt_missing_placeholder(self, mock_settings, mock_llm, sample_chunk):
        """Test LLM refinement fails gracefully when prompt lacks {text}."""
        refiner = ChunkRefiner(mock_settings, llm=mock_llm)
        refiner._prompt_template = "This prompt has no placeholder"
        refiner.use_llm = True
        
        result = refiner._llm_refine("test text")
        
        assert result is None


# Test Atomic Processing

class TestAtomicProcessing:
    """Test that individual chunk failures don't affect others."""
    
    def test_exception_in_one_chunk_preserves_original(self, mock_settings):
        """Test that exception in processing one chunk preserves its original."""
        refiner = ChunkRefiner(mock_settings)
        
        # Mock _rule_based_refine to fail on specific chunk
        original_method = refiner._rule_based_refine
        
        def failing_refine(text):
            if "fail" in text:
                raise ValueError("Intentional test error")
            return original_method(text)
        
        refiner._rule_based_refine = failing_refine
        
        chunks = [
            Chunk(id="c1", text="Normal text", metadata={"source_path": "test1.pdf"}),
            Chunk(id="c2", text="This should fail processing", metadata={"source_path": "test2.pdf"}),
            Chunk(id="c3", text="Another normal text", metadata={"source_path": "test3.pdf"})
        ]
        
        result = refiner.transform(chunks)
        
        # All chunks returned
        assert len(result) == 3
        # Failed chunk preserved as-is
        assert result[1].text == "This should fail processing"
        # Other chunks processed normally
        assert result[0].metadata.get('refined_by') == 'rule'
        assert result[2].metadata.get('refined_by') == 'rule'


# Test Configuration

class TestConfiguration:
    """Test configuration handling."""
    
    def test_use_llm_disabled_by_default(self):
        """Test that LLM is disabled when config missing."""
        settings = Mock(spec=Settings)
        settings.ingestion = None
        
        refiner = ChunkRefiner(settings)
        
        assert refiner.use_llm is False
    
    def test_lazy_llm_initialization(self, mock_settings_with_llm):
        """Test that LLM is only initialized when needed."""
        with patch('src.ingestion.transform.chunk_refiner.LLMFactory.create') as mock_factory:
            refiner = ChunkRefiner(mock_settings_with_llm)
            
            # LLM not initialized yet
            assert refiner._llm is None
            mock_factory.assert_not_called()
            
            # Access llm property triggers initialization
            _ = refiner.llm
            mock_factory.assert_called_once()
    
    def test_llm_init_failure_disables_llm(self, mock_settings_with_llm):
        """Test that LLM initialization failure disables LLM mode."""
        with patch('src.ingestion.transform.chunk_refiner.LLMFactory.create', side_effect=Exception("Init failed")):
            refiner = ChunkRefiner(mock_settings_with_llm)
            
            # Try to access LLM
            llm = refiner.llm
            
            # LLM mode disabled after failure
            assert llm is None
            assert refiner.use_llm is False
