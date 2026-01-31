"""Integration tests for ChunkRefiner with real LLM providers.

These tests require actual API keys and will make real API calls.
Run with: pytest tests/integration/test_chunk_refiner_llm.py -v -s

Required environment variables:
    - OPENAI_API_KEY: For OpenAI tests
    - AZURE_OPENAI_API_KEY: For Azure tests
    - OLLAMA_BASE_URL: For Ollama tests (default: http://localhost:11434)
"""

import os
import pytest
from unittest.mock import Mock

from src.core.settings import Settings, load_settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.chunk_refiner import ChunkRefiner


# Test data: Realistic noisy chunk from PDF extraction
NOISY_PDF_CHUNK = """
────────────────────────────
Page 42 | Technical Documentation
────────────────────────────


Chapter 5: System Architecture

The   microservices   architecture  consists  of  several  key  components.

<!-- Internal note: Update diagram -->

<div class="important">
Each service communicates via REST API or message queues.
</div>




The main   components   are:
- Gateway   Service  
- Authentication  Service  
- Data   Processing   Service


────────────────────────────
Footer: © 2024 Company | Confidential
────────────────────────────
"""

EXPECTED_CLEAN_RESULT_KEYWORDS = [
    "Chapter 5",
    "System Architecture",
    "microservices",
    "REST API",
    "message queues",
    "Gateway Service",
    "Authentication Service"
]


# Fixtures

@pytest.fixture
def sample_noisy_chunk():
    """Create a sample noisy chunk for testing."""
    return Chunk(
        id="test_pdf_chunk_001",
        text=NOISY_PDF_CHUNK,
        metadata={"source": "technical_doc.pdf", "source_path": "technical_doc.pdf", "page": 42},
        source_ref="doc_technical_2024"
    )


def create_settings_for_provider(provider: str) -> Settings:
    """Create settings object for specific provider.
    
    For Azure provider, loads actual settings from settings.yaml.
    For other providers, uses environment variables.
    
    Args:
        provider: One of 'openai', 'azure', 'ollama'
    
    Returns:
        Settings object configured for the provider
    """
    if provider == 'azure':
        # Load real settings from settings.yaml for Azure
        try:
            import yaml
            with open("config/settings.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # Inject into environment variables for AzureLLM
            if 'llm' in config and config['llm'].get('provider') == 'azure':
                os.environ["AZURE_OPENAI_API_KEY"] = config['llm'].get('api_key', '')
                os.environ["AZURE_OPENAI_ENDPOINT"] = config['llm'].get('azure_endpoint', '')
                os.environ["ENDPOINT"] = config['llm'].get('azure_endpoint', '')
                
                os.environ["AZURE_OPENAI_API_VERSION"] = config['llm'].get('api_version', '')
                os.environ["OPENAI_API_VERSION"] = config['llm'].get('api_version', '') 
            
            real_settings = load_settings("config/settings.yaml")

            # Create a Mock Settings object to allow modification (real settings are frozen)
            settings = Mock(spec=Settings)
            # Copy necessary frozen configs
            settings.llm = real_settings.llm
            settings.ingestion = Mock()
            # Enable LLM for chunk refiner
            settings.ingestion.chunk_refiner = {'use_llm': True}
            return settings
        except Exception as e:
            pytest.skip(f"Failed to load settings.yaml or configure Azure: {e}")
    
    # For non-Azure providers, use environment variables
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': True}
    
    # LLM configuration
    settings.llm = Mock()
    settings.llm.provider = provider
    
    if provider == 'openai':
        settings.llm.model = "gpt-3.5-turbo"
        settings.llm.api_key = os.getenv('OPENAI_API_KEY')
        settings.llm.temperature = 0.3
        settings.llm.max_tokens = 1000
        
    elif provider == 'ollama':
        settings.llm.model = "llama2"
        settings.llm.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        settings.llm.temperature = 0.3
        
    return settings


# Helper function to check provider availability

def is_provider_available(provider: str) -> tuple[bool, str]:
    """Check if provider credentials are available.
    
    Returns:
        (is_available, env_var_name)
    """
    if provider == 'azure':
        # For Azure, check if settings.yaml exists and has LLM config
        try:
            import yaml
            with open("config/settings.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            if 'llm' in config and config['llm'].get('provider') == 'azure':
                return True, 'settings.yaml'
        except:
            pass
        return False, 'settings.yaml'
        
    if provider == 'openai':
        env_var = 'OPENAI_API_KEY'
        return os.getenv(env_var) is not None, env_var
        
    elif provider == 'ollama':
        # Ollama assumed available if base_url is set or default
        return True, 'OLLAMA_BASE_URL'
        
    return False, ''


# Test Cases

@pytest.mark.integration
@pytest.mark.parametrize("provider,env_var", [
    ("openai", "OPENAI_API_KEY"),
    ("azure", "AZURE_OPENAI_API_KEY"),
    ("ollama", "OLLAMA_BASE_URL"),
])
def test_multiple_providers_if_available(provider, env_var, sample_noisy_chunk):
    """Test refinement with multiple LLM providers (if configured).
    
    This test will be skipped for providers without credentials.
    """
    available, env_name = is_provider_available(provider)
    
    if not available:
        pytest.skip(f"Skipping {provider} test: {env_name} not set")
    
    # Create settings for provider
    settings = create_settings_for_provider(provider)
    
    # Create refiner
    refiner = ChunkRefiner(settings)
    trace = TraceContext()
    
    # Perform refinement
    result = refiner.transform([sample_noisy_chunk], trace=trace)
    
    # Assertions
    assert len(result) == 1
    refined_chunk = result[0]
    
    # Should be marked as LLM-refined
    assert refined_chunk.metadata['refined_by'] == 'llm', \
        f"Expected LLM refinement for {provider}, got {refined_chunk.metadata.get('refined_by')}"
    
    # Refined text should be cleaner (no separator lines, no HTML)
    assert '────────────' not in refined_chunk.text
    assert '<!-- ' not in refined_chunk.text
    assert '<div' not in refined_chunk.text
    assert 'Footer:' not in refined_chunk.text.lower()
    
    # Should preserve key content
    for keyword in EXPECTED_CLEAN_RESULT_KEYWORDS:
        assert keyword in refined_chunk.text, \
            f"Expected keyword '{keyword}' not found in refined text"
    
    # Trace should record LLM usage
    stage_data = trace.get_stage_data('chunk_refiner')
    assert stage_data is not None
    assert stage_data['data']['llm_enhanced_count'] == 1
    assert stage_data['data']['fallback_count'] == 0
    
    # Print for manual review
    print(f"\n{'='*60}")
    print(f"Provider: {provider}")
    print(f"{'='*60}")
    print("ORIGINAL TEXT (first 200 chars):")
    print(sample_noisy_chunk.text[:200])
    print(f"\n{'-'*60}")
    print("REFINED TEXT:")
    print(refined_chunk.text)
    print(f"{'='*60}\n")


@pytest.mark.integration
def test_graceful_fallback_with_invalid_model(sample_noisy_chunk):
    """Test that refiner falls back to rule-based when LLM fails."""
    # Create settings with intentionally invalid model
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': True}
    settings.llm = Mock()
    settings.llm.provider = 'openai'
    settings.llm.model = 'nonexistent-model-xyz'
    settings.llm.api_key = os.getenv('OPENAI_API_KEY', 'fake-key')
    
    refiner = ChunkRefiner(settings)
    trace = TraceContext()
    
    # Should not crash, should fallback
    result = refiner.transform([sample_noisy_chunk], trace=trace)
    
    assert len(result) == 1
    # Should fallback to rule-based
    assert result[0].metadata['refined_by'] == 'rule'
    assert 'refine_fallback_reason' in result[0].metadata
    
    # Should still apply rule-based cleaning
    assert '────────────' not in result[0].text
    assert 'Footer:' not in result[0].text


@pytest.mark.integration
@pytest.mark.skipif(not is_provider_available('openai')[0], reason="OPENAI_API_KEY not set")
def test_refinement_quality_comparison(sample_noisy_chunk):
    """Compare rule-based vs LLM refinement quality.
    
    This test provides visual comparison for manual quality assessment.
    """
    # Rule-based only
    settings_rule = Mock(spec=Settings)
    settings_rule.ingestion = Mock()
    settings_rule.ingestion.chunk_refiner = {'use_llm': False}
    
    refiner_rule = ChunkRefiner(settings_rule)
    result_rule = refiner_rule.transform([sample_noisy_chunk])
    
    # LLM-enhanced
    settings_llm = create_settings_for_provider('openai')
    refiner_llm = ChunkRefiner(settings_llm)
    result_llm = refiner_llm.transform([sample_noisy_chunk])
    
    # Print comparison
    print(f"\n{'='*60}")
    print("QUALITY COMPARISON")
    print(f"{'='*60}")
    print("\nORIGINAL (first 300 chars):")
    print(sample_noisy_chunk.text[:300])
    print(f"\n{'-'*60}")
    print("RULE-BASED REFINEMENT:")
    print(result_rule[0].text)
    print(f"\n{'-'*60}")
    print("LLM-ENHANCED REFINEMENT:")
    print(result_llm[0].text)
    print(f"{'='*60}\n")
    
    # Basic assertions
    assert len(result_rule[0].text) > 0
    assert len(result_llm[0].text) > 0
    assert result_rule[0].metadata['refined_by'] == 'rule'
    assert result_llm[0].metadata['refined_by'] == 'llm'


@pytest.mark.integration
@pytest.mark.skipif(not is_provider_available('openai')[0], reason="OPENAI_API_KEY not set")
def test_batch_refinement_performance(sample_noisy_chunk):
    """Test refining multiple chunks in a batch."""
    settings = create_settings_for_provider('openai')
    refiner = ChunkRefiner(settings)
    trace = TraceContext()
    
    # Create 3 test chunks
    chunks = [
        sample_noisy_chunk,
        Chunk(
            id="chunk_002",
            text="Another  chunk   with   noise\n\n\n\nand issues.",
            metadata={"source": "test2.pdf", "source_path": "test2.pdf"}
        ),
        Chunk(
            id="chunk_003",
            text="Clean chunk without much noise.",
            metadata={"source": "test3.pdf", "source_path": "test3.pdf"}
        )
    ]
    
    # Refine all
    import time
    start_time = time.time()
    result = refiner.transform(chunks, trace=trace)
    elapsed_time = time.time() - start_time
    
    # Assertions
    assert len(result) == 3
    assert all(r.metadata['refined_by'] == 'llm' for r in result)
    
    # Trace
    stage_data = trace.get_stage_data('chunk_refiner')
    assert stage_data['data']['llm_enhanced_count'] == 3
    
    print(f"\n{'='*60}")
    print(f"Refined {len(chunks)} chunks in {elapsed_time:.2f} seconds")
    print(f"Average: {elapsed_time/len(chunks):.2f} seconds per chunk")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
