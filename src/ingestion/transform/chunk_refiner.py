"""Chunk refinement transform: rule-based cleaning + optional LLM enhancement."""

import re
from pathlib import Path
from typing import List, Optional

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import BaseLLM, Message
from src.observability.logger import get_logger

logger = get_logger(__name__)


class ChunkRefiner(BaseTransform):
    """Refines chunks through rule-based cleaning and optional LLM enhancement.
    
    Processing Pipeline:
        1. Rule-based refine: Remove noise (whitespace, headers/footers, HTML)
        2. (Optional) LLM refine: Intelligent content improvement
        3. On LLM failure: Gracefully fallback to rule-based result
    
    Configuration (via settings.yaml):
        - ingestion.chunk_refiner.use_llm: bool - Enable LLM enhancement
        - ingestion.chunk_refiner.prompt_path: str - Custom prompt file path
    
    Design Principles:
        - Graceful Degradation: LLM errors don't block ingestion
        - Atomic Processing: Each chunk processed independently
        - Observable: Records refined_by in metadata
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ):
        """Initialize ChunkRefiner.
        
        Args:
            settings: Application settings
            llm: Optional LLM instance (for testing; auto-created if None)
            prompt_path: Optional custom prompt file path
        """
        self.settings = settings
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._prompt_path = prompt_path or "config/prompts/chunk_refinement.txt"
        
        # Determine if LLM should be used
        self.use_llm = getattr(
            getattr(settings, 'ingestion', None), 
            'chunk_refiner', 
            {}
        ).get('use_llm', False) if hasattr(settings, 'ingestion') else False
        
    @property
    def llm(self) -> Optional[BaseLLM]:
        """Lazy-load LLM instance."""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for chunk refinement")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based only.")
                self.use_llm = False
        return self._llm
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Transform chunks through refinement pipeline.
        
        Args:
            chunks: List of chunks to refine
            trace: Optional trace context
            
        Returns:
            List of refined chunks (same length as input)
        """
        if not chunks:
            return []
        
        refined_chunks = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0
        
        for chunk in chunks:
            try:
                # Step 1: Rule-based refinement (always performed)
                rule_refined_text = self._rule_based_refine(chunk.text)
                
                # Step 2: Optional LLM enhancement
                if self.use_llm and self.llm:
                    llm_refined_text = self._llm_refine(rule_refined_text, trace)
                    
                    if llm_refined_text:
                        # LLM success
                        refined_text = llm_refined_text
                        refined_by = "llm"
                        llm_enhanced_count += 1
                    else:
                        # LLM failed, fallback to rule-based
                        refined_text = rule_refined_text
                        refined_by = "rule"
                        fallback_count += 1
                        if chunk.metadata:
                            chunk.metadata['refine_fallback_reason'] = "llm_failed"
                else:
                    # LLM disabled, use rule-based
                    refined_text = rule_refined_text
                    refined_by = "rule"
                
                # Create refined chunk
                refined_chunk = Chunk(
                    id=chunk.id,
                    text=refined_text,
                    metadata={
                        **(chunk.metadata or {}),
                        'refined_by': refined_by
                    },
                    source_ref=chunk.source_ref
                )
                refined_chunks.append(refined_chunk)
                success_count += 1
                
            except Exception as e:
                # Atomic failure: log and preserve original
                logger.error(f"Failed to refine chunk {chunk.id}: {e}")
                refined_chunks.append(chunk)
        
        # Record trace
        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm
            })
        
        logger.info(
            f"Refined {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )
        
        return refined_chunks
    
    def _rule_based_refine(self, text: str) -> str:
        """Apply rule-based text cleaning.
        
        Cleaning operations:
            1. Remove page headers/footers (separator lines + metadata)
            2. Remove HTML comments
            3. Remove HTML tags (preserve content)
            4. Normalize excessive whitespace
            5. Preserve code blocks and Markdown formatting
        
        Args:
            text: Raw chunk text
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Early return if only whitespace
        if not text.strip():
            return ""
        
        # Preserve code blocks (extract and restore later)
        code_blocks = []
        code_block_pattern = r'```[\s\S]*?```'
        
        def extract_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        text = re.sub(code_block_pattern, extract_code_block, text)
        
        # 1. Remove separator lines with page numbers/footers
        # Pattern: ────────────────
        # Followed by: Page XX, Footer text, etc.
        text = re.sub(
            r'─{10,}.*?(?:Page \d+|Footer|Section \d+|©|Confidential).*?─{10,}',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(r'─{10,}', '', text)  # Remove remaining separator lines
        
        # 2. Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # 3. Remove HTML tags (but preserve content)
        text = re.sub(r'<[^>]+>', '', text)
        
        # 4. Normalize whitespace
        # - Collapse multiple spaces to single space
        text = re.sub(r' {2,}', ' ', text)
        
        # - Collapse 3+ consecutive newlines to 2 (preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 5. Remove leading/trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # 6. Restore code blocks
        for i, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", code_block)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def _llm_refine(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """Apply LLM-based intelligent refinement.
        
        Args:
            text: Rule-refined text
            trace: Optional trace context
            
        Returns:
            LLM-refined text, or None if refinement failed
        """
        if not text or not text.strip():
            return text
        
        try:
            # Load prompt template
            prompt_template = self._load_prompt()
            if not prompt_template:
                logger.warning("Prompt template not found, skipping LLM refinement")
                return None
            
            # Fill prompt
            if '{text}' not in prompt_template:
                logger.error("Prompt template missing {text} placeholder")
                return None
            
            prompt = prompt_template.replace('{text}', text)
            
            # Call LLM with Message objects
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, trace=trace)
            
            # Extract text from ChatResponse
            if isinstance(response, str):
                refined_text = response
            else:
                # response is ChatResponse object
                refined_text = response.content
            
            if refined_text and refined_text.strip():
                return refined_text.strip()
            else:
                logger.warning("LLM returned empty result")
                return None
                
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
            return None
    
    def _load_prompt(self) -> Optional[str]:
        """Load prompt template from file.
        
        Returns:
            Prompt template string, or None if file not found
        """
        if self._prompt_template is not None:
            return self._prompt_template
        
        try:
            prompt_path = Path(self._prompt_path)
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found: {self._prompt_path}")
                return None
            
            self._prompt_template = prompt_path.read_text(encoding='utf-8')
            logger.debug(f"Loaded prompt template from {self._prompt_path}")
            return self._prompt_template
            
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            return None
