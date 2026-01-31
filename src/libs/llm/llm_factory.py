"""Factory for creating LLM provider instances.

This module implements the Factory Pattern to instantiate the appropriate
LLM provider based on configuration, enabling configuration-driven selection
of different backends without code changes.

Supports both text-only LLMs and Vision LLMs (multimodal).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.base_vision_llm import BaseVisionLLM

if TYPE_CHECKING:
    from src.core.settings import Settings


# Import and register Vision LLM providers at module load time
def _register_vision_providers() -> None:
    """Register all Vision LLM provider implementations.
    
    This function is called at module import time to populate the
    Vision LLM provider registry. Add new providers here as they
    are implemented.
    """
    try:
        from src.libs.llm.azure_vision_llm import AzureVisionLLM
        from src.libs.llm.llm_factory import LLMFactory
        LLMFactory.register_vision_provider("azure", AzureVisionLLM)
    except ImportError:
        # Provider not yet implemented, skip registration
        pass


class LLMFactory:
    """Factory for creating LLM provider instances.
    
    This factory reads the provider configuration from settings and instantiates
    the corresponding LLM implementation. Supports both text-only LLMs and
    Vision LLMs (multimodal).
    
    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    - Separation: Text and Vision LLM registries are separate.
    """
    
    # Registry of supported text-only LLM providers (to be populated in B7.x tasks)
    _PROVIDERS: dict[str, type[BaseLLM]] = {}
    
    # Registry of supported Vision LLM providers (to be populated in B9+ tasks)
    _VISION_PROVIDERS: dict[str, type[BaseVisionLLM]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLM]) -> None:
        """Register a new LLM provider implementation.
        
        This method allows provider implementations to register themselves
        with the factory, supporting extensibility.
        
        Args:
            name: The provider identifier (e.g., 'openai', 'azure', 'ollama').
            provider_class: The BaseLLM subclass implementing the provider.
        
        Raises:
            ValueError: If provider_class doesn't inherit from BaseLLM.
        """
        if not issubclass(provider_class, BaseLLM):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseLLM"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseLLM:
        """Create an LLM instance based on configuration.
        
        Args:
            settings: The application settings containing LLM configuration.
            **override_kwargs: Optional parameters to override config values.
        
        Returns:
            An instance of the configured LLM provider.
        
        Raises:
            ValueError: If the configured provider is not supported.
            AttributeError: If required configuration fields are missing.
        
        Example:
            >>> settings = Settings.load('config/settings.yaml')
            >>> llm = LLMFactory.create(settings)
            >>> response = llm.chat([Message(role='user', content='Hello')])
        """
        # Extract provider name from settings
        try:
            provider_name = settings.llm.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.llm.provider. "
                "Please ensure 'llm.provider' is specified in settings.yaml"
            ) from e
        
        # Look up provider class in registry
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Provider implementations will be added in tasks B7.1-B7.2."
            )
        
        # Instantiate the provider
        # Provider classes should accept settings and optional kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate LLM provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.
        
        Returns:
            Sorted list of available provider identifiers.
        """
        return sorted(cls._PROVIDERS.keys())
    
    @classmethod
    def register_vision_provider(
        cls,
        name: str,
        provider_class: type[BaseVisionLLM]
    ) -> None:
        """Register a new Vision LLM provider implementation.
        
        This method allows Vision LLM provider implementations to register
        themselves with the factory, supporting extensibility.
        
        Args:
            name: The provider identifier (e.g., 'azure', 'ollama').
            provider_class: The BaseVisionLLM subclass implementing the provider.
        
        Raises:
            ValueError: If provider_class doesn't inherit from BaseVisionLLM.
        """
        if not issubclass(provider_class, BaseVisionLLM):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseVisionLLM"
            )
        cls._VISION_PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create_vision_llm(
        cls,
        settings: Settings,
        **override_kwargs: Any
    ) -> BaseVisionLLM:
        """Create a Vision LLM instance based on configuration.
        
        Vision LLMs support multimodal input (text + image) and are used for
        tasks like image captioning, visual question answering, and document
        understanding with embedded images.
        
        Args:
            settings: The application settings containing Vision LLM configuration.
            **override_kwargs: Optional parameters to override config values.
        
        Returns:
            An instance of the configured Vision LLM provider.
        
        Raises:
            ValueError: If the configured provider is not supported or configuration is missing.
            RuntimeError: If provider instantiation fails.
        
        Example:
            >>> settings = Settings.load('config/settings.yaml')
            >>> vision_llm = LLMFactory.create_vision_llm(settings)
            >>> image = ImageInput(path="diagram.png")
            >>> response = vision_llm.chat_with_image("Describe this", image)
        """
        # Extract provider name from settings
        # Vision LLM config may be nested under settings.vision_llm or settings.llm
        try:
            # Try vision_llm section first
            if hasattr(settings, 'vision_llm') and hasattr(settings.vision_llm, 'provider'):
                provider_name = settings.vision_llm.provider.lower()
            # Fallback to llm.provider (some providers support both text and vision)
            elif hasattr(settings, 'llm') and hasattr(settings.llm, 'provider'):
                provider_name = settings.llm.provider.lower()
            else:
                raise AttributeError("No vision_llm or llm provider configuration found")
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vision_llm.provider or settings.llm.provider. "
                "Please ensure 'vision_llm.provider' or 'llm.provider' is specified in settings.yaml"
            ) from e
        
        # Look up provider class in vision registry
        provider_class = cls._VISION_PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._VISION_PROVIDERS.keys())) if cls._VISION_PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Vision LLM provider: '{provider_name}'. "
                f"Available Vision LLM providers: {available}. "
                f"Vision LLM implementations will be added in tasks B9+."
            )
        
        # Instantiate the provider
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Vision LLM provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_vision_providers(cls) -> list[str]:
        """List all registered Vision LLM provider names.
        
        Returns:
            Sorted list of available Vision LLM provider identifiers.
        """
        return sorted(cls._VISION_PROVIDERS.keys())


# Register Vision LLM providers at module load time
_register_vision_providers()
