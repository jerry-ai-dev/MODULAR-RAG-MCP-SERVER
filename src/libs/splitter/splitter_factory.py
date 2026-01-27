"""Factory for creating Splitter instances.

This module implements the Factory Pattern to instantiate the appropriate
Splitter provider based on configuration, enabling configuration-driven selection
of different splitting strategies without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from src.core.settings import Settings


class SplitterFactory:
    """Factory for creating Splitter provider instances.
    
    This factory reads the splitter configuration from settings and instantiates
    the corresponding Splitter implementation. Supported providers will be added
    in subsequent tasks (B7.5).
    
    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    """
    
    _PROVIDERS: dict[str, type[BaseSplitter]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseSplitter]) -> None:
        """Register a new Splitter provider implementation.
        
        Args:
            name: The provider identifier (e.g., 'recursive', 'semantic', 'fixed').
            provider_class: The BaseSplitter subclass implementing the provider.
        
        Raises:
            ValueError: If provider_class doesn't inherit from BaseSplitter.
        """
        if not issubclass(provider_class, BaseSplitter):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseSplitter"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseSplitter:
        """Create a Splitter instance based on configuration.
        
        Args:
            settings: The application settings containing ingestion configuration.
            **override_kwargs: Optional parameters to override config values.
        
        Returns:
            An instance of the configured Splitter provider.
        
        Raises:
            ValueError: If the configured provider is not supported or missing.
        """
        try:
            ingestion_settings = settings.ingestion
            if ingestion_settings is None:
                raise AttributeError("settings.ingestion is None")
            provider_name = ingestion_settings.splitter.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.ingestion.splitter. "
                "Please ensure 'ingestion.splitter' is specified in settings.yaml"
            ) from e
        
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Splitter provider: '{provider_name}'. "
                f"Available providers: {available}. "
                "Provider implementations will be added in task B7.5."
            )
        
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Splitter provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.
        
        Returns:
            Sorted list of available provider identifiers.
        """
        return sorted(cls._PROVIDERS.keys())