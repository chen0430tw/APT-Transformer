#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Registry - Provider Registration and Management System

This module implements the core Provider pattern and Registry system for APT's
microkernel architecture. It enables dynamic loading, fallback mechanisms, and
plugin extensibility.

Key Features:
- Dynamic provider registration and lookup
- Automatic fallback to default implementations
- Singleton pattern for instances
- Dependency checking and conflict detection
- Version management
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional, List, Callable
import warnings
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Provider Base Class
# ============================================================================

class Provider(ABC):
    """
    Base class for all Providers in the APT system.

    A Provider is a factory that creates instances of a specific component type
    (e.g., attention layers, FFN layers, routers, etc.). Providers enable
    swappable implementations and plugin architecture.

    Subclasses must implement:
    - get_name(): Return the provider's unique name
    - get_version(): Return the semantic version string

    Optional overrides:
    - validate_config(): Validate configuration before instantiation
    - get_dependencies(): List required dependencies
    - get_priority(): Return priority for automatic selection
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return the unique name of this provider."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Return the semantic version string (e.g., '1.0.0')."""
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the configuration before instantiation.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid, False otherwise
        """
        return True

    def get_dependencies(self) -> List[str]:
        """
        Return list of required dependencies.

        Returns:
            List of dependency strings (e.g., ['torch>=1.9.0', 'flash-attn>=2.0'])
        """
        return []

    def get_priority(self) -> int:
        """
        Return priority for automatic provider selection (higher = preferred).

        Returns:
            Priority integer (default: 0)
        """
        return 0

    def __repr__(self):
        return f"{self.__class__.__name__}({self.get_name()}, v{self.get_version()})"


# ============================================================================
# Registry Implementation
# ============================================================================

class Registry:
    """
    Global Provider Registry for APT's microkernel architecture.

    The Registry manages all Provider implementations and provides:
    - Registration of new providers
    - Lazy instantiation with singleton pattern
    - Automatic fallback to default implementations
    - Dependency and conflict checking
    - Version management

    Usage:
        # Register a provider
        registry.register('attention', 'tva_default', TVAAttention, default=True)

        # Get a provider instance
        provider = registry.get('attention', 'tva_default', config)

        # Create component using provider
        layer = provider.create_layer(d_model=768, num_heads=12)
    """

    def __init__(self):
        """Initialize the registry with empty collections."""
        # Storage for provider classes: {kind: {name: cls}}
        self._providers: Dict[str, Dict[str, Type[Provider]]] = {}

        # Storage for singleton instances: {kind:name: instance}
        self._instances: Dict[str, Provider] = {}

        # Default provider names for each kind
        self._defaults: Dict[str, str] = {}

        # Exclusion rules: {provider_key: [excluded_keys]}
        self._exclusions: Dict[str, List[str]] = {}

        # Initialization hooks: {kind: [callback]}
        self._init_hooks: Dict[str, List[Callable]] = {}

        # Track enabled providers for conflict checking
        self._enabled: List[str] = []

        logger.info("Registry initialized")

    def register(
        self,
        kind: str,
        name: str,
        provider_cls: Type[Provider],
        default: bool = False,
        excludes: Optional[List[str]] = None,
        override: bool = False
    ) -> None:
        """
        Register a provider implementation.

        Args:
            kind: Provider category (e.g., 'attention', 'ffn', 'router')
            name: Unique name within the category (e.g., 'tva_default', 'flash_v2')
            provider_cls: Provider class (must inherit from Provider)
            default: If True, set as default for this kind
            excludes: List of provider names that conflict with this one
            override: If True, allow overriding existing registrations

        Raises:
            TypeError: If provider_cls doesn't inherit from Provider
            ValueError: If provider already registered and override=False
        """
        # Validate provider class
        if not issubclass(provider_cls, Provider):
            raise TypeError(
                f"{provider_cls.__name__} must inherit from Provider"
            )

        # Initialize kind if needed
        if kind not in self._providers:
            self._providers[kind] = {}

        # Check for duplicate registration
        key = f"{kind}:{name}"
        if name in self._providers[kind] and not override:
            warnings.warn(
                f"Provider {key} already registered. Use override=True to replace.",
                UserWarning
            )
            return

        # Register the provider
        self._providers[kind][name] = provider_cls

        # Create temporary instance for logging version info
        try:
            temp_instance = provider_cls({})
            version = temp_instance.get_version()
            logger.info(f"Registered {key} (v{version})")
        except Exception as e:
            logger.warning(f"Registered {key} (version unavailable: {e})")

        # Set as default if requested
        if default or kind not in self._defaults:
            self._defaults[kind] = name
            logger.info(f"Set {key} as default for '{kind}'")

        # Register exclusions
        if excludes:
            full_excludes = [f"{kind}:{e}" for e in excludes]
            self._exclusions[key] = full_excludes
            logger.debug(f"Exclusions for {key}: {full_excludes}")

    def get(
        self,
        kind: str,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        fallback: bool = True
    ) -> Provider:
        """
        Get or create a provider instance (singleton pattern).

        Args:
            kind: Provider category
            name: Provider name
            config: Configuration dictionary
            fallback: If True, fallback to default on failure

        Returns:
            Provider instance

        Raises:
            ValueError: If provider not found and fallback disabled
            RuntimeError: If provider creation fails and fallback exhausted
        """
        key = f"{kind}:{name}"
        config = config or {}

        # Return cached instance if available
        if key in self._instances:
            logger.debug(f"Returning cached instance: {key}")
            return self._instances[key]

        # Find provider class
        provider_cls = self._find_provider_class(kind, name)

        if provider_cls is None:
            if fallback:
                return self._fallback_to_default(kind, config)
            else:
                raise ValueError(f"Provider {key} not registered")

        # Create instance
        try:
            instance = self._create_instance(provider_cls, key, config)

            # Track as enabled
            if key not in self._enabled:
                self._enabled.append(key)

            return instance

        except Exception as e:
            logger.error(f"Failed to create {key}: {e}", exc_info=True)

            if fallback and name != self._defaults.get(kind):
                logger.warning(f"Attempting fallback for {key}")
                return self._fallback_to_default(kind, config)
            else:
                raise RuntimeError(f"Failed to create {key}: {e}")

    def _find_provider_class(
        self,
        kind: str,
        name: str
    ) -> Optional[Type[Provider]]:
        """Find provider class in registry."""
        if kind in self._providers and name in self._providers[kind]:
            return self._providers[kind][name]
        return None

    def _create_instance(
        self,
        provider_cls: Type[Provider],
        key: str,
        config: Dict[str, Any]
    ) -> Provider:
        """Create and validate provider instance."""
        # Instantiate
        instance = provider_cls(config)

        # Validate configuration
        if not instance.validate_config(config):
            raise ValueError(f"Configuration validation failed for {key}")

        # Check dependencies (warn only, don't fail)
        deps = instance.get_dependencies()
        if deps:
            logger.debug(f"Dependencies for {key}: {deps}")
            # TODO: Implement actual dependency checking

        # Cache instance
        self._instances[key] = instance
        logger.info(f"Created instance: {key}")

        # Run init hooks
        kind = key.split(':')[0]
        if kind in self._init_hooks:
            for hook in self._init_hooks[kind]:
                try:
                    hook(instance)
                except Exception as e:
                    logger.warning(f"Init hook failed for {key}: {e}")

        return instance

    def _fallback_to_default(
        self,
        kind: str,
        config: Dict[str, Any]
    ) -> Provider:
        """Fallback to default provider for the given kind."""
        default_name = self._defaults.get(kind)

        if default_name is None:
            raise ValueError(f"No default provider for kind '{kind}'")

        default_key = f"{kind}:{default_name}"

        if default_key in self._instances:
            logger.info(f"Using cached default: {default_key}")
            return self._instances[default_key]

        logger.warning(f"Falling back to default: {default_key}")
        return self.get(kind, default_name, config, fallback=False)

    def list_providers(
        self,
        kind: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        List all registered providers.

        Args:
            kind: If specified, only list providers of this kind

        Returns:
            Dictionary mapping kind to list of provider names
        """
        if kind:
            return {kind: list(self._providers.get(kind, {}).keys())}
        else:
            return {k: list(v.keys()) for k, v in self._providers.items()}

    def get_info(self, kind: str, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a provider.

        Args:
            kind: Provider category
            name: Provider name

        Returns:
            Dictionary with provider metadata

        Raises:
            ValueError: If provider not found
        """
        key = f"{kind}:{name}"
        provider_cls = self._find_provider_class(kind, name)

        if provider_cls is None:
            raise ValueError(f"Provider {key} not found")

        # Create temporary instance for metadata
        try:
            temp = provider_cls({})
            return {
                'name': name,
                'kind': kind,
                'version': temp.get_version(),
                'class': provider_cls.__name__,
                'module': provider_cls.__module__,
                'dependencies': temp.get_dependencies(),
                'priority': temp.get_priority(),
                'is_default': self._defaults.get(kind) == name,
                'excludes': self._exclusions.get(key, []),
                'is_cached': key in self._instances
            }
        except Exception as e:
            logger.error(f"Failed to get info for {key}: {e}")
            return {
                'name': name,
                'kind': kind,
                'error': str(e)
            }

    def check_conflicts(
        self,
        providers: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Check for conflicts in provider list.

        Args:
            providers: List of provider keys ('kind:name'). If None, check enabled.

        Returns:
            Conflict description string, or None if no conflicts
        """
        check_list = providers if providers is not None else self._enabled

        for p1 in check_list:
            if p1 in self._exclusions:
                for p2 in self._exclusions[p1]:
                    if p2 in check_list:
                        return f"Conflict: {p1} excludes {p2}"

        return None

    def register_init_hook(self, kind: str, callback: Callable) -> None:
        """
        Register a hook to be called after provider instantiation.

        Args:
            kind: Provider category to hook
            callback: Function(provider_instance) to call after init
        """
        if kind not in self._init_hooks:
            self._init_hooks[kind] = []
        self._init_hooks[kind].append(callback)
        logger.debug(f"Registered init hook for '{kind}'")

    def clear_cache(self, kind: Optional[str] = None, name: Optional[str] = None):
        """
        Clear cached provider instances.

        Args:
            kind: If specified, only clear this kind
            name: If specified with kind, only clear this specific provider
        """
        if kind and name:
            key = f"{kind}:{name}"
            if key in self._instances:
                del self._instances[key]
                logger.info(f"Cleared cache for {key}")
        elif kind:
            keys_to_remove = [k for k in self._instances if k.startswith(f"{kind}:")]
            for key in keys_to_remove:
                del self._instances[key]
            logger.info(f"Cleared {len(keys_to_remove)} cached instances for '{kind}'")
        else:
            count = len(self._instances)
            self._instances.clear()
            logger.info(f"Cleared all {count} cached instances")

    def set_default(self, kind: str, name: str) -> None:
        """
        Set the default provider for a kind.

        Args:
            kind: Provider category
            name: Provider name to set as default

        Raises:
            ValueError: If provider not registered
        """
        if kind not in self._providers or name not in self._providers[kind]:
            raise ValueError(f"Provider {kind}:{name} not registered")

        old_default = self._defaults.get(kind)
        self._defaults[kind] = name
        logger.info(f"Changed default for '{kind}': {old_default} -> {name}")


# ============================================================================
# Global Registry Singleton
# ============================================================================

# Global registry instance - import this in other modules
registry = Registry()


# ============================================================================
# Helper Functions
# ============================================================================

def register_provider(
    kind: str,
    name: str,
    default: bool = False,
    excludes: Optional[List[str]] = None
):
    """
    Decorator for registering a provider class.

    Usage:
        @register_provider('attention', 'tva_default', default=True)
        class TVAAttention(Provider):
            ...

    Args:
        kind: Provider category
        name: Provider name
        default: If True, set as default
        excludes: List of conflicting providers
    """
    def decorator(cls):
        registry.register(kind, name, cls, default=default, excludes=excludes)
        return cls
    return decorator


def get_provider(
    kind: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Provider:
    """
    Convenience function to get a provider.

    Args:
        kind: Provider category
        name: Provider name (if None, uses default)
        config: Configuration dictionary

    Returns:
        Provider instance
    """
    if name is None:
        name = registry._defaults.get(kind)
        if name is None:
            raise ValueError(f"No default provider for kind '{kind}'")

    return registry.get(kind, name, config)
