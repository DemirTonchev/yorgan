from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import importlib
from pathlib import Path
from typing import Any, Optional, TypeVar, Generic, Union, cast
import inspect
from .base import BaseService, ParseService, LLMStructuredOutputService, ParseExtractService

T = TypeVar("T", bound=BaseService)


@dataclass
class RegisteredServiceInfo(Generic[T]):
    module: str
    service: type[T]


class Registry(Generic[T]):
    """Registry for Service subclasses with dynamic discovery"""

    def __init__(self) -> None:
        self._registry: dict[str, RegisteredServiceInfo[T]] = {}

    def register(self, service_class: type[T], name: Optional[str] = None) -> None:
        """
        Register a service class.
        """
        # we expect all services to be inherited from BaseService and have service_name but __qualname__ allows protocol class
        service_name = name or getattr(service_class, 'service_name', service_class.__qualname__)

        self._registry[service_name] = RegisteredServiceInfo(
            module=service_class.__module__.rsplit('.', 1)[-1],
            service=service_class
        )

    def list_modules(self) -> list[str]:
        """
        Get a list of all module names for registered services.

        Returns:
            list[str]: A list of loaded module names (e.g., ['gemini', 'landing', 'openai']) extracted from
                      the registered service classes. Module names are derived from the
                      last component of the service's __module__ attribute.

            Example:
                >>> registry.list_modules()
                ['gemini', 'landing', 'openai']
        """
        return [v.module for v in self._registry.values()]

    @lru_cache(maxsize=3)
    def get_module_members(self, module: str) -> list[RegisteredServiceInfo[T]]:
        return [v for v in self._registry.values() if v.module == module]

    @lru_cache(maxsize=1)
    def get_module_memebers_map(self) -> dict[str, list[RegisteredServiceInfo[T]]]:
        registered_modules = self.list_modules()
        return {mod: self.get_module_members(mod) for mod in registered_modules}

    @lru_cache(maxsize=1)
    def get_module_service_map(self) -> dict[str, list[type[T]]]:
        registered_modules = self.list_modules()
        return {mod: [member.service for member in self.get_module_members(mod)] for mod in registered_modules}

    def get(self, service_name: str) -> RegisteredServiceInfo[T]:
        """Get a registered service by name"""
        return self._registry[service_name]

    def get_all(self) -> dict[str, RegisteredServiceInfo[T]]:
        """Get all registered services"""
        return self._registry.copy()

    def list_services(self) -> list[str]:
        """List all registered service names"""
        return list(self._registry.keys())

    def clear(self) -> None:
        """Clear all registrations"""
        self._registry.clear()

    def _is_valid_service(self, obj: Any, service_class: type[T]) -> bool:
        if inspect.isabstract(obj):
            return False

        if obj is service_class:
            return False

        if not inspect.isclass(obj) or not issubclass(obj, service_class):
            return False

        return True

    def discover_services(
        self,
        package_path: str,
        service_class: type[T],
    ) -> None:
        try:
            module = importlib.import_module(package_path)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_valid_service(obj, service_class):
                    typed_obj = cast(type[T], obj)
                    self.register(typed_obj)

        except ImportError:
            pass

    def discover_local_services(self, service_class: type[T], base_path: Optional[Path] = None) -> None:
        """Auto-discover services from local modules in the same directory"""
        if base_path is None:
            base_path = Path(__file__).parent

        for file_path in base_path.glob("*.py"):
            if file_path.stem in ('__init__', 'registry', 'base', 'multipage', 'utils'):
                continue

            module_name = f"{self.__module__.rsplit('.', 1)[0]}.{file_path.stem}"
            self.discover_services(package_path=module_name, service_class=service_class)

    def __repr__(self) -> str:
        """
        Returns a nice string showing the current state.

        Example:
            <Registry services=['GeminiParseService', 'AgenticDocParseService'>
        """
        keys = list(self._registry.keys())
        return f"<{self.__class__.__name__} services={keys}>"


AnyServiceRegistry = Union[
    Registry[ParseService],
    Registry[LLMStructuredOutputService],
    Registry[ParseExtractService]
]


class ServiceRegistries:
    """
    Central manager for all service registries.
    """

    def __init__(self) -> None:
        self._parse: Optional[Registry[ParseService]] = None
        self._structured: Optional[Registry[LLMStructuredOutputService]] = None
        self._parse_extract: Optional[Registry[ParseExtractService]] = None

    def _load_registry(
        self,
        current: Optional[Registry[T]],
        service_class: type[T]
    ) -> Registry[T]:
        """
        Generic helper to lazy-load a registry if it doesn't exist.
        """
        if current is not None:
            return current

        registry = Registry[T]()  # type: ignore
        registry.discover_local_services(service_class=service_class)
        return registry

    @property
    def parse(self) -> Registry[ParseService]:
        """Get or create the parse registry"""
        self._parse = self._load_registry(self._parse, ParseService)
        return self._parse

    @property
    def structured(self) -> Registry[LLMStructuredOutputService]:
        """Get or create the structured output registry"""
        self._structured = self._load_registry(self._structured, LLMStructuredOutputService)
        return self._structured

    @property
    def parse_extract(self) -> Registry[ParseExtractService]:
        """Get or create the parse extract registry"""
        self._extract = self._load_registry(self._parse_extract, ParseExtractService)
        return self._extract

    def initialize_all(self) -> None:
        """
        Force initialization of all registries at once.
        Useful for eager loading during application startup to catch import errors early.
        """
        _ = self.parse
        _ = self.structured
        _ = self.parse_extract

    def clear_all(self) -> None:
        """
        Clear and nullify all registries.
        Useful for testing to force re-discovery on next access.
        """
        if self._parse:
            self._parse.clear()
        if self._structured:
            self._structured.clear()
        if self._extract:
            self._extract.clear()

        self._parse = None
        self._structured = None
        self._extract = None

    def get_all(self) -> dict[str, AnyServiceRegistry]:
        """Get all initialized registries as a dictionary."""
        return {
            "parse": self.parse,
            "structured": self.structured,
            "parse_extract": self.parse_extract,
        }

    def list_all_services(self) -> dict[str, list[str]]:
        """List all services across all registries."""
        return {
            name: registry.list_services()
            for name, registry in self.get_all().items()
        }

    def __repr__(self) -> str:
        parts: list[str] = []
        for name, reg in (
            ("parse", self.parse),
            ("structured", self.structured),
            ("parse_extract", self.parse_extract),
        ):
            parts.append(f"{name}={repr(reg)}")

        return f"<{self.__class__.__name__} with {' '.join(parts)}>"


@lru_cache(maxsize=1)
def get_registries() -> ServiceRegistries:
    """
    Get the global ServiceRegistries instance.
    Uses lru_cache to guarantee a singleton pattern without global variables.
    """
    return ServiceRegistries()


def get_parse_registry() -> Registry[ParseService]:
    return get_registries().parse


def get_structured_registry() -> Registry[LLMStructuredOutputService]:
    return get_registries().structured


def get_extract_registry() -> Registry[ParseExtractService]:
    return get_registries().parse_extract
