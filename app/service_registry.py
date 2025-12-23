from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, Optional, TYPE_CHECKING, Type, TypeVar, overload, Literal, cast
from pydantic import BaseModel
from enum import StrEnum
from yorgan.services import ParseService, StructuredOutputService, ParseExtractService
from yorgan.services.registry import ServiceRegistries

if TYPE_CHECKING:
    from yorgan.cache import BaseCache
    from yorgan.services.landingai import AgenticDocParseService


T = TypeVar("T", bound=BaseModel)

_registy = ServiceRegistries()


ParseServiceOptions = StrEnum("ParseServiceOptions", " ".join(_registy.parse.list_modules()))
ExctractServiceOptions = StrEnum("ExctractServiceOptions", " ".join(_registy.structured.list_modules()))
ParseExctractServiceOptions = StrEnum("ParseExctractServiceOptions", " ".join(_registy.parse_extract.list_modules()))


@overload
def get_parse_service(
    option: Literal["landingai", "auto"], cache: Optional[BaseCache] = None
) -> AgenticDocParseService: ...


@overload
def get_parse_service(
    option: ParseServiceOptions, cache: Optional[BaseCache] = None
) -> ParseService: ...


@lru_cache(maxsize=10)
def get_parse_service(
        option: ParseServiceOptions,
        cache: Optional[BaseCache] = None,
        **kwargs: Any
) -> ParseService:
    # gets mapping of module: list of service classes, curretnly we only have one per module but that might change
    _parse_registry = _registy.parse.get_module_service_map()

    if option not in _parse_registry:
        raise ValueError(f"Unknown Parse service or service can't load: {option}")

    parse_service_class = _parse_registry[option][0]
    return parse_service_class(cache=cache, **kwargs)  # type: ignore


@lru_cache(maxsize=10)
def get_extract_service(
    option: ExctractServiceOptions,
    response_type: Type[T],
    cache: Optional[BaseCache] = None,
    **kwargs: Any
) -> StructuredOutputService[T]:
    _extract_registry = _registy.structured.get_module_service_map()

    if option not in _extract_registry:
        raise ValueError(f"Unknown Extract service or service can't load: {option}")

    extract_service_class = _extract_registry[option][0]
    return extract_service_class(response_type=response_type, cache=cache, **kwargs)


@lru_cache(maxsize=10)
def get_parse_extract_service(
        option: ParseExctractServiceOptions,
        response_type: Type[T],
        cache: Optional[BaseCache] = None,
        **kwargs: Any
) -> ParseExtractService[T]:
    _parse_extract_registry = _registy.parse_extract.get_module_service_map()

    if option not in _parse_extract_registry:
        raise ValueError(f"Unknown Extract service or service can't load: {option}")

    parse_extract_service = _parse_extract_registry[option][0]
    return parse_extract_service(response_type=response_type, cache=cache, **kwargs)
