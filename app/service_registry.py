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
def get_parse_service(option: ParseServiceOptions, cache: Optional[BaseCache] = None) -> ParseService:

    # gets mapping of module: list of service classes, curretnly we only have one per module but that might change
    _parse_registry = _registy.parse.get_module_service_map()
    if option == "auto":  # TODO Remove auto
        option = "landingai"  # type: ignore
    if option not in _parse_registry:
        raise ValueError(f"Unknown Parse service or service can't load: {option}")

    parse_service = _parse_registry[option][0]

    if option in ["landingai"]:
        return parse_service(cache=cache)  # type: ignore
    else:
        # model = ..... we need to inject model from settings or endpoint?
        # return parse_service(model=model, cache=cache)
        return parse_service(cache=cache)  # type: ignore


@lru_cache(maxsize=10)
def get_extract_service(
    option: ExctractServiceOptions,
    response_type: Type[T],
    cache: Optional[BaseCache] = None
) -> StructuredOutputService[T]:

    _extract_registry = _registy.structured.get_module_service_map()
    if option == "auto":  # TODO Remove auto
        option = "gemini"  # type: ignore
    if option not in _extract_registry:
        raise ValueError(f"Unknown Extract service or service can't load: {option}")

    extract_service = _extract_registry[option][0]
    # TODO
    # if moded ...# model = ..... we need to inject model from settings or endpoint?
    # extract_service(response_type=response_type, cache=cache, model = model)

    return extract_service(response_type=response_type, cache=cache)


def get_parse_extract_service(
        option: ParseExctractServiceOptions,
        response_type: Type[T],
        cache: Optional[BaseCache] = None
) -> ParseExtractService[T]:

    _parse_extract_registry = _registy.parse_extract.get_module_service_map()

    if option not in _parse_extract_registry:
        raise ValueError(f"Unknown Extract service or service can't load: {option}")

    parse_extract_service = _parse_extract_registry[option][0]
    # TODO
    # if moded ...# model = ..... we need to inject model from settings or endpoint?
    # extract_service(response_type=response_type, cache=cache, model = model)

    return parse_extract_service(response_type=response_type, cache=cache)
