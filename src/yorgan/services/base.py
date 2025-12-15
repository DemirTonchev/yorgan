
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel

from yorgan.cache import NullCache, BaseCache, SimpleMemoryCacheWithPersistence, cache_result
from yorgan.datamodels import ParseResponse, Metadata

if TYPE_CHECKING:
    OptionalCacheType = Optional[BaseCache | SimpleMemoryCacheWithPersistence]


class HasMarkdown(Protocol):
    """A protocol for any object that has a 'markdown' string attribute."""
    markdown: str


T = TypeVar('T', bound=BaseModel)
PARSE_T = TypeVar('PARSE_T', bound=ParseResponse)


class BaseService(ABC, Generic[T]):
    """
    Class Abstract base class for all Services that have a response type.
    """

    def __init__(self, response_type: Type[T], cache: OptionalCacheType = None):
        self.response_type = response_type
        self.cache = cache or NullCache()

    def __init_subclass__(subclass):  # type: ignore
        super().__init_subclass__()
        subclass.service_name = subclass.__qualname__


class ParseService(BaseService[PARSE_T]):
    """
    Class for a Service that performs Parsing or Optical Character Recognition (OCR) of a document.
    """

    @abstractmethod
    async def parse(
            self,
            filename: str,
            content: bytes,
            *kwargs,
    ) -> PARSE_T:
        """Processes document content and returns a parsed representation."""
        ...

    # maybe add call method for all services so that they can be used by some other abstraction that
    # does not know about specific methods, ocr, parse.... etc
    # suggestion: if we keep the decorator, change __call__ to something else, for consistency with other services
    @cache_result(key_params=['filename'])
    async def __call__(
            self,
            filename: str,
            content: bytes,
            **kwargs
    ) -> PARSE_T:
        return await self.parse(filename, content, **kwargs)


class LLMParseService(ParseService[PARSE_T]):
    """
    Abstract base for parse services that use an LLM/VLM with a prompt template.
    """
    DEFAULT_PROMPT: str = """\
Your task is to extract the text from the attached document. Format it nicely as a markdown.
Insert the following page break between consecutive pages:

<!-- PAGE BREAK -->

"""

    def __init__(
        self,
        response_type: Type[PARSE_T],
        model: str,
        prompt: Optional[str] = None,
        cache: OptionalCacheType = None,
    ):
        super().__init__(response_type=response_type, cache=cache)
        self.model = model
        self.prompt = prompt if prompt is not None else self.DEFAULT_PROMPT


class StructuredOutputService(BaseService[T]):
    """
    Class for a Service that extracts structured output using a language model.
    """

    @abstractmethod
    async def extract(
        self,
        filename: str,
        parse_response: ParseResponse,
    ) -> T:
        """Generates a structured Pydantic object from parsed content."""
        ...

    @cache_result(key_params=['filename'])
    async def __call__(
            self,
            filename: str,
            parse_response: ParseResponse,
            **kwargs
    ) -> T:
        return await self.extract(filename, parse_response, **kwargs)


class LLMStructuredOutputService(StructuredOutputService[T]):
    """
    Abstract base for structured output services that use an LLM with a prompt template.
    """
    DEFAULT_PROMPT: str = """\
You extract structured information from documents. Review the following document and return the extracted data in the specified format.
Document:
{parse_response_markdown}
"""

    def __init__(
        self,
        response_type: Type[T],
        model: str,
        prompt: Optional[str] = None,
        cache: OptionalCacheType = None,
    ):
        super().__init__(response_type=response_type, cache=cache)
        self.model = model
        self.prompt = prompt if prompt is not None else self.DEFAULT_PROMPT

    def format_prompt(self, parse_response: ParseResponse) -> str:
        """Format the prompt template with the parsed document content."""
        return self.prompt.format(parse_response_markdown=parse_response.markdown)


R = TypeVar("R", bound=BaseModel, covariant=True)


class ParseExtractProtocol(Protocol[R]):
    """Protocol defining the interface for ParseExtract services.

    Any object implementing these methods can be used as ParseExtractService.
    """

    async def parse_extract(self, filename: str, content: bytes) -> R: ...
    async def __call__(self, filename: str, content: bytes) -> R: ...


class ParseExtractService(BaseService[T]):
    """
    Class for a Service that parses and extracts structured output using a language model in one step.
    """

    @abstractmethod
    async def parse_extract(
        self,
        filename: str,
        content: bytes,
        mime_type: Optional[str] = None
    ) -> T:
        """Generates a structured Pydantic object from file content."""
        ...

    @cache_result(['filename'])
    async def __call__(
            self,
            filename: str,
            content: bytes
    ) -> T:
        return await self.parse_extract(filename, content)


class ParseExtractPipelineService(Generic[T]):
    """
    An end-to-end document parsing Service using two steps: parse and Structured Output.
    It orchestrates a ParseService and a StructuredOutputService.
    """

    def __init__(
        self,
        parse_service: ParseService[PARSE_T],
        structured_output_service: StructuredOutputService[T],
    ):
        self.parse_service = parse_service
        self.structured_output_service = structured_output_service

    async def parse_extract(
        self,
        filename: str,
        content: bytes,
    ) -> T:
        """
        Performs end-to-end document parsing and structuring. Save immediate intermediate results in the object.
        """
        ocr_response = await self.parse_service(filename, content)
        self.ocr_response = ocr_response
        structured_output_response = await self.structured_output_service.extract(filename, ocr_response)
        self.structured_output_response = structured_output_response

        return structured_output_response

    async def __call__(
            self,
            filename: str,
            content: bytes,
    ) -> T:
        structured_output = await self.parse_extract(filename, content)
        return structured_output
