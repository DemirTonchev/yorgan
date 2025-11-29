from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Type, TypeVar, override, cast
from functools import lru_cache
from google import genai
from google.genai import types
from .base import (
    ParseService,
    ParseResponse,
    StructuredOutputService,
    ParseExtractService,
    BaseModel
)
from .utils import get_mime_type

T = TypeVar('T', bound=BaseModel)
# ie has markdown attribute
ParseResponseT = TypeVar("ParseResponseT", bound=ParseResponse)

if TYPE_CHECKING:
    from .base import OptionalCacheType

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@lru_cache(maxsize=1)
def get_default_client():
    return genai.Client()


class GeminiParseService(ParseService[ParseResponse]):
    """
    A Parse Service implementation using Gemini.
    """

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        prompt: Optional[str] = None,
        cache: OptionalCacheType = None
    ):
        super().__init__(response_type=ParseResponse, cache=cache)
        self.client = client if client is not None else get_default_client()
        self.model = model
        if prompt is None:
            prompt = """Your task is to extract the text from the attached document. Format it nicely as a markdown."""
        self.prompt = prompt

    @override
    async def parse(
            self,
            filename: str,
            content: bytes,
            **kwargs
    ) -> ParseResponse:
        """Processes a document using Gemini as OCR engine."""

        # Determine the MIME type based on the file extension
        mime_type = get_mime_type(filename)

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=content,
                    mime_type=mime_type,
                ),
                self.prompt,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": self.response_type,
            },
        )
        # for bookkeeping purposes at some point like
        # metadata - token counts etc.. whatever the api returns and we might want to log
        self._last_raw_response = response
        parsed_document = response.parsed
        if parsed_document is None:
            raise ValueError(f"Extraction error: {self.service_name} failed to parse document - no output received")
        return cast(ParseResponse, parsed_document)


class GeminiStructuredOutputService(StructuredOutputService[T]):
    """
    A structured output Service implementation using Gemini.
    """

    def __init__(
        self,
        response_type: Type[T],
        client: Optional[genai.Client] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        prompt: Optional[str] = None,
        cache: OptionalCacheType = None
    ):
        super().__init__(response_type=response_type, cache=cache)
        self.client = client if client is not None else get_default_client()
        self.model = model
        if prompt is None:
            prompt = """\
You are an accountant that extracts information from documents. Please look at this document and extract the needed information.
Document:
{parse_response_markdown}
"""
        self.prompt = prompt

    @override
    async def extract(
        self,
        filename: str,
        parse_response: ParseResponse,
    ) -> T:
        """
        Generates a structured Pydantic object from a parsed document.
        """
        prompt = self.prompt.format(parse_response_markdown=parse_response.markdown)

        # client.aio uses async stuff
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": self.response_type,
            },
        )
        # for bookkeeping purposes at some point like
        # metadata - token counts etc.. whatever the api returns and we might want to log
        self._last_raw_response = response
        structured_output = response.parsed
        if structured_output is None:
            raise ValueError(f"Extraction error: {self.service_name} failed to parse document - no output received")
        return cast(T, structured_output)


class GeminiParseExtractService(ParseExtractService[T]):
    """
    An end-to-end document parsing Service using Gemini.
    """

    prompt: str = """\
You are an accountant that extracts information from documents. Please look at this document and extract the needed information.
"""

    def __init__(
        self,
        response_type: Type[T],
        client: Optional[genai.Client] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        cache: OptionalCacheType = None
    ):
        super().__init__(response_type=response_type, cache=cache)
        self.client = client if client is not None else get_default_client()
        self.model = model

    @override
    async def parse_extract(
        self,
        filename: str,
        content: bytes,
        mime_type: Optional[str] = None,
    ) -> T:
        """
        Performs end-to-end document parsing and structuring using Gemini's native functionality.
        """
        if not mime_type:
            # Determine the MIME type based on the file extension
            mime_type = get_mime_type(filename)

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=content,
                    mime_type=mime_type,
                ),
                self.prompt,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": self.response_type,
            },
        )
        # for bookkeeping purposes at some point like
        # metadata - token counts etc.. whatever the api returns and we might want to log
        self._last_raw_response = response

        structured_output = response.parsed
        if structured_output is None:
            raise ValueError(f"Extraction error: {self.service_name} failed to parse document - no output received")
        return cast(T, structured_output)
