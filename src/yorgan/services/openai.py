from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Type, TypeVar, override
import openai
from .base import (
    ParseService,
    ParseResponse,
    LLMStructuredOutputService,
    ParseExtractService,
    BaseModel
)
from .utils import get_mime_type, encode_bytes_for_transfer

if TYPE_CHECKING:
    from .base import OptionalCacheType

DEFAULT_OPENAI_MODEL = "gpt-4.1"

T = TypeVar('T', bound=BaseModel)
# ie has markdown attribute
ParseResponseT = TypeVar("ParseResponseT", bound=ParseResponse)


@lru_cache(maxsize=1)
def get_default_client():
    return openai.AsyncOpenAI()


class OpenAIParseService(ParseService[ParseResponse]):
    """
    A Parse Service implementation using OpenAI.
    """

    def __init__(
        self,
        client: Optional[openai.AsyncClient] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        prompt: Optional[str] = None,
        cache: OptionalCacheType = None
    ):
        super().__init__(response_type=ParseResponse, cache=cache)
        self.client = client if client is not None else get_default_client()
        self.model = model
        if prompt is None:
            prompt = """Your task is to extract text from documents. Format it nicely as a markdown."""
        self.prompt = prompt
        self._supported_file_types = {"png", "jpeg", "jpg", "gif", "pdf"}

    @override
    async def parse(
            self,
            filename: str,
            content: bytes,
            **kwargs
    ) -> ParseResponse:
        """Processes a document using OpenAI as OCR engine."""

        base64_content = encode_bytes_for_transfer(content)
        mime_type = get_mime_type(filename)

        if (extention := mime_type.split("/")[-1]) not in self._supported_file_types:
            raise ValueError(f"OpenAI: unsupported file type: {mime_type}")

        if extention == "pdf":
            input_content = {
                "type": "input_file",
                "filename": filename,
                "file_data": f"data:{mime_type};base64,{base64_content}",
            }
        else:
            input_content = {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{base64_content}",
            }
        response = await self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Extract the text from the attached document."
                        },
                        input_content,
                    ],
                },  # pyright: ignore[reportArgumentType]
            ],
            text_format=self.response_type,
        )
        # for bookkeeping purposes at some point
        self._last_raw_response = response
        parsed_document = response.output_parsed
        if parsed_document is None:
            raise ValueError(f"Parsing error: {self.service_name} failed to parse document - no output received")
        return parsed_document


class OpenAIStructuredOutputService(LLMStructuredOutputService[T]):
    """
    A structured output Service implementation using OpenAI.
    """

    def __init__(
        self,
        response_type: Type[T],
        client: Optional[openai.AsyncClient] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        prompt: Optional[str] = None,
        cache: OptionalCacheType = None
    ):
        if prompt is None:
            prompt = """\
You extract structured information from documents. Review the following document and return the extracted data in the specified format.
Document:
{parse_response_markdown}
"""
        super().__init__(response_type=response_type, cache=cache, model=model, prompt=prompt)
        self.client = client if client is not None else get_default_client()

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

        response = await self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": prompt},
            ],
            text_format=self.response_type,
        )

        # TODO: future bookkeeping
        self._last_raw_response = response

        structured_output = response.output_parsed
        if structured_output is None:
            raise ValueError(f"Extraction error: {self.service_name} failed to parse document - no output received")
        return structured_output


class OpenAIParseExtractService(ParseExtractService[T]):
    """
    An end-to-end document parsing Service using OpenAI. Usually you would prefer to use the Pipeline but this is also an option.
    """

    def __init__(
        self,
        response_type: Type[T],
        client: Optional[openai.AsyncClient] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        prompt: Optional[str] = None,
        cache: OptionalCacheType = None
    ):
        super().__init__(response_type=response_type, cache=cache)
        self.client = client if client is not None else get_default_client()
        self.model = model
        if prompt is None:
            prompt = """\
You extract structured information from documents.Review the following document and return the extracted data in the specified format.
"""
        self.prompt = prompt
        self._supported_file_types = {"png", "jpeg", "jpg", "gif", "pdf"}

    @override
    async def parse_extract(
        self,
        filename: str,
        content: bytes,
    ) -> T:
        """
        Performs end-to-end document parsing and structuring using OpenAI native functionality.
        """

        base64_content = encode_bytes_for_transfer(content)
        mime_type = get_mime_type(filename)
        if (extention := mime_type.split("/")[-1]) not in self._supported_file_types:
            raise ValueError(f"OpenAI: unsupported file type: {mime_type}")

        if extention == "pdf":
            input_content = {
                "type": "input_file",
                "filename": filename,
                "file_data": f"data:{mime_type};base64,{base64_content}",
            }
        else:
            input_content = {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{base64_content}",
            }

        response = await self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Extract the text from the attached document.",
                        },
                        input_content,
                    ],
                }  # pyright: ignore[reportArgumentType]
            ],
            text_format=self.response_type,
        )
        # for bookkeeping purposes at some point like
        # metadata - token counts etc.. whatever the api returns and we might want to log
        self._last_raw_response = response

        structured_output = response.output_parsed
        if structured_output is None:
            raise ValueError(f"Extraction error: {self.service_name} failed to parse document - no output received")
        return structured_output
