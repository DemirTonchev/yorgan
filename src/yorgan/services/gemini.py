from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Type, TypeVar, cast, override

from google import genai
from google.genai import types

from .base import (BaseLLM, BaseModel, LLMParseExtractPipelineService,
                   LLMParseService, LLMStructuredOutputService, ParseResponse)
from .utils import get_mime_type

if TYPE_CHECKING:
    from .base import OptionalCacheType

T = TypeVar('T', bound=BaseModel)


@lru_cache(maxsize=1)
def get_default_client():
    """Get or create a singleton Gemini client."""
    return genai.Client()


class GeminiLLM(BaseLLM):
    """
    LLM implementation for Google Gemini models.

    Handles structured output generation from prompts and optional document content.
    """
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        client: Optional[genai.Client] = None,
    ):
        """
        Args:
            client: Optional Gemini client instance. If not provided, uses a default singleton client.
        """
        self.client = client if client is not None else get_default_client()
        self._supported_file_types = {"png", "jpeg", "jpg", "pdf"}

    @override
    async def generate(
        self,
        model: str,
        prompt: str,
        response_type: Type[T],
        filename: Optional[str] = None,
        content: Optional[bytes] = None,
        **kwargs
    ) -> T:
        """
        Generate structured output from content using Gemini.

        Args:
            model: Name/identifier of the Gemini model to use.
            prompt: The prompt to send to the model.
            response_type: The Pydantic model class for the response.
            filename: Name of the file being processed. The extension is used
                to guess the MIME type.
            content: Document content as bytes, or None if only using the prompt.
            **kwargs: Additional arguments for the Gemini API.

        Returns:
            Instance of response_type with the model's generated output.

        Raises:
            ValueError: If the file type is unsupported or if generation fails.
        """
        # Build the contents list
        contents = []

        # If content is provided, add it as a Part
        if content is not None and filename is not None:
            mime_type = get_mime_type(filename)
            extension = mime_type.split("/")[-1]

            if extension not in self._supported_file_types:
                raise ValueError(f"Gemini: unsupported file type: {mime_type}")

            contents.append(
                types.Part.from_bytes(
                    data=content,
                    mime_type=mime_type,
                )
            )

        # Add the prompt
        contents.append(prompt)

        # Make the API call
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_type,
            },
            **kwargs
        )

        # Store raw response for bookkeeping (token counts, metadata, etc.)
        self._last_raw_response = response

        # Parse and validate the response
        structured_output = response.parsed
        if structured_output is None:
            raise ValueError(
                f"Generation error: Gemini failed to generate output - no parsed response received"
            )

        return cast(T, structured_output)


class GeminiParseService(LLMParseService[ParseResponse]):
    LLM_TYPE = GeminiLLM


class GeminiStructuredOutputService(LLMStructuredOutputService[T]):
    LLM_TYPE = GeminiLLM


class GeminiParseExtractPipelineService(LLMParseExtractPipelineService[T]):
    LLM_PARSE_SERVICE_TYPE = GeminiParseService
    LLM_STRUCTURED_OUTPUT_SERVICE_TYPE = GeminiStructuredOutputService
