from __future__ import annotations

from functools import lru_cache
from typing import Optional, Type, TypeVar, cast, override

from google import genai
from google.genai import types
from pydantic import BaseModel

from yorgan.datamodels import ParseResponse, ResponseMetadata

from .base import (BaseLLM, LLMParseExtractPipelineService, LLMParseService,
                   LLMStructuredOutputService)
from .utils import get_mime_type

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
            client: Optional Google client instance. If not provided, uses a default singleton client.
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
    ) -> tuple[T, ResponseMetadata]:
        """
        Generate structured output from content using Google Gemini models.

        Args:
            model: Name/identifier of the Gemini model to use.
            prompt: The prompt to send to the model.
            response_type: The Pydantic model class for the response.
            filename: Name of the file being processed. The extension is used
                to guess the MIME type.
            content: Document content as bytes, or None if only using the prompt.
            **kwargs: Additional arguments for the Google API.

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
        llm_response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_type,
            },
            **kwargs
        )

        # Parse and validate the response
        structured_output = llm_response.parsed
        if structured_output is None:
            raise ValueError(
                "Generation error: Gemini failed to generate output - no parsed response received"
            )
        response = cast(T, structured_output)

        usage_metadata = llm_response.usage_metadata
        if usage_metadata:
            metadata = ResponseMetadata(
                input_token_count=usage_metadata.prompt_token_count,
                output_token_count=cast(int, usage_metadata.total_token_count) - cast(int, usage_metadata.prompt_token_count),
            )
        else:
            metadata = ResponseMetadata()

        return response, metadata


class GeminiParseService(LLMParseService[ParseResponse]):
    LLM_TYPE = GeminiLLM


class GeminiStructuredOutputService(LLMStructuredOutputService[T]):
    LLM_TYPE = GeminiLLM


class GeminiParseExtractPipelineService(LLMParseExtractPipelineService[T]):
    LLM_PARSE_SERVICE_TYPE = GeminiParseService
    LLM_STRUCTURED_OUTPUT_SERVICE_TYPE = GeminiStructuredOutputService
