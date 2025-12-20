from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Type, TypeVar, override

import openai

from .base import (BaseLLM, BaseModel, LLMParseExtractPipelineService,
                   LLMParseService, LLMStructuredOutputService, ParseResponse)
from .utils import encode_bytes_for_transfer, get_mime_type

if TYPE_CHECKING:
    from .base import OptionalCacheType

T = TypeVar('T', bound=BaseModel)


@lru_cache(maxsize=1)
def get_default_client():
    """Get or create a singleton OpenAI async client."""
    return openai.AsyncOpenAI()


class OpenAILLM(BaseLLM):
    """
    LLM implementation for OpenAI models.

    Handles structured output generation from prompts and optional document content
    using OpenAI's responses.parse API.
    """

    DEFAULT_MODEL = "gpt-4.1"

    def __init__(
        self,
        client: Optional[openai.AsyncClient] = None,
    ):
        """
        Args:
            client: Optional OpenAI async client instance. If not provided, uses a default singleton client.
        """
        self.client = client if client is not None else get_default_client()
        self._supported_file_types = {"png", "jpeg", "jpg", "gif", "pdf"}

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
        Generate structured output from content using OpenAI.

        Args:
            model: Name/identifier of the OpenAI model to use.
            prompt: The prompt to send to the model.
            response_type: The Pydantic model class for the response.
            filename: Name of the file being processed. The extension is used
                to guess the MIME type.
            content: Document content as bytes, or None if only using the prompt.
            **kwargs: Additional arguments for the OpenAI API.

        Returns:
            Instance of response_type with the model's generated output.

        Raises:
            ValueError: If the file type is unsupported or if generation fails.
        """
        # Build the input messages
        input_messages = [
            {"role": "system", "content": prompt}
        ]

        # If content is provided, add it to the user message
        if content is not None and filename is not None:
            mime_type = get_mime_type(filename)
            extension = mime_type.split("/")[-1]

            if extension not in self._supported_file_types:
                raise ValueError(f"OpenAI: unsupported file type: {mime_type}")

            base64_content = encode_bytes_for_transfer(content)

            # OpenAI uses different content types for PDFs vs images
            if extension == "pdf":
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

            input_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract the text from the attached document.",
                    },
                    input_content,
                ],
            })

        # Make the API call
        response = await self.client.responses.parse(
            model=model,
            input=input_messages,
            text_format=response_type,
            **kwargs
        )

        # Store raw response for bookkeeping (token counts, metadata, etc.)
        self._last_raw_response = response

        # Parse and validate the response
        structured_output = response.output_parsed
        if structured_output is None:
            raise ValueError(
                f"Generation error: OpenAI failed to generate output - no parsed response received"
            )

        return structured_output


class OpenAIParseService(LLMParseService[ParseResponse]):
    LLM_TYPE = OpenAILLM


class OpenAIStructuredOutputService(LLMStructuredOutputService[T]):
    LLM_TYPE = OpenAILLM


class OpenAIParseExtractPipelineService(LLMParseExtractPipelineService[T]):
    LLM_PARSE_SERVICE_TYPE = OpenAIParseService
    LLM_STRUCTURED_OUTPUT_SERVICE_TYPE = OpenAIStructuredOutputService
