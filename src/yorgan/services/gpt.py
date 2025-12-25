from __future__ import annotations

from functools import lru_cache
from typing import Optional, Type, TypeVar, cast, override

import openai
from pydantic import BaseModel

from yorgan.datamodels import ParseResponse, ResponseMetadata

from .base import (BaseLLM, LLMParseExtractPipelineService, LLMParseService,
                   LLMStructuredOutputService)
from .utils import encode_bytes_for_transfer, get_mime_type

T = TypeVar('T', bound=BaseModel)


@lru_cache(maxsize=1)
def get_default_client():
    """Get or create a singleton OpenAI async client."""
    return openai.AsyncOpenAI()


class GPTLLM(BaseLLM):
    """
    LLM implementation for OpenAI GPT models.

    Handles structured output generation from prompts and optional document content
    using OpenAI responses.parse API.
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
    ) -> tuple[T, ResponseMetadata]:
        """
        Generate structured output from content using OpenAI GPT models.

        Args:
            model: Name/identifier of the OpenAI GPT model to use.
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
                raise ValueError(f"GPT: unsupported file type: {mime_type}")

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
        llm_response = await self.client.responses.parse(
            model=model,
            input=input_messages,
            text_format=response_type,
            **kwargs
        )

        # Parse and validate the response
        structured_output = llm_response.output_parsed
        if structured_output is None:
            raise ValueError(
                "Generation error: GPT failed to generate output - no parsed response received"
            )
        response = cast(T, structured_output)

        # Extract usage metadata from the response
        usage = getattr(llm_response, 'usage', None)
        if usage:
            metadata = ResponseMetadata(
                input_token_count=getattr(usage, 'input_tokens', None),
                output_token_count=getattr(usage, 'output_tokens', None),
            )
        else:
            metadata = ResponseMetadata()

        return response, metadata


class GPTParseService(LLMParseService[ParseResponse]):
    LLM_TYPE = GPTLLM


class GPTStructuredOutputService(LLMStructuredOutputService[T]):
    LLM_TYPE = GPTLLM


class GPTParseExtractPipelineService(LLMParseExtractPipelineService[T]):
    LLM_PARSE_SERVICE_TYPE = GPTParseService
    LLM_STRUCTURED_OUTPUT_SERVICE_TYPE = GPTStructuredOutputService
