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

DEFAULT_OPENAI_MODEL = "gpt-4.1"


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
    """
    Parse Service implementation using OpenAI.

    Converts documents (images, PDFs) into markdown using OpenAI's vision capabilities.
    """

    def __init__(
        self,
        llm: Optional[OpenAILLM] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        prompt: Optional[str] = None,
        batch_prompt: Optional[str] = None,
        cache: OptionalCacheType = None,
        page_threshold: int = 10
    ):
        """
        Args:
            llm: Optional OpenAILLM instance. If not provided, creates a default instance.
            model: Name/identifier of the OpenAI model to use. Defaults to DEFAULT_OPENAI_MODEL.
            prompt: Custom prompt template. Defaults to DEFAULT_PROMPT.
            batch_prompt: Custom batch prompt template. Defaults to DEFAULT_BATCH_PROMPT.
            cache: Optional cache instance for storing results.
            page_threshold: Maximum number of pages to parse in one call. PDFs exceeding
                this are split and parsed page-by-page. Defaults to 10.
        """
        llm = llm if llm else OpenAILLM()

        super().__init__(
            response_type=ParseResponse,
            llm=llm,
            model=model,
            prompt=prompt,
            batch_prompt=batch_prompt,
            cache=cache,
            page_threshold=page_threshold
        )


class OpenAIStructuredOutputService(LLMStructuredOutputService[T]):
    """
    Structured output Service implementation using OpenAI.

    Extracts structured data from parsed markdown documents.
    """

    def __init__(
        self,
        response_type: Type[T],
        llm: Optional[OpenAILLM] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        prompt: Optional[str] = None,
        batch_prompt: Optional[str] = None,
        page_threshold: int = 10,
        batch_window: int = 6,
        batch_overlap: int = 1,
        cache: OptionalCacheType = None,
    ):
        """
        Args:
            response_type: The Pydantic model class for the structured output.
            llm: Optional OpenAILLM instance. If not provided, creates a default instance.
            model: Name/identifier of the OpenAI model to use. Defaults to DEFAULT_OPENAI_MODEL.
            prompt: Custom prompt template for single-page extraction. Defaults to DEFAULT_PROMPT.
            batch_prompt: Custom prompt template for multi-page extraction. Defaults to DEFAULT_BATCH_PROMPT.
            page_threshold: Maximum number of pages to extract in one call. Documents exceeding
                this are processed page-by-page with sliding context. Defaults to 10.
            batch_window: The length of the sliding window. Defaults to 6.
            batch_overlap: The length of the overlap. Defaults to 1.
            cache: Optional cache instance for storing results.
        """
        llm = llm if llm else OpenAILLM()

        super().__init__(
            response_type=response_type,
            llm=llm,
            model=model,
            prompt=prompt,
            batch_prompt=batch_prompt,
            page_threshold=page_threshold,
            batch_window=batch_window,
            batch_overlap=batch_overlap,
            cache=cache,
        )


class OpenAIParseExtractPipelineService(LLMParseExtractPipelineService[T]):
    """
    End-to-end document parsing Service using OpenAI.

    Performs both parsing and extraction in a single step using OpenAI's native capabilities.
    """

    def __init__(
        self,
        response_type: Type[T],
        parse_service: Optional[OpenAIParseService[ParseResponse]] = None,
        structured_output_service: Optional[OpenAIStructuredOutputService[T]] = None,
        cache: OptionalCacheType = None,
    ):
        """
        Args:
            response_type: The Pydantic model class for the service's response
            parse_service: Service instance for the parse step
            structured_output_service: Service instance for the extract step
            cache: Optional cache instance for storing results. Defaults to NullCache
        """
        parse_service = parse_service if parse_service else OpenAIParseService(cache=cache)
        structured_output_service = structured_output_service if structured_output_service else OpenAIStructuredOutputService(
            response_type=response_type, cache=cache)

        super().__init__(
            response_type=response_type,
            parse_service=parse_service,
            structured_output_service=structured_output_service,
            cache=cache,
        )
