from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, TypeVar, override
from huggingface_hub import AsyncInferenceClient
from .base import (
    ParseService,
    ParseResponse,
    BaseModel
)
from .utils import get_mime_type, encode_bytes_for_transfer

if TYPE_CHECKING:
    from .base import OptionalCacheType

DEFAULT_HUGGINGFACE_MODEL = "allenai/olmOCR-2-7B-1025"

T = TypeVar('T', bound=BaseModel)
# ie has markdown attribute
ParseResponseT = TypeVar("ParseResponseT", bound=ParseResponse)


@lru_cache(maxsize=1)
def get_default_client():
    return AsyncInferenceClient()


class HuggingFaceParseService(ParseService[ParseResponse]):
    """
    A Parse Service implementation using Hugging Face.
    """

    def __init__(
        self,
        client: Optional[AsyncInferenceClient] = None,
        model: str = DEFAULT_HUGGINGFACE_MODEL,
        prompt: Optional[str] = None,
        cache: OptionalCacheType = None
    ):
        super().__init__(response_type=ParseResponse, cache=cache)
        self.client = client if client is not None else get_default_client()
        self.model = model
        if prompt is None:
            prompt = """Your task is to extract text from documents. Format it nicely as a markdown."""
        self.prompt = prompt
        self._supported_file_types = {"png", "jpeg", "jpg", "gif"}

    @override
    async def parse(
            self,
            filename: str,
            content: bytes,
            **kwargs
    ) -> ParseResponse:
        """Processes a document using Hugging Face as OCR engine."""

        base64_content = encode_bytes_for_transfer(content)
        mime_type = get_mime_type(filename)

        if (extention := mime_type.split("/")[-1]) not in self._supported_file_types:
            if extention == "pdf":
                raise ValueError(f"HuggingFace: PDF support is not yet implemented")
            raise ValueError(f"HuggingFace: unsupported file type: {mime_type}")

        image_url = f"data:{mime_type};base64,{base64_content}"

        response = await self.client.chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the text from the attached document."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                    ],
                },
            ],
        )

        # for bookkeeping purposes at some point
        self._last_raw_response = response

        # Extract text from response
        if not response.choices or not response.choices[0].message.content:
            raise ValueError(f"Parsing error: {self.service_name} failed to parse document - no output received")

        markdown_text = response.choices[0].message.content
        parsed_document = self.response_type(markdown=markdown_text)

        return parsed_document
