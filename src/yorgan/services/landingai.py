from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, override

from landingai_ade import AsyncLandingAIADE
from landingai_ade.types import ParseResponse as LandingParseResponse

from .base import (
    ParseService,
    StructuredOutputService,
    ParseExtractService,
)

if TYPE_CHECKING:
    from aiocache.base import BaseCache
    from yorgan.cache import SimpleMemoryCacheWithPersistence
    OptionalCacheType = Optional[BaseCache | SimpleMemoryCacheWithPersistence]


# ignoring type this is necessary due to how we typed stuff in services/base.py does not break runtime
class AgenticDocParseService(ParseService[LandingParseResponse]):  # type: ignore
    """
    An Parse Service implementation using landingai_ade.
    """
    available_models = {"dpt-2", "dpt-2-mini"}

    def __init__(self, cache: OptionalCacheType = None, **kwargs: Any):
        super().__init__(response_type=LandingParseResponse, cache=cache)

        model = kwargs.pop("model", "dpt-2")
        if model not in self.available_models:
            raise ValueError(f"Invalid model: {model}, available models: {self.available_models}")
        self.model = model

        self._client = AsyncLandingAIADE(**kwargs)

    @override
    async def parse(
        self,
        filename: str,
        content: bytes,
        **kwargs
    ) -> LandingParseResponse:
        """Processes a document using landingai_ade's OCR engine."""
        results = await self._client.parse(document=content, model=self.model, **kwargs)
        return results
