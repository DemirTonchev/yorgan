from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, Optional, override

from landingai_ade import AsyncLandingAIADE
from landingai_ade.types.parse_job_create_response import ParseJobCreateResponse
from landingai_ade.types.parse_job_get_response import ParseJobGetResponse
# from landingai_ade.types import ParseResponse as LandingParseResponse
from yorgan.datamodels import ParseResponse
from yorgan.cache import cache_result
from yorgan.services.utils import count_pdf_pages, download_blob

from .base import ParseService  # , StructuredOutputService future implementation

if TYPE_CHECKING:
    from aiocache.base import BaseCache
    from yorgan.cache import SimpleMemoryCacheWithPersistence
    OptionalCacheType = Optional[BaseCache | SimpleMemoryCacheWithPersistence]


# ignoring type this is necessary due to how we typed stuff in services/base.py does not break runtime
class AgenticDocParseService(ParseService[ParseResponse]):  # type: ignore
    """
    An Parse Service implementation using landingai_ade.
    """
    available_models = {"dpt-2", "dpt-2-mini"}

    def __init__(
        self,
        environment: Literal["production", "eu"] = "production",
        page_threshold: int = 50,
        cache: OptionalCacheType = None,
        **kwargs: Any
    ):
        super().__init__(response_type=ParseResponse, cache=cache)

        model = kwargs.pop("model", "dpt-2")
        if model not in self.available_models:
            raise ValueError(f"Invalid model: {model}, available models: {self.available_models}")
        self.model = model
        self.page_threshold = page_threshold

        self._client = AsyncLandingAIADE(environment=environment, **kwargs)

    async def _create_job(self, content: bytes, **kwargs) -> ParseJobCreateResponse:
        return await self._client.parse_jobs.create(document=content, model=self.model, **kwargs)

    async def _poll_job(self, job_id: str, poll_interval=5, **kwargs) -> ParseJobGetResponse:
        """Poll until job completes or fails."""
        while True:
            response = await self._client.parse_jobs.get(job_id)
            if response.status == "completed":
                return response
            if response.status in ["failed", "cancelled"]:
                raise RuntimeError(f"Parse job {job_id} failed or was cancelled")
            await asyncio.sleep(poll_interval)

    async def _run_job(self, content: bytes | None = None, job_id: str | None = None, **kwargs) -> ParseResponse:
        """
        Run an async parse job and wait for completion.

        Either creates a new job from content or resumes polling an existing job by ID.

        Args:
            content: Document bytes to parse. Creates a new job if provided.
            job_id: Existing job ID to resume polling. Use to recover from interrupted jobs.
            **kwargs: Additional arguments passed to job creation or job polling.

        Returns:
            Parsed document response.

        Raises:
            ValueError: If neither content nor job_id is provided, or if the
                completed job has no result data.
        """
        if content and not job_id:
            job = await self._create_job(content, **kwargs)
            job_id = job.job_id

        if job_id is None:
            raise ValueError("Either 'content' or 'job_id' must be provided")

        completed_job = await self._poll_job(job_id, **kwargs)
        if completed_job.output_url:
            buffer = await download_blob(completed_job.output_url)
            response = ParseResponse.model_validate_json(buffer.getvalue())
            return response
        elif completed_job.data:
            response = ParseResponse.model_validate(completed_job.data, from_attributes=True)
            return response
        else:
            raise ValueError(
                f"Parse job {job_id} completed but returned no result. "
                f"Expected 'output_url' or 'data' in response, got neither."
            )

    @override
    @cache_result(key_params=['filename'])
    async def __call__(
        self,
        filename: str,
        content: bytes,
        **kwargs
    ) -> ParseResponse:
        """Processes a document using landingai_ade's OCR engine."""
        page_count = count_pdf_pages(content)
        if page_count < self.page_threshold:
            landing_response = await self._client.parse(document=content, model=self.model, **kwargs)
            response = ParseResponse.model_validate(landing_response, from_attributes=True)
        else:
            response = await self._run_job(content=content)
        return response
