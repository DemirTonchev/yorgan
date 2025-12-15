from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from .base import LLMParseService, PARSE_T
from .utils import split_pdf_pages, count_pdf_pages, get_mime_type

if TYPE_CHECKING:
    from yorgan.cache import BaseCache, SimpleMemoryCacheWithPersistence
    OptionalCacheType = Optional[BaseCache | SimpleMemoryCacheWithPersistence]


class MultipageLLMParseService(LLMParseService[PARSE_T]):
    """
    Parse service that automatically handles multi-page PDFs.

    For multi-page PDFs (more than 1 page), each page is parsed separately 
    using the underlying parse service, then combined with page breaks.
    For images and other documents, the base parse service is used directly.

    This class wraps an existing LLMParseService (like GeminiParseService 
    or OpenAIParseService) and adds automatic multipage PDF processing.
    """

    PAGE_BREAK = "\n\n<!-- PAGE BREAK -->\n\n"

    def __init__(
        self,
        base_parse_service: LLMParseService[PARSE_T],
    ):
        """
        Initialize with an existing parse service to use for individual pages.

        Args:
            base_parse_service: The underlying parse service (e.g., GeminiParseService)
        """
        super().__init__(
            response_type=base_parse_service.response_type,
            model=base_parse_service.model,
            prompt=base_parse_service.prompt,
            cache=base_parse_service.cache
        )
        self.base_parse_service = base_parse_service
        # Perhaps we should add "Multipage" to the service name
        self.service_name = base_parse_service.service_name

    def _should_process_multipage(self, filename: str, content: bytes) -> bool:
        """
        Check if the content should be processed as multi-page.

        Returns True only for PDFs with more than 1 page.
        Images and other document types return False.

        Args:
            filename (str): Name or path of the file to check. The extension is used
                to guess the MIME type.

            content: Document content as bytes

        Returns:
            True if it's a PDF with more than 1 page, False otherwise
        """
        mime_type = get_mime_type(filename)

        # Only process PDFs as multipage
        if mime_type != "application/pdf":
            return False

        # Check if PDF has more than 1 page
        page_count = count_pdf_pages(content)
        return page_count > 1

    async def parse(
        self,
        filename: str,
        content: bytes,
        **kwargs
    ) -> PARSE_T:
        """
        Parse a document. If it's a multi-page PDF, process each page individually.
        Otherwise, use the base parse service directly.

        Args:
            filename: Name of the file
            content: Full document content as bytes
            **kwargs: Additional arguments passed to base parse service

        Returns:
            ParseResponse with markdown content
        """
        # Check if this should be processed as multi-page
        if not self._should_process_multipage(filename, content):
            # Not a multi-page PDF (could be image, single-page PDF, etc.)
            # Use base service directly
            return await self.base_parse_service.parse(
                filename=filename,
                content=content,
                **kwargs
            )

        # Multi-page PDF: split and process each page
        pages = split_pdf_pages(content)

        # Parse each page using the base parse service
        page_markdowns = []
        for page_num, page_content in enumerate(pages, start=1):
            # Create a filename for this page
            # This is not important, the cache uses the full filename
            name, dot, ext = filename.rpartition(".")
            page_filename = f"{name}_page_{page_num}.{ext}"

            # Use the base parse service to parse this single page
            page_response = await self.base_parse_service.parse(
                filename=page_filename,
                content=page_content,
                **kwargs
            )

            # Extract markdown from the response
            page_markdowns.append(page_response.markdown)

        # Combine all pages with page breaks
        combined_markdown = self.PAGE_BREAK.join(page_markdowns)

        # Create the response object
        return self.response_type(markdown=combined_markdown)
