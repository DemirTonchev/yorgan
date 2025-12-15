from __future__ import annotations

from typing import TYPE_CHECKING, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field, create_model

from yorgan.datamodels import ParseResponse

from .base import PAGE_BREAK, PARSE_T, LLMParseService, LLMStructuredOutputService
from .utils import count_pdf_pages, get_mime_type, split_pdf_pages

if TYPE_CHECKING:
    from yorgan.cache import BaseCache, SimpleMemoryCacheWithPersistence
    OptionalCacheType = Optional[BaseCache | SimpleMemoryCacheWithPersistence]

T = TypeVar('T', bound=BaseModel)


class MultipageLLMParseService(LLMParseService[PARSE_T]):
    """
    Parse service that automatically handles multi-page PDFs.

    For multi-page PDFs (more than 1 page), each page is parsed separately 
    using the underlying parse service, then combined with page breaks.
    For images and other documents, the base parse service is used directly.

    This class wraps an existing LLMParseService (like GeminiParseService 
    or OpenAIParseService) and adds automatic multipage PDF processing.
    """

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
        combined_markdown = f"\n\n{PAGE_BREAK}\n\n".join(page_markdowns)

        # Create the response object
        return self.response_type(markdown=combined_markdown)


class StructuredOutputWrapper(BaseModel, Generic[T]):
    """Wrapper model that includes the structured output and notes for iterative extraction."""
    extracted_data: T = Field(description="The structured data extracted so far")
    notes: str = Field(
        default="",
        description="Notes about information that may be useful for extraction in subsequent pages"
    )


class MultipageLLMStructuredOutputService(LLMStructuredOutputService[T], Generic[T]):
    """
    Structured output service that processes multi-page documents page by page.

    For documents with many pages (configurable threshold), this service processes
    pages iteratively with a sliding window context (previous, current, next page).
    The model incrementally builds the structured output and maintains notes
    about information that may be relevant for subsequent pages.
    """

    DEFAULT_MULTIPAGE_PROMPT: str = """\
You are extracting structured information from a multi-page document, processing it page by page.
Unless it is the first or the last page, you will receive the previous, current, and next pages.
You will also receive the extracted data so far, and relevant notes from previous pages.

Your task:
1. Add any NEW values based on information found on the current page to the extracted_data
2. PRESERVE previously extracted data, unless you are completely certain that the values need to be changed based on new information
3. Update the notes by adding anything important for subsequent pages (e.g., "table continues", etc),
   or removing notes that are no longer relevant (e.g. remove "table continues" once the table is fully extracted)
4. Return the updated extracted_data and notes

Extracted data so far:
{extracted_data}

Notes from previous pages:
{notes}

Document:
{parse_response_markdown}
"""

    def __init__(
        self,
        base_structured_output_service: LLMStructuredOutputService[T],
        page_threshold: int = 1,
        multipage_prompt: Optional[str] = None,
    ):
        """
        Initialize with an existing structured output service.

        Args:
            base_structured_output_service: The underlying structured output service
            page_threshold: Number of pages above which to use multipage processing
            multipage_prompt: Custom prompt template for multipage processing
        """
        super().__init__(
            response_type=base_structured_output_service.response_type,
            model=base_structured_output_service.model,
            prompt=base_structured_output_service.prompt,
            cache=base_structured_output_service.cache
        )
        self.base_structured_output_service = base_structured_output_service
        self.page_threshold = page_threshold
        self.multipage_prompt = multipage_prompt if multipage_prompt is not None else self.DEFAULT_MULTIPAGE_PROMPT
        # Perhaps we should add "Multipage" to the service name
        self.service_name = base_structured_output_service.service_name

    def _get_page_context(
        self,
        pages: List[str],
        page_num: int,
        total_pages: int,
    ) -> str:
        """
        Get the context window for a page (previous, current, next).

        Args:
            pages: List of all page markdowns
            page_num: Index of the current page (1-indexed)
            total_pages: Total number of pages

        Returns:
            Formatted context with page markers
        """
        context_parts = []

        # Previous page (if exists)
        if page_num > 1:
            context_parts.append(
                f"--- PREVIOUS PAGE ({page_num - 1}/{total_pages}) ---\n{pages[page_num - 2]}\n"
            )

        # Current page
        context_parts.append(
            f"--- CURRENT PAGE ({page_num}/{total_pages}) ---\n{pages[page_num - 1]}\n"
        )

        # Next page (if exists)
        if page_num < len(pages):
            context_parts.append(
                f"--- NEXT PAGE ({page_num+1}/{total_pages}) ---\n{pages[page_num]}\n"
            )

        return "\n\n".join(context_parts)

    def _create_wrapper_type(self) -> Type[BaseModel]:
        """
        Create a wrapper type where extracted_data has all fields as optional.
        This allows the LLM to gradually fill in required fields across multiple pages.

        Returns:
            Wrapper model with optional extracted_data fields
        """
        from typing import Optional as TypingOptional

        # Create a version of response_type with all fields optional
        optional_fields = {}
        for field_name, field_info in self.response_type.model_fields.items():
            optional_fields[field_name] = (
                TypingOptional[field_info.annotation],
                Field(default=None, description=field_info.description)
            )

        OptionalResponseType = create_model(
            f"Optional{self.response_type.__name__}",
            **optional_fields
        )

        return StructuredOutputWrapper[OptionalResponseType]

    async def _extract_multipage(
        self,
        filename: str,
        pages: List[str],
    ) -> T:
        """
        Extract structured data from multiple pages iteratively.

        Args:
            filename: Name of the source file
            pages: List of page markdowns

        Returns:
            Final structured output with all required fields populated
        """
        # Initialize with empty data and notes
        current_data = None
        notes = ""

        total_pages = len(pages)

        # Create the extraction wrapper type with optional fields
        wrapper_type = self._create_wrapper_type()

        # Save original prompt and response type
        original_prompt = self.base_structured_output_service.prompt
        original_response_type = self.base_structured_output_service.response_type

        try:
            # Process each page
            for page_num, page_content in enumerate(pages, start=1):
                # Get context window (previous, current, next pages)
                page_context = self._get_page_context(pages, page_num, total_pages)

                # Format the multipage prompt with current state
                current_data_str = str(current_data.model_dump()) if current_data else "None"
                notes_str = notes if notes else "None"

                formatted_prompt = self.multipage_prompt.format(
                    current_page=page_num,
                    total_pages=total_pages,
                    extracted_data=current_data_str,
                    notes=notes_str,
                    parse_response_markdown="{parse_response_markdown}"  # Keep placeholder for format_prompt
                )

                # Temporarily override the prompt and response type
                self.base_structured_output_service.prompt = formatted_prompt
                self.base_structured_output_service.response_type = wrapper_type

                # Create ParseResponse with the page context
                temp_parse_response = ParseResponse(markdown=page_context)

                # Create a filename for this page
                # This is not important, the cache uses the full filename
                name, dot, ext = filename.rpartition(".")
                page_filename = f"{name}_page_{page_num}.{ext}"

                # Extract using the base service
                result = await self.base_structured_output_service.extract(
                    filename=page_filename,
                    parse_response=temp_parse_response
                )

                # Update current data and notes
                current_data = result.extracted_data
                notes = result.notes
        finally:
            # Restore original prompt and response type
            self.base_structured_output_service.prompt = original_prompt
            self.base_structured_output_service.response_type = original_response_type

        # Convert the optional model to the final required model
        # This will validate that all required fields are now populated
        if current_data is None:
            raise ValueError("No data was extracted from the document")

        final_data_dict = current_data.model_dump()
        return self.response_type(**final_data_dict)

    async def extract(
        self,
        filename: str,
        parse_response: ParseResponse,
    ) -> T:
        """
        Extract structured data from parsed content.

        For documents with pages above the threshold, uses iterative multipage processing.
        Otherwise, uses the base structured output service directly.

        Args:
            filename: Name of the source file
            parse_response: Parsed document with markdown content

        Returns:
            Structured output
        """
        # Split by pages
        pages = parse_response.markdown.split(PAGE_BREAK)

        # Check if we should use multipage processing
        if len(pages) < self.page_threshold:
            # Use base service directly
            return await self.base_structured_output_service.extract(
                filename=filename,
                parse_response=parse_response
            )

        # Use multipage processing
        return await self._extract_multipage(filename, pages)
