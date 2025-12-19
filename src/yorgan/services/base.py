from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, List, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel, Field, create_model

from yorgan.cache import NullCache, BaseCache, SimpleMemoryCacheWithPersistence, cache_result
from yorgan.datamodels import ParseResponse, Metadata
from yorgan.services.utils import count_pdf_pages, get_mime_type, split_pdf_pages

if TYPE_CHECKING:
    OptionalCacheType = Optional[BaseCache | SimpleMemoryCacheWithPersistence]


class HasMarkdown(Protocol):
    """A protocol for any object that has a 'markdown' string attribute."""
    markdown: str


T = TypeVar('T', bound=BaseModel)
PARSE_T = TypeVar('PARSE_T', bound=ParseResponse)

PAGE_BREAK = "<!-- PAGE BREAK -->"


class BaseService(ABC, Generic[T]):
    """
    Abstract base class for all Services that have a response type.
    """

    def __init__(self, response_type: Type[T], cache: OptionalCacheType = None):
        """
        Args:
            response_type: The Pydantic model class for the service's response
            cache: Optional cache instance for storing results. Defaults to NullCache.
        """
        self.response_type = response_type
        self.cache = cache or NullCache()

    def __init_subclass__(subclass):  # type: ignore
        """
        Automatically set the service_name attribute for all subclasses.
        """
        super().__init_subclass__()
        subclass.service_name = subclass.__qualname__


class ParseService(BaseService[PARSE_T]):
    """
    Service that performs Parsing or Optical Character Recognition (OCR) of a document.
    """

    @abstractmethod
    async def parse(
            self,
            filename: str,
            content: bytes,
            *kwargs,
    ) -> PARSE_T:
        """
        Processes document content and returns a parsed representation.

        Args:
            filename: Name of the file being parsed
            content: Raw document content as bytes
            **kwargs: Additional arguments for parsing

        Returns:
            Parsed response containing markdown representation of the document
        """
        ...

    @cache_result(key_params=['filename'])
    async def __call__(
            self,
            filename: str,
            content: bytes,
            **kwargs
    ) -> PARSE_T:
        """
        Cached entry point for parsing a document.

        Args:
            filename: Name of the file being parsed
            content: Raw document content as bytes
            **kwargs: Additional arguments passed to parse()

        Returns:
            Parsed response containing markdown representation of the document
        """
        return await self.parse(filename, content, **kwargs)


class LLMParseService(ParseService[PARSE_T]):
    """
    Abstract base for parse services that use an LLM/VLM with a prompt template.

    For PDFs with many pages (configurable threshold),
    each page is parsed separately, then combined with page breaks.
    For images and other documents, the LLM/VLM is prompted directly.
    """

    DEFAULT_PROMPT: str = f"""\
Your task is to extract the text from the attached document.
Format it nicely as a markdown.
Make sure to include the newline symbols.
Parse tables properly.
Insert the following page break between consecutive pages:

{PAGE_BREAK}

"""

    def __init__(
        self,
        response_type: Type[PARSE_T],
        model: str,
        prompt: Optional[str] = None,
        cache: OptionalCacheType = None,
        page_threshold: int = 10
    ):
        """
        Args:
            response_type: The Pydantic model class for the service's response
            model: Name/identifier of the LLM/VLM model to use
            prompt: Custom prompt template. Defaults to DEFAULT_PROMPT.
            cache: Optional cache instance for storing results
            page_threshold: Maximum number of pages to parse in one call. PDFs exceeding
                this are split and parsed page-by-page. Defaults to 10.
        """
        super().__init__(response_type=response_type, cache=cache)
        self.model = model
        self.prompt = prompt if prompt is not None else self.DEFAULT_PROMPT
        self.page_threshold = page_threshold

    def _should_process_multipage(self, filename: str, content: bytes) -> bool:
        """
        Check if the content should be processed as multi-page.

        Returns True only for PDFs with more than page_threshold pages.
        Images and other document types return False.

        Args:
            filename: Name or path of the file to check. The extension is used
                to guess the MIME type.
            content: Document content as bytes

        Returns:
            True if it's a PDF with more than page_threshold pages, False otherwise
        """
        mime_type = get_mime_type(filename)

        # Only process PDFs as multipage
        if mime_type != "application/pdf":
            return False

        # Check if PDF has more than page_threshold pages
        page_count = count_pdf_pages(content)
        return page_count > self.page_threshold

    @cache_result(key_params=['filename'])
    async def _parse_single_page(
        self,
        filename: str,
        content: bytes,
        **kwargs
    ) -> PARSE_T:
        """
        Cached wrapper for parsing a single page.

        Args:
            filename: Name of the file being parsed
            content: Page content as bytes
            **kwargs: Additional arguments passed to parse()

        Returns:
            Parsed response for the single page
        """
        return await self.parse(filename, content, **kwargs)

    @cache_result(key_params=['filename'])
    async def __call__(
            self,
            filename: str,
            content: bytes,
            **kwargs
    ) -> PARSE_T:
        """
        Parse a document. If it's a multi-page PDF, each page is parsed individually.
        Otherwise, the LLM/VLM is prompted directly.

        Args:
            filename: Name of the file
            content: Full document content as bytes
            **kwargs: Additional arguments passed to parse()

        Returns:
            ParseResponse with markdown content
        """
        # Check if this should be processed as multi-page
        if not self._should_process_multipage(filename, content):
            # Not a long PDF (could be image, small PDF, etc.)
            return await self.parse(
                filename=filename,
                content=content,
                **kwargs
            )

        # Multi-page PDF: split and process each page
        pages = split_pdf_pages(content)

        # Parse each page separately
        page_markdowns = []
        for page_num, page_content in enumerate(pages, start=1):
            print(f"Page {page_num}/{len(pages)}")
            # Create a filename for this page
            name, dot, ext = filename.rpartition(".")
            page_filename = f"{name}_page_{page_num}.{ext}"

            # Parse this single page
            page_response = await self._parse_single_page(
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


class StructuredOutputService(BaseService[T]):
    """
    Service that extracts structured output using a language model.
    """

    @abstractmethod
    async def extract(
        self,
        filename: str,
        parse_response: ParseResponse,
    ) -> T:
        """
        Generates a structured Pydantic object from parsed content.

        Args:
            filename: Name of the source file
            parse_response: Parsed document with markdown content

        Returns:
            Structured output as defined by response_type
        """
        ...

    @cache_result(key_params=['filename'])
    async def __call__(
            self,
            filename: str,
            parse_response: ParseResponse,
            **kwargs
    ) -> T:
        """
        Cached entry point for extracting structured data.

        Args:
            filename: Name of the source file
            parse_response: Parsed document with markdown content
            **kwargs: Additional arguments (currently unused)

        Returns:
            Structured output as defined by response_type
        """
        return await self.extract(filename, parse_response, **kwargs)


class LLMStructuredOutputService(StructuredOutputService[T]):
    """
    Abstract base for structured output services that use an LLM with a prompt template.

    For PDFs with many pages (configurable threshold),
    this service processes pages iteratively with a sliding window context
    (previous, current, next page). The model incrementally builds the structured output
    and maintains notes about information that may be relevant for subsequent pages.
    """

    DEFAULT_PROMPT: str = """\
You extract structured information from documents. Review the following document and return the extracted data in the specified format.
When you extract numbers normalize every extracted numeric value to its true numeric form by applying the unit multiplier,
(e.g., thousands x1000; millions x1000000; milli x0.001; micro x0.000001),
so the output contains the correctly scaled integer or floating-point value.

Document:
{parse_response_markdown}
"""

    DEFAULT_MULTIPAGE_PROMPT: str = """\
You are extracting structured information from a multi-page document, processing it page by page.
Unless it is the first or the last page, you will receive the previous, current, and next pages.
You will also receive the extracted data so far, and relevant notes from previous pages.

When you extract numbers normalize every extracted numeric value to its true numeric form by applying the unit multiplier,
(e.g., thousands x1000; millions x1000000; milli x0.001; micro x0.000001),
so the output contains the correctly scaled integer or floating-point value.

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
        response_type: Type[T],
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        multipage_prompt: Optional[str] = None,
        page_threshold: int = 10,
        cache: OptionalCacheType = None,
    ):
        """
        Args:
            response_type: The Pydantic model class for the structured output
            model: Name/identifier of the LLM model to use
            prompt: Custom prompt template for single-page extraction. Defaults to DEFAULT_PROMPT.
            multipage_prompt: Custom prompt template for multi-page extraction. Defaults to DEFAULT_MULTIPAGE_PROMPT.
            page_threshold: Maximum number of pages to extract in one call. Documents exceeding
                this are processed page-by-page with sliding context. Defaults to 10.
            cache: Optional cache instance for storing results
        """
        super().__init__(response_type=response_type, cache=cache)
        if model is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a model. "
                "Must pass 'model' to __init__"
            )
        self.model = model
        self.prompt = prompt if prompt is not None else self.DEFAULT_PROMPT
        self.multipage_prompt = multipage_prompt if multipage_prompt is not None else self.DEFAULT_MULTIPAGE_PROMPT
        self.page_threshold = page_threshold

    def format_prompt(self, parse_response: ParseResponse) -> str:
        """
        Format the prompt template with the parsed document content.

        Args:
            parse_response: Parsed document containing markdown

        Returns:
            Formatted prompt string with markdown content inserted
        """
        # markdown may contain {} brackets
        return self.prompt.replace("{parse_response_markdown}", parse_response.markdown)

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
        Create a wrapper type where extracted_data has all fields as optional,
        and where notes are added to keep track of information that is relevant.
        This allows the LLM to gradually fill in required fields across multiple pages.

        Returns:
            Wrapper model with optional extracted_data fields and notes field
        """

        class StructuredOutputWrapper(BaseModel, Generic[T]):
            """Wrapper model that includes the structured output and notes for iterative extraction."""
            extracted_data: T = Field(description="The structured data extracted so far")
            notes: str = Field(
                default="",
                description="Notes about information that may be useful for extraction in subsequent pages"
            )

        # Create a version of response_type with all fields optional
        optional_fields = {}
        for field_name, field_info in self.response_type.model_fields.items():
            optional_fields[field_name] = (
                Optional[field_info.annotation],
                Field(default=None, description=field_info.description)
            )

        OptionalResponseType = create_model(
            f"Optional{self.response_type.__name__}",
            **optional_fields
        )

        return StructuredOutputWrapper[OptionalResponseType]

    @cache_result(key_params=['filename'])
    async def _extract_single_page(
            self,
            filename: str,
            parse_response: ParseResponse,
            **kwargs
    ) -> T:
        """
        Cached wrapper for extracting from a single page.

        Args:
            filename: Name of the file being processed
            parse_response: Parsed content for this page
            **kwargs: Additional arguments passed to extract()

        Returns:
            Structured output for the single page
        """
        return await self.extract(filename, parse_response, **kwargs)

    @cache_result(key_params=['filename'])
    async def __call__(
        self,
        filename: str,
        parse_response: ParseResponse,
    ) -> T:
        """
        Extract structured data from parsed content.

        For documents with pages above the threshold, extracts each page separately
        using a sliding window context (previous, current, next page). The model
        maintains accumulated data and notes across pages.

        For shorter documents, prompts the LLM directly.

        Args:
            filename: Name of the source file
            parse_response: Parsed document with markdown content

        Returns:
            Structured output as defined by response_type
        """
        # Split by pages
        pages = parse_response.markdown.split(PAGE_BREAK)

        # Check if we should use multipage processing
        if len(pages) < self.page_threshold:
            # Not a long PDF (could be image, small PDF, etc.)
            return await self.extract(
                filename=filename,
                parse_response=parse_response
            )

        # Initialize with empty data and notes
        current_data = None
        notes = ""

        total_pages = len(pages)

        # Create the extraction wrapper type with optional fields
        wrapper_type = self._create_wrapper_type()

        # Save original prompt and response type
        original_prompt = self.prompt
        original_response_type = self.response_type

        try:
            # Process each page
            for page_num, page_content in enumerate(pages, start=1):
                print(f"Page {page_num}/{len(pages)}")
                # Get context window (previous, current, next pages)
                page_context = self._get_page_context(pages, page_num, total_pages)

                # Format the multipage prompt with current state
                current_data_str = str(current_data.model_dump(mode="json")) if current_data else "None"
                notes_str = notes if notes else ""

                formatted_prompt = self.multipage_prompt.format(
                    current_page=page_num,
                    total_pages=total_pages,
                    extracted_data=current_data_str,
                    notes=notes_str,
                    parse_response_markdown="{parse_response_markdown}"  # Keep placeholder for format_prompt
                )

                # Temporarily override the prompt and response type
                self.prompt = formatted_prompt
                self.response_type = wrapper_type

                # Create ParseResponse with the page context
                temp_parse_response = ParseResponse(markdown=page_context)

                # Create a filename for this page
                name, dot, ext = filename.rpartition(".")
                page_filename = f"{name}_page_{page_num}.{ext}"

                # Extract this single page
                result = await self._extract_single_page(
                    filename=page_filename,
                    parse_response=temp_parse_response
                )

                # Update current data and notes
                current_data = result.extracted_data
                notes = result.notes
        finally:
            # Restore original prompt and response type
            self.prompt = original_prompt
            self.response_type = original_response_type

        # Convert the optional model to the final required model
        # This will validate that all required fields are now populated
        if current_data is None:
            raise ValueError("No data was extracted from the document")

        final_data_dict = current_data.model_dump()
        return self.response_type(**final_data_dict)


R = TypeVar("R", bound=BaseModel, covariant=True)


class ParseExtractProtocol(Protocol[R]):
    """
    Protocol defining the interface for ParseExtract services.

    Any object implementing these methods can be used as ParseExtractService.
    """

    async def parse_extract(self, filename: str, content: bytes) -> R:
        """
        Parse and extract structured data in one step.

        Args:
            filename: Name of the file being processed
            content: Raw document content as bytes

        Returns:
            Structured output
        """
        ...

    async def __call__(self, filename: str, content: bytes) -> R:
        """
        Cached entry point for parse and extract.

        Args:
            filename: Name of the file being processed
            content: Raw document content as bytes

        Returns:
            Structured output
        """
        ...


class ParseExtractService(BaseService[T]):
    """
    Service that parses and extracts structured output using a language model in one step.
    """

    @abstractmethod
    async def parse_extract(
        self,
        filename: str,
        content: bytes,
        mime_type: Optional[str] = None
    ) -> T:
        """
        Generates a structured Pydantic object from file content.

        Args:
            filename: Name of the file being processed
            content: Raw document content as bytes
            mime_type: Optional MIME type override

        Returns:
            Structured output as defined by response_type
        """
        ...

    @cache_result(['filename'])
    async def __call__(
            self,
            filename: str,
            content: bytes
    ) -> T:
        """
        Cached entry point for parse and extract.

        Args:
            filename: Name of the file being processed
            content: Raw document content as bytes

        Returns:
            Structured output as defined by response_type
        """
        return await self.parse_extract(filename, content)


class ParseExtractPipelineService(Generic[T]):
    """
    An end-to-end document parsing Service using two steps: parse and structured output.
    It orchestrates a ParseService and a StructuredOutputService.
    """

    def __init__(
        self,
        parse_service: ParseService[PARSE_T],
        structured_output_service: StructuredOutputService[T],
    ):
        """
        Args:
            parse_service: Service instance for the parse step
            structured_output_service: Service instance for the extract step
        """
        self.parse_service = parse_service
        self.structured_output_service = structured_output_service

    async def parse_extract(
        self,
        filename: str,
        content: bytes,
    ) -> T:
        """
        Performs end-to-end document parsing and structuring.
        Saves immediate intermediate results in the object.

        Args:
            filename: Name of the file being processed
            content: Raw document content as bytes

        Returns:
            Structured output as defined by the structured_output_service's response type
        """
        ocr_response = await self.parse_service(filename, content)
        self.ocr_response = ocr_response
        structured_output_response = await self.structured_output_service(filename, ocr_response)
        self.structured_output_response = structured_output_response

        return structured_output_response

    async def __call__(
            self,
            filename: str,
            content: bytes,
    ) -> T:
        """
        Entry point for the pipeline.

        Args:
            filename: Name of the file being processed
            content: Raw document content as bytes

        Returns:
            Structured output from the complete pipeline
        """
        structured_output = await self.parse_extract(filename, content)
        return structured_output
