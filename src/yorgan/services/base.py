from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel, Field, create_model

from yorgan.cache import (BaseCache, NullCache,
                          SimpleMemoryCacheWithPersistence, cache_result)
from yorgan.datamodels import ParseResponse, add_explicit_page_numbering
from yorgan.services.utils import count_pdf_pages, get_mime_type, split_pdf

if TYPE_CHECKING:
    OptionalCacheType = Optional[BaseCache | SimpleMemoryCacheWithPersistence]


class HasMarkdown(Protocol):
    """A protocol for any object that has a 'markdown' string attribute."""
    markdown: str


T = TypeVar('T', bound=BaseModel)
PARSE_T = TypeVar('PARSE_T', bound=ParseResponse)


class BaseService(ABC, Generic[T]):
    """
    Abstract base class for all Services that have a response type.
    """

    def __init__(self, response_type: Type[T], cache: OptionalCacheType = None):
        """
        Args:
            response_type: The Pydantic model class for the service's response
            cache: Optional cache instance for storing results. Defaults to NullCache
        """
        self.response_type = response_type
        self.cache = cache or NullCache()

    def __init_subclass__(subclass):  # type: ignore
        """
        Automatically set the service_name attribute for all subclasses.
        """
        super().__init_subclass__()
        subclass.service_name = subclass.__qualname__


class BaseLLM(ABC):
    """
    Abstract base class for LLM interactions.
    Subclasses implement the generate method for specific LLM providers.
    """

    @abstractmethod
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
        Generate structured output from content using an LLM.

        Args:
            model: Name/identifier of the LLM model to use
            prompt: The prompt to send to the LLM
            response_type: The Pydantic model class for the response
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            content: Raw document content as bytes, or None if only using the prompt
            **kwargs: Additional arguments for the LLM

        Returns:
            Structured output as defined by response_type
        """
        ...


class ParseService(BaseService[PARSE_T]):
    """
    Service that performs Parsing or Optical Character Recognition (OCR) of a document.
    """

    @cache_result(key_params=['filename'])
    async def __call__(
        self,
        filename: str,
        content: bytes,
        **kwargs
    ) -> PARSE_T:
        """
        Cached entry point for parsing a document.
        Processes document content and returns a parsed representation.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            content: Raw document content as bytes
            **kwargs: Additional arguments

        Returns:
            Parsed response containing markdown representation of the document
        """
        ...


class LLMParseService(ParseService[PARSE_T]):
    """
    Parse service that uses an LLM/VLM with a prompt template.

    For PDFs with many pages (configurable threshold),
    the pages are parsed in batches, then combined with page breaks.
    For images and other documents, the LLM/VLM is prompted directly.
    """

    DEFAULT_PROMPT: str = f"""\
Your task is to extract the text from the attached document.
Format it nicely as a markdown.
Make sure to include the newline symbols.
Parse tables properly.
Insert the following page break between consecutive pages:

{ParseResponse.PAGE_BREAK}

"""

    DEFAULT_BATCH_PROMPT: str = DEFAULT_PROMPT

    def __init__(
        self,
        response_type: Type[PARSE_T],
        llm: BaseLLM,
        model: str,
        prompt: Optional[str] = None,
        batch_prompt: Optional[str] = None,
        page_threshold: int = 10,
        batch_window: int = 1,
        batch_overlap: int = 0,
        cache: OptionalCacheType = None
    ):
        """
        Args:
            response_type: The Pydantic model class for the service's response
            llm: BaseLLM instance for LLM interactions
            model: Name/identifier of the LLM/VLM model to use
            prompt: Custom prompt template. Defaults to DEFAULT_PROMPT
            batch_prompt: Custom batch prompt template. Defaults to DEFAULT_BATCH_PROMPT
            page_threshold: Maximum number of pages to parse in one call. PDFs exceeding
                this are split and parsed in batches. Defaults to 10
            batch_window: The length of the sliding window. Defaults to 1.
                Note that pricing is based on tokens. No reason to use higher window
            batch_overlap: The length of the overlap. Defaults to 0. Only 0 is currently supported
            cache: Optional cache instance for storing results
        """
        super().__init__(response_type=response_type, cache=cache)
        self.llm = llm
        self.model = model
        self.prompt = prompt if prompt else self.DEFAULT_PROMPT
        self.batch_prompt = batch_prompt if batch_prompt else self.DEFAULT_BATCH_PROMPT
        self.page_threshold = page_threshold
        self.batch_window = batch_window
        self.batch_overlap = batch_overlap
        if self.batch_window != 1:
            raise NotImplementedError("This service currently only supports batch_window=1. Note that pricing is based on tokens.")
        if self.batch_overlap != 0:
            raise NotImplementedError("This service currently only supports batch_overlap=0.")

    def _should_process_in_batches(self, filename: str, content: bytes) -> bool:
        """
        Check if the content should be processed in batches.

        Returns True only for PDFs with more than page_threshold pages.
        Images and other document types return False.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            content: Document content as bytes

        Returns:
            True if it's a PDF with more than page_threshold pages, False otherwise
        """
        mime_type = get_mime_type(filename)

        # Only process PDFs in batches
        if mime_type != "application/pdf":
            return False

        # Check if PDF has more than page_threshold pages
        page_count = count_pdf_pages(content)
        return page_count > self.page_threshold

    def _create_batches(
        self,
        filename: str,
        content: bytes
    ) -> list[tuple[str, bytes]]:
        """
        Splits the PDF content into batches and prepares filename for every batch.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            content: Document content as bytes

        Returns:
            A list of filenames and content batches
        """
        batches = []
        batch_contents = split_pdf(content, self.batch_window, self.batch_overlap)
        page_count = count_pdf_pages(content)
        for batch_index, batch_content in enumerate(batch_contents):
            start_page = batch_index * (self.batch_window - self.batch_overlap) + 1
            end_page = min(start_page + self.batch_window - 1, page_count)

            name, dot, ext = filename.rpartition(".")
            batch_filename = f"{name}_pages_{start_page}_{end_page}.{ext}"

            batches.append((batch_filename, batch_content))
        return batches

    async def _parse(
        self,
        filename: str,
        content: bytes,
        **kwargs
    ) -> PARSE_T:
        """
        Parse a document using the LLM.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            content: Raw document content as bytes
            **kwargs: Additional arguments passed to LLM

        Returns:
            Parsed response containing markdown representation
        """
        return await self.llm.generate(
            model=self.model,
            prompt=self.prompt,
            response_type=self.response_type,
            filename=filename,
            content=content,
            **kwargs
        )

    @cache_result(key_params=['batch_filename'])
    async def _parse_single_batch(
        self,
        batch_filename: str,
        batch_content: bytes,
        **kwargs
    ) -> PARSE_T:
        """
        Parse a single batch of pages from a document using the LLM.

        Args:
            batch_filename: Name of the file being processed. The extension is used
                to guess the MIME type
            batch_content: Content of the current batch as bytes
            **kwargs: Additional arguments passed to LLM

        Returns:
            Parsed response for the single batch
        """
        return await self.llm.generate(
            model=self.model,
            prompt=self.batch_prompt,
            response_type=self.response_type,
            filename=batch_filename,
            content=batch_content,
            **kwargs
        )

    @cache_result(key_params=['filename'])
    async def __call__(
        self,
        filename: str,
        content: bytes,
        **kwargs
    ) -> PARSE_T:
        """
        Parse a document. If it's a long PDF, the pages are parsed in batches.
        Otherwise, the LLM/VLM is prompted directly.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            content: Full document content as bytes
            **kwargs: Additional arguments passed to LLM

        Returns:
            ParseResponse with markdown content
        """
        # Check if this document should be processed in batches
        if not self._should_process_in_batches(filename, content):
            # Process the whole document
            return await self._parse(
                filename=filename,
                content=content,
                **kwargs
            )

        # Prepare batches
        batches = self._create_batches(filename, content)

        # Process the batches
        batch_markdowns = []
        for batch_filename, batch_content in batches:
            print(f"Parsing {batch_filename} ...")
            # Parse this single batch
            batch_response = await self._parse_single_batch(
                batch_filename=batch_filename,
                batch_content=batch_content,
                **kwargs
            )

            # Extract markdown from the response
            batch_markdowns.append(batch_response.markdown)

        # Combine all batches with page breaks
        combined_markdown = f"\n\n{ParseResponse.PAGE_BREAK}\n\n".join(batch_markdowns)

        # Create the response object
        return self.response_type(markdown=combined_markdown)


class StructuredOutputService(BaseService[T]):
    """
    Service that extracts structured output using a language model.
    """

    @cache_result(key_params=['filename'])
    async def __call__(
        self,
        filename: str,
        parse_response: ParseResponse,
        **kwargs
    ) -> T:
        """
        Cached entry point for extracting structured data.
        Generates a structured Pydantic object from parsed content.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            parse_response: Parsed document with markdown content
            **kwargs: Additional arguments

        Returns:
            Structured output as defined by response_type
        """
        return await self.extract(filename, parse_response, **kwargs)


class LLMStructuredOutputService(StructuredOutputService[T]):
    """
    Structured output service that uses an LLM with a prompt template.

    For PDFs with many pages (configurable threshold),
    this service processes pages iteratively in batches with a sliding window context.
    The model incrementally builds the structured output and
    maintains notes about information that may be relevant for subsequent batches.
    """

    DEFAULT_PROMPT: str = """\
You extract structured information from documents. Review the following document and return the extracted data in the specified format.
When you extract numbers normalize every extracted numeric value to its true numeric form by applying the unit multiplier,
(e.g., thousands x1000; millions x1000000; milli x0.001; micro x0.000001),
so the output contains the correctly scaled integer or floating-point value.

Document:
{parse_response_markdown}
"""

    DEFAULT_BATCH_PROMPT: str = """\
You are extracting structured information from a multi-page document, processing it page by page.
You will receive a window of parsed pages.
You will also receive the extracted data so far, and relevant notes from previous pages.

When you extract numbers normalize every extracted numeric value to its true numeric form by applying the unit multiplier,
(e.g., thousands x1000; millions x1000000; milli x0.001; micro x0.000001),
so the output contains the correctly scaled integer or floating-point value.

Your task:
1. Add any NEW values based on information found on the current batch of pages to the extracted_data
2. PRESERVE previously extracted data, unless you are completely certain that the values need to be changed based on new information
3. Update the notes by adding anything important for subsequent pages (e.g., "table continues", etc),
   or removing notes that are no longer relevant (e.g. remove "table continues" once the table is fully extracted)
4. Return the updated extracted_data and notes

Extracted data so far:
{extracted_data}

Notes from previous pages:
{notes}

Pages:

{batch_parse_response_markdown}
"""

    def __init__(
        self,
        response_type: Type[T],
        llm: BaseLLM,
        model: str,
        prompt: Optional[str] = None,
        batch_prompt: Optional[str] = None,
        page_threshold: int = 10,
        batch_window: int = 6,
        batch_overlap: int = 1,
        cache: OptionalCacheType = None,
    ):
        """
        Args:
            response_type: The Pydantic model class for the structured output
            llm: BaseLLM instance for LLM interactions
            model: Name/identifier of the LLM model to use
            prompt: Custom prompt template. Defaults to DEFAULT_PROMPT
            batch_prompt: Custom batch prompt template. Defaults to DEFAULT_BATCH_PROMPT
            page_threshold: Maximum number of pages to extract in one call. Documents exceeding
                this are processed in batches with sliding context. Defaults to 10
            batch_window: The length of the sliding window. Defaults to 6
            batch_overlap: The length of the overlap. Defaults to 1
            cache: Optional cache instance for storing results
        """
        super().__init__(response_type=response_type, cache=cache)
        self.llm = llm
        self.model = model
        self.prompt = prompt if prompt else self.DEFAULT_PROMPT
        self.batch_prompt = batch_prompt if batch_prompt else self.DEFAULT_BATCH_PROMPT
        self.page_threshold = page_threshold
        self.batch_window = batch_window
        self.batch_overlap = batch_overlap

        # Create response_type_wrapper with optional fields and notes
        self._response_type_wrapper = self._create_response_type_wrapper()

    class _StructuredOutputWrapper(BaseModel, Generic[T]):
        """
        Wrapper model that includes the structured output and notes for iterative extraction.
        Used for batch processing.
        """
        extracted_data: T | None = Field(
            default=None,
            description="The structured data extracted so far"
        )
        notes: str = Field(
            default="",
            description="Notes about information that may be useful for extraction in subsequent pages"
        )

    def _create_response_type_wrapper(self) -> Type[_StructuredOutputWrapper]:
        """
        Create a wrapper type where all response_type fields are optional,
        and where notes are added to keep track of information that is relevant.
        This allows the LLM to gradually fill in required fields across multiple pages.

        Returns:
            Wrapper type with optional response_type fields and notes
        """

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

        return self._StructuredOutputWrapper[OptionalResponseType]

    def _should_process_in_batches(self, filename: str, parse_response: ParseResponse) -> bool:
        """
        Check if the parse response should be processed in batches.

        Returns True only for responses with more than page_threshold pages.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            parse_response: the response from a parse service

        Returns:
            True if it's a response with more than page_threshold pages, False otherwise
        """
        # this will be different once we have chunks
        page_count = len(parse_response.markdown.split(ParseResponse.PAGE_BREAK))

        return page_count > self.page_threshold

    def _create_batches(
        self,
        filename: str,
        parse_response: ParseResponse,
    ) -> list[tuple[str, ParseResponse]]:
        """
        Splits the parse response into batches and prepares filename for every batch.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            parse_response: the response from a parse service

        Returns:
            A list of filenames and parse response batches
        """
        batches = []

        pages = parse_response.markdown.split(ParseResponse.PAGE_BREAK)
        page_count = len(pages)
        for page_num in range(0, page_count - self.batch_overlap, self.batch_window - self.batch_overlap):
            start_page = page_num + 1
            end_page = min(start_page + self.batch_window - 1, page_count)

            name, dot, ext = filename.rpartition(".")
            batch_filename = f"{name}_pages_{start_page}_{end_page}.{ext}"

            parse_responses_window = pages[start_page - 1:end_page]
            combined_markdown = ParseResponse.PAGE_BREAK.join(parse_responses_window)
            batch_parse_response = ParseResponse(markdown=combined_markdown)

            batches.append((batch_filename, batch_parse_response))

        return batches

    def format_prompt(
        self,
        parse_response: ParseResponse
    ) -> str:
        """
        Format the prompt template with the parsed document content.

        Args:
            parse_response: Parsed document containing markdown

        Returns:
            Formatted prompt string with markdown content inserted
        """
        return self.prompt.format(parse_response_markdown=parse_response.markdown)

    def format_batch_prompt(
        self,
        batch_parse_response: ParseResponse,
        current_output: _StructuredOutputWrapper,
    ) -> str:
        """
        Format the batch prompt template with the current batch of pages and extracted data so far.

        Args:
            batch_parse_response: The current batch of pages as ParseResponse
            current_output: Extracted data so far with notes

        Returns:
            Formatted prompt string with markdown content inserted
        """
        return self.batch_prompt.format(
            extracted_data=str(current_output.extracted_data.model_dump(mode="json")) if current_output.extracted_data else "None",
            notes=current_output.notes if current_output.notes else "",
            batch_parse_response_markdown=batch_parse_response.markdown
        )

    async def _extract(
        self,
        filename: str,
        parse_response: ParseResponse,
        **kwargs
    ) -> T:
        """
        Extract structured data from parsed content using the LLM.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            parse_response: Parsed document with markdown content
            **kwargs: Additional arguments passed to LLM

        Returns:
            Structured output as defined by response_type
        """
        # Format the prompt with the markdown content
        formatted_prompt = self.format_prompt(parse_response)

        # Use the LLM to generate structured output
        return await self.llm.generate(
            model=self.model,
            prompt=formatted_prompt,
            response_type=self.response_type,
            **kwargs
        )

    @cache_result(key_params=['batch_filename'])
    async def _extract_single_batch(
        self,
        batch_filename: str,
        batch_parse_response: ParseResponse,
        current_output: _StructuredOutputWrapper,
        **kwargs
    ) -> T:
        """
        Cached wrapper for extracting from a single batch of pages.

        Args:
            batch_filename: Name of the file being processed. The extension is used
                to guess the MIME type
            batch_parse_response: Parsed batch with markdown content 
            current_output: Contains the extracted data so far and notes
            **kwargs: Additional arguments passed to LLM

        Returns:
            Structured output wrapper for the single batch
        """
        # Format the prompt with the markdown content
        formatted_prompt = self.format_batch_prompt(batch_parse_response, current_output)

        # Use the LLM to generate structured output
        return await self.llm.generate(
            model=self.model,
            prompt=formatted_prompt,
            response_type=self._response_type_wrapper,
            **kwargs
        )

    @cache_result(key_params=['filename'])
    async def __call__(
        self,
        filename: str,
        parse_response: ParseResponse,
        **kwargs
    ) -> T:
        """
        Extract structured data from parsed content.

        For documents with pages above the threshold, process the pages in batches.
        The model maintains accumulated data and notes across pages.

        For shorter documents, prompts the LLM directly.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            parse_response: Parsed document with markdown content
            **kwargs: Additional arguments passed to LLM

        Returns:
            Structured output as defined by response_type
        """
        # add explicit page numbering
        # this won't be needed once we have chunks
        add_explicit_page_numbering(parse_response)

        # Check if we should process in batches
        if not self._should_process_in_batches(filename, parse_response):
            # Process the whole document
            return await self._extract(
                filename=filename,
                parse_response=parse_response,
                **kwargs
            )

        # Prepare batches
        batches = self._create_batches(filename, parse_response)

        # Initialize current output with empty extracted data and notes
        current_output = self._response_type_wrapper()

        # Process the batches
        for batch_filename, batch_parse_response in batches:
            print(f"Extracting {batch_filename} ...")

            # Extract this single batch
            current_output = await self._extract_single_batch(
                batch_filename=batch_filename,
                batch_parse_response=batch_parse_response,
                current_output=current_output,
                **kwargs
            )

        final_extracted_data_dict = current_output.extracted_data.model_dump()
        final_response = self.response_type(**final_extracted_data_dict)
        return final_response


R = TypeVar("R", bound=BaseModel, covariant=True)


class ParseExtractProtocol(Protocol[R]):
    """
    Protocol defining the interface for ParseExtract services.

    Any object implementing these methods can be used as ParseExtractService.
    """

    async def __call__(
        self,
        filename: str,
        content: bytes
    ) -> R:
        """
        Cached entry point for parse and extract.
        Performs two step document parsing and extracting.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            content: Raw document content as bytes

        Returns:
            Structured output
        """
        ...


class ParseExtractService(BaseService[T]):
    """
    Service that parses and extracts structured output in one step.
    """

    @cache_result(['filename'])
    async def __call__(
        self,
        filename: str,
        content: bytes
    ) -> T:
        """
        Cached entry point for parse and extract.
        Performs two step document parsing and extracting.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            content: Raw document content as bytes

        Returns:
            Structured output as defined by response_type
        """
        ...


class ParseExtractPipeline(Generic[T]):
    """
    An end-to-end document parsing pipeline using two steps: parse and extract.
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

    async def __call__(
        self,
        filename: str,
        content: bytes,
    ) -> T:
        """
        Cached entry point for parse and extract.
        Performs two step document parsing and extracting.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            content: Raw document content as bytes

        Returns:
            Structured output as defined by the structured_output_service's response type
        """
        ocr_response = await self.parse_service(filename, content)
        self.ocr_response = ocr_response
        structured_output_response = await self.structured_output_service(filename, ocr_response)
        self.structured_output_response = structured_output_response

        return structured_output_response


class LLMParseExtractPipelineService(ParseExtractService[T]):
    """
    A parse extract pipeline that uses an LLM.
    """

    def __init__(
        self,
        response_type: Type[T],
        parse_service: LLMParseService[ParseResponse],
        structured_output_service: LLMStructuredOutputService[T],
        cache: OptionalCacheType = None,
    ):
        """
        Args:
            response_type: The Pydantic model class for the service's response
            parse_service: Service instance for the parse step
            structured_output_service: Service instance for the extract step
            cache: Optional cache instance for storing results. Defaults to NullCache
        """
        super().__init__(response_type=response_type, cache=cache)
        self.pipeline = ParseExtractPipeline(parse_service=parse_service, structured_output_service=structured_output_service)

    async def __call__(
        self,
        filename: str,
        content: bytes,
    ) -> T:
        """
        Cached entry point for parse and extract.
        Performs two step document parsing and extracting.

        Args:
            filename: Name of the file being processed. The extension is used
                to guess the MIME type
            content: Raw document content as bytes

        Returns:
            Structured output as defined by the structured_output_service's response type
        """
        return await self.pipeline(filename, content)
