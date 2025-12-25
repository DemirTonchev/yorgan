from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, Protocol, Type, TypeVar, cast

from pydantic import BaseModel, Field, create_model

from yorgan.cache import (BaseCache, NullCache,
                          SimpleMemoryCacheWithPersistence, cache_result)
from yorgan.datamodels import Markdown, ParseMetadata, ParseResponse, add_explicit_page_numbering
from yorgan.services.utils import count_pdf_pages, get_mime_type, split_pdf

if TYPE_CHECKING:
    OptionalCacheType = Optional[BaseCache | SimpleMemoryCacheWithPersistence]


class HasMarkdown(Protocol):
    """A protocol for any object that has a 'markdown' string attribute."""
    markdown: str


T = TypeVar('T', bound=BaseModel)
S = TypeVar('S', bound=BaseModel)
R = TypeVar('R', bound=BaseModel)
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
    # Subclasses should define a default model
    DEFAULT_MODEL: str

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

    @abstractmethod
    def get_metadata(self) -> ParseMetadata:
        ...


class LLMServiceMixin:
    """
    Mixin that provides LLM and model management for LLM-based services.

    Subclasses can set LLM_TYPE as class attribute to enable provider shortcuts.
    """

    # Class attributes for provider-specific subclasses
    LLM_TYPE: Optional[Type[BaseLLM]] = None

    def _initialize_llm_params(
        self,
        llm: Optional[BaseLLM] = None,
        model: Optional[str] = None
    ) -> tuple[BaseLLM, str]:
        """
        Initialize LLM and model parameters with fallback to defaults.

        If llm is not provided, attempts to instantiate LLM_TYPE.
        If model is not provided, uses llm.DEFAULT_MODEL.

        Args:
            llm: BaseLLM instance for LLM interactions. If None, uses LLM_TYPE.
            model: Name/identifier of the LLM model to use. If None, uses llm.DEFAULT_MODEL.

        Returns:
            Tuple of (llm, model) with defaults applied.

        Raises:
            ValueError: If llm is None and LLM_TYPE is not set as a class attribute.
        """
        if llm is None:
            if self.LLM_TYPE is not None:
                llm = self.LLM_TYPE()
            else:
                raise ValueError(
                    f"{self.__class__.__name__} requires either an 'llm' parameter or LLM_TYPE to be set as a class attribute"
                )

        if model is None:
            model = llm.DEFAULT_MODEL

        return llm, model


class ParseService(BaseService[PARSE_T]):
    """
    Service that performs Parsing or Optical Character Recognition (OCR) of a document.
    """
    @abstractmethod
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


class LLMParseService(ParseService[PARSE_T], LLMServiceMixin):
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
        response_type: Type[PARSE_T] = ParseResponse,
        llm: Optional[BaseLLM] = None,
        model: Optional[str] = None,
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
        self.llm, self.model = self._initialize_llm_params(llm, model)
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
        markdown = await self.llm.generate(
            model=self.model,
            prompt=self.prompt,
            response_type=Markdown,
            filename=filename,
            content=content,
            **kwargs
        )
        metadata = self.llm.get_metadata()
        return self.response_type(**markdown.model_dump(), metadata=metadata)

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
        markdown = await self.llm.generate(
            model=self.model,
            prompt=self.batch_prompt,
            response_type=Markdown,
            filename=batch_filename,
            content=batch_content,
            **kwargs
        )
        metadata = self.llm.get_metadata()
        return self.response_type(**markdown.model_dump(), metadata=metadata)

    def _combine_batch_responses(self, batch_responses: list[PARSE_T]) -> PARSE_T:
        combined_markdown = f"\n\n{ParseResponse.PAGE_BREAK}\n\n".join([batch_response.markdown for batch_response in batch_responses])
        combined_metadata = ParseMetadata(
            credit_usage=sum(b.metadata.credit_usage for b in batch_responses if b.metadata.credit_usage is not None) or None,
            duration_ms=sum(b.metadata.duration_ms for b in batch_responses if b.metadata.duration_ms is not None) or None,
            filename=batch_responses[0].metadata.filename if batch_responses else None,
            job_id=None,  # Multiple jobs, no single ID
            org_id=batch_responses[0].metadata.org_id if batch_responses else None,
            page_count=sum(b.metadata.page_count for b in batch_responses if b.metadata.page_count is not None) or None,
            version=batch_responses[0].metadata.version if batch_responses else None,
            failed_pages=[
                page
                for b in batch_responses
                if b.metadata.failed_pages
                for page in b.metadata.failed_pages
            ] or None,
            input_token_count=sum(b.metadata.input_token_count for b in batch_responses if b.metadata.input_token_count is not None) or None,
            output_token_count=sum(b.metadata.output_token_count for b in batch_responses if b.metadata.output_token_count is not None) or None,
        )
        # Create the response object
        return self.response_type(markdown=combined_markdown, metadata=combined_metadata)

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
        batch_responses = []
        for batch_filename, batch_content in batches:
            print(f"Parsing {batch_filename} ...")
            # Parse this single batch
            batch_response = await self._parse_single_batch(
                batch_filename=batch_filename,
                batch_content=batch_content,
                **kwargs
            )
            batch_responses.append(batch_response)

        # Combine all batches
        combined_response = self._combine_batch_responses(batch_responses)
        return combined_response


class StructuredOutputService(BaseService[T]):
    """
    Service that extracts structured output using a language model.
    """

    @abstractmethod
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
        ...


class _StructuredOutputWrapper(BaseModel, Generic[R]):
    """
    Wrapper model that includes the structured output and notes for iterative extraction.
    Used for batch processing.
    In principle can be used for wrapping any BaseModel
    """
    extracted_data: R | None = Field(
        default=None,
        description="The structured data extracted so far"
    )
    notes: str = Field(
        default="",
        description="Notes about information that may be useful for extraction in subsequent pages"
    )


class LLMStructuredOutputService(StructuredOutputService[T], LLMServiceMixin):
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
        llm: Optional[BaseLLM] = None,
        model: Optional[str] = None,
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
        self.llm, self.model = self._initialize_llm_params(llm, model)
        self.prompt = prompt if prompt else self.DEFAULT_PROMPT
        self.batch_prompt = batch_prompt if batch_prompt else self.DEFAULT_BATCH_PROMPT
        self.page_threshold = page_threshold
        self.batch_window = batch_window
        self.batch_overlap = batch_overlap

        # Create a helper service for processing a single batch
        self._extract_single_batch = self._ExtractSingleBatchService(
            parent_type=type(self),
            original_response_type=self.response_type,
            llm=self.llm,
            model=self.model,
            batch_prompt=self.batch_prompt,
            cache=self.cache,
        )

    class _ExtractSingleBatchService(BaseService[_StructuredOutputWrapper[S]], Generic[S]):
        """
        A helper service to extract a single batch.
        We need this because the cache uses self.response_type for serialization.
        """

        def __init__(
            self,
            parent_type: Type[LLMStructuredOutputService[T]],
            original_response_type: Type[T],
            llm: BaseLLM,
            model: str,
            batch_prompt: str,
            cache: OptionalCacheType = None,
        ):
            # Create response_type_wrapper with optional fields and notes
            wrapper_type = self._create_response_type_wrapper(original_response_type)

            super().__init__(response_type=cast(type[_StructuredOutputWrapper[S]], wrapper_type), cache=cache)

            # This service should cache in the same namespace as its parent
            # TODO remove this as meta init deals with this
            self.service_name = parent_type.__qualname__

            self.batch_prompt = batch_prompt
            self.llm = llm
            self.model = model

        def _create_response_type_wrapper(self, response_type: Type[T]) -> Type[_StructuredOutputWrapper]:
            """
            Create a wrapper type where all response_type fields are optional,
            and where notes are added to keep track of information that is relevant.
            This allows the LLM to gradually fill in required fields across multiple pages.

            Returns:
                Wrapper type with optional response_type fields and notes
            """

            # Create a version of response_type with all fields optional
            optional_fields = {}
            for field_name, field_info in response_type.model_fields.items():
                optional_fields[field_name] = (
                    Optional[field_info.annotation],
                    Field(default=None, description=field_info.description)
                )

            OptionalResponseType = create_model(
                f"Optional{response_type.__name__}",
                **optional_fields
            )

            return _StructuredOutputWrapper[OptionalResponseType]

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

        @cache_result(key_params=['batch_filename'])
        async def __call__(
            self,
            batch_filename: str,
            batch_parse_response: ParseResponse,
            current_output: Optional[_StructuredOutputWrapper] = None,
            **kwargs
        ) -> _StructuredOutputWrapper[S]:
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
            # If it is the first call, we need to create an empty current_output
            if not current_output:
                current_output = self.response_type()

            # Format the prompt with the markdown content
            formatted_prompt = self.format_batch_prompt(batch_parse_response, current_output)

            # Use the LLM to generate structured output
            return await self.llm.generate(
                model=self.model,
                prompt=formatted_prompt,
                response_type=self.response_type,
                **kwargs
            )

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

        # Extracted data and notes
        current_output = None

        # Process the batches
        for batch_filename, batch_parse_response in batches:
            print(f"Extracting {batch_filename} ...")

            # Extract this single batch
            current_output: _StructuredOutputWrapper[T] = await self._extract_single_batch(
                batch_filename=batch_filename,
                batch_parse_response=batch_parse_response,
                current_output=current_output,
                **kwargs
            )

        final_response = self.response_type.model_validate(current_output.extracted_data, from_attributes=True)
        return final_response


Q = TypeVar("Q", bound=BaseModel, covariant=True)


class ParseExtractProtocol(Protocol[Q]):
    """
    Protocol defining the interface for ParseExtract services.

    Any object implementing these methods can be used as ParseExtractService.
    """

    async def __call__(
        self,
        filename: str,
        content: bytes
    ) -> Q:
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

    # Class attributes for provider-specific subclasses
    LLM_PARSE_SERVICE_TYPE: Optional[Type[LLMParseService[ParseResponse]]] = None
    LLM_STRUCTURED_OUTPUT_SERVICE_TYPE: Optional[Type[LLMStructuredOutputService[T]]] = None

    def __init__(
        self,
        response_type: Type[T],
        parse_service: Optional[LLMParseService[ParseResponse]] = None,
        structured_output_service: Optional[LLMStructuredOutputService[T]] = None,
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
        self.parse_service, self.structured_output_service = self._initialize_services(parse_service, structured_output_service)
        self.pipeline = ParseExtractPipeline(parse_service=self.parse_service, structured_output_service=self.structured_output_service)

    def _initialize_services(
        self,
        parse_service: Optional[LLMParseService[ParseResponse]] = None,
        structured_output_service: Optional[LLMStructuredOutputService[T]] = None,
    ) -> tuple[BaseLLM, str]:
        """
        Initialize parse_service and structured_output service with fallback to defaults.

        If parse_service is not provided, attempts to instantiate LLM_PARSE_SERVICE_TYPE.
        If structured_output_service is not provided, attempts to instantiate LLM_STRUCTURED_OUTPUT_SERVICE_TYPE.

        Args:
            parse_service: Service instance for the parse step.
                If None, uses LLM_PARSE_SERVICE_TYPE
            structured_output_service: Service instance for the extract step.
                If None, uses LLM_STRUCTURED_OUTPUT_SERVICE_TYPE  

        Returns:
            Tuple of (parse_service, structured_output_service) with defaults applied.

        Raises:
            ValueError: If parse_service is None and LLM_PARSE_SERVICE_TYPE is not set as a class attribute.
            ValueError: If structured_output_service is None and LLM_STRUCTURED_OUTPUT_SERVICE_TYPE is not set as a class attribute.
        """
        if parse_service is None:
            if self.LLM_PARSE_SERVICE_TYPE is not None:
                parse_service = self.LLM_PARSE_SERVICE_TYPE(cache=self.cache)
            else:
                raise ValueError(
                    f"{self.__class__.__name__} requires either an 'parse_service' parameter or LLM_PARSE_SERVICE_TYPE to be set as a class attribute"
                )

        if structured_output_service is None:
            if self.LLM_STRUCTURED_OUTPUT_SERVICE_TYPE is not None:
                structured_output_service = self.LLM_STRUCTURED_OUTPUT_SERVICE_TYPE(response_type=self.response_type, cache=self.cache)
            else:
                raise ValueError(
                    f"{self.__class__.__name__} requires either an 'structured_output_service' parameter or LLM_STRUCTURED_OUTPUT_SERVICE_TYPE to be set as a class attribute"
                )

        return parse_service, structured_output_service

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
