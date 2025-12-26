from typing import Any, ClassVar, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, model_validator


class SchemaDict(TypedDict, total=False):
    properties: dict[str, Any]
    required: list[str]
    title: str
    type: Literal["object"]


class ResponseMetadata(BaseModel):
    credit_usage: Optional[float] = None

    duration_ms: Optional[int] = None

    filename: Optional[str] = None

    job_id: Optional[str] = None

    org_id: Optional[str] = None

    page_count: Optional[int] = None

    version: Optional[str] = None

    failed_pages: Optional[list[int]] = None
    # --- above is same as langing parse response

    input_token_count: Optional[int] = None

    output_token_count: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class BoundingBox(BaseModel):
    # coordinates are normalized between 0 and 1
    bottom: float

    left: float

    right: float

    top: float


class ChunkGrounding(BaseModel):
    box: BoundingBox

    page: int


class Chunk(BaseModel):
    id: str

    grounding: ChunkGrounding

    markdown: str

    type: str


# class Split(BaseModel):
#     chunks: list[str]

#     class_: str = Field(alias="class")

#     identifier: str

#     markdown: str

#     pages: list[int]

#     model_config = ConfigDict(populate_by_name=True)


# class Grounding(BaseModel):
#     box: ParseGroundingBox

#     page: int

#     type: Literal[
#         "chunkLogo",
#         "chunkCard",
#         "chunkAttestation",
#         "chunkScanCode",
#         "chunkForm",
#         "chunkTable",
#         "chunkFigure",
#         "chunkText",
#         "chunkMarginalia",
#         "chunkTitle",
#         "chunkPageHeader",
#         "chunkPageFooter",
#         "chunkPageNumber",
#         "chunkKeyValue",
#         "table",
#         "tableCell",
#     ]


class Markdown(BaseModel):
    PAGE_BREAK: ClassVar[str] = "<!-- PAGE BREAK -->"

    markdown: str


class ParseResponse(Markdown):

    chunks: Optional[list[Chunk]] = None

    metadata: ResponseMetadata = ResponseMetadata()

    # we dont use the splits right now...
    # splits: List[Split]

    # grounding: Optional[dict[str, Grounding]] = None


# TODO: this information will be available in chunks
def add_explicit_page_numbering(parse_response: ParseResponse):
    """
    Adds explicit page numbering in the markdown of the parse response.

    Args:
        parse_response: a response from parse service

    Returns:
        The parse response with modified markdown
    """
    page_markdowns = parse_response.markdown.split(ParseResponse.PAGE_BREAK)
    total = len(page_markdowns)
    edited_page_markdowns = []
    for page_num, page_markdown in enumerate(page_markdowns, start=1):
        edited_page_markdown = f"--- PAGE ({page_num}/{total}) ---" + "\n\n" + page_markdown
        edited_page_markdowns.append(edited_page_markdown)
    edited_markdown = ParseResponse.PAGE_BREAK.join(edited_page_markdowns)
    parse_response.markdown = edited_markdown


class APIParseResponse(ParseResponse):

    model_config = ConfigDict(extra="allow")

    # exclude attributes that are None by default.
    def model_dump(self, **kwargs) -> dict:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)

    @model_validator(mode="after")
    def debug_usage(self):
        print("created pydantic model")

        return self


class APIExtractResponse[T](BaseModel):

    extraction: T
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def debug_usage(self):
        print("created pydantic model")

        return self
