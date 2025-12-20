from datetime import datetime
from typing import Any, ClassVar, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SchemaDict(TypedDict, total=False):
    properties: dict[str, Any]
    required: list[str]
    title: str
    type: Literal["object"]


class Metadata(BaseModel):
    filename: Optional[str] = None
    duration_ms: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class ParseResponse(BaseModel):
    PAGE_BREAK: ClassVar[str] = "<!-- PAGE BREAK -->"

    markdown: str

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


class ParseResponseMetaData(ParseResponse):
    metadata: Metadata = Field(default=Metadata())


class APIParseResponse(ParseResponseMetaData):

    model_config = ConfigDict(extra="allow")

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


class CompanyInfo(BaseModel):
    name: str
    address: str
    email: Optional[str] = None
    vat_id: Optional[str] = None


class LineItem(BaseModel):
    id: str = Field(..., description="Order of the line item, order by top to bottom")
    description: str
    quantity: float
    unit_price: float

    @property
    def total(self) -> float:
        return self.quantity * self.unit_price


class Invoice(BaseModel):
    invoice_number: str
    issue_date: datetime
    due_date: datetime
    vendor: CompanyInfo
    receiver: CompanyInfo
    currency: str
    shipping: Optional[float]
    discount: Optional[float]
    tax: Optional[float]
    tax_rate: Optional[float]
    line_items: list[LineItem]

    @property
    def subtotal(self) -> float:
        return sum(item.quantity * item.unit_price for item in self.line_items)

    @property
    def tax_total(self) -> float:
        return (self.tax_rate or 0) * self.subtotal

    @property
    def total(self) -> float:
        return self.subtotal + self.tax_total - (self.discount or 0)


class Category(BaseModel):
    name: str = Field(..., min_length=1, description="The name of the category.")


class CategoryPrediction(BaseModel):
    id: Optional[str] = Field(..., description="Id of line item.")
    # category: str = Field(..., min_length=1, description="The name of the category.")
    category: Category
    score: float = Field(..., ge=0, le=1, description="A confidence score between 0 and 1.")


class CategoryPredictions(BaseModel):
    predictions: list[CategoryPrediction]
