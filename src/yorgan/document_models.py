from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


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
