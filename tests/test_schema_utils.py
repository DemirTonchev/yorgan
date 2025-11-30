from enum import Enum, StrEnum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from yorgan.utils import flat_json_schema_to_pydantic_model, resolve_openAPI_json_schema, json_schema_to_pydantic_model
from yorgan.datamodels import Invoice
import pytest


class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"


class Address(BaseModel):
    street: str
    city: str


class UserProfile(BaseModel):
    username: str = Field(description="User's unique name")
    age: Optional[int] = 18
    role: UserRole = Field(...)
    contact: list[Union[Address, str]]

    model_config = ConfigDict(use_enum_values=True)


"""Provides the resolved JSON schema for the UserProfile model."""
user_profile_schema = {'properties':
                       {'username': {
                           'description': "User's unique name",
                           'title': 'Username',
                           'type': 'string'},
                           'age': {
                           'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                           'default': 18,
                           'title': 'Age'},
                           'role': {
                           'enum': ['admin', 'user'], 'title': 'UserRole', 'type': 'string'},
                           'contact': {
                           'items': {
                               'anyOf': [
                                   {
                                       'properties': {
                                           'street': {'title': 'Street',
                                                      'type': 'string'},
                                           'city': {'title': 'City', 'type': 'string'}},
                                       'required': ['street', 'city'],
                                       'title': 'Address',
                                       'type': 'object'},
                                   {'type': 'string'}
                               ]},
                           'title': 'Contact',
                           'type': 'array'}},
                       'required': ['username', 'role', 'contact'],
                       'title': 'UserProfile',
                       'type': 'object'}

invoice_schema = {
    'properties':
    {
        'invoice_number': {'type': 'string'},
        'issue_date': {'format': 'date-time', 'type': 'string'},
        'due_date': {'format': 'date-time', 'type': 'string'},
        'vendor': {
            'properties': {
                'name': {'type': 'string'},
                'address': {'type': 'string'},
                'email': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
                          'default': None},
                'vat_id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
                           'default': None}
            },
            'required': ['name', 'address'],
            'title': 'CompanyInfo',
            'type': 'object'
        },
        'receiver': {
            'properties': {
                'name': {'type': 'string'},
                'address': {'type': 'string'},
                'email': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
                          'default': None,
                          },
                'vat_id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
                           'default': None,
                           }
            },
            'required': ['name', 'address'],
            'title': 'CompanyInfo',
            'type': 'object'
        },
        'currency': {'type': 'string'},
        'shipping': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        'discount': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        'tax': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        'tax_rate': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        'line_items': {
            'items': {
                'properties': {
                    'id': {'description': 'Order of the line item, order by top to bottom',
                           'title': 'Id',
                           'type': 'string'},
                    'description': {'type': 'string'},
                    'quantity': {'type': 'number'},
                    'unit_price': {'type': 'number'}
                },
                'required': ['id', 'description', 'quantity', 'unit_price'],
                'title': 'LineItem',
                'type': 'object'},
            'type': 'array'}
    },
    'required': ['invoice_number',
                 'issue_date',
                 'due_date',
                 'vendor',
                 'receiver',
                 'currency',
                 'shipping',
                 'discount',
                 'tax',
                 'tax_rate',
                 'line_items'],
    'title': 'Invoice',
    'type': 'object'}


def test_schema_resolve():
    schema = UserProfile.model_json_schema()

    assert resolve_openAPI_json_schema(schema) == user_profile_schema
    # f(g(x)) = x
    assert json_schema_to_pydantic_model(UserProfile.model_json_schema()).model_json_schema() == UserProfile.model_json_schema()


def test_json_schema_to_base_model_creation():
    """Tests that a Pydantic model class is created correctly."""
    UserProfile = flat_json_schema_to_pydantic_model(user_profile_schema)
    assert set(UserProfile.model_fields.keys()) == {"username", "age", "role", "contact"}


@pytest.mark.parametrize("schema, data, pure_model",
                         [
                             (
                                 user_profile_schema,
                                 {
                                     'username': 'testuser',
                                     'age': 20,
                                     'role': 'user',
                                     'contact': ['real street', {'street': 'krum', 'city': 'sofia'}]
                                 },
                                 UserProfile
                             ),
                             (
                                 invoice_schema,
                                 {
                                     "invoice_number": "INV-2025-001",
                                     "issue_date": "2025-08-30T00:00:00",
                                     "due_date": "2025-09-30T00:00:00",
                                     "vendor": {
                                         "name": "Tech Solutions Inc.",
                                         "address": "123 Silicon Valley",
                                         "email": "vendor@example.com",
                                         "vat_id": "bg1234"
                                     },
                                     "receiver": {
                                         "name": "Tech Solutions BG Inc.",
                                         "address": "1000 Sofia",
                                         "email": "vendor@example.com",
                                         "vat_id": "bg1234"
                                     },
                                     "currency": "USD",
                                     "shipping": 25.0,
                                     "discount": 50.0,
                                     "tax": 25.0,
                                     "tax_rate": 0.08,
                                     "line_items": [
                                         {
                                             "id": "id1",
                                             "description": "Software License (Annual)",
                                             "quantity": 1.0,
                                             "unit_price": 1200.0
                                         },
                                         {
                                             "id": "id2",
                                             "description": "On-site Consulting",
                                             "quantity": 8.0,
                                             "unit_price": 150.0
                                         }
                                     ]
                                 },
                                 Invoice
                             )
                         ])
def test_dynamic_model_instantiation_and_validation(schema, data, pure_model):
    """Tests that the dynamically created model can be instantiated and validates data."""
    DynamicModel = flat_json_schema_to_pydantic_model(schema, model_config=ConfigDict(use_enum_values=True))

    dyn_instance = DynamicModel(**data)
    orig_instance = pure_model(**data)

    assert dyn_instance.model_dump() == orig_instance.model_dump()
    assert DynamicModel.model_json_schema() == pure_model.model_json_schema()


@pytest.mark.parametrize("schema, pure_model",
                         [
                             (
                                 user_profile_schema,
                                 UserProfile
                             ),
                             (
                                 invoice_schema,
                                 Invoice
                             )
                         ])
def test_dynamic_model_field_types(schema, pure_model):
    """Tests that the field types of the dynamic model are correct."""
    DynamicModel = flat_json_schema_to_pydantic_model(schema)
    dyn_fields = DynamicModel.model_fields
    orig_fields = pure_model.model_fields

    assert set(dyn_fields.keys()) == set(orig_fields.keys())
    assert resolve_openAPI_json_schema(DynamicModel.model_json_schema()) == resolve_openAPI_json_schema(pure_model.model_json_schema())


def test_required_field_optional_behavior():
    DynamicModel = flat_json_schema_to_pydantic_model(user_profile_schema)

    # Missing all required fields: should fail
    with pytest.raises(ValidationError):
        DynamicModel()
    with pytest.raises(ValidationError):
        DynamicModel(username="testuser", role="admin")

    # All required fields present, optional omitted
    instance = DynamicModel(username="testuser", role="admin", contact=["foo"])

    # All required and optional fields present
    instance2 = DynamicModel(username="testuser", role="user", contact=["bar"], age=42)
