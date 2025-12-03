"""This module supports converting json to pydantic v2 model.
It is joint work of google and microsoft to tackle the problem. Becuase Pyndatic founder says fuck you here:
https://github.com/pydantic/pydantic/discussions/6598#discussioncomment-7610330
Msf autogen:
https://microsoft.github.io/autogen/dev//_modules/autogen_core/utils/_json_to_pydantic.html#schema_to_pydantic_model
"""
import copy
import datetime
from enum import Enum
from json import loads as json_loads  # we can use other parser in the future (orjson)
from json import dumps as json_dumps
from typing import Annotated, Any, Dict, ForwardRef, List, Literal, Optional, Type, Union, cast, TypedDict
from ipaddress import IPv4Address, IPv6Address
from pathlib import Path

from pydantic import (
    UUID1,
    UUID3,
    UUID4,
    UUID5,
    AnyUrl,
    BaseModel,
    EmailStr,
    Field,
    Json,
    conbytes,
    constr,
    confloat,
    conint,
    conlist,
    create_model,
)
from pydantic.fields import FieldInfo


class SchemaConversionError(Exception):
    """Base class for schema conversion exceptions."""

    pass


class ReferenceNotFoundError(SchemaConversionError):
    """Raised when a $ref cannot be resolved."""

    pass


class FormatNotSupportedError(SchemaConversionError):
    """Raised when a format is not supported."""

    pass


class UnsupportedKeywordError(SchemaConversionError):
    """Raised when an unsupported JSON Schema keyword is encountered."""

    pass


class TypeNotSupportedError(SchemaConversionError):
    """Raised when an unsupported JSON Schema keyword is encountered."""

    pass


TYPE_MAPPING: dict[str, Type[Any]] = {
    "string": str,
    "integer": int,
    "boolean": bool,
    "number": float,
    "array": list,
    "object": dict,
    "null": type(None),
}

FORMAT_MAPPING: dict[str, Any] = {
    "uuid": UUID4,
    "uuid1": UUID1,
    "uuid2": UUID4,
    "uuid3": UUID3,
    "uuid4": UUID4,
    "uuid5": UUID5,
    "email": EmailStr,
    "uri": AnyUrl,
    "hostname": constr(strict=True),
    "ipv4": IPv4Address,
    "ipv6": IPv6Address,
    "ipv4-network": IPv4Address,
    "ipv6-network": IPv6Address,
    "date-time": datetime.datetime,
    "date": datetime.date,
    "time": datetime.time,
    "duration": datetime.timedelta,
    "int32": Annotated[int, Field(..., strict=True, ge=-(2**31), le=2**31 - 1)],
    "int64": Annotated[int, Field(..., strict=True, ge=-(2**63), le=2**63 - 1)],
    "float": Annotated[float, Field(..., strict=True)],
    "double": float,
    "decimal": float,
    "byte": conbytes(strict=True),
    "binary": conbytes(strict=True),
    "password": str,
    "path": str,
    "json": Json,
}


class SchemaDict(TypedDict, total=False):
    properties: dict[str, Any]
    required: list[str]
    title: str
    type: Literal["object"]


def _resolve_references(openapi_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolves all $ref references in an OpenAPI specification.

    Handles circular references correctly.

    Args:
        openapi_spec: A dictionary representing the OpenAPI specification.

    Returns:
        A dictionary representing the OpenAPI specification with all references
        resolved.
    """
    # copied from googleadk due to extremely slow import. IDK WHAT THEY DO.
    # from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_spec_parser import OpenApiSpecParser
    # resolved_schema = OpenApiSpecParser()._resolve_references(model_json_schema)

    openapi_spec = copy.deepcopy(openapi_spec)  # Work on a copy
    resolved_cache = {}  # Cache resolved references

    def resolve_ref(ref_string, current_doc):
        """Resolves a single $ref string."""
        parts = ref_string.split("/")
        if parts[0] != "#":
            raise ValueError(f"External references not supported: {ref_string}")

        current = current_doc
        for part in parts[1:]:
            if part in current:
                current = current[part]
            else:
                return None  # Reference not found
        return current

    def recursive_resolve(obj, current_doc, seen_refs=None):
        """Recursively resolves references, handling circularity.

        Args:
            obj: The object to traverse.
            current_doc:  Document to search for refs.
            seen_refs: A set to track already-visited references (for circularity
              detection).

        Returns:
            The resolved object.
        """
        if seen_refs is None:
            seen_refs = set()  # Initialize the set if it's the first call

        if isinstance(obj, dict):
            if "$ref" in obj and isinstance(obj["$ref"], str):
                ref_string = obj["$ref"]

                # Check for circularity
                if ref_string in seen_refs and ref_string not in resolved_cache:
                    # Circular reference detected! Return a *copy* of the object,
                    # but *without* the $ref.  This breaks the cycle while
                    # still maintaining the overall structure.
                    return {k: v for k, v in obj.items() if k != "$ref"}

                seen_refs.add(ref_string)  # Add the reference to the set

                # Check if we have a cached resolved value
                if ref_string in resolved_cache:
                    return copy.deepcopy(resolved_cache[ref_string])

                resolved_value = resolve_ref(ref_string, current_doc)
                if resolved_value is not None:
                    # Recursively resolve the *resolved* value,
                    # passing along the 'seen_refs' set
                    resolved_value = recursive_resolve(
                        resolved_value, current_doc, seen_refs
                    )
                    resolved_cache[ref_string] = resolved_value
                    return copy.deepcopy(resolved_value)  # return the cached result
                else:
                    return obj  # return original if no resolved value.

            else:
                new_dict = {}
                for key, value in obj.items():
                    new_dict[key] = recursive_resolve(value, current_doc, seen_refs)
                return new_dict

        elif isinstance(obj, list):
            return [recursive_resolve(item, current_doc, seen_refs) for item in obj]
        else:
            return obj

    return recursive_resolve(openapi_spec, openapi_spec)


def resolve_openAPI_json_schema(model_json_schema: dict, remove_defs=True) -> SchemaDict:
    resolved_schema = _resolve_references(model_json_schema)
    if remove_defs:
        resolved_schema.pop("$defs", None)
    return resolved_schema


def pydantic_to_json_schema(model: Type[BaseModel], indent=None) -> str:

    return json_dumps(
        resolve_openAPI_json_schema(model.model_json_schema()),
        indent=indent
    )


def flat_json_schema_to_pydantic_model(schema: dict[str, Any], model_config=None) -> Type[BaseModel]:
    """Convert a flattened/expanded JSON Schema to a Pydantic model.

    This function handles JSON schemas where all $refs have already been resolved/expanded.
    It creates Pydantic models that match the schema structure.

    **Supported Schema Features**:
    - Basic types: string, number, integer, boolean, array, object
    - Special string formats (email, date-time, etc)
    - Arrays with type validation
    - Nested object structures
    - Optional fields with defaults
    - Field titles and descriptions

    Args:
        schema: A dict containing the expanded JSON schema with no $refs
        model_config: Optional Pydantic model configuration

    Returns:
        Type[BaseModel]: A generated Pydantic model class

    Example:
        ```python
        schema = {
            "title": "User",
            "type": "object",
            "properties": {
                "name": {"type": "string", "title": "Full Name"},
                "age": {"type": "integer"},
                "email": {
                    "type": "string",
                    "format": "email"
                }
            },
            "required": ["name", "email"]
        }

        UserModel = flat_json_schema_to_pydantic_model(schema)
        ```

    Note:
        This function expects schemas where $refs are already resolved. For schemas
        with $refs, use json_schema_to_pydantic_model() instead or resolve them with resolve_openAPI_json_schema.
    """

    properties = schema.get("properties", {})
    model_fields = {}
    required_fields = set(schema.get("required", []))

    def process_field(field_name, field_props: dict[str, Any]) -> tuple:
        """Recursively processes a field and returns its type and Field instance."""
        json_type = field_props.get("type")
        # if json_type is None:
        #     raise ValueError(f"Missing type for `{field_name}`, object: `{field_props}`")
        enum_values = field_props.get("enum")

        # Handle Enums - caveat enum expects string type.
        if enum_values:
            enum_name: str = field_props['title']
            field_type = Enum(enum_name, {v.upper(): v for v in enum_values})

        elif "anyOf" in field_props:
            union_types = tuple(TYPE_MAPPING[props['type']] for props in field_props["anyOf"])
            field_type = Union[union_types]

        # Handle Arrays with Nested Objects
        elif json_type == "array" and "items" in field_props:
            item_props = field_props["items"]
            if "anyOf" in item_props:
                union_types = tuple(
                    process_field(field_name, sub_schema)[0]
                    for sub_schema in item_props["anyOf"]
                )
                item_type = Union[union_types]

            elif item_props.get("type") == "object":
                item_type: type[BaseModel] = flat_json_schema_to_pydantic_model(item_props)
            else:
                item_type: type = TYPE_MAPPING.get(item_props.get("type"), Any)
            field_type = list[item_type]
        # Handle Nested Objects
        elif json_type == "object" and "properties" in field_props:
            field_type = flat_json_schema_to_pydantic_model(field_props)  # Recursively create submodel

        # handle primitives and end recursion here if any
        else:
            if 'format' in field_props:
                # TODO:
                # "type" is ignored if "format" is present
                fmt_type = field_props["format"]
                field_type = FORMAT_MAPPING.get(fmt_type)
                if field_type is None:
                    raise FormatNotSupportedError(f"Unknown format `{fmt_type}` for `{field_name}` object: `{field_props}`")
            else:
                if json_type not in TYPE_MAPPING:
                    # TODO:
                    # if "type" is missing, error message needs to be different
                    raise TypeNotSupportedError(
                        f"Unsupported item type `{json_type}`, supported types: {list(TYPE_MAPPING.keys())}.\nLoc: field `{field_name}` object `{field_props}`"
                    )
                field_type = TYPE_MAPPING[json_type]

        field_options = {}

        default_value = field_props.get("default")
        is_required = field_name in required_fields

        if not is_required and default_value is None:
            field_type = Optional[field_type]

        field_options['default'] = default_value if not is_required else ...
        field_options['description'] = field_props.get("description")
        return field_type, Field(**field_options)

    # Process each field
    for field_name, field_props in properties.items():
        model_fields[field_name] = process_field(field_name, field_props)

    model = create_model(schema.get("title", "GeneratedModel"), **model_fields, __config__=model_config)
    model.model_rebuild()
    return create_model(schema.get("title", "GeneratedModel"), **model_fields, __config__=model_config)


def json_schema_to_pydantic_model(schema: dict[str, Any] | str | Path, model_config=None) -> Type[BaseModel]:
    """
    Convert a JSON Schema dictionary to a fully-typed Pydantic model.

    This function handles schema translation and validation logic to produce
    a Pydantic model.
    """
    if isinstance(schema, (str, Path)):
        schema = Path(schema)
        if schema.exists():
            schema = json_loads(schema.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(f"File not found: {schema}")

    schema = cast(dict[str, Any], schema)
    return flat_json_schema_to_pydantic_model(
        resolve_openAPI_json_schema(
            schema  # type: ignore
        ),
        model_config=model_config
    )


def _make_field(
    default: Any,
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Construct a Pydantic Field with proper typing."""
    field_kwargs: Dict[str, Any] = {}
    if title is not None:
        field_kwargs["title"] = title
    if description is not None:
        field_kwargs["description"] = description
    return Field(default, **field_kwargs)


class _JSONSchemaToPydantic:
    def __init__(self) -> None:
        self._model_cache: Dict[str, Optional[Union[Type[BaseModel], ForwardRef]]] = {}

    def _resolve_ref(self, ref: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        ref_key = ref.split("/")[-1]
        definitions = cast(dict[str, dict[str, Any]], schema.get("$defs", {}))

        if ref_key not in definitions:
            raise ReferenceNotFoundError(
                f"Reference `{ref}` not found in `$defs`. Available keys: {list(definitions.keys())}"
            )

        return definitions[ref_key]

    def get_ref(self, ref_name: str) -> Any:
        if ref_name not in self._model_cache:
            raise ReferenceNotFoundError(
                f"Reference `{ref_name}` not found in cache. Available: {list(self._model_cache.keys())}"
            )

        if self._model_cache[ref_name] is None:
            return ForwardRef(ref_name)

        return self._model_cache[ref_name]

    def _get_item_model_name(self, array_field_name: str, parent_model_name: str) -> str:
        """Generate hash-based model names for array items to keep names short and unique."""
        import hashlib

        # Create a short hash of the full path to ensure uniqueness
        full_path = f"{parent_model_name}_{array_field_name}"
        hash_suffix = hashlib.md5(full_path.encode()).hexdigest()[:6]

        # Use field name as-is with hash suffix
        return f"{array_field_name}_{hash_suffix}"

    def _process_definitions(self, root_schema: Dict[str, Any]) -> None:
        if "$defs" in root_schema:
            for model_name in root_schema["$defs"]:
                if model_name not in self._model_cache:
                    self._model_cache[model_name] = None

            for model_name, model_schema in root_schema["$defs"].items():
                if self._model_cache[model_name] is None:
                    self._model_cache[model_name] = self.json_schema_to_pydantic(model_schema, model_name, root_schema)

    def json_schema_to_pydantic(
        self, schema: Dict[str, Any], model_name: str = "GeneratedModel", root_schema: Optional[Dict[str, Any]] = None
    ) -> Type[BaseModel]:
        if root_schema is None:
            root_schema = schema
            self._process_definitions(root_schema)

        if "$ref" in schema:
            resolved = self._resolve_ref(schema["$ref"], root_schema)
            schema = {**resolved, **{k: v for k, v in schema.items() if k != "$ref"}}

        if "allOf" in schema:
            merged: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            for s in schema["allOf"]:
                part = self._resolve_ref(s["$ref"], root_schema) if "$ref" in s else s
                merged["properties"].update(part.get("properties", {}))
                merged["required"].extend(part.get("required", []))
            for k, v in schema.items():
                if k not in {"allOf", "properties", "required"}:
                    merged[k] = v
            merged["required"] = list(set(merged["required"]))
            schema = merged

        return self._json_schema_to_model(schema, model_name, root_schema)

    def _resolve_union_types(self, schemas: List[Dict[str, Any]]) -> List[Any]:
        types: List[Any] = []
        for s in schemas:
            if "$ref" in s:
                types.append(self.get_ref(s["$ref"].split("/")[-1]))
            elif "enum" in s:
                types.append(Literal[tuple(s["enum"])] if len(s["enum"]) > 0 else Any)
            else:
                json_type = s.get("type")
                if json_type not in TYPE_MAPPING:
                    raise UnsupportedKeywordError(f"Unsupported or missing type `{json_type}` in union")

                # Handle array types with items specification
                if json_type == "array" and "items" in s:
                    item_schema = s["items"]
                    if "$ref" in item_schema:
                        item_type = self.get_ref(item_schema["$ref"].split("/")[-1])
                    else:
                        item_type_name = item_schema.get("type")
                        if item_type_name is None:
                            item_type = str
                        elif item_type_name not in TYPE_MAPPING:
                            raise UnsupportedKeywordError(f"Unsupported item type `{item_type_name}` in union array")
                        else:
                            item_type = TYPE_MAPPING[item_type_name]

                    constraints = {}
                    if "minItems" in s:
                        constraints["min_length"] = s["minItems"]
                    if "maxItems" in s:
                        constraints["max_length"] = s["maxItems"]

                    array_type = conlist(item_type, **constraints) if constraints else List[item_type]  # type: ignore[valid-type]
                    types.append(array_type)
                else:
                    types.append(TYPE_MAPPING[json_type])
        return types

    def _extract_field_type(self, key: str, value: Dict[str, Any], model_name: str, root_schema: Dict[str, Any]) -> Any:
        json_type = value.get("type")
        if json_type not in TYPE_MAPPING:
            raise UnsupportedKeywordError(
                f"Unsupported or missing type `{json_type}` for field `{key}` in `{model_name}`"
            )

        base_type = TYPE_MAPPING[json_type]
        constraints: Dict[str, Any] = {}

        if json_type == "string":
            if "minLength" in value:
                constraints["min_length"] = value["minLength"]
            if "maxLength" in value:
                constraints["max_length"] = value["maxLength"]
            if "pattern" in value:
                constraints["pattern"] = value["pattern"]
            if constraints:
                base_type = constr(**constraints)

        elif json_type == "integer":
            if "minimum" in value:
                constraints["ge"] = value["minimum"]
            if "maximum" in value:
                constraints["le"] = value["maximum"]
            if "exclusiveMinimum" in value:
                constraints["gt"] = value["exclusiveMinimum"]
            if "exclusiveMaximum" in value:
                constraints["lt"] = value["exclusiveMaximum"]
            if constraints:
                base_type = conint(**constraints)

        elif json_type == "number":
            if "minimum" in value:
                constraints["ge"] = value["minimum"]
            if "maximum" in value:
                constraints["le"] = value["maximum"]
            if "exclusiveMinimum" in value:
                constraints["gt"] = value["exclusiveMinimum"]
            if "exclusiveMaximum" in value:
                constraints["lt"] = value["exclusiveMaximum"]
            if constraints:
                base_type = confloat(**constraints)

        elif json_type == "array":
            if "minItems" in value:
                constraints["min_length"] = value["minItems"]
            if "maxItems" in value:
                constraints["max_length"] = value["maxItems"]
            item_schema = value.get("items", {"type": "string"})
            if "$ref" in item_schema:
                item_type = self.get_ref(item_schema["$ref"].split("/")[-1])
            elif item_schema.get("type") == "object" and "properties" in item_schema:
                # Handle array items that are objects with properties - create a nested model
                # Use hash-based naming to keep names short and unique
                item_model_name = self._get_item_model_name(key, model_name)
                item_type = self._json_schema_to_model(item_schema, item_model_name, root_schema)
            else:
                item_type_name = item_schema.get("type")
                if item_type_name is None:
                    item_type = str
                elif item_type_name not in TYPE_MAPPING:
                    raise UnsupportedKeywordError(
                        f"Unsupported or missing item type `{item_type_name}` for array field `{key}` in `{model_name}`"
                    )
                else:
                    item_type = TYPE_MAPPING[item_type_name]

            base_type = conlist(item_type, **constraints) if constraints else List[item_type]  # type: ignore[valid-type]

        if "format" in value:
            format_type = FORMAT_MAPPING.get(value["format"])
            if format_type is None:
                raise FormatNotSupportedError(f"Unknown format `{value['format']}` for `{key}` in `{model_name}`")
            if not isinstance(format_type, type):
                return format_type
            if not issubclass(format_type, str):
                return format_type
            return format_type

        return base_type

    def _json_schema_to_model(
        self, schema: Dict[str, Any], model_name: str, root_schema: Dict[str, Any]
    ) -> Type[BaseModel]:
        if "allOf" in schema:
            merged: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            for s in schema["allOf"]:
                part = self._resolve_ref(s["$ref"], root_schema) if "$ref" in s else s
                merged["properties"].update(part.get("properties", {}))
                merged["required"].extend(part.get("required", []))
            for k, v in schema.items():
                if k not in {"allOf", "properties", "required"}:
                    merged[k] = v
            merged["required"] = list(set(merged["required"]))
            schema = merged

        fields: Dict[str, tuple[Any, FieldInfo]] = {}
        required_fields = set(schema.get("required", []))

        for key, value in schema.get("properties", {}).items():
            if "$ref" in value:
                ref_name = value["$ref"].split("/")[-1]
                field_type = self.get_ref(ref_name)
            elif "anyOf" in value:
                sub_models = self._resolve_union_types(value["anyOf"])
                field_type = Union[tuple(sub_models)]
            elif "oneOf" in value:
                sub_models = self._resolve_union_types(value["oneOf"])
                field_type = Union[tuple(sub_models)]
                if "discriminator" in value:
                    discriminator = value["discriminator"]["propertyName"]
                    field_type = Annotated[field_type, Field(discriminator=discriminator)]
            elif "enum" in value:
                field_type = Literal[tuple(value["enum"])]
            elif "allOf" in value:
                merged = {"type": "object", "properties": {}, "required": []}
                for s in value["allOf"]:
                    part = self._resolve_ref(s["$ref"], root_schema) if "$ref" in s else s
                    merged["properties"].update(part.get("properties", {}))
                    merged["required"].extend(part.get("required", []))
                for k, v in value.items():
                    if k not in {"allOf", "properties", "required"}:
                        merged[k] = v
                merged["required"] = list(set(merged["required"]))
                field_type = self._json_schema_to_model(merged, f"{model_name}_{key}", root_schema)
            elif value.get("type") == "object" and "properties" in value:
                field_type = self._json_schema_to_model(value, f"{model_name}_{key}", root_schema)
            else:
                field_type = self._extract_field_type(key, value, model_name, root_schema)

            if field_type is None:
                raise UnsupportedKeywordError(f"Unsupported or missing type for field `{key}` in `{model_name}`")

            default_value = value.get("default")
            is_required = key in required_fields

            if not is_required and default_value is None:
                field_type = Optional[field_type]

            field_args = {
                "default": default_value if not is_required else ...,
            }
            if "title" in value:
                field_args["title"] = value["title"]
            if "description" in value:
                field_args["description"] = value["description"]

            fields[key] = (
                field_type,
                _make_field(
                    default_value if not is_required else ...,
                    title=value.get("title"),
                    description=value.get("description"),
                ),
            )

        model: Type[BaseModel] = create_model(model_name, **cast(dict[str, Any], fields))
        model.model_rebuild()
        return model


def autogen_json_schema_to_pydantic_model(schema: Dict[str, Any], model_name: str = "GeneratedModel") -> Type[BaseModel]:
    """
    This is autogen function!
    Convert a JSON Schema dictionary to a fully-typed Pydantic model.

    This function handles schema translation and validation logic to produce
    a Pydantic model.

    **Supported JSON Schema Features**

    - **Primitive types**: `string`, `integer`, `number`, `boolean`, `object`, `array`, `null`
    - **String formats**:
        - `email`, `uri`, `uuid`, `uuid1`, `uuid3`, `uuid4`, `uuid5`
        - `hostname`, `ipv4`, `ipv6`, `ipv4-network`, `ipv6-network`
        - `date`, `time`, `date-time`, `duration`
        - `byte`, `binary`, `password`, `path`
    - **String constraints**:
        - `minLength`, `maxLength`, `pattern`
    - **Numeric constraints**:
        - `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`
    - **Array constraints**:
        - `minItems`, `maxItems`, `items`
    - **Object schema support**:
        - `properties`, `required`, `title`, `description`, `default`
    - **Enums**:
        - Converted to Python `Literal` type
    - **Union types**:
        - `anyOf`, `oneOf` supported with optional `discriminator`
    - **Inheritance and composition**:
        - `allOf` merges multiple schemas into one model
    - **$ref and $defs resolution**:
        - Supports references to sibling definitions and self-referencing schemas

    .. code-block:: python

        # Example 1: Simple user model
        schema = {
            "title": "User",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name", "email"],
        }

        UserModel = json_schema_to_pydantic_model(schema)
        user = UserModel(name="Alice", email="alice@example.com", age=30)

    .. code-block:: python

        # Example 2: Nested model
        schema = {
            "title": "BlogPost",
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "author": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "email": {"type": "string", "format": "email"}},
                    "required": ["name"],
                },
            },
            "required": ["title", "author"],
        }

        BlogPost = json_schema_to_pydantic_model(schema)


    .. code-block:: python

        # Example 3: allOf merging with $refs
        schema = {
            "title": "EmployeeWithDepartment",
            "allOf": [{"$ref": "#/$defs/Employee"}, {"$ref": "#/$defs/Department"}],
            "$defs": {
                "Employee": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                    "required": ["id", "name"],
                },
                "Department": {
                    "type": "object",
                    "properties": {"department": {"type": "string"}},
                    "required": ["department"],
                },
            },
        }

        Model = json_schema_to_pydantic_model(schema)

    .. code-block:: python

        from autogen_core.utils import json_schema_to_pydantic_model

        # Example 4: Self-referencing (recursive) model
        schema = {
            "title": "Category",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "subcategories": {"type": "array", "items": {"$ref": "#/$defs/Category"}},
            },
            "required": ["name"],
            "$defs": {
                "Category": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "subcategories": {"type": "array", "items": {"$ref": "#/$defs/Category"}},
                    },
                    "required": ["name"],
                }
            },
        }

        Category = json_schema_to_pydantic_model(schema)

    .. code-block:: python

        # Example 5: Serializing and deserializing with Pydantic

        from uuid import uuid4
        from pydantic import BaseModel, EmailStr, Field
        from typing import Optional, List, Dict, Any
        from autogen_core.utils import json_schema_to_pydantic_model


        class Address(BaseModel):
            street: str
            city: str
            zipcode: str


        class User(BaseModel):
            id: str
            name: str
            email: EmailStr
            age: int = Field(..., ge=18)
            address: Address


        class Employee(BaseModel):
            id: str
            name: str
            manager: Optional["Employee"] = None


        class Department(BaseModel):
            name: str
            employees: List[Employee]


        class ComplexModel(BaseModel):
            user: User
            extra_info: Optional[Dict[str, Any]] = None
            sub_items: List[Employee]


        # Convert ComplexModel to JSON schema
        complex_schema = ComplexModel.model_json_schema()

        # Rebuild a new Pydantic model from JSON schema
        ReconstructedModel = json_schema_to_pydantic_model(complex_schema, "ComplexModel")

        # Instantiate reconstructed model
        reconstructed = ReconstructedModel(
            user={
                "id": str(uuid4()),
                "name": "Alice",
                "email": "alice@example.com",
                "age": 30,
                "address": {"street": "123 Main St", "city": "Wonderland", "zipcode": "12345"},
            },
            sub_items=[{"id": str(uuid4()), "name": "Bob", "manager": {"id": str(uuid4()), "name": "Eve"}}],
        )

        print(reconstructed.model_dump())


    Args:
        schema (Dict[str, Any]): A valid JSON Schema dictionary.
        model_name (str, optional): The name of the root model. Defaults to "GeneratedModel" or to 'title' if present in the root schema.

    Returns:
        Type[BaseModel]: A dynamically generated Pydantic model class.

    Raises:
        ReferenceNotFoundError: If a `$ref` key references a missing entry.
        FormatNotSupportedError: If a `format` keyword is unknown or unsupported.
        UnsupportedKeywordError: If the schema contains an unsupported `type`.

    See Also:
        - :class:`pydantic.BaseModel`
        - :func:`pydantic.create_model`
        - https://json-schema.org/
    """
    model_name = schema.get('title')
    return _JSONSchemaToPydantic().json_schema_to_pydantic(schema, model_name)
