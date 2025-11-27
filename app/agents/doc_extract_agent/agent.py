import base64
import io
import json
from typing import Optional

import httpx
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Part
from pydantic import BaseModel, Field

from yorgan.utils import (json_schema_to_pydantic_model,
                          resolve_openAPI_json_schema)

APP_NAME = "doc_extractor_app"
MODEL = "gemini-2.5-flash"

_client = httpx.AsyncClient(base_url="http://localhost:8000")


async def get_last_uploaded_filename(tool_context: ToolContext) -> str:
    """
    Lists artifacts for the current invocation's user session
    """
    artifacts = await tool_context.list_artifacts()
    if not artifacts:
        raise RuntimeError("No artifacts uploaded")

    # assume the last one is the most recent by default
    filename = artifacts[-1].filename
    return filename


async def get_artifact(tool_context: ToolContext) -> Part:
    """
    Loads the most recently uploaded artifact (by filename) in the current invocation scope.
    Uses tool_context.load_artifact(...) to ensure the correct user/session scope is used.
    """
    filename = await get_last_uploaded_filename(tool_context)
    part = await tool_context.load_artifact(filename=filename)
    if part is None:
        raise RuntimeError(f"Artifact not found: {filename}")
    return part


# --- Tool Functions for Agent ---
async def parse_document(tool_context: ToolContext) -> dict:
    """
    Parses the most recently uploaded document.

    Returns:
        dict: The extracted text in markdown format or a dictionary with error information.
    """
    try:
        part = await get_artifact(tool_context)
        # inline_data.data is bytes-like; ensure we have bytes
        data = bytes(part.inline_data.data)
        mime = part.inline_data.mime_type or "application/octet-stream"
        filename = await get_last_uploaded_filename(tool_context)
        files = {"file": (filename, io.BytesIO(data), mime)}

        response = await _client.post("/parse", files=files)
        response.raise_for_status()
        result = response.json()

        tool_context.state['parsed_document'] = result
        tool_context.state['file_uploaded'] = True
        return result

    except Exception as e:
        return {"error": f"Document parsing failed: {str(e)}"}


async def verify_document_schema(tool_context: ToolContext, document_schema: str) -> dict:
    """
    Verifies that the document schema can be converted to a model.

    Args:
        document_schema: The JSON schema for the Pydantic model.

    Returns:
        dict: A dictionary with the status and result of the update.
    """
    try:
        json_schema = json.loads(document_schema)
        new_model = json_schema_to_pydantic_model(json_schema)
        resolved_schema = resolve_openAPI_json_schema(new_model.model_json_schema())

        tool_context.state['extraction_schema'] = resolved_schema
        tool_context.state['schema_defined'] = True
        return {
            "success": True,
            "message": "Schema is valid",
            "document_schema": resolved_schema
        }

    except Exception as e:
        return {"error": f"Schema update failed: {str(e)}"}


class AgentResponse(BaseModel):
    """Structured response from the agent to the UI."""
    schema_defined: bool = Field(description="Whether an extraction schema has been defined")
    status_message: str = Field(description="Fixed status message based on current state")
    assistant_message: str = Field(description="Natural language message from the assistant")
    document_schema: Optional[dict] = Field(None, description="Resolved JSON schema for the extraction model")


# --- Agent and Runner Configuration ---
INSTRUCTION = """\
You are a document extraction specialist. Help users define structured output schemas and extract data from documents.

Key capabilities:
- Design JSON schemas for Pydantic models based on user requirements
- Parse documents using OCR to understand their content and structure
- Provide guidance on schema design best practices

Guidelines:
- Only OCR documents when you need to understand their structure or content.
- Create schemas that are robust and handle common variations.
- When the user is asking for help, take the initiative. Avoid asking follow up questions unless absolutely necessary.

Tool Usage:
- ALWAYS verify the schema using the verify_document_schema tool before returning it.
- Use tools strategically based on user needs, not in a fixed sequence.

Rules for JSON schemas for Pydantic models:
- Produce a single, well-formed JSON Schema (draft 2020-12 / OpenAPI compatible)
- Root: {"title": "<ModelName>", "type":"object", "properties": {...}, "required": [...]}
- Nested objects must include a "title".
- Use types: string, integer, number, boolean, object, array.
- For optional fields omit them from "required".
- For date/time use strings with format date/date-time/time.
- Arrays must include "items" that describe the element schema.
- Enums: use "enum": [...].
- Keep descriptions short.

Schema and code mapping examples:
```python

class Car(BaseModel):
    id: int
    name: str | None = Field(max_length=50)
```
has schema:

```json
{
    "title": "Car",
    "type": "object",
    "properties": {
        "id": {
            "title": "Id",
            "type": "integer"
        },
        "name": {
            "anyOf": [
                {
                    "maxLength": 50,
                    "type": "string"
                },
                {
                    "type": "null"
                }
            ],
            "title": "Name"
        }
    },
    "required": [
        "id",
        "name"
    ],
}
```
another example with submodels with reference is:

```json
{
    "$defs": {
        "Subtest": {
            "properties": {
                "name": {
                    "default": "default",
                    "type": "string"
                }
            },
            "title": "Subtest",
            "type": "object"
        }
    },
    "properties": {
        "thing": {
            "description": "The thing that is in the test",
            "title": "Thing",
            "type": "string"
        },
        "tags": {
            "items": {
                "type": "string"
            },
            "title": "Tags",
            "type": "array"
        },
        "subtest": {
            "$ref": "#/$defs/Subtest"
        }
    },
    "required": [
        "tags",
        "subtest"
    ],
    "title": "Test",
    "type": "object"
}
```
Prefer creating schema as defined by RFC 3986 and implemented in JSON Schema and OpenAPI and allows for modularity and reusability of schema definitions:
In JSON Schema, the $defs keyword provides a standardized, reusable location within a schema to define subschemas for reuse elsewhere in the document via the $ref keyword.
Always output any JSON Schema in ```json ...``` block.

You must ALWAYS respond with the following structure:
- schema_defined: Set to true after successfully calling update_document_schema, false otherwise
- status_message: Choose the appropriate message:
  * "Welcome! Upload a document or define what you want to extract to get started." - for initial greeting or when starting fresh
  * "Provide details to define what data you want to extract." - when working on defining the schema
  * "Document schema ready. Upload a document to extract structured data." - after schema is successfully defined
- assistant_message: Your natural conversational response to the user

"""

# TODO:
# make sure schemas are always verified using the tool

root_agent = Agent(
    model=MODEL,
    name="doc_extractor_agent",
    instruction=INSTRUCTION,
    tools=[
        verify_document_schema,
        parse_document
    ],
    output_schema=AgentResponse
)
