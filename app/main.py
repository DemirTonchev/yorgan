# type: ignore
from json import (
    # we will change this with faster json parser such as orjson
    loads as json_loads,
    dumps as json_dumps,
    JSONDecodeError
)
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pprint import pprint  # yeas I am lazy to do proper logging
from typing import Annotated, Any
from pathlib import Path

import pydantic
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from google.adk.cli.fast_api import get_fast_api_app as adk_get_fast_api_app
from pydantic import BaseModel, ConfigDict, model_validator
from aiocache import RedisCache
from aiocache.serializers import PickleSerializer

from yorgan.cache import SimpleMemoryCacheWithPersistence
from yorgan.datamodels import ParseResponse, ParseResponseMetaData
from yorgan.datamodels import APIParseResponse, APIExtractResponse
from yorgan.services.gemini import (
    GeminiStructuredOutputService,
)
from yorgan.utils import SchemaDict, json_schema_to_pydantic_model, SchemaConversionError
from app.app_settings import settings
from app.app_utils import generate_hashed_filename
from app.service_registry import (
    ParseServiceOptions,
    ExctractServiceOptions,
    ParseExctractServiceOptions,
    get_parse_service,
    get_extract_service,
    get_parse_extract_service,
)


# turn off logging to gemini and other outbound services
# logging.getLogger("httpx").setLevel(logging.ERROR)


# load cache and global app state, in normal app cache would be redis or at least something like google adk memory cache.
# needs to be cli param
CACHE = SimpleMemoryCacheWithPersistence(persist_dir=Path(__file__).parent / './cache')
# CACHE = RedisCache(serializer=PickleSerializer())


app = FastAPI(default_response_class=ORJSONResponse)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Skip if already an HTTPException
    if isinstance(exc, HTTPException):
        raise exc

    return ORJSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Unhandled Server error: {str(exc)}"}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(category_router)

# need to start adk web on another port
# adk web --port 8001

# agents_app = adk_get_fast_api_app(
#     agents_dir="./agents",
#     session_service_uri="sqlite:///./sessions_db.db",
#     web=True,
# )

# app.mount("", agents_app)


@app.post("/parse")
async def parse(
        file: Annotated[UploadFile, File(...)],
        option: ParseServiceOptions = "landingai",
        service_options: Annotated[str, Form()] = "{}",
) -> APIParseResponse:
    """
    Parses an uploaded document using OCR and returns the response from OCR.

    Parameters
    - file (UploadFile): Required. The uploaded file to parse (e.g. PDF, image). Provided by FastAPI via multipart/form-data.
    - option (str | ParseServiceOptions): Optional. Service selector for parsing backend. Supported values: "landingai", "gemini", "auto".
        Defaults to "auto". Controls which underlying parse service is used.
    - service_options (str): Optional. A JSON string containing options for the service (e.g. {"model": "...", "prompt": "..."}).

    Returns
    - APIParseResponse: A Pydantic model (extends ParseResponse) containing parsed text/markdown, detected entities and metadata.

    Errors
    - Raises HTTPException(400) when parsing fails; detail contains the underlying error message.

    Example (curl)
    curl -X POST "http://localhost:8000/parse" \
      -F "file=@/path/to/document.pdf" \
      -F "option=gemini" \
      -F 'service_options={"model": "gemini-2.5-pro", "prompt": "You are ..."}'
    """

    try:
        service_kwargs = json_loads(service_options)
    except (JSONDecodeError, TypeError):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON for service_options: {service_options}"
        )

    content = await file.read()
    filename = file.filename
    
    parse_service = get_parse_service(option, cache=CACHE, **service_kwargs)

    filename_key = generate_hashed_filename(filename, content)
    try:
        parsed_response = await parse_service(filename=filename_key, content=content)
        api_response = APIParseResponse(**parsed_response.model_dump())
        api_response.metadata.filename = filename

    except Exception as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Parsing ERROR: {str(e)}"
        )

    return ORJSONResponse(api_response.model_dump(mode='json'))


@app.post("/extract")
async def extract(
    markdown: Annotated[str, Form()],
    schema: Annotated[str, Form()],
    metadata: Annotated[str, Form()] = "{}",
    option: ExctractServiceOptions = "gemini",
    service_options: Annotated[str, Form()] = "{}",
) -> APIExtractResponse:
    """
    Extract structured data from a document markdown using a provided JSON Schema.

    Parameters
    - markdown (str): Required. The document text (e.g. OCR markdown) to extract data from.
    - schema (str | None): Optional. A JSON string describing the expected output JSON Schema.
        If provided, it will be parsed into a Pydantic model and used to validate / shape the
        structured output. Must be a valid JSON document.
    - metadata (str): Optional. A JSON string containing arbitrary metadata about the request
        (for example {"filename": "invoice.pdf"}). This metadata is merged into the response
        and used when generating cache keys.
    - option (str): Optional. Parse/extraction service selector. Currently accepted but not actively used inside this endpoint.

    Returns
    - APIExtractResponse: JSONResponse containing:
        - extraction: The structured object produced by the model, validated against the
          provided schema (converted to a Pydantic model).
        - metadata: A dict containing the merged input metadata plus runtime info such as
          used_service and duration_ms.

    Errors
    - Raises HTTPException(400) when input JSON (schema or metadata) is invalid or when schema
      parsing fails. The error detail includes the underlying exception information.

    Example (curl)
    curl -X POST "http://localhost:8000/extract" \
      -F 'markdown=...document text...' \
      -F 'schema={"title": "Invoice", "type": "object", "properties": {"invoice_number": {"title": "Invoice Number", "type": "string"}, "issue_date": {"format": "date-time", "title": "Issue Date", "type": "string"}}, "required": ["invoice_number", "issue_date"]}' \
      -F 'metadata={"filename":"invoice.pdf"}'
    """
    start_ts = time.perf_counter()

    try:
        schema = json_loads(schema)
        ExtractionModel = json_schema_to_pydantic_model(schema)
        metadata = json_loads(metadata)
        service_kwargs = json_loads(service_options)
    except (JSONDecodeError, Exception) as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"ERROR decoding input: {str(e)}\nfor input: {e.doc}"
        )

    filename = metadata.get("filename", "unknown")

    extract_service = get_extract_service(option, response_type=ExtractionModel, cache=CACHE, **service_kwargs)

    # this way we guarantee that the hit the correct cache if any on model change back and forth.
    # json_dumps again is not ideal but we want to sort the keys right to guarantee same hash
    filename_key = generate_hashed_filename(filename, content=json_dumps(schema, sort_keys=True).encode() + markdown.encode())

    structured_output = await extract_service(
        filename=filename_key,
        parse_response=ParseResponse(markdown=markdown)
    )

    metadata.update(
        {
            "used_service": extract_service.service_name,
            "duration_ms": (time.perf_counter() - start_ts) * 1000,
        }
    )
    api_response = APIExtractResponse(extraction=structured_output, metadata=metadata)
    return ORJSONResponse(api_response.model_dump(mode='json'))


@app.post("/parse-extract")
async def parse_extract(
    file: Annotated[UploadFile, File()],
    schema: Annotated[str, Form()],
    option: str = "gemini",
    service_options: Annotated[str, Form()] = "{}",
) -> APIExtractResponse:
    """
    Parses an uploaded document and returns a structured DocumentModel based on the provided JSON Schema.

    Parameters
    - file (UploadFile): Required. The uploaded file to parse (PDF/image). Provided via multipart/form-data.
    - schema (str): Required. A JSON string describing the expected output JSON Schema. This is converted
      into a Pydantic model (flat schema -> Pydantic).
    - option (str): Optional. Service selector for parsing backend. Supported values: TBD
      Defaults to "auto". Controls which underlying parse-extract service is used.

    Returns
    - APIExtractResponse: Contains:
        - extraction: The structured object produced by the model, validated against the provided schema model.
        - metadata: Dict with runtime info such as filename, used_service, and duration_ms.

    Errors
    - Raises HTTPException(400) when input schema is invalid or extraction/parsing fails.

    Example (curl)
    curl -X POST "http://localhost:8000/parse-extract" \
      -F "file=@/path/to/invoice.pdf" \
      -F 'schema={"type":"object","properties":{"total":{"type":"number"}}}' \
      -F "option=auto"
    """
    start_ts = time.perf_counter()

    content = await file.read()
    filename = file.filename
    try:
        ExtractionModel = json_schema_to_pydantic_model(json_loads(schema))
        service_kwargs = json_loads(service_options)
    except (JSONDecodeError, ) as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"ERROR decoding input: {str(e)}\nfor input: {e.doc}"
        )
    except Exception as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"ERROR creating Pydantic model: {e}"
        )

    filename_key = generate_hashed_filename(filename, content=json_dumps(schema, sort_keys=True).encode() + content)

    parse_extract_service = get_parse_extract_service(option, response_type=ExtractionModel, cache=CACHE, **service_kwargs)
    structured_output = await parse_extract_service(filename=filename_key, content=content)
    metadata = {
        "filename": filename,
        "used_service": parse_extract_service.service_name,
        "duration_ms": (time.perf_counter() - start_ts) * 1000,
    }
    api_response = APIExtractResponse(extraction=structured_output, metadata=metadata)
    return ORJSONResponse(api_response.model_dump(mode='json'))


@app.post("/schema/validate")
async def validate_schema(schema: Annotated[str, Form()]):
    try:
        ExtractionModel = json_schema_to_pydantic_model(json_loads(schema))
        return ORJSONResponse(
            content={
                "success": True,
                "message": "Schema is valid",
                # "openapi_schema":ExtractionModel.model_json_schema()
            }
        )
    except JSONDecodeError as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Failed decoding input: {str(e)}\nfor input: {e.doc}"
        )
    except SchemaConversionError as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Schema validation failed, can't create Pydantic model: {e}"
        )


@app.get("/options")
async def get_parse_options():
    return {
        "parse": ParseServiceOptions._member_names_,
        "extract": ExctractServiceOptions._member_names_,
        "parse-extract": ParseExctractServiceOptions._member_names_,
    }


@app.get("/info")
async def info():
    return ORJSONResponse(content=settings.model_dump(mode='json'))
