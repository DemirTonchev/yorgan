import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Type, TypeVar, cast

import httpx
from pydantic import BaseModel

from yorgan.datamodels import APIExtractResponse, APIParseResponse, SchemaDict

T = TypeVar("T", bound=BaseModel)


class _RequestBuilder:
    """Helper class to build requests - shared between sync and async"""

    @staticmethod
    def build_parse_request(
        document: Optional[Path] = None,
        document_url: Optional[str] = None,
        option: str = "landingai",
        service_options: Optional[dict] = None,
    ) -> tuple[Optional[dict], Optional[dict]]:
        if document is None and document_url is None:
            raise ValueError("Either document or document_url must be provided")

        if document_url is not None:
            raise NotImplementedError("document_url not yet supported")
        document = cast(Path, document)
        files = {"file": (document.name, document.read_bytes())}  # read_bytes closes the file
        data = {"option": option}
        if service_options is not None:
            data.update({"service_options": json.dumps(service_options)})
        return files, data

    @staticmethod
    def build_extract_request(
        markdown: Optional[str] = None,
        markdown_file: Optional[Path] = None,
        schema: Optional[SchemaDict] = None,
        schema_file: Optional[Path] = None,
        response_model: Optional[Type[T]] = None,
        metadata: Optional[dict] = None,
        option: str = "gemini",
        service_options: Optional[dict] = None,
    ) -> dict:
        if markdown is None and markdown_file is None:
            raise ValueError("Either markdown or markdown_file must be provided")

        if schema is None and schema_file is None and response_model is None:
            raise ValueError("One of schema, schema_file, or response_model must be provided")

        if markdown_file is not None:
            with open(markdown_file, "r") as f:
                markdown = f.read()

        if response_model is not None:
            schema = cast(SchemaDict, response_model.model_json_schema())
        elif schema_file is not None:
            with open(schema_file, "r") as f:
                schema = json.load(f)

        data = {
            "markdown": markdown,
            "schema": json.dumps(schema),
            "metadata": json.dumps(metadata or {}),
            "option": option,
        }
        if service_options is not None:
            data.update({"service_options": json.dumps(service_options)})
        return data

    @staticmethod
    def build_parse_extract_request(
        document: Optional[Path] = None,
        document_url: Optional[str] = None,
        schema: Optional[dict] = None,
        schema_file: Optional[Path] = None,
        response_model: Optional[Type[T]] = None,
        option: str = "gemini",
    ) -> tuple[Optional[dict], Optional[dict]]:
        if document is None and document_url is None:
            raise ValueError("Either document or document_url must be provided")

        if schema is None and schema_file is None and response_model is None:
            raise ValueError("One of schema, schema_file, or response_model must be provided")

        if response_model is not None:
            schema = response_model.model_json_schema()
        elif schema_file is not None:
            with open(schema_file, "r") as f:
                schema = json.load(f)

        if document_url is not None:
            raise NotImplementedError("document_url not yet supported")

        document = cast(Path, document)
        files = {"file": (document.name, document.read_bytes())}
        data = {
            "schema": json.dumps(schema),
            "option": option,
        }
        return files, data

    @staticmethod
    def build_validate_schema_request(schema: dict) -> dict:
        return {"schema": json.dumps(schema)}


class YorganSyncClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self._builder = _RequestBuilder()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def close(self):
        self.client.close()

    def parse(
        self,
        document: Optional[Path] = None,
        document_url: Optional[str] = None,
        option: str = "landingai",
        service_options: Optional[dict] = None,
    ) -> APIParseResponse:
        files, data = self._builder.build_parse_request(document, document_url, option, service_options)
        response = self.client.post(
            f"{self.base_url}/parse",
            files=files,
            data=data,
        )
        response.raise_for_status()
        return APIParseResponse.model_validate_json(response.text)

    def extract(
        self,
        markdown: Optional[str] = None,
        markdown_file: Optional[Path] = None,
        schema: Optional[SchemaDict] = None,
        schema_file: Optional[Path] = None,
        response_model: Optional[Type[T]] = None,
        metadata: Optional[dict] = None,
        option: str = "gemini",
        service_options: Optional[dict] = None,
    ) -> APIExtractResponse:
        data = self._builder.build_extract_request(
            markdown, markdown_file, schema, schema_file,
            response_model, metadata, option, service_options,
        )
        response = self.client.post(f"{self.base_url}/extract", data=data)
        response.raise_for_status()
        return APIExtractResponse.model_validate_json(response.text)

    def parse_extract(
        self,
        document: Optional[Path] = None,
        document_url: Optional[str] = None,
        schema: Optional[dict] = None,
        schema_file: Optional[Path] = None,
        response_model: Optional[Type[T]] = None,
        option: str = "gemini",
    ) -> APIExtractResponse:
        files, data = self._builder.build_parse_extract_request(
            document, document_url, schema, schema_file, response_model, option
        )
        response = self.client.post(
            f"{self.base_url}/parse-extract",
            files=files,
            data=data,
        )
        response.raise_for_status()
        return APIExtractResponse.model_validate_json(response.text)

    def validate_schema(self, schema: dict) -> dict:
        data = self._builder.build_validate_schema_request(schema)
        response = self.client.post(f"{self.base_url}/schema/validate", data=data)
        response.raise_for_status()
        return response.json()

    @lru_cache(maxsize=1)
    def get_options(self) -> dict[str, list[str]]:
        response = self.client.get(f"{self.base_url}/options")
        response.raise_for_status()
        return response.json()

    def get_info(self) -> dict:
        response = self.client.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()


class YorganAsyncClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)
        self._builder = _RequestBuilder()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def close(self):
        await self.client.aclose()

    async def parse(
        self,
        document: Optional[Path] = None,
        document_url: Optional[str] = None,
        option: str = "landingai",
        service_options: Optional[dict] = None,
    ) -> APIParseResponse:
        files, data = self._builder.build_parse_request(document, document_url, option, service_options)
        response = await self.client.post(
            f"{self.base_url}/parse",
            files=files,
            data=data,
        )
        response.raise_for_status()
        return APIParseResponse.model_validate_json(response.text)

    async def extract(
        self,
        markdown: Optional[str] = None,
        markdown_file: Optional[Path] = None,
        schema: Optional[SchemaDict] = None,
        schema_file: Optional[Path] = None,
        response_model: Optional[Type[T]] = None,
        metadata: Optional[dict] = None,
        option: str = "gemini",
        service_options: Optional[dict] = None,
    ) -> APIExtractResponse:
        data = self._builder.build_extract_request(
            markdown, markdown_file, schema, schema_file,
            response_model, metadata, option, service_options
        )
        response = await self.client.post(f"{self.base_url}/extract", data=data)
        response.raise_for_status()
        return APIExtractResponse.model_validate_json(response.text)

    async def parse_extract(
        self,
        document: Optional[Path] = None,
        document_url: Optional[str] = None,
        schema: Optional[dict] = None,
        schema_file: Optional[Path] = None,
        response_model: Optional[Type[T]] = None,
        option: str = "gemini",
    ) -> APIExtractResponse:
        files, data = self._builder.build_parse_extract_request(
            document, document_url, schema, schema_file, response_model, option
        )
        response = await self.client.post(
            f"{self.base_url}/parse-extract",
            files=files,
            data=data,
        )
        response.raise_for_status()
        return APIExtractResponse.model_validate_json(response.text)

    async def validate_schema(self, schema: dict) -> dict:
        data = self._builder.build_validate_schema_request(schema)
        response = await self.client.post(f"{self.base_url}/schema/validate", data=data)
        response.raise_for_status()
        return response.json()

    @lru_cache(maxsize=1)
    async def get_options(self) -> dict:
        response = await self.client.get(f"{self.base_url}/options")
        response.raise_for_status()
        return response.json()

    async def get_info(self) -> dict:
        response = await self.client.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()
