import pytest
import httpx
import json
from pydantic import BaseModel
from yorgan.client import YorganSyncClient, YorganAsyncClient
from yorgan.datamodels import ParseResponse


# Fake responses keyed by path suffix
RESPONSES = {
    "/parse": {"markdown": "# Title: Yorgan", "metadata": {"filename": "doc.pdf", "duration_ms": 1000}},
    "/extract": {"extraction": {"title": "Yorgan"}, "metadata": {"filename": "doc.pdf", "duration_ms": 1000, "used_service": "GeminiExtractService"}},
    "/parse-extract": {"extraction": {"title": "Yorgan"}, "metadata": {"filename": "doc.pdf", "duration_ms": 1000, "used_service": "GeminiParseExtractService"}},
    "/schema/validate": {"success": True, "message": "Schema is valid"},
    "/options": {"parse": ["gemini", "gpt", "landingai"], "extract": ["gemini", "gpt"], "parse-extract": ["gemini", "gpt"]},
    "/info": {"version": "0.1.0"},
}


class Extraction(BaseModel):
    title: str


class FakeResponse:
    def __init__(self, data, status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


class FakeSyncClient:
    def __init__(self, mapping):
        self._mapping = mapping

    def post(self, url, files=None, data=None):
        for key in self._mapping:
            if url.endswith(key):
                return FakeResponse(self._mapping[key])
        return FakeResponse({"detail": "not found"}, status_code=404)

    def get(self, url):
        for key in self._mapping:
            if url.endswith(key):
                return FakeResponse(self._mapping[key])
        return FakeResponse({"detail": "not found"}, status_code=404)

    def close(self):
        pass


class FakeAsyncClient:
    def __init__(self, mapping):
        self._mapping = mapping

    async def post(self, url, files=None, data=None):
        for key in self._mapping:
            if url.endswith(key):
                return FakeResponse(self._mapping[key])
        return FakeResponse({"detail": "not found"}, status_code=404)

    async def get(self, url):
        for key in self._mapping:
            if url.endswith(key):
                return FakeResponse(self._mapping[key])
        return FakeResponse({"detail": "not found"}, status_code=404)

    async def aclose(self):
        pass


def _make_sync_client(mapping=RESPONSES):
    return FakeSyncClient(mapping)


def _make_async_client(mapping=RESPONSES):
    return FakeAsyncClient(mapping)


def test_sync_client_endpoints(monkeypatch, tmp_path):
    # Replace httpx.Client with our fake sync client
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: _make_sync_client())

    # Create a temporary file because parse expects a Path that is opened by the request builder
    fpath = tmp_path / "doc.pdf"
    fpath.write_bytes(b"dummy")

    client = YorganSyncClient(base_url="http://fake")

    # parse
    parsed = client.parse(document=fpath, option="any")
    assert isinstance(parsed, ParseResponse)
    assert parsed.markdown == "# Title: Yorgan"
    assert parsed.metadata.filename == "doc.pdf"
    assert parsed.metadata.duration_ms == 1000

    # extract (use markdown + response_model)
    extracted = client.extract(markdown="# hello", response_model=Extraction)
    # The fake extract returns extraction.markdown == "# Title"
    assert extracted.extraction["title"] == "Yorgan"
    assert extracted.metadata["filename"] == "doc.pdf"
    assert extracted.metadata["used_service"] == "GeminiExtractService"

    # parse_extract
    parsed_extracted = client.parse_extract(document=fpath, response_model=ParseResponse)
    assert parsed_extracted.extraction["title"] == "Yorgan"
    assert parsed_extracted.metadata["used_service"] == "GeminiParseExtractService"

    # validate_schema
    v = client.validate_schema({"type": "object"})
    assert v["success"] is True
    assert v["message"] == "Schema is valid"

    # options & info
    opts = client.get_options()
    assert "parse" in opts and "extract" in opts and "parse-extract" in opts
    info = client.get_info()
    assert info["version"] == "1.0.0"


# async test?
@pytest.mark.asyncio
async def test_async_client_endpoints(monkeypatch, tmp_path):
    # Replace httpx.AsyncClient with our fake async client
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: _make_async_client())

    fpath = tmp_path / "doc.pdf"
    fpath.write_bytes(b"dummy-async")

    async with YorganAsyncClient(base_url="http://fake") as client:
        parsed = await client.parse(document=fpath, option="any")
        assert isinstance(parsed, ParseResponse)
        assert parsed.markdown == "# Title: Yorgan"
        assert parsed.metadata.filename == "doc.pdf"

        extracted = await client.extract(markdown="# hello", response_model=Extraction)
        assert extracted.extraction["title"] == "Yorgan"
        assert extracted.metadata["used_service"] == "GeminiExtractService"

        parsed_extracted = await client.parse_extract(document=fpath, response_model=Extraction)
        assert parsed_extracted.extraction["title"] == "Yorgan"
        assert parsed_extracted.metadata["used_service"] == "GeminiParseExtractService"

        v = await client.validate_schema({"type": "object"})
        assert v["success"] is True
        assert v["message"] == "Schema is valid"

        opts = await client.get_options()
        assert "parse" in opts and "extract" in opts and "parse-extract" in opts
        info = await client.get_info()
        assert info["version"] == "1.0.0"
