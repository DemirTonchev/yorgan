import base64
import mimetypes
from io import BytesIO
import httpx

import pymupdf


def encode_bytes_for_transfer(
    data: bytes,
) -> str:
    """
    Encode bytes for safe transfer in text-based formats like JSON.

    Args:
        data: Raw bytes to encode

    Returns:
        Encoded string safe for JSON/text transmission
    """
    return base64.b64encode(data).decode("utf-8")


def get_mime_type(filename):
    """
    Determine the MIME type of a file based on its filename or extension.

    Args:
        filename (str): Name or path of the file to check. The extension is used
            to guess the MIME type.

    Returns:
        str: The MIME type string (e.g. 'application/pdf', 'image/jpeg')

    Raises:
        ValueError: If the file type cannot be determined from the extension
            or if the file type is not supported.

    Example:
        >>> get_mime_type('document.pdf')
        'application/pdf'
        >>> get_mime_type('image.jpg')
        'image/jpeg'
    """
    # strict is false so we catch webp
    mime_type, _ = mimetypes.guess_type(filename, strict=False)
    if mime_type is None:
        raise ValueError(f"Unsupported file type: {filename}")

    return mime_type


def count_pdf_pages(content: bytes) -> int:
    """
    Return number of pages in a PDF document.

    Args:
        content: Document content as bytes

    Returns:
        The number of pages
    """
    doc = pymupdf.open(stream=content, filetype="pdf")
    page_count = len(doc)
    doc.close()
    return page_count


def split_pdf(
    content: bytes,
    window: int = 1,
    overlap: int = 0
) -> list[bytes]:
    """
    Split a PDF document into batches using PyMuPDF.

    Args:
        content: Full PDF document as bytes.
        window: Number of pages per returned batch (default 1).
        overlap: Number of overlapping pages between consecutive batches (default 0).

    Returns:
        List of PDF bytes, each entry contains `window` pages (last batch may be smaller).
    """
    if window <= 0:
        raise ValueError("window must be greater than 0")

    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must satisfy 0 <= overlap < window")

    doc = pymupdf.open(stream=content, filetype="pdf")
    total = len(doc)
    batches = []

    for start in range(0, total - overlap, window - overlap):
        end = min(start + window - 1, total - 1)
        batch = pymupdf.open()
        batch.insert_pdf(doc, from_page=start, to_page=end)
        batches.append(batch.tobytes())
        batch.close()

    doc.close()
    return batches


async def download_blob(presigned_url: str) -> BytesIO:
    """
    Downloads a blob from a provided presigned URL and returns it as a byte stream.

    This function uses a streaming GET request to efficiently handle data transfer,
    reading the response in chunks and writing them into a memory buffer.

    Args:
        presigned_url: A temporary, authenticated URL (e.g., from AWS S3,
            GCS, or Azure) that provides read access to a specific blob/object.

    Returns:
        BytesIO: An in-memory binary stream containing the downloaded content.

    Raises:
        httpx.HTTPStatusError: If the request fails (e.g., 403 Forbidden, 404 Not Found).
        httpx.RequestError: If a network-related error occurs during the download.
    """
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", presigned_url) as response:
            response.raise_for_status()
            buffer = BytesIO()
            async for chunk in response.aiter_bytes():
                buffer.write(chunk)
            return buffer
