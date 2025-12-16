import base64
import mimetypes
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


def split_pdf_pages(content: bytes) -> list[bytes]:
    """
    Split a PDF document into individual pages using PyMuPDF.

    Args:
        content: Full PDF document as bytes

    Returns:
        List of individual page PDFs as bytes
    """
    doc = pymupdf.open(stream=content, filetype="pdf")
    pages = []

    for page_num in range(len(doc)):
        # Create a new PDF with just this page
        single_page_doc = pymupdf.open()
        single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        # Convert to bytes
        page_bytes = single_page_doc.tobytes()
        single_page_doc.close()

        pages.append(page_bytes)

    doc.close()
    return pages
