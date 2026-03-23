"""PDF text extraction — direct text first, OCR fallback for scanned PDFs."""

import io
from PIL import Image

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

MAX_PAGES = 3
MIN_TEXT_CHARS = 50  # below this, treat as scanned image


def extract_text_from_pdf(pdf_bytes: bytes) -> str | None:
    """
    Try to extract embedded text directly from a PDF (instant, no OCR).

    Returns:
        Combined text string if the PDF has selectable text, else None.
    """
    if not PDFPLUMBER_AVAILABLE:
        return None

    pages_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages[:MAX_PAGES]:
            pages_text.append(page.extract_text() or "")

    total_chars = sum(len(t) for t in pages_text)
    if total_chars >= MIN_TEXT_CHARS:
        return "\n".join(pages_text).strip()

    return None


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[Image.Image]:
    """
    Convert PDF bytes into PIL Images for OCR (used only for scanned PDFs).

    Args:
        pdf_bytes: Raw PDF file content.
        dpi: Render resolution. 150 DPI is sufficient for OCR.

    Returns:
        List of PIL Image objects (max 3 pages).

    Raises:
        RuntimeError: If pdf2image/poppler is not available.
    """
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError(
            "pdf2image is not installed. Install it with: pip install pdf2image\n"
            "Also install poppler: brew install poppler (macOS) or apt install poppler-utils (Linux)"
        )

    return convert_from_bytes(pdf_bytes, dpi=dpi, last_page=MAX_PAGES)
