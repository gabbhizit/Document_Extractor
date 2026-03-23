"""OCR service — Google Cloud Vision API (DOCUMENT_TEXT_DETECTION)."""

import base64
import io
import logging
import os

import requests
from PIL import Image

logger = logging.getLogger(__name__)


def extract_text_from_image(image: Image.Image) -> dict:
    """
    Extract text from a PIL Image using Google Cloud Vision API.

    Uses DOCUMENT_TEXT_DETECTION — optimised for structured documents
    (PAN cards, Aadhaar cards, study certificates). Google Vision handles
    document orientation automatically so no local rotation correction is needed.

    Env var required: GOOGLE_VISION_API_KEY

    Returns:
        {
            "text": str,            # full extracted text
            "lines": list[str],     # non-empty lines
            "bounding_boxes": list  # empty — not needed downstream
        }
    """
    api_key = os.getenv("GOOGLE_VISION_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GOOGLE_VISION_API_KEY not set. Add it to your .env file. "
            "Get a key at: https://console.cloud.google.com → Cloud Vision API → Credentials."
        )

    # Convert PIL Image to base64-encoded PNG
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    content = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }]
    }

    try:
        resp = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

        response_data = resp.json()
        annotation = response_data["responses"][0].get("fullTextAnnotation", {})
        full_text = annotation.get("text", "")
        lines = [ln for ln in full_text.splitlines() if ln.strip()]

        logger.info("Vision API OCR: %d lines extracted (%d chars).", len(lines), len(full_text))
        return {"text": full_text, "lines": lines, "bounding_boxes": []}

    except requests.exceptions.Timeout:
        logger.error("Vision API request timed out after 30s.")
        return {"text": "", "lines": [], "bounding_boxes": []}
    except requests.exceptions.HTTPError as e:
        logger.error("Vision API HTTP error: %s — %s", e, resp.text[:200])
        return {"text": "", "lines": [], "bounding_boxes": []}
    except requests.exceptions.RequestException as e:
        logger.error("Vision API request failed: %s", e)
        return {"text": "", "lines": [], "bounding_boxes": []}
    except (KeyError, IndexError) as e:
        logger.error("Vision API response parse error: %s", e)
        return {"text": "", "lines": [], "bounding_boxes": []}
