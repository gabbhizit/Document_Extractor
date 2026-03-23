"""OCR service using PaddleOCR 3.x — optimised for speed."""

import logging
import os

import numpy as np
from PIL import Image

# Skip slow network connectivity check on every startup
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

_ocr_instance = None


def get_ocr() -> PaddleOCR:
    """
    Singleton PaddleOCR instance.

    Disabled heavy optional models not needed for flat document scans:
    - doc_orientation_classify: handled by our custom correct_rotation() in image_utils
    - doc_unwarping (UVDoc): corrects curved/warped pages (unnecessary for flat scans)
    - textline_orientation: per-line rotation (unnecessary for standard printed docs)

    Uses mobile detection model for faster inference.
    """
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("Initialising PaddleOCR singleton...")
        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        )
        logger.info("PaddleOCR singleton ready.")
    return _ocr_instance


def extract_text_from_image(image: Image.Image) -> dict:
    """
    Run OCR on a PIL Image using PaddleOCR 3.x API.

    Returns:
        {
            "text": str,                  # full concatenated text
            "lines": list[str],           # individual text lines
            "bounding_boxes": list[dict]  # [{text, bbox, confidence}]
        }
    """
    ocr = get_ocr()
    img_array = np.array(image.convert("RGB"))

    try:
        results = ocr.ocr(img_array)
    except Exception as e:
        logger.error("PaddleOCR inference failed: %s", e)
        return {"text": "", "lines": [], "bounding_boxes": []}

    lines = []
    bounding_boxes = []

    for page in results:
        rec_texts = page.get("rec_texts", [])
        rec_scores = page.get("rec_scores", [])
        rec_polys = page.get("rec_polys", [])

        for i, text in enumerate(rec_texts):
            if not text.strip():
                continue
            conf = rec_scores[i] if i < len(rec_scores) else 0.0
            bbox = rec_polys[i].tolist() if i < len(rec_polys) else []
            lines.append(text)
            bounding_boxes.append({"text": text, "bbox": bbox, "confidence": round(conf, 4)})

    logger.debug("OCR result: %d lines extracted.", len(lines))

    return {
        "text": "\n".join(lines),
        "lines": lines,
        "bounding_boxes": bounding_boxes,
    }
