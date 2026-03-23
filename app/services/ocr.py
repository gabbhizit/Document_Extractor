"""OCR service — version-aware for PaddleOCR 2.7.x (deployed) and 3.x (local)."""

import logging
import os

# Must be set before paddleocr is imported — skips slow model-hoster connectivity check.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import numpy as np
import paddleocr as _paddleocr_module
from PIL import Image
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

# Detect PaddleOCR major version once at import time.
# 3.x removed use_angle_cls / show_log and changed result format to dict-based.
# 2.7.x uses the classic list-based result format.
_PADDLE_V3: bool = getattr(_paddleocr_module, "__version__", "2.0.0").startswith("3.")

_ocr_instance = None


def get_ocr() -> PaddleOCR:
    """
    Singleton PaddleOCR instance.
    Branches on installed version so the same code works locally (3.x)
    and on the deployed server (2.7.x).
    """
    global _ocr_instance
    if _ocr_instance is None:
        version_tag = "v3" if _PADDLE_V3 else "v2"
        logger.info("Initialising PaddleOCR singleton (%s)...", version_tag)

        if _PADDLE_V3:
            # 3.x API — use_angle_cls and show_log are removed
            _ocr_instance = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            )
        else:
            # 2.7.x API
            _ocr_instance = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                show_log=False,
            )

        logger.info("PaddleOCR singleton ready (%s).", version_tag)
    return _ocr_instance


def extract_text_from_image(image: Image.Image) -> dict:
    """
    Run OCR on a PIL Image.

    PaddleOCR 3.x result format  — list of dicts per page:
        [{"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]}, ...]

    PaddleOCR 2.7.x result format — list of pages, each page is a list of lines:
        [[[bbox, (text, confidence)], ...], ...]

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
        if _PADDLE_V3:
            results = ocr.ocr(img_array)
        else:
            results = ocr.ocr(img_array, cls=False)
    except Exception as e:
        logger.error("PaddleOCR inference failed: %s", e)
        return {"text": "", "lines": [], "bounding_boxes": []}

    lines = []
    bounding_boxes = []

    if _PADDLE_V3:
        # 3.x: each element is a dict with rec_texts / rec_scores / rec_polys
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
                bounding_boxes.append(
                    {"text": text, "bbox": bbox, "confidence": round(float(conf), 4)}
                )
    else:
        # 2.7.x: each element is a list of [bbox, (text, confidence)]
        for page in results:
            if page is None:
                continue
            for line in page:
                text = line[1][0]
                if not text.strip():
                    continue
                conf = float(line[1][1])
                bbox = line[0]
                lines.append(text)
                bounding_boxes.append(
                    {"text": text, "bbox": bbox, "confidence": round(conf, 4)}
                )

    logger.debug("OCR result: %d lines extracted.", len(lines))

    return {
        "text": "\n".join(lines),
        "lines": lines,
        "bounding_boxes": bounding_boxes,
    }
