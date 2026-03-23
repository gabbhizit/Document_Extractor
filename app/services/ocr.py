"""OCR service using PaddleOCR 2.7.x."""

import logging

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

_ocr_instance = None


def get_ocr() -> PaddleOCR:
    """
    Singleton PaddleOCR 2.7.x instance.

    use_angle_cls=False  — skips angle classifier (faster, sufficient for flat docs)
    lang='en'            — English recognition model
    show_log=False       — suppresses PaddleOCR verbose output
    """
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("Initialising PaddleOCR singleton...")
        _ocr_instance = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            show_log=False,
        )
        logger.info("PaddleOCR singleton ready.")
    return _ocr_instance


def extract_text_from_image(image: Image.Image) -> dict:
    """
    Run OCR on a PIL Image using PaddleOCR 2.7.x API.

    PaddleOCR 2.7.x result format:
        list of pages → each page is a list of lines →
        each line is [bbox, (text, confidence)]

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
        results = ocr.ocr(img_array, cls=False)
    except Exception as e:
        logger.error("PaddleOCR inference failed: %s", e)
        return {"text": "", "lines": [], "bounding_boxes": []}

    lines = []
    bounding_boxes = []

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
