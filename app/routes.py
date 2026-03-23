"""API routes for document extraction."""

import logging
import time

from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.ocr import extract_text_from_image
from app.services.classifier import classify_document
from app.services.extractor import extract_fields
from app.services.validator import validate_and_score
from app.services.cost_tracker import calculate_cost, GOOGLE_VISION_COST_USD, USD_TO_INR
from app.utils.pdf_parser import pdf_to_images, extract_text_from_pdf
from app.utils.image_utils import (
    load_image_from_bytes,
    preprocess_image,
    correct_rotation,
    correct_skew,
)

logger = logging.getLogger(__name__)

router = APIRouter()

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/tiff"}
SUPPORTED_PDF_TYPE = "application/pdf"


@router.post("/extract", summary="Extract structured data from a document image or PDF")
async def extract_document(file: UploadFile = File(...)) -> dict:
    """
    Accepts an image or PDF upload and returns structured extraction results.

    Returns:
        {
            "document_type": str,
            "extracted_data": dict,
            "validation": dict,
            "confidence": float,
            "cost": dict,
            "ocr_text": str,
            "processing_time_seconds": float
        }
    """
    t_start = time.perf_counter()
    filename = file.filename or "unknown"
    content_type = file.content_type or ""

    logger.info("Request received — file: %s | type: %s", filename, content_type)

    file_bytes = await file.read()

    try:
        is_pdf = content_type == SUPPORTED_PDF_TYPE or filename.lower().endswith(".pdf")
        is_image = content_type in SUPPORTED_IMAGE_TYPES or any(
            filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".tiff"]
        )

        if not is_pdf and not is_image:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{content_type}'. Accepted: JPEG, PNG, WEBP, TIFF, PDF.",
            )

        full_text = ""
        ocr_page_count = 0  # tracks Vision API calls; 0 for PDF direct-text path

        # --- Step 1: For PDFs, try direct text extraction first (instant) ---
        if is_pdf:
            full_text = extract_text_from_pdf(file_bytes) or ""
            if full_text.strip():
                logger.info("PDF direct text extraction succeeded (%d chars).", len(full_text))

        # --- Step 2: Fall back to OCR (images or scanned PDFs) ---
        if not full_text.strip():
            if is_pdf:
                images = pdf_to_images(file_bytes)
                logger.info("PDF converted to %d page image(s) for OCR.", len(images))
            else:
                images = [load_image_from_bytes(file_bytes)]

            all_text_parts = []
            for idx, image in enumerate(images):
                image = preprocess_image(image)

                # --- Layer 1: correct cardinal rotation (90°/180°/270°) ---
                try:
                    image, rotation_applied = correct_rotation(image)
                    if rotation_applied:
                        logger.info("Page %d — rotation correction applied: %d°", idx + 1, rotation_applied)
                except Exception as exc:
                    logger.warning("Page %d — rotation correction failed, using original: %s", idx + 1, exc)

                # --- Layer 2: correct slight skew (±0.5°–15°) ---
                try:
                    image, skew_applied = correct_skew(image)
                    if skew_applied:
                        logger.info("Page %d — skew correction applied: %.2f°", idx + 1, skew_applied)
                except Exception as exc:
                    logger.warning("Page %d — skew correction failed, using original: %s", idx + 1, exc)

                ocr_result = extract_text_from_image(image)
                ocr_page_count += 1
                page_text = ocr_result["text"]
                line_count = len(ocr_result["lines"])
                logger.info("Page %d — OCR complete: %d lines, %d chars.", idx + 1, line_count, len(page_text))
                all_text_parts.append(page_text)

            full_text = "\n".join(all_text_parts)

        if not full_text.strip():
            raise HTTPException(status_code=422, detail="OCR extracted no text from the document.")

        # --- Step 3: Classify ---
        classification = classify_document(full_text)
        document_type = classification["document_type"]
        logger.info("Document classified as: %s", document_type)

        if document_type == "UNKNOWN":
            elapsed = round(time.perf_counter() - t_start, 3)
            logger.warning("Document type could not be determined. Elapsed: %.3fs", elapsed)
            return {
                "document_type": "UNKNOWN",
                "extracted_data": {},
                "validation": {"is_valid": False, "errors": ["Could not classify document type."]},
                "confidence": 0.0,
                "cost": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cost_inr": 0.0},
                "ocr_text": full_text,
                "processing_time_seconds": elapsed,
            }

        # --- Step 4: LLM Extraction ---
        extracted_data, cost_info = extract_fields(full_text, document_type)

        # Augment cost with Google Vision charges (0 for PDF direct-text path)
        vision_cost_usd = GOOGLE_VISION_COST_USD * ocr_page_count
        cost_info["vision_api_calls"] = ocr_page_count
        cost_info["vision_cost_inr"] = round(vision_cost_usd * USD_TO_INR, 4)
        cost_info["cost_usd"] = round(cost_info.get("cost_usd", 0.0) + vision_cost_usd, 6)
        cost_info["cost_inr"] = round(cost_info.get("cost_inr", 0.0) + vision_cost_usd * USD_TO_INR, 4)

        logger.info(
            "Extraction complete — tokens in: %d, out: %d | LLM+Vision cost: ₹%.4f (Vision calls: %d)",
            cost_info.get("input_tokens", 0),
            cost_info.get("output_tokens", 0),
            cost_info.get("cost_inr", 0.0),
            ocr_page_count,
        )

        # --- Step 5: Validate + Score ---
        validation_result, confidence = validate_and_score(document_type, extracted_data)
        is_valid = validation_result.get("is_valid", False)
        logger.info("Validation: %s | Confidence: %.2f", "PASS" if is_valid else "FAIL", confidence)

        elapsed = round(time.perf_counter() - t_start, 3)
        logger.info("Request complete — %s | %.3fs", filename, elapsed)

        return {
            "document_type": document_type,
            "extracted_data": extracted_data,
            "validation": validation_result,
            "confidence": confidence,
            "cost": cost_info,
            "ocr_text": full_text,
            "processing_time_seconds": elapsed,
        }

    except HTTPException:
        raise
    except Exception as e:
        elapsed = round(time.perf_counter() - t_start, 3)
        logger.exception("Unhandled error processing '%s' after %.3fs: %s", filename, elapsed, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", summary="Health check")
async def health_check() -> dict:
    """Returns service health status."""
    return {"status": "ok", "service": "document-extractor"}
