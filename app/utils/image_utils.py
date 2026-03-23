"""Image loading, preprocessing, rotation correction, and skew correction utilities."""

import io
import math
import logging
import os

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Layer 1 constants ────────────────────────────────────────────────────────
FAST_PATH_MIN_CONFIDENCE = 0.85
FAST_PATH_MIN_LINES = 5
KEYWORD_BOOST = 0.2

# All classifier keywords — used for fast-path orientation gate
_PAN_KEYWORDS = {"INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER"}
_AADHAAR_KEYWORDS = {"GOVERNMENT OF INDIA", "UIDAI", "AADHAAR", "AADHAR"}
_STUDY_KEYWORDS = {
    "CERTIFICATE", "SCHOOL", "COLLEGE", "UNIVERSITY", "CBSE",
    "SSC", "HSC", "ICSE", "DEGREE", "DIPLOMA",
    "BOARD OF INTERMEDIATE", "MATRICULATION",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _keyword_hit(text: str) -> bool:
    """
    Return True if OCR text contains any strong document classification keyword.

    PAN / Aadhaar: 1 keyword match is sufficient (highly specific strings).
    Study Certificate: requires 2+ keyword matches (same threshold as classifier).
    """
    upper = text.upper()
    if any(kw in upper for kw in _PAN_KEYWORDS):
        return True
    if any(kw in upper for kw in _AADHAAR_KEYWORDS):
        return True
    study_hits = sum(1 for kw in _STUDY_KEYWORDS if kw in upper)
    return study_hits >= 2


def _composite_score(bounding_boxes: list[dict], text: str) -> float:
    """
    Composite orientation score for a set of OCR bounding boxes.

        score = mean(confidences) × log1p(line_count) + keyword_boost

    - mean confidence captures OCR quality  (garbage tokens score low)
    - log1p(line_count) captures quantity   (diminishing returns above ~20 lines)
    - +0.2 keyword_boost decisively favours semantically correct orientations
    """
    if not bounding_boxes:
        return 0.0
    confs = [b["confidence"] for b in bounding_boxes]
    mean_conf = sum(confs) / len(confs)
    score = mean_conf * math.log1p(len(bounding_boxes))
    if _keyword_hit(text):
        score += KEYWORD_BOOST
    return score


# ── Public utilities ──────────────────────────────────────────────────────────

def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """
    Load a PIL Image from raw bytes.

    Args:
        file_bytes: Raw image file content.

    Returns:
        PIL Image object in RGB mode.
    """
    image = Image.open(io.BytesIO(file_bytes))
    return image.convert("RGB")


def preprocess_image(image: Image.Image, max_size: int = 1000) -> Image.Image:
    """
    Resize image so its longest side does not exceed max_size, preserving aspect ratio.

    Capping the longest side (not just width) prevents portrait phone photos from
    being resized to 1400×1867 — which is still large and slow for OCR.

    Args:
        image: PIL Image.
        max_size: Maximum pixels on the longest side.

    Returns:
        Resized PIL Image.
    """
    longest = max(image.width, image.height)
    if longest > max_size:
        ratio = max_size / longest
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return image


def correct_rotation(image: Image.Image) -> tuple[Image.Image, int]:
    """
    Detect and correct cardinal rotation (0°, 90°, 180°, 270°) via
    confidence-scored OCR voting.

    Fast path — returns immediately (zero extra OCR passes) when ALL hold:
      1. mean OCR confidence ≥ 0.85
      2. ≥ 5 text lines detected
      (keyword check removed — confidence + line count is sufficient for orientation)

    Slow path — scores all 4 orientations:
      score = mean_conf × log1p(lines) + 0.2 if keyword matched
      Picks the orientation with the highest composite score.

    Set SKIP_ORIENTATION_CORRECTION=true in .env to bypass entirely.

    Args:
        image: PIL Image (RGB), already preprocessed/resized.

    Returns:
        (corrected_image, rotation_applied_degrees)  — rotation ∈ {0, 90, 180, 270}
    """
    if os.getenv("SKIP_ORIENTATION_CORRECTION", "false").lower() == "true":
        return image, 0

    # Lazy import — prevents circular dependency at module load time
    from app.services.ocr import extract_text_from_image  # noqa: PLC0415

    # ── Pass 0: original orientation ──────────────────────────────────────────
    result_0 = extract_text_from_image(image)
    bboxes_0 = result_0.get("bounding_boxes", [])
    text_0 = result_0["text"]

    # Fast path: 2-gate check (confidence + line count).
    # Keyword check removed — it was too fragile (single OCR misread caused slow path).
    # High confidence across ≥5 lines is sufficient proof of correct orientation.
    if bboxes_0:
        mean_conf_0 = sum(b["confidence"] for b in bboxes_0) / len(bboxes_0)
        if (
            mean_conf_0 >= FAST_PATH_MIN_CONFIDENCE
            and len(bboxes_0) >= FAST_PATH_MIN_LINES
        ):
            logger.debug("Rotation fast path: original orientation accepted (conf=%.3f, lines=%d).",
                         mean_conf_0, len(bboxes_0))
            return image, 0

    # ── Slow path: score all 4 rotations ─────────────────────────────────────
    best_score = _composite_score(bboxes_0, text_0)
    best_rotation = 0
    best_image = image

    for angle in [90, 180, 270]:
        rotated = image.rotate(angle, expand=True)
        result = extract_text_from_image(rotated)
        score = _composite_score(result.get("bounding_boxes", []), result["text"])
        logger.debug("Rotation %d° composite score: %.4f", angle, score)
        if score > best_score:
            best_score = score
            best_rotation = angle
            best_image = rotated

    if best_rotation != 0:
        logger.info("Rotation correction applied: %d°", best_rotation)

    return best_image, best_rotation


def correct_skew(image: Image.Image) -> tuple[Image.Image, float]:
    """
    Detect and correct slight document skew (±0.5° to ±15°) using dual-method
    angle estimation with agreement checking.

    Primary method:
        Horizontal morphological dilation (40×1 kernel) merges individual
        characters into text-line blobs. minAreaRect on the largest qualifying
        blob returns the baseline skew angle.

    Fallback method:
        HoughLinesP on near-horizontal Canny edges → median angle.
        Used when no large contour is found (e.g. plain-paper study certificates).

    Agreement check:
        - Both agree within ±2°  → use their average (more precise estimate)
        - Disagree              → prefer minAreaRect if its contour > 20% of
                                   image area, otherwise prefer Hough median
        - Only one available    → use it directly

    Decision window:
        < ±0.5°   → skip (imperceptible, introduces resampling noise)
        ±0.5–15°  → apply warpAffine correction
        > ±15°    → skip (likely false detection from diagonal/decorative elements)

    Set SKIP_ORIENTATION_CORRECTION=true in .env to bypass entirely.

    Args:
        image: PIL Image (RGB), already rotation-corrected.

    Returns:
        (corrected_image, skew_angle_applied_degrees)
    """
    if os.getenv("SKIP_ORIENTATION_CORRECTION", "false").lower() == "true":
        return image, 0.0

    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    image_area = float(h * w)

    # Shared preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Primary: horizontal dilation → largest text-line blob → minAreaRect ──
    mar_angle: float | None = None
    mar_contour_area: float = 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    qualifying = [c for c in contours if cv2.contourArea(c) > 0.05 * image_area]
    if qualifying:
        largest = max(qualifying, key=cv2.contourArea)
        mar_contour_area = float(cv2.contourArea(largest))
        rect = cv2.minAreaRect(largest)
        angle = float(rect[-1])
        # minAreaRect returns [-90, 0] — normalise to [-45, +45]
        if angle < -45.0:
            angle += 90.0
        mar_angle = angle
        logger.debug("minAreaRect angle: %.2f° (contour area: %.0f px²)", angle, mar_contour_area)

    # ── Fallback: HoughLinesP → median near-horizontal angle ─────────────────
    hough_angle: float | None = None
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(w * 0.3),
        maxLineGap=20,
    )
    if lines is not None:
        h_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(a) < 30.0:   # near-horizontal lines only
                h_angles.append(a)
        if h_angles:
            hough_angle = float(np.median(h_angles))
            logger.debug("Hough median angle: %.2f° (%d qualifying lines)", hough_angle, len(h_angles))

    # ── Dual-method agreement check ───────────────────────────────────────────
    final_angle: float | None = None

    if mar_angle is not None and hough_angle is not None:
        if abs(mar_angle - hough_angle) <= 2.0:
            final_angle = (mar_angle + hough_angle) / 2.0
            logger.debug("Methods agree — using average: %.2f°", final_angle)
        else:
            if mar_contour_area > 0.20 * image_area:
                final_angle = mar_angle
                logger.debug("Methods disagree — minAreaRect preferred (large contour): %.2f°", final_angle)
            else:
                final_angle = hough_angle
                logger.debug("Methods disagree — Hough preferred (small contour): %.2f°", final_angle)
    elif mar_angle is not None:
        final_angle = mar_angle
    elif hough_angle is not None:
        final_angle = hough_angle

    if final_angle is None:
        return image, 0.0

    # ── Decision window ───────────────────────────────────────────────────────
    abs_angle = abs(final_angle)
    if abs_angle < 0.5 or abs_angle > 15.0:
        logger.debug("Skew angle %.2f° outside correction window [±0.5°, ±15°] — skipping.", final_angle)
        return image, 0.0

    # ── Apply warpAffine with BORDER_REPLICATE to avoid black border artifacts ─
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
    corrected_np = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # img_np is RGB (from PIL) — no channel swap needed when converting back
    corrected_pil = Image.fromarray(corrected_np)

    logger.info("Skew correction applied: %.2f°", final_angle)
    return corrected_pil, round(final_angle, 2)
