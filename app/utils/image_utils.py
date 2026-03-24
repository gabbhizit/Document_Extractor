"""Image loading, preprocessing, and skew correction utilities."""

import io
import logging
import os

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


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
