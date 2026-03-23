"""Keyword-based document type classifier."""

import re

DOCUMENT_TYPES = ["PAN", "AADHAAR", "STUDY_CERTIFICATE", "UNKNOWN"]

PAN_KEYWORDS = ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER"]
AADHAAR_KEYWORDS = ["GOVERNMENT OF INDIA", "UIDAI", "AADHAAR", "AADHAR"]
STUDY_KEYWORDS = [
    "CERTIFICATE", "SCHOOL", "COLLEGE", "UNIVERSITY",
    "BOARD OF INTERMEDIATE", "CBSE", "SSC", "HSC",
    "ICSE", "MATRICULATION", "DEGREE", "DIPLOMA",
]

AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")


def classify_document(text: str) -> dict:
    """
    Classify document type from OCR text using keyword matching.

    Args:
        text: Raw OCR-extracted text.

    Returns:
        {"document_type": "PAN" | "AADHAAR" | "STUDY_CERTIFICATE" | "UNKNOWN"}
    """
    upper = text.upper()

    if any(kw in upper for kw in PAN_KEYWORDS):
        return {"document_type": "PAN"}

    if any(kw in upper for kw in AADHAAR_KEYWORDS) or AADHAAR_PATTERN.search(text):
        return {"document_type": "AADHAAR"}

    study_hits = sum(1 for kw in STUDY_KEYWORDS if kw in upper)
    if study_hits >= 2:
        return {"document_type": "STUDY_CERTIFICATE"}

    return {"document_type": "UNKNOWN"}
