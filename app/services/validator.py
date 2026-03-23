"""Field validation and confidence scoring."""

import re

PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
AADHAAR_REGEX = re.compile(r"^\d{12}$")


def validate_and_score(document_type: str, extracted_data: dict) -> tuple[dict, float]:
    """
    Validate extracted fields and compute confidence score.

    Rules:
        - Base confidence: 0.9
        - Each missing required field: -0.1
        - Each validation failure: -0.15

    Args:
        document_type: PAN | AADHAAR | STUDY_CERTIFICATE
        extracted_data: Dict of extracted fields.

    Returns:
        (validation_result: dict, confidence: float)
    """
    errors = []
    confidence = 0.9

    if document_type == "PAN":
        required = ["name", "pan_number", "date_of_birth"]
        for field in required:
            if not extracted_data.get(field):
                errors.append(f"Missing field: {field}")
                confidence -= 0.1

        pan = extracted_data.get("pan_number", "")
        clean_pan = pan.replace(" ", "").upper()
        if pan and not PAN_REGEX.match(clean_pan):
            errors.append(f"Invalid PAN format: {pan}")
            confidence -= 0.15

    elif document_type == "AADHAAR":
        required = ["name", "aadhaar_number", "date_of_birth", "gender"]
        for field in required:
            if not extracted_data.get(field):
                errors.append(f"Missing field: {field}")
                confidence -= 0.1

        aadhaar = extracted_data.get("aadhaar_number", "")
        clean_aadhaar = re.sub(r"\s", "", aadhaar)
        if aadhaar and not AADHAAR_REGEX.match(clean_aadhaar):
            errors.append(f"Invalid Aadhaar format (must be 12 digits): {aadhaar}")
            confidence -= 0.15

    elif document_type == "STUDY_CERTIFICATE":
        required = ["name", "institution", "course", "year_of_passing"]
        for field in required:
            if not extracted_data.get(field):
                errors.append(f"Missing field: {field}")
                confidence -= 0.1

    confidence = max(round(confidence, 2), 0.0)

    validation_result = {
        "is_valid": len(errors) == 0,
        "errors": errors,
    }

    return validation_result, confidence
