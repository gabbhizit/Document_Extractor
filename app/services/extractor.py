"""LLM-based structured data extractor using OpenAI API."""

import json
import logging
import os

import openai
from dotenv import load_dotenv

from app.services.cost_tracker import calculate_cost

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PROMPTS = {
    "PAN": (
        "Extract the following fields from this PAN card OCR text and return strict JSON only:\n"
        '{{"name": "", "pan_number": "", "date_of_birth": ""}}\n\n'
        "OCR Text:\n{text}"
    ),
    "AADHAAR": (
        "Extract the following fields from this Aadhaar card OCR text and return strict JSON only:\n"
        '{{"name": "", "aadhaar_number": "", "date_of_birth": "", "gender": ""}}\n\n'
        "OCR Text:\n{text}"
    ),
    "STUDY_CERTIFICATE": (
        "Extract the following fields from this study certificate OCR text and return strict JSON only:\n"
        '{{"name": "", "institution": "", "course": "", "year_of_passing": "", "grade": ""}}\n\n'
        "OCR Text:\n{text}"
    ),
}


def extract_fields(text: str, document_type: str) -> tuple[dict, dict]:
    """
    Extract structured fields from OCR text using OpenAI LLM.

    Args:
        text: OCR-extracted text.
        document_type: One of PAN, AADHAAR, STUDY_CERTIFICATE.

    Returns:
        (extracted_data: dict, cost_info: dict)

    Raises:
        ValueError: If document_type is unsupported or LLM returns non-JSON.
    """
    if document_type not in PROMPTS:
        raise ValueError(f"Unsupported document type: {document_type}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    prompt = PROMPTS[document_type].format(text=text)
    model = os.getenv("OPENAI_MODEL", OPENAI_MODEL)

    logger.info("Calling OpenAI API — model: %s | doc type: %s", model, document_type)

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document data extraction assistant. "
                    "Return ONLY valid JSON. No explanation, no markdown, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        extracted_data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("LLM returned non-JSON: %s", raw[:200])
        raise ValueError(f"LLM returned non-JSON response: {raw}") from e

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    cost_info = calculate_cost(usage)

    logger.info(
        "OpenAI response — in: %d tokens | out: %d tokens | cost: $%.6f (₹%.5f)",
        cost_info["input_tokens"],
        cost_info["output_tokens"],
        cost_info["cost_usd"],
        cost_info["cost_inr"],
    )

    return extracted_data, cost_info
