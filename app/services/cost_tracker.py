"""API cost tracker — OpenAI (LLM extraction) + Google Vision (OCR)."""

import os
from dotenv import load_dotenv

load_dotenv()

INPUT_TOKEN_COST_USD = float(os.getenv("INPUT_TOKEN_COST_USD", "0.00015"))
OUTPUT_TOKEN_COST_USD = float(os.getenv("OUTPUT_TOKEN_COST_USD", "0.0006"))
USD_TO_INR = float(os.getenv("USD_TO_INR", "83.5"))

# Google Vision API: $1.50 per 1,000 images after free tier (first 1,000/month free)
GOOGLE_VISION_COST_USD = float(os.getenv("GOOGLE_VISION_COST_USD", "0.0015"))


def calculate_cost(usage: dict, vision_api_calls: int = 0) -> dict:
    """
    Calculate total API cost from OpenAI usage + Google Vision API calls.

    Args:
        usage: {"prompt_tokens": int, "completion_tokens": int, ...}
        vision_api_calls: Number of Vision API calls made (0 for PDF direct-text path).

    Returns:
        {
            "input_tokens": int,
            "output_tokens": int,
            "cost_usd": float,          # total (LLM + Vision)
            "cost_inr": float,          # total in INR
            "vision_api_calls": int,
            "vision_cost_inr": float,   # Vision-only cost in INR
        }
    """
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    llm_cost_usd = (input_tokens * INPUT_TOKEN_COST_USD / 1000) + (output_tokens * OUTPUT_TOKEN_COST_USD / 1000)
    vision_cost_usd = GOOGLE_VISION_COST_USD * vision_api_calls
    total_cost_usd = llm_cost_usd + vision_cost_usd

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(total_cost_usd, 6),
        "cost_inr": round(total_cost_usd * USD_TO_INR, 4),
        "vision_api_calls": vision_api_calls,
        "vision_cost_inr": round(vision_cost_usd * USD_TO_INR, 4),
    }
