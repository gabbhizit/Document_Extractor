"""OpenAI API cost tracker."""

import os
from dotenv import load_dotenv

load_dotenv()

INPUT_TOKEN_COST_USD = float(os.getenv("INPUT_TOKEN_COST_USD", "0.00015"))
OUTPUT_TOKEN_COST_USD = float(os.getenv("OUTPUT_TOKEN_COST_USD", "0.0006"))
USD_TO_INR = float(os.getenv("USD_TO_INR", "83.5"))


def calculate_cost(usage: dict) -> dict:
    """
    Calculate API cost from OpenAI usage dict.

    Args:
        usage: {"prompt_tokens": int, "completion_tokens": int, ...}

    Returns:
        {
            "input_tokens": int,
            "output_tokens": int,
            "cost_usd": float,
            "cost_inr": float
        }
    """
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    cost_usd = (input_tokens * INPUT_TOKEN_COST_USD / 1000) + (output_tokens * OUTPUT_TOKEN_COST_USD / 1000)
    cost_inr = round(cost_usd * USD_TO_INR, 6)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "cost_inr": cost_inr,
    }
