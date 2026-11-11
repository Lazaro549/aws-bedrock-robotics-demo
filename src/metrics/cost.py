"""Approximate Bedrock token cost estimates (USD).

Prices are indicative defaults for portfolio demos and can be overridden
via environment variables. They are not a billing API.
"""

from __future__ import annotations

import os

# Default: Claude 3.5 Sonnet-class on Bedrock (per 1M tokens, USD)
# Override with BEDROCK_INPUT_PRICE_PER_1M / BEDROCK_OUTPUT_PRICE_PER_1M
_DEFAULT_INPUT_PER_1M = 3.0
_DEFAULT_OUTPUT_PER_1M = 15.0


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    input_price_per_1m: float | None = None,
    output_price_per_1m: float | None = None,
) -> float:
    """Return estimated USD cost for a single model call."""
    in_price = input_price_per_1m
    if in_price is None:
        in_price = float(os.getenv("BEDROCK_INPUT_PRICE_PER_1M", _DEFAULT_INPUT_PER_1M))
    out_price = output_price_per_1m
    if out_price is None:
        out_price = float(os.getenv("BEDROCK_OUTPUT_PRICE_PER_1M", _DEFAULT_OUTPUT_PER_1M))

    cost = (input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price
    return round(cost, 8)
