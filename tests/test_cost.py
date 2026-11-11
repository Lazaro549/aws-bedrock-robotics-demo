"""Cost estimation unit tests."""

from src.metrics.cost import estimate_cost_usd


def test_zero_tokens_is_zero():
    assert estimate_cost_usd(0, 0) == 0.0


def test_cost_scales_with_tokens():
    c1 = estimate_cost_usd(1_000_000, 0, input_price_per_1m=3.0, output_price_per_1m=15.0)
    c2 = estimate_cost_usd(0, 1_000_000, input_price_per_1m=3.0, output_price_per_1m=15.0)
    assert c1 == 3.0
    assert c2 == 15.0


def test_small_call_is_tiny():
    cost = estimate_cost_usd(300, 150, input_price_per_1m=3.0, output_price_per_1m=15.0)
    assert 0 < cost < 0.01
