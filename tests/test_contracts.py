"""Tests for contract specifications."""
import pytest
from prop_backtest.contracts.specs import CONTRACT_REGISTRY, ContractSpec, get_contract


def test_es_spec():
    es = get_contract("ES")
    assert es.symbol == "ES"
    assert es.point_value == 50.0
    assert es.tick_size == 0.25
    assert es.tick_value == 12.50
    assert es.yfinance_ticker == "ES=F"


def test_nq_spec():
    nq = get_contract("NQ")
    assert nq.point_value == 20.0
    assert nq.tick_size == 0.25
    assert nq.tick_value == 5.0


def test_all_contracts_tick_value():
    """tick_value must equal tick_size * point_value for all contracts."""
    for sym, spec in CONTRACT_REGISTRY.items():
        expected = round(spec.tick_size * spec.point_value, 6)
        assert abs(spec.tick_value - expected) < 1e-6, (
            f"{sym}: tick_value {spec.tick_value} != tick_size * point_value {expected}"
        )


def test_pnl_long_es():
    es = get_contract("ES")
    # Long 1 contract, entry 5000, exit 5004 (16 ticks)
    pnl = es.pnl(5000.0, 5004.0, 1, is_short=False)
    assert pnl == pytest.approx(16 * 12.50)  # 16 ticks * $12.50


def test_pnl_short_es():
    es = get_contract("ES")
    # Short 1 contract, entry 5000, exit 4996 (16 ticks in favour)
    pnl = es.pnl(5000.0, 4996.0, 1, is_short=True)
    assert pnl == pytest.approx(16 * 12.50)


def test_pnl_loss_es():
    es = get_contract("ES")
    # Long 1, entry 5000, exit 4990 (-40 ticks)
    pnl = es.pnl(5000.0, 4990.0, 1, is_short=False)
    assert pnl == pytest.approx(-40 * 12.50)


def test_case_insensitive_lookup():
    assert get_contract("es").symbol == "ES"
    assert get_contract("Nq").symbol == "NQ"


def test_unknown_contract():
    with pytest.raises(ValueError, match="Unknown contract"):
        get_contract("ZZZ")
