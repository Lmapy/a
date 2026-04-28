"""Tests for prop firm rule configurations."""
import pytest
from prop_backtest.firms import get_firm, TOPSTEP_RULES, MFF_RULES, LUCID_RULES
from prop_backtest.firms.base import DrawdownType


def test_topstep_tiers():
    assert len(TOPSTEP_RULES.tiers) == 3
    t50 = TOPSTEP_RULES.get_tier("50K")
    assert t50.starting_balance == 50_000
    assert t50.profit_target == 3_000
    assert t50.max_trailing_drawdown == 2_000
    assert t50.daily_loss_limit == 1_000


def test_topstep_drawdown_type():
    assert TOPSTEP_RULES.drawdown_type == DrawdownType.TRAILING_INTRADAY


def test_mff_drawdown_type():
    assert MFF_RULES.drawdown_type == DrawdownType.TRAILING_EOD


def test_lucid_drawdown_type():
    assert LUCID_RULES.drawdown_type == DrawdownType.TRAILING_INTRADAY


def test_get_firm_case_insensitive():
    assert get_firm("TOPSTEP").firm_name == "TopStep"
    assert get_firm("topstep").firm_name == "TopStep"
    assert get_firm("mff").firm_name == "MyFundedFutures"


def test_get_firm_unknown():
    with pytest.raises(ValueError, match="Unknown firm"):
        get_firm("unknown_firm_xyz")


def test_tier_not_found():
    with pytest.raises(ValueError, match="Tier '999K' not found"):
        TOPSTEP_RULES.get_tier("999K")


def test_all_tiers_have_positive_values():
    for rules in [TOPSTEP_RULES, MFF_RULES, LUCID_RULES]:
        for tier in rules.tiers:
            assert tier.starting_balance > 0
            assert tier.profit_target > 0
            assert tier.max_trailing_drawdown > 0
            assert tier.daily_loss_limit > 0
            assert tier.min_trading_days > 0
