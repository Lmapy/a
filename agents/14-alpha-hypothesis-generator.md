# Agent 14 — alpha hypothesis generator (agent 02 in v3 spec)

## Role

Generate strategy hypotheses from named market-structure principles —
not from random indicator combinations. Each hypothesis comes with an
explicit failure mode, required data, and the family it belongs to.

## Source of truth

`core/strategy_families.py:STRATEGY_FAMILIES`. The v3 catalogue:

| family | class | required data | preferred entry TFs |
| --- | --- | --- | --- |
| strong_body_continuation | continuation | H4, M15 | M15, M5 |
| exhaustion_reversal | reversal | H4, M15 | M15, M5 |
| sweep_reclaim_back_to_value | reversal | H4, M5 | M5, M3 |
| fib_continuation | continuation | H4, M15 | M15, M5 |
| asia_compression_session_breakout | breakout | H4, M15 | M5, M15 |
| vwap_mean_reversion | mean_reversion | H4, M15 | M15, M5 |
| compression_breakout_continuation | breakout | H4, M15 | M15, M5 |

Adding a family requires editing `STRATEGY_FAMILIES` and adding a test
that the spec builder produces at least one runnable spec for it on
available data.

## Output

```json
{
  "generated_by": "agent_02_hypothesis_generator",
  "n_hypotheses": 7,
  "hypotheses": [
    {
      "hypothesis_id": "fib_continuation",
      "market_logic": "Trade shallow-to-deep retracement of previous H4 candle only when trend/regime agrees.",
      "expected_failure_mode": "Deep retracements that turn into full reversals -- 32% of bars retrace fully.",
      "required_data": ["H4", "M15"],
      "preferred_entry_timeframes": ["M15", "M5"],
      "strategy_family": "continuation"
    }
  ]
}
```

Written to `results/hypotheses.json`.

## Run

```bash
python3 scripts/agent_02_hypothesis.py
```
