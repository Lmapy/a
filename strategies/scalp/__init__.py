"""CBR-style gold scalp strategy module (Batch K).

This is a MECHANICAL APPROXIMATION of the publicly-described
TomTrades / CBR-style gold scalping model. It is NOT the proprietary
TomTrades system. The implementation here is for backtesting and
research only; the actual TomTrades methodology involves discretionary
elements (live order-flow read, news context, intuition built over
years of trading) that this engine cannot replicate.

What this module reproduces, mechanically:

  * 1H higher-timeframe bias from completed bars only
  * 1m execution inside a configurable session window
    (default: Asia "second hour", 10:00-11:00 Australia/Melbourne)
  * 20+ minutes of one-sided expansion detection
  * Sweep of previous 1H high/low OR rebalance to expansion midpoint
  * Confirmed-pivot market structure shift (no unconfirmed-pivot peeks)
  * 50% retracement limit entry on the MSB impulse
  * Recent-swing / expansion-extreme stops
  * Fixed-R / equilibrium / session-extreme targets

What this module does NOT reproduce:

  * any real volume / order-flow / DOM analysis
  * Tom's discretionary "feel" for trend days vs rotation days
  * news filters
  * exact session timing (real traders adjust per holiday)
  * any proprietary indicator that wasn't publicly described

Architecture:

    config.py       CBRGoldScalpConfig dataclass
    sessions.py     Australia/Melbourne session window logic
    detectors.py    expansion / sweep / rebalance / MSB
    bias.py         1H bias modes + DXY filter (graceful when no DXY)
    entries.py      50% retrace / market entry + stop / target
    engine.py       no-lookahead 1m bar loop + state machine
    metrics.py      per-trade + per-setup metrics
    sweep.py        parameter sweep runner

The engine runs entirely separately from the existing H4 executor /
walk-forward / prop-passing path -- nothing in those modules depends
on or is changed by this package.
"""
