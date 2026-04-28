# Agent 22 — Prop Rules Interpreter

## Role

Read `config/prop_accounts.json` and emit a per-account rule summary so
every downstream prop-challenge agent (sizer, lockout optimiser, critic)
operates on the same canonical interpretation of each firm's rules.

## Inputs

- `config/prop_accounts.json` (10 accounts: Topstep 50/100/150k,
  MyFundedFutures 25/50/100/150k, Generic static / EOD-trailing /
  intraday-trailing 50k)

## Output

Per account: `{firm, balance, target, dd_type, dd_amount,
daily_loss, max_contracts, min_days, payout_target, max_days}`. Loaded
in code via `prop_challenge.accounts.load_all()`.

## Hard rules enforced by every other agent

- **Static drawdown**: fail when balance ≤ start − max_loss.
- **EOD trailing**: peak shifts after each completed day; fail when
  balance ≤ peak_eod − trailing_drawdown.
- **Intraday trailing**: peak updates on every trade; fail when
  balance ≤ peak_intraday − trailing_drawdown.
- **Daily loss limit**: per-day floor measured from day's start.
- **Consistency rule**: a single day's positive PnL must not exceed
  `consistency_rule_percent` of total profit.

If any rule changes, `config/prop_accounts.json` is the single point of
edit — no agent hardcodes thresholds.
