# Local research control room (Batch J)

Minimal local UI on top of the prop-passing engine. Reads from the
files the pipeline already writes — `results/runs/<run_id>/{progress.json,
events.jsonl, summary.json}` and `results/prop_passing_leaderboard.{csv,
meta.json}`.

## What's here

```
ui/
  __init__.py      docstring + module docs
  server.py        FastAPI backend; falls back to stdlib http.server
                    if FastAPI / uvicorn aren't installed
  static/
    index.html     5 pages: runs, leaderboard, candidate, failures, audit
    app.js         vanilla JS — no build step, no node_modules
    style.css      dark mono CSS
```

## Run

```bash
make ui                  # FastAPI (if installed); falls back to stdlib
make ui-stdlib           # always use the stdlib server
python3 ui/server.py --port 9000
```

The default address is `http://127.0.0.1:8765`. The server binds
locally only — there is no auth.

## Pages

1. **runs** — newest run at the top; click a row to load its
   `progress.json` + recent events. Polls every 4 s while a run is
   active.
2. **leaderboard** — `prop_passing_leaderboard.csv` rendered as a
   table, sorted by `prop_passing_score`. Cells are colour-coded by
   `certification_level`. Click a row to load the candidate detail
   page.
3. **candidate** — full candidate JSON for a single id. Useful
   when the leaderboard row is truncated.
4. **failures** — failure-reason histogram aggregated from the
   currently-loaded leaderboard.
5. **audit** — contents of `results/audit.txt`. Re-run
   `python3 scripts/audit.py` from the repo root to refresh.

## Backend API

```
GET /api/runs                          list of recent runs (newest first)
GET /api/runs/{run_id}/progress        progress.json
GET /api/runs/{run_id}/events          events.jsonl as list[dict]
                                        ?after=<n>  ?limit=<n>
GET /api/runs/{run_id}/summary         summary.json
GET /api/leaderboard                   leaderboard CSV + sidecar
                                        ?stem=<stem>  override CSV name
GET /api/candidate/{candidate_id}      one row from the leaderboard
GET /api/audit                         results/audit.txt as text
GET /                                  static index.html
GET /static/{path}                     served from ui/static/
```

All endpoints are read-only. The pipeline writes; the UI reads.

## What's NOT yet here

The user's brief lists a future "animated research network view"
with per-stage nodes (strategy_generator → feature_capability_auditor
→ ohlc_backtest → entry_model_lab → walk_forward → risk_sweep →
daily_rule_optimiser → prop_firm_simulator → robustness_critic →
judge → leaderboard → report). The current `runs` page shows the
*linear* event stream, but the graph view itself is not yet built.
The events.jsonl contract supports it — every event carries a
`stage` field — so it's a frontend-only addition when needed.

The TPO / Session Explorer is not yet implemented either; the
backend would need to load + render TPO profiles per session, which
is a fair amount of frontend work beyond the current scope.

## Implementation notes

* **No build step.** Vanilla JS, vanilla CSS, single index.html.
  No webpack, no React, no node_modules. The tradeoff is simpler
  visuals; the contract files are designed so swapping in React
  later is straightforward (every page reads from one or two API
  endpoints).
* **FastAPI is optional.** The same routes are served by a minimal
  `http.server.BaseHTTPRequestHandler` fallback so anyone who
  hasn't run `pip install fastapi uvicorn` can still launch the
  UI. Tests run against the file-reader functions directly so
  CI doesn't need FastAPI either.
* **Polling, not websockets.** The frontend polls every 4 seconds.
  A WebSocket / SSE upgrade would be straightforward; keeping
  polling means there's no async server state to manage.
