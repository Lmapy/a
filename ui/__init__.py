"""UI scaffold (Batch J).

Local research control room. A FastAPI backend (`ui/server.py`)
serves the data the frontend needs; a single-page HTML +
vanilla-JS frontend (`ui/static/`) renders it. No node_modules,
no build step, no React.

Backend endpoints
-----------------

  GET /api/runs                      list of recent runs (newest first)
  GET /api/runs/{run_id}/progress    progress.json
  GET /api/runs/{run_id}/events      events.jsonl as list[dict]
                                      ?after=<n> -> events at index>=n
  GET /api/runs/{run_id}/summary     summary.json
  GET /api/leaderboard               prop_passing_leaderboard.csv as list[dict]
                                      ?stem=<other>  -> different leaderboard
  GET /api/candidate/{candidate_id}  one row from the leaderboard
  GET /api/audit                     results/audit.txt as text
  GET /                              static index.html

Run with:
    make ui    or    python3 ui/server.py
"""
