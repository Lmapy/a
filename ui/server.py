"""FastAPI backend for the local research control room (Batch J).

Reads from `results/runs/<run_id>/{progress.json,events.jsonl,summary.json}`
and `results/<leaderboard>.csv` to expose a small JSON API the
frontend consumes. Stateless and read-only — the pipeline writes
files; the UI reads them.

Run:
    python3 ui/server.py             # http://127.0.0.1:8765
    python3 ui/server.py --port 9000

Optional: install `uvicorn` for production-quality reload, but the
default `app.run()` via `uvicorn.run(...)` works fine on a
single-laptop research setup.

The frontend lives at `ui/static/index.html` + `ui/static/app.js`
+ `ui/static/style.css`. The backend serves these static files at
the root path (`/`) so there's no separate dev server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --- lazy import: FastAPI is optional unless you actually run the server.
def _try_import_fastapi():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
        from fastapi.staticfiles import StaticFiles
        return FastAPI, HTTPException, HTMLResponse, JSONResponse, PlainTextResponse, StaticFiles
    except ImportError:
        return None


RESULTS = ROOT / "results"
RUNS_DIR = RESULTS / "runs"
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------- file readers -----------------------------------------------

def list_runs() -> list[dict]:
    """Return run-id directories under results/runs/, newest first."""
    if not RUNS_DIR.exists():
        return []
    out = []
    for p in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        progress = read_progress(p.name)
        out.append({
            "run_id": p.name,
            "status": progress.get("status") if progress else None,
            "current_stage": progress.get("current_stage") if progress else None,
            "started_at": progress.get("started_at") if progress else None,
            "updated_at": progress.get("updated_at") if progress else None,
            "counts": progress.get("counts") if progress else None,
        })
    return out


def read_progress(run_id: str) -> dict | None:
    p = RUNS_DIR / run_id / "progress.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_summary(run_id: str) -> dict | None:
    p = RUNS_DIR / run_id / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_events(run_id: str, after: int = 0, limit: int = 1000) -> list[dict]:
    p = RUNS_DIR / run_id / "events.jsonl"
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < after:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
    return out


def read_leaderboard(stem: str = "prop_passing_leaderboard") -> dict:
    """Load the leaderboard CSV + sidecar meta as a single payload."""
    import pandas as pd
    csv_path = RESULTS / f"{stem}.csv"
    meta_path = RESULTS / f"{stem}.meta.json"
    if not csv_path.exists():
        return {"rows": [], "meta": None,
                "error": f"{csv_path.name} not found"}
    df = pd.read_csv(csv_path).fillna("")
    rows = df.to_dict(orient="records")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return {"rows": rows, "meta": meta, "stem": stem}


def read_candidate(candidate_id: str,
                    stem: str = "prop_passing_leaderboard") -> dict | None:
    """Find one row in the leaderboard by candidate_id."""
    lb = read_leaderboard(stem)
    for r in lb["rows"]:
        if str(r.get("candidate_id")) == candidate_id:
            return r
    return None


def read_audit() -> str:
    p = RESULTS / "audit.txt"
    if not p.exists():
        return "(audit.txt not found — run `python3 scripts/audit.py`)"
    return p.read_text(encoding="utf-8", errors="replace")


# ---------------- FastAPI app builder ---------------------------------------

def build_app():
    fa = _try_import_fastapi()
    if fa is None:
        raise RuntimeError(
            "FastAPI is not installed. Install with `pip install "
            "fastapi uvicorn` or run with `--no-fastapi` to use the "
            "stdlib stub server.")
    FastAPI, HTTPException, HTMLResponse, JSONResponse, PlainTextResponse, StaticFiles = fa

    app = FastAPI(title="XAUUSD prop-firm passing — control room",
                  version="0.1.0")

    @app.get("/api/runs")
    def _runs():
        return list_runs()

    @app.get("/api/runs/{run_id}/progress")
    def _progress(run_id: str):
        p = read_progress(run_id)
        if p is None:
            raise HTTPException(404, f"run {run_id} not found")
        return p

    @app.get("/api/runs/{run_id}/summary")
    def _summary(run_id: str):
        s = read_summary(run_id)
        if s is None:
            raise HTTPException(404, f"summary.json not found for {run_id}")
        return s

    @app.get("/api/runs/{run_id}/events")
    def _events(run_id: str, after: int = 0, limit: int = 1000):
        return read_events(run_id, after=after, limit=limit)

    @app.get("/api/leaderboard")
    def _leaderboard(stem: str = "prop_passing_leaderboard"):
        return read_leaderboard(stem)

    @app.get("/api/candidate/{candidate_id}")
    def _candidate(candidate_id: str,
                    stem: str = "prop_passing_leaderboard"):
        row = read_candidate(candidate_id, stem)
        if row is None:
            raise HTTPException(404, f"candidate {candidate_id} not found")
        return row

    @app.get("/api/audit", response_class=PlainTextResponse)
    def _audit():
        return read_audit()

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)),
                  name="static")
    @app.get("/", response_class=HTMLResponse)
    def _index():
        idx = STATIC_DIR / "index.html"
        if idx.exists():
            return idx.read_text(encoding="utf-8")
        return ("<h1>UI not built</h1>"
                "<p>Frontend files are at ui/static/. The backend is "
                "serving the JSON API at /api/* — try /api/runs.</p>")

    return app


# ---------------- stdlib stub fallback --------------------------------------
#
# When FastAPI / uvicorn aren't installed we fall back to a minimal
# stdlib http.server. Same routes, same shape.

def _stdlib_app(host: str, port: int) -> int:
    import http.server
    import socketserver
    import urllib.parse as _up

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, status: int = 200) -> None:
            data = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_text(self, text: str, status: int = 200,
                        ctype: str = "text/plain") -> None:
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            url = _up.urlparse(self.path)
            qs = _up.parse_qs(url.query)
            path = url.path

            if path == "/" or path == "/index.html":
                idx = STATIC_DIR / "index.html"
                if idx.exists():
                    self._send_text(idx.read_text(encoding="utf-8"),
                                     ctype="text/html")
                    return
                self._send_text("<h1>UI not built</h1>", ctype="text/html")
                return
            if path.startswith("/static/"):
                rel = path[len("/static/"):]
                f = STATIC_DIR / rel
                if not f.exists() or not f.is_file():
                    self._send_text("not found", status=404)
                    return
                ctype = ("text/css" if f.suffix == ".css"
                          else "application/javascript" if f.suffix == ".js"
                          else "text/plain")
                self._send_text(f.read_text(encoding="utf-8"), ctype=ctype)
                return
            if path == "/api/runs":
                self._send_json(list_runs()); return
            if path.startswith("/api/runs/") and path.endswith("/progress"):
                rid = path.split("/")[3]
                p = read_progress(rid)
                if p is None:
                    self._send_json({"error": "not found"}, status=404)
                else:
                    self._send_json(p)
                return
            if path.startswith("/api/runs/") and path.endswith("/summary"):
                rid = path.split("/")[3]
                s = read_summary(rid)
                if s is None:
                    self._send_json({"error": "not found"}, status=404)
                else:
                    self._send_json(s)
                return
            if path.startswith("/api/runs/") and path.endswith("/events"):
                rid = path.split("/")[3]
                after = int(qs.get("after", ["0"])[0])
                limit = int(qs.get("limit", ["1000"])[0])
                self._send_json(read_events(rid, after=after, limit=limit))
                return
            if path == "/api/leaderboard":
                stem = qs.get("stem", ["prop_passing_leaderboard"])[0]
                self._send_json(read_leaderboard(stem)); return
            if path.startswith("/api/candidate/"):
                cid = path[len("/api/candidate/"):]
                stem = qs.get("stem", ["prop_passing_leaderboard"])[0]
                row = read_candidate(cid, stem)
                if row is None:
                    self._send_json({"error": "not found"}, status=404)
                else:
                    self._send_json(row)
                return
            if path == "/api/audit":
                self._send_text(read_audit(), ctype="text/plain")
                return
            self._send_text("not found", status=404)

        def log_message(self, fmt, *args):
            return  # suppress default access logs

    with socketserver.TCPServer((host, port), Handler) as httpd:
        print(f"[ui] stdlib server listening on http://{host}:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-fastapi", action="store_true",
                    help="use the stdlib http.server fallback (no FastAPI)")
    args = ap.parse_args(argv)

    fa = _try_import_fastapi()
    if args.no_fastapi or fa is None:
        if fa is None:
            print("[ui] FastAPI not installed; using stdlib server.")
        return _stdlib_app(args.host, args.port)

    import uvicorn
    app = build_app()
    print(f"[ui] FastAPI server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port,
                 log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
