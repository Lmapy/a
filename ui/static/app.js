// Minimal vanilla-JS frontend for the prop-passing control room.
// Polls the backend every few seconds; no SSE / websockets to keep
// the deployment simple.

const PAGES = ["runs", "leaderboard", "candidate", "failure", "audit"];
const POLL_MS = 4000;

let currentRun = null;
let lastEventIndex = 0;
let leaderboardCache = null;

// ---- nav -----------------------------------------------------------------

function showPage(name) {
  PAGES.forEach(p => {
    const sec = document.getElementById("page-" + p);
    if (sec) sec.classList.toggle("active", p === name);
  });
  document.querySelectorAll("header nav button").forEach(b => {
    b.classList.toggle("active", b.dataset.page === name);
  });
  if (name === "runs")        loadRuns();
  if (name === "leaderboard") loadLeaderboard();
  if (name === "failure")     renderFailureHistogram();
  if (name === "audit")       loadAudit();
}

document.querySelectorAll("header nav button").forEach(b => {
  b.addEventListener("click", () => showPage(b.dataset.page));
});

// ---- API helper ----------------------------------------------------------

async function api(path) {
  try {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(resp.status + " " + resp.statusText);
    setConn(true, path);
    const ct = resp.headers.get("content-type") || "";
    return ct.includes("json") ? resp.json() : resp.text();
  } catch (e) {
    setConn(false, e.message);
    return null;
  }
}

function setConn(ok, msg) {
  const el = document.getElementById("conn");
  el.classList.toggle("ok", ok);
  el.classList.toggle("bad", !ok);
  el.textContent = ok ? "✓ " + (msg || "") : "✗ " + msg;
}

// ---- runs page ----------------------------------------------------------

async function loadRuns() {
  const runs = await api("/api/runs");
  if (!runs) return;
  const tbody = document.querySelector("#runs-table tbody");
  tbody.innerHTML = "";
  runs.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.run_id}</td>
      <td>${r.status ?? ""}</td>
      <td>${r.current_stage ?? ""}</td>
      <td>${(r.started_at ?? "").slice(0, 19)}</td>
      <td>${(r.updated_at ?? "").slice(0, 19)}</td>`;
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => selectRun(r.run_id));
    tbody.appendChild(tr);
  });
  if (!currentRun && runs.length) selectRun(runs[0].run_id);
}

async function selectRun(rid) {
  if (currentRun !== rid) {
    currentRun = rid;
    lastEventIndex = 0;
    document.querySelector("#run-events").innerHTML = "";
  }
  await refreshSelectedRun();
}

async function refreshSelectedRun() {
  if (!currentRun) return;
  const progress = await api(`/api/runs/${currentRun}/progress`);
  if (progress)
    document.getElementById("run-progress").textContent =
      JSON.stringify(progress, null, 2);
  // poll new events
  const events = await api(
    `/api/runs/${currentRun}/events?after=${lastEventIndex}&limit=200`);
  if (Array.isArray(events) && events.length) {
    const ul = document.getElementById("run-events");
    events.forEach(e => {
      const li = document.createElement("li");
      li.classList.add(e.status || "");
      const ts = (e.timestamp || "").slice(11, 19);
      const cid = e.candidate_id ? ` [${e.candidate_id}]` : "";
      const fail = e.failure_reasons && e.failure_reasons.length
                    ? ` (${e.failure_reasons.join(", ")})` : "";
      li.textContent = `${ts}  ${e.stage || ""}  ${e.status || ""}${cid}${fail}`;
      ul.insertBefore(li, ul.firstChild);
    });
    lastEventIndex += events.length;
  }
}

// ---- leaderboard page ---------------------------------------------------

async function loadLeaderboard() {
  const data = await api("/api/leaderboard");
  if (!data) return;
  leaderboardCache = data;
  document.getElementById("lb-meta").textContent =
    data.meta ? `produced ${data.meta.produced_at_utc}, ` +
                  `git ${(data.meta.git_head || "").slice(0, 7)}, ` +
                  `mode=${data.meta.mode}, ` +
                  `n=${data.rows.length}`
              : "(no .meta.json sidecar)";
  const tbody = document.querySelector("#lb-table tbody");
  tbody.innerHTML = "";
  data.rows.forEach(r => {
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => loadCandidate(r.candidate_id));
    const cert = r.certification_level || "";
    const fail = (r.fail_reasons || "").slice(0, 80);
    tr.innerHTML = `
      <td>${r.candidate_id}</td>
      <td>${r.family || ""}</td>
      <td class="cert-${cert}">${cert}</td>
      <td>${fmtNum(r.prop_passing_score)}</td>
      <td>${fmtNum(r.pass_probability)}</td>
      <td>${fmtNum(r.blowup_probability)}</td>
      <td>${r.wf_folds ?? ""}</td>
      <td>${fmtNum(r.wf_median_sharpe)}</td>
      <td>${fmtNum(r.holdout_return)}</td>
      <td>${fmtNum(r.label_perm_p)}</td>
      <td title="${r.fail_reasons || ""}">${fail}</td>`;
    tbody.appendChild(tr);
  });
}

function fmtNum(x) {
  if (x === null || x === undefined || x === "") return "";
  if (typeof x === "number") return Number(x.toFixed(4)).toString();
  const n = Number(x);
  return Number.isFinite(n) ? Number(n.toFixed(4)).toString() : x;
}

// ---- candidate detail --------------------------------------------------

async function loadCandidate(cid) {
  if (!cid) return;
  showPage("candidate");
  const detail = await api(`/api/candidate/${cid}`);
  if (detail) {
    document.getElementById("cand-detail").textContent =
      JSON.stringify(detail, null, 2);
    document.getElementById("cand-id-input").value = cid;
  }
}

document.getElementById("cand-load-btn").addEventListener("click", () => {
  const cid = document.getElementById("cand-id-input").value.trim();
  if (cid) loadCandidate(cid);
});

// ---- failure histogram --------------------------------------------------

function renderFailureHistogram() {
  const tbody = document.querySelector("#fail-table tbody");
  tbody.innerHTML = "";
  if (!leaderboardCache || !leaderboardCache.rows) {
    tbody.innerHTML = "<tr><td colspan=\"2\">load leaderboard first</td></tr>";
    return;
  }
  const counts = {};
  leaderboardCache.rows.forEach(r => {
    const tokens = (r.fail_reasons || "").split(";")
        .map(s => s.trim()).filter(Boolean);
    tokens.forEach(t => { counts[t] = (counts[t] || 0) + 1; });
  });
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    tbody.innerHTML = "<tr><td colspan=\"2\">no failures</td></tr>";
    return;
  }
  entries.forEach(([reason, n]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${reason}</td><td>${n}</td>`;
    tbody.appendChild(tr);
  });
}

// ---- audit --------------------------------------------------------------

async function loadAudit() {
  const text = await api("/api/audit");
  if (typeof text === "string")
    document.getElementById("audit-log").textContent = text;
}

// ---- main ---------------------------------------------------------------

showPage("runs");
setInterval(refreshSelectedRun, POLL_MS);
setInterval(() => {
  if (document.getElementById("page-runs").classList.contains("active")) {
    loadRuns();
  }
}, POLL_MS * 2);
