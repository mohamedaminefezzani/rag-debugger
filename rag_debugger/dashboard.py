"""
rd.dashboard() — local web dashboard served via FastAPI.

Usage:
    import rag_debugger as rd
    rd.init(project="my-app")
    rd.dashboard()  # opens http://localhost:7842
"""
import json
import time
import threading
import webbrowser
from typing import Optional
from .store import SQLiteStore


HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>rag-debugger</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d0f12;--surface:#161920;--surface2:#1e2229;--border:#2a2f3a;
  --accent:#4ade9f;--text:#e8eaf0;--muted:#7a8099;--warn:#f0a04a;
  --err:#f06a6a;--blue:#6aa4f0;--purple:#a48af0;--r:6px;
  --mono:'JetBrains Mono',monospace;--sans:'Inter',system-ui,sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.6}
header{background:var(--surface);border-bottom:1px solid var(--border);padding:14px 28px;display:flex;align-items:center;gap:16px}
header h1{font-size:15px;font-weight:600;color:var(--accent);font-family:var(--mono)}
header .sub{color:var(--muted);font-size:13px}
.stat-bar{display:flex;gap:1px;background:var(--border);border-bottom:1px solid var(--border)}
.stat{flex:1;background:var(--surface);padding:16px 24px}
.stat .val{font-size:24px;font-weight:600;font-family:var(--mono);color:var(--text)}
.stat .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.stat .val.warn{color:var(--warn)}.stat .val.err{color:var(--err)}.stat .val.ok{color:var(--accent)}
.layout{display:flex;height:calc(100vh - 113px)}
.sidebar{width:280px;min-width:280px;border-right:1px solid var(--border);overflow-y:auto;background:var(--surface)}
.sidebar-hdr{padding:12px 16px;font-size:11px;font-weight:600;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;border-bottom:1px solid var(--border)}
.sess-item{padding:12px 16px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .1s}
.sess-item:hover{background:var(--surface2)}
.sess-item.active{background:var(--surface2);border-left:3px solid var(--accent)}
.sess-id{font-family:var(--mono);font-size:12px;color:var(--blue);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sess-meta{font-size:11px;color:var(--muted);margin-top:3px;display:flex;gap:8px}
.badge{display:inline-block;font-size:10px;padding:1px 6px;border-radius:10px;font-weight:500}
.badge-ok{background:#1a3d2e;color:var(--accent)}.badge-gap{background:#3a1e1e;color:var(--err)}
.badge-warn{background:#2a2210;color:var(--warn)}
.main{flex:1;overflow-y:auto;padding:24px 28px}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--muted);gap:8px}
.empty .icon{font-size:32px}.empty p{font-size:13px}
.section-hdr{font-size:11px;font-weight:600;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin:24px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.section-hdr:first-child{margin-top:0}
.summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);border-radius:var(--r);overflow:hidden;margin-bottom:24px}
.summary-cell{background:var(--surface2);padding:14px 16px}
.summary-cell .v{font-size:20px;font-weight:600;font-family:var(--mono)}
.summary-cell .l{font-size:11px;color:var(--muted);margin-top:2px}
.event-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);margin-bottom:12px;overflow:hidden}
.event-hdr{display:flex;align-items:center;gap:10px;padding:12px 16px;background:var(--surface2);border-bottom:1px solid var(--border);cursor:pointer}
.event-hdr:hover{background:#252a33}
.event-query{flex:1;font-size:13px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.event-meta{font-size:11px;color:var(--muted);display:flex;gap:10px;flex-shrink:0}
.event-body{padding:16px}
.chunk-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.chunk-label{font-size:11px;color:var(--muted);width:52px;flex-shrink:0;font-family:var(--mono)}
.chunk-bar-wrap{flex:1;height:20px;background:var(--surface2);border-radius:3px;overflow:hidden;position:relative}
.chunk-bar{height:100%;border-radius:3px;transition:width .3s}
.chunk-score{position:absolute;right:6px;top:50%;transform:translateY(-50%);font-size:11px;font-family:var(--mono);color:var(--text)}
.chunk-content{font-size:11px;color:var(--muted);margin-top:4px;padding:6px 10px;background:var(--surface2);border-radius:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
.gap-flag{background:#3a1e1e;border:1px solid #5a2a2a;border-radius:var(--r);padding:10px 14px;margin-top:12px;font-size:12px;color:var(--err)}
.gap-flag strong{display:block;margin-bottom:4px}
.refresh-btn{background:var(--surface2);border:1px solid var(--border);color:var(--muted);font-size:12px;padding:5px 12px;border-radius:var(--r);cursor:pointer;font-family:var(--sans)}
.refresh-btn:hover{color:var(--text)}
.toggle-icon{color:var(--muted);font-size:10px;transition:transform .2s}
.collapsed .toggle-icon{transform:rotate(-90deg)}
.collapsed .event-body{display:none}
pre{font-family:var(--mono);font-size:11px;background:var(--surface2);padding:10px;border-radius:var(--r);overflow-x:auto;color:#c0c4d0}
</style>
</head>
<body>
<header>
  <h1>rag-debugger</h1>
  <span class="sub" id="project-name">loading…</span>
  <div style="flex:1"></div>
  <button class="refresh-btn" onclick="load()">↻ Refresh</button>
</header>

<div class="stat-bar">
  <div class="stat"><div class="val" id="s-sessions">—</div><div class="lbl">Sessions</div></div>
  <div class="stat"><div class="val" id="s-events">—</div><div class="lbl">Total events</div></div>
  <div class="stat"><div class="val" id="s-avg">—</div><div class="lbl">Avg top score</div></div>
  <div class="stat"><div class="val" id="s-gaps">—</div><div class="lbl">Gap events</div></div>
</div>

<div class="layout">
  <div class="sidebar">
    <div class="sidebar-hdr">Sessions</div>
    <div id="session-list"></div>
  </div>
  <div class="main" id="main-panel">
    <div class="empty"><div class="icon">⬡</div><p>Select a session to inspect</p></div>
  </div>
</div>

<script>
let allData = null;
let activeSession = null;

async function load() {
  const r = await fetch('/api/data');
  allData = await r.json();
  renderStats(allData.stats);
  renderSidebar(allData.sessions);
  document.getElementById('project-name').textContent = allData.project || '';
  if (activeSession) renderMain(activeSession);
}

function renderStats(s) {
  document.getElementById('s-sessions').textContent = s.total_sessions;
  document.getElementById('s-events').textContent = s.total_events;
  const avgEl = document.getElementById('s-avg');
  avgEl.textContent = s.avg_top_score.toFixed(3);
  avgEl.className = 'val ' + (s.avg_top_score >= 0.72 ? 'ok' : s.avg_top_score >= 0.55 ? 'warn' : 'err');
  const gapEl = document.getElementById('s-gaps');
  gapEl.textContent = s.gap_events;
  gapEl.className = 'val ' + (s.gap_events === 0 ? 'ok' : s.gap_events < 5 ? 'warn' : 'err');
}

function renderSidebar(sessions) {
  const el = document.getElementById('session-list');
  if (!sessions.length) { el.innerHTML = '<div style="padding:16px;color:var(--muted);font-size:13px">No sessions yet</div>'; return; }
  el.innerHTML = sessions.map(s => `
    <div class="sess-item ${activeSession === s.id ? 'active' : ''}" onclick="selectSession('${s.id}')">
      <div class="sess-id">${s.id}</div>
      <div class="sess-meta">
        <span>${s.event_count} event${s.event_count !== 1 ? 's' : ''}</span>
        <span class="badge ${s.gap_count > 0 ? 'badge-gap' : 'badge-ok'}">${s.gap_count > 0 ? s.gap_count + ' gap' + (s.gap_count > 1 ? 's' : '') : '✓ ok'}</span>
        ${s.user ? '<span>' + s.user + '</span>' : ''}
      </div>
    </div>
  `).join('');
}

function selectSession(id) {
  activeSession = id;
  renderSidebar(allData.sessions);
  renderMain(id);
}

function renderMain(sessionId) {
  const sess = allData.sessions.find(s => s.id === sessionId);
  if (!sess) return;
  const panel = document.getElementById('main-panel');

  const avgScore = sess.avg_top_score || 0;
  const worstScore = sess.lowest_top_score || 0;

  panel.innerHTML = `
    <div class="section-hdr">Session summary — ${sessionId}</div>
    <div class="summary-grid">
      <div class="summary-cell"><div class="v">${sess.event_count}</div><div class="l">Events</div></div>
      <div class="summary-cell"><div class="v" style="color:${scoreColor(avgScore)}">${avgScore.toFixed(3)}</div><div class="l">Avg top score</div></div>
      <div class="summary-cell"><div class="v" style="color:${scoreColor(worstScore)}">${worstScore.toFixed(3)}</div><div class="l">Worst score</div></div>
      <div class="summary-cell"><div class="v" style="color:${sess.gap_count > 0 ? 'var(--err)' : 'var(--accent)'}">${sess.gap_count}</div><div class="l">Gaps detected</div></div>
    </div>
    <div class="section-hdr">Retrieval events</div>
    ${sess.events.map((ev, i) => renderEvent(ev, i)).join('')}
  `;
}

function renderEvent(ev, i) {
  const chunks = ev.chunks || [];
  const topScore = chunks.length ? Math.max(...chunks.map(c => c.score || 0)) : 0;
  const hasGapMeta = ev.metadata?.has_gap;
  const isGap = hasGapMeta !== undefined ? hasGapMeta : (topScore > 0 && topScore < 0.65);
  const label = ev.label || 'retrieval';
  const ms = ev.metadata?.elapsed_ms;

  const chunkBars = chunks.slice(0,8).map((c, ci) => {
    const score = c.score || 0;
    const pct = Math.round(score * 100);
    const color = score >= 0.72 ? '#1D9E75' : score >= 0.55 ? '#BA7517' : '#A32D2D';
    const content = (c.content || '').slice(0, 120);
    return `
      <div class="chunk-row">
        <div class="chunk-label">#${ci+1}</div>
        <div style="flex:1">
          <div class="chunk-bar-wrap">
            <div class="chunk-bar" style="width:${pct}%;background:${color}"></div>
            <span class="chunk-score">${score.toFixed(3)}</span>
          </div>
          <div class="chunk-content" title="${content}">${content}</div>
        </div>
      </div>`;
  }).join('');

  return `
    <div class="event-card" id="ev-${i}">
      <div class="event-hdr" onclick="toggleEvent('ev-${i}')">
        <span class="toggle-icon">▼</span>
        <span class="event-query" title="${ev.query}">${ev.query}</span>
        <div class="event-meta">
          <span class="badge ${isGap ? 'badge-gap' : 'badge-ok'}">${isGap ? '✗ gap' : '✓ ok'}</span>
          <span>${label}</span>
          ${ms !== undefined ? '<span>' + ms + 'ms</span>' : ''}
          <span style="color:${scoreColor(topScore)}">${topScore.toFixed(3)}</span>
        </div>
      </div>
      <div class="event-body">
        ${chunkBars || '<div style="color:var(--muted);font-size:12px">No chunks logged</div>'}
        ${isGap ? `<div class="gap-flag">
  <strong>⚠ Gap detected</strong>
  ${ev.metadata?.missing_topics?.length 
    ? 'Missing: ' + ev.metadata.missing_topics.join(', ') 
    : 'Best chunk score ' + topScore.toFixed(3) + ' is below threshold 0.65.'}
  ${ev.metadata?.suggestion ? '<br>' + ev.metadata.suggestion : ''}
        </div>` : ''}
      </div>
    </div>`;
}

function toggleEvent(id) {
  document.getElementById(id).classList.toggle('collapsed');
}

function scoreColor(s) {
  return s >= 0.72 ? 'var(--accent)' : s >= 0.55 ? 'var(--warn)' : 'var(--err)';
}

load();
setInterval(load, 10000);
</script>
</body>
</html>'''


def _build_app(store: SQLiteStore, project: str):
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError:
        raise ImportError("Run: pip install fastapi uvicorn")

    app = FastAPI(title="rag-debugger")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML

    @app.get("/api/data")
    async def data():
        events = store.recent(limit=500)

        # group by session
        sessions: dict[str, dict] = {}
        ungrouped = []
        all_top_scores = []
        gap_events = 0

        for ev in reversed(events):
            sid = ev.session_id or "__ungrouped__"
            if sid not in sessions:
                sessions[sid] = {
                    "id": sid,
                    "user": ev.user,
                    "event_count": 0,
                    "gap_count": 0,
                    "top_scores": [],
                    "events": [],
                }
            s = sessions[sid]
            s["event_count"] += 1

            chunks = ev.chunks or []
            scores = [c.get("score") or 0 for c in chunks if c.get("score") is not None]
            top = max(scores) if scores else 0.0
            s["top_scores"].append(top)
            all_top_scores.append(top)

            if top > 0 and top < 0.65:
                s["gap_count"] += 1
                gap_events += 1

            s["events"].append({
                "query": ev.query,
                "label": ev.label,
                "chunks": chunks,
                "metadata": ev.metadata,
                "timestamp": ev.timestamp,
            })

        session_list = []
        for s in sessions.values():
            ts = s["top_scores"]
            s["avg_top_score"] = round(sum(ts) / len(ts), 4) if ts else 0.0
            s["lowest_top_score"] = round(min(ts), 4) if ts else 0.0
            del s["top_scores"]
            session_list.append(s)

        avg = round(sum(all_top_scores) / len(all_top_scores), 4) if all_top_scores else 0.0

        return JSONResponse({
            "project": project,
            "stats": {
                "total_sessions": len([s for s in session_list if s["id"] != "__ungrouped__"]),
                "total_events": len(events),
                "avg_top_score": avg,
                "gap_events": gap_events,
            },
            "sessions": session_list,
        })

    return app


def launch(
    store: SQLiteStore,
    project: str,
    host: str = "127.0.0.1",
    port: int = 7842,
    open_browser: bool = True,
) -> None:
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Run: pip install fastapi uvicorn")

    app = _build_app(store, project)
    url = f"http://{host}:{port}"

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"\n  rag-debugger dashboard → {url}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
