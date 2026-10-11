"""HyperQwen-Mem server: MCP tool endpoint + human review page.

One process, stdlib only:
  POST /mcp   JSON-RPC 2.0 (MCP-style): tools/call -> memory_search,
              memory_remember, memory_list
  GET  /      review page: list/search memories, see superseded history
  GET  /stats service stats (memory counts, db size, last search latency)

Run:  python server.py   (reads .env or env vars)
Env:  MEM_DB=/path/to/memory.db   MEM_PORT=8321
      HYPERQWEN_URL=http://127.0.0.1:18020   HYPERQWEN_API_KEY=...
      CONSOLIDATE=1|0     (default 1: background worker on)
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from memcore import Embedder, MemoryStore
from consolidator import Consolidator

# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(BASE, ".env")) if os.path.exists(
        os.path.join(BASE, ".env")) else []:
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

DB_PATH = os.path.abspath(os.environ.get("MEM_DB", os.path.join(BASE, "data", "memory.db")))
PORT = int(os.environ.get("MEM_PORT", "8321"))
HYPERQWEN_URL = os.environ.get("HYPERQWEN_URL", "http://127.0.0.1:18020")

embedder = Embedder(os.environ.get("MEM_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
store = MemoryStore(DB_PATH, embedder)
consolidator = Consolidator(
    store,
    llm_url=HYPERQWEN_URL,
    llm_api_key=os.environ.get("HYPERQWEN_API_KEY", ""),
    batch_size=int(os.environ.get("MEM_BATCH", "4")),
    batch_interval=float(os.environ.get("MEM_BATCH_INTERVAL", "30")),
    enabled=os.environ.get("CONSOLIDATE", "1") == "1",
)
if consolidator.enabled:
    consolidator.start()
    print(f"[server] consolidator on (batch {consolidator.batch_size} "
          f"every {consolidator.batch_interval}s, llm={HYPERQWEN_URL})", flush=True)
else:
    print("[server] consolidator off (naive mode only)", flush=True)


# ---------------------------------------------------------------------------
# MCP-ish JSON-RPC
# ---------------------------------------------------------------------------

def _tool_search(args: dict):
    k = int(args.get("k", 5))
    res = store.search(args.get("query", ""), k=k,
                       session=args.get("session"),
                       include_superseded=args.get("include_superseded", False))
    return [{
        "id": m.id, "text": m.text, "session": m.session, "status": m.status,
        "created_at": m.created_at, "score": round(s, 4),
        "superseded_by": m.superseded_by,
    } for m, s in res]


def _tool_remember(args: dict):
    mid = store.add_memory(args.get("text", ""),
                           session=args.get("session", "direct"))
    if mid is None:
        return {"ok": True, "duplicate": True,
                "note": "an equivalent memory already exists; it was refreshed"}
    return {"ok": True, "id": mid}


def _tool_list(args: dict):
    ms = store.list_memories(status=args.get("status", "remembered"),
                             limit=int(args.get("limit", 50)),
                             offset=int(args.get("offset", 0)))
    return [{
        "id": m.id, "text": m.text, "session": m.session, "status": m.status,
        "created_at": m.created_at, "accesses": m.accesses,
        "superseded_by": m.superseded_by,
    } for m in ms]


TOOLS = [
    {"name": "memory_search", "fn": _tool_search,
     "description": "Semantic + keyword search over long-term memories. "
                    "Returns the most relevant durable facts (recency-decayed).",
     "schema": {"query": "str", "k": "int=5", "session": "str?",
                "include_superseded": "bool=false"}},
    {"name": "memory_remember", "fn": _tool_remember,
     "description": "Store an atomic durable fact directly (deduplicated by "
                    "embedding). Use for things the agent learned and must "
                    "not lose.",
     "schema": {"text": "str", "session": "str=direct"}},
    {"name": "memory_list", "fn": _tool_list,
     "description": "List stored memories (default: active ones, newest "
                    "first). status: remembered | superseded | forgotten.",
     "schema": {"status": "str=remembered", "limit": "int=50",
                "offset": "int=0"}},
]

TOOL_BY_NAME = {t["name"]: t for t in TOOLS}

REVIEW_PAGE = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>HyperQwen-Mem — recuerdos</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#0e1116;color:#e6e6e6}
 .wrap{max-width:860px;margin:0 auto;padding:24px}
 h1{font-size:20px;font-weight:600}
 .sub{color:#8a93a3;margin-bottom:18px;font-size:12px}
 input[type=text]{width:100%;box-sizing:border-box;padding:10px;border-radius:8px;
   border:1px solid #2a3342;background:#151a22;color:#e6e6e6;font-size:14px}
 .f{background:#151a22;border:1px solid #232c3a;border-radius:10px;
   padding:10px 14px;margin:8px 0}
 .f .m{color:#e6e6e6}
 .f .meta{color:#8a93a3;font-size:12px;margin-top:4px}
 .f.sup{opacity:.55}
 .f.sup .m text-decoration: line-through;
 .f .sup-by{color:#5f8fd4}
 .pill{display:inline-block;font-size:11px;padding:1px 8px;border-radius:99px;
   background:#232c3a;margin-right:6px}
 .pill.s{background:#3a2d14;color:#e0b45f}
 .pill.f{background:#3a1a1a;color:#e07f7f}
 table{font-size:12px;color:#8a93a3;margin-top:24px;border-collapse:collapse}
 td{padding:2px 14px 2px 0}
 .err{color:#e07f7f}
</style></head><body><div class="wrap">
<h1>Recuerdos</h1>
<div class="sub">HyperQwen-Mem — memoria de largo plazo del agente.
Solo se muestran los activos; los obsoletos se marcan.</div>
<input type="text" id="q" placeholder="Buscar recuerdos... (vacío = todos)"
  oninput="go()">
<div id="out"></div>
<table id="st"></table>
<script>
async function go(){
  const q = document.getElementById('q').value.trim();
  let url;
  if(q) url = '/api/search?q=' + encodeURIComponent(q) + '&k=25';
  else url = '/api/list?limit=250';
  const r = await fetch(url); const d = await r.json();
  const out = document.getElementById('out');
  out.innerHTML = d.map(m => `<div class="f ${m.status}">
    <div class="m">${esc(m.text)}</div>
    <div class="meta">
      <span class="pill ${m.status==='remembered'?'s':m.status}">${m.status}</span>
      ${fmt(m.created_at)} · ${m.session} · ${m.accesses||0} usos
      ${m.score?` · score ${m.score}`:''}
      ${m.superseded_by?`<span class="sup-by">→ reemplazado por #${m.superseded_by}</span>`:''}
    </div></div>`).join('') || '<div class="f"><div class="m">Sin recuerdos aún.</div></div>';
  st();
}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;')}
function fmt(t){return new Date(t*1000).toLocaleString('es-CL')}
async function st(){
  const r = await fetch('/stats'); const d = await r.json();
  document.getElementById('st').innerHTML =
   `<tr><td>BD</td><td>${(d.db_bytes/1024/1024).toFixed(1)} MB</td></tr>`+
   `<tr><td>recuerdos</td><td>activos ${d.memories.remembered||0} · `+
   `obsoletos ${d.memories.superseded||0} · olvidados ${d.memories.forgotten||0}</td></tr>`+
   `<tr><td>episodios</td><td>pendientes ${d.episodes.pending||0} · `+
   `consolidados ${d.episodes.consolidated||0}</td></tr>`+
   `<tr><td>embeddings</td><td>${d.embedding}</td></tr>`+
   `<tr><td>última búsqueda</td><td>${d.last_search||'-'}</td></tr>`+
   `<tr><td>consolidador</td><td>${d.consolidator||'off'}</td></tr>`;
}
go();
</script></div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = REVIEW_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/stats":
            d = store.stats()
            c = consolidator
            d["consolidator"] = ("on" if c.enabled else "off") + (
                f" | llm={'yes' if c.stats['llm'] else 'no'} | "
                f"batches={c.stats['batches']} facts={c.stats['facts']} "
                f"superseded={c.stats['superseded']} err={c.stats['errors']}")
            self._json(d)
        elif u.path == "/api/search":
            q = parse_qs(u.query)
            res = store.search(q.get("q", [""])[0], k=int(q.get("k", ["5"])[0]),
                               include_superseded=True)
            self._json([{"id": m.id, "text": m.text, "session": m.session,
                         "status": m.status, "created_at": m.created_at,
                         "accesses": m.accesses, "score": round(s, 4),
                         "superseded_by": m.superseded_by}
                        for m, s in res])
        elif u.path == "/api/list":
            q = parse_qs(u.query)
            ms = store.list_memories(limit=int(q.get("limit", ["100"])[0]))
            self._json([{"id": m.id, "text": m.text, "session": m.session,
                         "status": m.status, "created_at": m.created_at,
                         "accesses": m.accesses,
                         "superseded_by": m.superseded_by} for m in ms])
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path in ("/api/episode", "/api/supersede", "/api/consolidate"):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if path == "/api/episode":
                    eid = store.add_episode(body.get("text", ""),
                                            session=body.get("session", "default"))
                    self._json({"id": eid})
                elif path == "/api/supersede":
                    old, new = int(body["old"]), int(body["new"])
                    # similarity guard: only supersede when the two memories
                    # are close enough (mirrors the consolidator's guard)
                    all_m = store.list_memories(status="any", limit=10**6)
                    old_m = next((m for m in all_m if m.id == old), None)
                    new_m = next((m for m in all_m if m.id == new), None)
                    if not old_m or not new_m:
                        self._json({"error": "memory not found"}, 404)
                        return
                    v = store.embedder.encode([old_m.text, new_m.text])
                    sim = max(0.0, min(1.0, sum(a*b for a, b in zip(v[0], v[1]))))
                    if sim < 0.70:
                        self._json({"error": f"refused: similarity {sim:.2f} < 0.70",
                                    "similarity": round(sim, 3)}, 409)
                        return
                    store.supersede(old, new)
                    self._json({"ok": True, "similarity": round(sim, 3)})
                elif path == "/api/consolidate":
                    eps = store.pending_episodes(
                        int(os.environ.get("MEM_BATCH", "4")) * 2)
                    n_facts = consolidator.consolidate(eps) if eps else 0
                    self._json({"consolidated_episodes": len(eps),
                                "new_facts": n_facts,
                                "stats": consolidator.stats})
            except KeyError as e:
                self._json({"error": f"missing field {e}"}, 400)
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return
        if path != "/mcp":
            self._json({"error": "not found"}, 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json({"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": "parse error"}}, 400)
            return
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params", {})
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": params.get("protocolVersion", "2025-03-26"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "hyperqwen-mem", "version": "0.1.0"},
                }
            elif method == "tools/list":
                result = {"tools": [{
                    "name": t["name"], "description": t["description"],
                    "inputSchema": {"type": "object",
                                    "properties": {
                                        k: {"type": v.split("=")[0]}
                                        for k, v in t["schema"].items()}},
                } for t in TOOLS]}
            elif method == "tools/call":
                t = TOOL_BY_NAME.get(params.get("name", ""))
                if not t:
                    raise ValueError(f"unknown tool {params.get('name')}")
                args = params.get("arguments", {})
                result = {"content": [
                    {"type": "text",
                     "text": json.dumps(t["fn"](args), ensure_ascii=False)}]}
            elif method in ("notifications/initialized",):
                self._json({})
                return
            else:
                raise ValueError(f"unsupported method {method}")
            self._json({"jsonrpc": "2.0", "id": rid, "result": result})
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)
            self._json({"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32000, "message": traceback.format_exc(limit=8)}})


if __name__ == "__main__":
    print(f"[server] HyperQwen-Mem on http://127.0.0.1:{PORT} "
          f"(mcp=/mcp review=/ stats=/stats) db={DB_PATH}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
