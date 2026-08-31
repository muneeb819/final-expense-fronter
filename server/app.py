"""FastAPI server: health, simple dashboard, call-log API, and streaming mode toggle."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .config import Config
from .service import build_stasis_handler

app = FastAPI(title="Final Expense Voice Fronter")
CALL_LOG: list = []
CONFIG = Config.load()
MODE = "turn-based"


@app.on_event("startup")
def _startup():
    global MODE
    MODE = "streaming" if CONFIG.tts_provider == "elevenlabs" else "turn-based"


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE, "tts": CONFIG.tts_provider, "time": datetime.now(timezone.utc).isoformat()}


@app.post("/api/calls")
async def add_call(request: Request):
    data = await request.json()
    data["id"] = data.get("id") or os.urandom(8).hex()
    data["ts"] = data.get("ts") or datetime.now(timezone.utc).isoformat()
    CALL_LOG.append(data)
    return {"ok": True, "id": data["id"]}


@app.get("/api/calls")
def list_calls(limit: int = 50):
    return {"calls": CALL_LOG[-limit:]}


@app.get("/api/config")
def config_view():
    c = CONFIG
    return {
        "mode": MODE,
        "llm_provider": c.llm_provider,
        "stt_provider": c.stt_provider,
        "tts_provider": c.tts_provider,
        "elevenlabs_voice": c.elevenlabs_voice if c.tts_provider == "elevenlabs" else None,
        "elevenlabs_model": c.elevenlabs_model if c.tts_provider == "elevenlabs" else None,
        "agent_name": c.agent_name,
        "min_age": c.min_age,
        "max_age": c.max_age,
        "licensed_states": c.licensed_states,
        "closer_queue": c.closer_queue,
        "vicidial_enabled": bool(c.vicidial_url and c.vicidial_user),
    }


DASHBOARD = """
<!doctype html><html><head><meta charset="utf-8"><title>Final Expense Fronter — Lina</title>
<style>body{font-family:system-ui;margin:2rem;background:#0f172a;color:#e2e8f0}
h1{color:#38bdf8}h2{color:#94a3b8;font-size:.9rem;font-weight:normal;margin-top:.2rem}
table{border-collapse:collapse;width:100%;margin-top:1rem}
th,td{border:1px solid #334155;padding:.5rem;text-align:left;font-size:.85rem}
th{background:#1e293b}.tag{padding:.1rem .4rem;border-radius:.3rem;background:#334155}
.tag.ok{background:#166534}.tag.hangup{background:#991b1b}
.badge{display:inline-block;padding:.2rem .6rem;border-radius:1rem;font-size:.75rem;font-weight:600}
.badge.live{background:#166534;color:#bbf7d0}.badge.off{background:#991b1b;color:#fecaca}
a{color:#38bdf8}.meta{display:flex;gap:2rem;margin:1rem 0;flex-wrap:wrap}
.meta div{background:#1e293b;padding:.8rem 1.2rem;border-radius:.5rem}
.meta .label{font-size:.7rem;text-transform:uppercase;color:#64748b;letter-spacing:.05em}
.meta .val{font-size:1.1rem;font-weight:600;color:#e2e8f0}</style></head><body>
<h1>Final Expense Voice Fronter</h1>
<h2>Powered by ElevenLabs v3 Expressive Voice &bull; Lina</h2>
<div class="meta">
<div><div class="label">Pipeline</div><div class="val" id="mode">...</div></div>
<div><div class="label">TTS</div><div class="val" id="tts">...</div></div>
<div><div class="label">Voice</div><div class="val" id="voice">...</div></div>
<div><div class="label">STT</div><div class="val" id="stt">...</div></div>
<div><div class="label">Status</div><div class="val" id="status">...</div></div>
</div>
<p>Live call qualification log (fronter &rarr; closer handoff).</p>
<table id="t"><thead><tr><th>Time</th><th>Lead</th><th>Action</th><th>Disposition</th><th>State</th><th>Age</th></tr></thead>
<tbody></tbody></table>
<script>async function load(){
const [cr,cl]=await Promise.all([fetch('/api/config'),fetch('/api/calls')]);
const c=await cr.json();const d=await cl.json();
document.getElementById('mode').textContent=c.mode||'turn-based';
document.getElementById('tts').textContent=c.tts_provider||'console';
document.getElementById('voice').textContent=c.elevenlabs_voice||c.tts_provider||'-';
document.getElementById('stt').textContent=c.stt_provider||'console';
document.getElementById('status').innerHTML='<span class="badge live">LIVE</span>';
const tb=document.querySelector('#t tbody');tb.innerHTML='';for(const x of d.calls){const f=x.facts||{};
const tag=x.action==='transfer'?'tag ok':'tag hangup';
tb.insertAdjacentHTML('beforeend',`<tr><td>${x.ts}</td><td>${x.lead_id||'-'}</td>
<td><span class='${tag}'>${x.action}</span></td><td>${x.disposition}</td>
<td>${f.state||'-'}</td><td>${f.age||'-'}</td></tr>`)}}
load();setInterval(load,4000);</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD)


def run_ari():
    from .ari import ARI

    handler = build_stasis_handler(CONFIG, CALL_LOG)
    ARI(CONFIG).run_forever(CONFIG.ari_app, handler)
