"""FastAPI server: health, simple dashboard, call-log API, streaming mode toggle,
and an interactive DEBUG console for testing the bot without a PBX/keys.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, Response, JSONResponse

from agent.config import Config
from agent.service import build_stasis_handler

app = FastAPI(title="Final Expense Voice Fronter")
CALL_LOG: list = []
CONFIG = Config.load()
MODE = "turn-based"
DEBUG = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")

# ---- Debug sessions: session_id -> Conversation ----
_DEBUG_SESSIONS: dict = {}


def _debug_llm(config: Config):
    """LLM with MockLLM fallback so debug mode works with zero keys."""
    from agent.llm import LLM

    llm = LLM(config)
    try:
        if llm.models_available():
            return llm, config.llm_provider
    except Exception:
        pass
    from agent.mock_llm import MockLLM

    return MockLLM(config), f"{config.llm_provider} (mock fallback)"


@app.on_event("startup")
def _startup():
    global MODE
    MODE = "streaming" if CONFIG.tts_provider == "elevenlabs" else "turn-based"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": MODE,
        "debug": DEBUG,
        "tts": CONFIG.tts_provider,
        "time": datetime.now(timezone.utc).isoformat(),
    }


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
        "debug": DEBUG,
        "llm_provider": c.llm_provider,
        "stt_provider": c.stt_provider,
        "tts_provider": c.tts_provider,
        "elevenlabs_voice": c.elevenlabs_voice if c.tts_provider == "elevenlabs" else None,
        "elevenlabs_model": c.elevenlabs_model if c.tts_provider == "elevenlabs" else None,
        "agent_name": c.agent_name,
        "business_name": c.business_name,
        "min_age": c.min_age,
        "max_age": c.max_age,
        "licensed_states": c.licensed_states,
        "closer_queue": c.closer_queue,
        "vicidial_enabled": bool(c.vicidial_url and c.vicidial_user),
    }


# ---------------- DEBUG API (bot testing without PBX) ----------------

@app.post("/api/debug/start")
def debug_start():
    """Start a new debug conversation. Returns greeting + session_id."""
    from agent.conversation import Conversation

    llm, llm_used = _debug_llm(CONFIG)
    convo = Conversation(CONFIG, llm)
    greeting = convo.start()
    sid = uuid.uuid4().hex[:12]
    _DEBUG_SESSIONS[sid] = convo
    print(f"[DEBUG] session {sid} started (llm={llm_used}) state={convo.ctx.state}")
    return {
        "session_id": sid,
        "agent": greeting,
        "state": convo.ctx.state,
        "llm": llm_used,
        "ended": False,
    }


@app.post("/api/debug/chat")
async def debug_chat(request: Request):
    """Send a caller message, get the bot's reply + beat/state debug info."""
    from agent.qualifier import extract_age

    data = await request.json()
    sid = data.get("session_id", "")
    text = (data.get("text") or "").strip()
    convo = _DEBUG_SESSIONS.get(sid)
    if convo is None:
        return {"error": "unknown session_id — POST /api/debug/start first"}
    if not text:
        return {"error": "empty text"}

    state_before = convo.ctx.state
    age_dbg = extract_age(text)
    result = convo.hear(text)
    print(
        f"[DEBUG] session {sid} | {state_before} -> {convo.ctx.state} | "
        f"caller={text!r} | agent={result.ai_speech!r} | "
        f"end={result.end_call} disp={result.disposition}"
    )

    payload = {
        "session_id": sid,
        "agent": result.ai_speech,
        "state_before": state_before,
        "state": convo.ctx.state,
        "ended": result.end_call,
        "disposition": result.disposition,
        "classification": result.classification,
        "debug": {
            "age_detected": age_dbg.detected,
            "age": age_dbg.age,
            "age_eligible": age_dbg.eligible,
            "age_confirmed": age_dbg.confirmed,
            "ctx_age": convo.ctx.age,
            "transcript_len": len(convo.ctx.transcript),
        },
    }
    if result.end_call:
        action = "transfer" if result.disposition == "AI-TRANSFER" else "hangup"
        payload["action"] = action
        CALL_LOG.append({
            "id": uuid.uuid4().hex,
            "ts": datetime.now(timezone.utc).isoformat(),
            "lead_id": f"debug-{sid}",
            "action": action,
            "disposition": result.disposition,
            "facts": {**(result.classification or {}), "age": convo.ctx.age},
        })
    return payload


@app.get("/api/debug/transcript")
def debug_transcript(session_id: str = ""):
    convo = _DEBUG_SESSIONS.get(session_id)
    if convo is None:
        return {"error": "unknown session_id"}
    return {
        "session_id": session_id,
        "state": convo.ctx.state,
        "ended": convo.ended,
        "disposition": convo.ctx.disposition,
        "age": convo.ctx.age,
        "transcript": convo.ctx.transcript,
    }


@app.post("/api/debug/speak")
async def debug_speak(request: Request):
    """Synthesize bot speech (Lina's voice) — needs TTS_PROVIDER=elevenlabs + key.

    Returns audio/mpeg bytes, or JSON {error} when voice is not configured.
    """
    from agent.tts import TTS

    data = await request.json()
    text = (data.get("text") or "").strip()[:800]
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    if CONFIG.tts_provider != "elevenlabs" or not CONFIG.elevenlabs_api_key:
        return JSONResponse(
            {"error": "voice not configured — set TTS_PROVIDER=elevenlabs and ELEVENLABS_API_KEY in .env, then restart"},
            status_code=400,
        )
    try:
        audio = TTS(CONFIG).synthesize(text)
    except Exception as e:
        return JSONResponse({"error": f"TTS failed: {e}"}, status_code=502)
    if not audio:
        return JSONResponse({"error": "TTS returned no audio"}, status_code=502)
    print(f"[DEBUG] speak: {len(text)} chars -> {len(audio)} bytes")
    return Response(content=audio, media_type="audio/mpeg")


@app.post("/api/debug/transcribe")
async def debug_transcribe(file: UploadFile = File(...)):
    """Transcribe caller audio (mic upload) — needs STT_PROVIDER=deepgram + key.

    Accepts webm/wav/mp3 up to ~6 MB. Returns {text}.
    """
    import tempfile

    from agent.stt import STT

    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    if len(raw) > 6 * 1024 * 1024:
        return JSONResponse({"error": "audio too large (max ~6 MB, a few seconds is enough)"}, status_code=400)
    if CONFIG.stt_provider != "deepgram" or not CONFIG.deepgram_api_key:
        return JSONResponse(
            {"error": "mic STT not configured — set STT_PROVIDER=deepgram and DEEPGRAM_API_KEY in .env, then restart (or just type instead)"},
            status_code=400,
        )
    suffix = ".webm"
    ct = (file.content_type or "").lower()
    if "wav" in ct:
        suffix = ".wav"
    elif "mpeg" in ct or "mp3" in ct:
        suffix = ".mp3"
    elif "ogg" in ct:
        suffix = ".ogg"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            path = tmp.name
        text = STT(CONFIG).transcribe_file(path)
    except Exception as e:
        return JSONResponse({"error": f"STT failed: {e}"}, status_code=502)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    print(f"[DEBUG] transcribe: {len(raw)} bytes -> {text!r}")
    return {"text": text or ""}


@app.get("/api/debug/voice")
def debug_voice_status():
    """Report which voice features are available (for the debug console UI)."""
    return {
        "speak": bool(CONFIG.tts_provider == "elevenlabs" and CONFIG.elevenlabs_api_key),
        "mic": bool(CONFIG.stt_provider == "deepgram" and CONFIG.deepgram_api_key),
        "tts_provider": CONFIG.tts_provider,
        "stt_provider": CONFIG.stt_provider,
        "voice": CONFIG.elevenlabs_voice,
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
.badge.debug{background:#7c3aed;color:#ede9fe}
a{color:#38bdf8}.meta{display:flex;gap:2rem;margin:1rem 0;flex-wrap:wrap}
.meta div{background:#1e293b;padding:.8rem 1.2rem;border-radius:.5rem}
.meta .label{font-size:.7rem;text-transform:uppercase;color:#64748b;letter-spacing:.05em}
.meta .val{font-size:1.1rem;font-weight:600;color:#e2e8f0}
.nav{margin:1rem 0;display:flex;gap:1rem}
.nav a{background:#1e293b;padding:.6rem 1.2rem;border-radius:.5rem;text-decoration:none;font-weight:600}
.nav a:hover{background:#334155}</style></head><body>
<h1>Final Expense Voice Fronter</h1>
<h2>Powered by ElevenLabs v3 Expressive Voice &bull; Lina</h2>
<div class="nav"><a href="/">&#128202; Dashboard</a><a href="/debug">&#129516; Debug Console (test the bot)</a></div>
<div class="meta">
<div><div class="label">Pipeline</div><div class="val" id="mode">...</div></div>
<div><div class="label">TTS</div><div class="val" id="tts">...</div></div>
<div><div class="label">Voice</div><div class="val" id="voice">...</div></div>
<div><div class="label">STT</div><div class="val" id="stt">...</div></div>
<div><div class="label">Debug</div><div class="val" id="debug">...</div></div>
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
document.getElementById('debug').innerHTML=c.debug?'<span class="badge debug">DEBUG ON</span>':'off';
document.getElementById('status').innerHTML='<span class="badge live">LIVE</span>';
const tb=document.querySelector('#t tbody');tb.innerHTML='';for(const x of d.calls){const f=x.facts||{};
const tag=x.action==='transfer'?'tag ok':'tag hangup';
tb.insertAdjacentHTML('beforeend',`<tr><td>${x.ts}</td><td>${x.lead_id||'-'}</td>
<td><span class='${tag}'>${x.action}</span></td><td>${x.disposition}</td>
<td>${f.state||'-'}</td><td>${f.age??'-'}</td></tr>`)}}
load();setInterval(load,4000);</script></body></html>
"""


DEBUG_CONSOLE = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Debug Console — Test the Bot</title>
<style>
body{font-family:system-ui;margin:0;background:#0f172a;color:#e2e8f0}
header{background:#1e293b;padding:1rem 2rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem}
header h1{margin:0;font-size:1.2rem;color:#38bdf8}
header a{color:#38bdf8;text-decoration:none;font-weight:600}
main{max-width:900px;margin:0 auto;padding:1.5rem}
#chat{background:#1e293b;border-radius:.75rem;padding:1rem;min-height:300px;max-height:50vh;overflow-y:auto;margin-bottom:1rem}
.msg{margin:.6rem 0;padding:.6rem .9rem;border-radius:.6rem;max-width:80%;line-height:1.4}
.msg.agent{background:#0e7490;align-self:flex-start}
.msg.caller{background:#334155;margin-left:auto;text-align:right}
.msg.sys{background:transparent;border:1px dashed #475569;color:#94a3b8;font-size:.8rem;max-width:100%}
#chat{display:flex;flex-direction:column}
.row{display:flex;gap:.5rem}
#text{flex:1;padding:.7rem;border-radius:.5rem;border:1px solid #334155;background:#0f172a;color:#e2e8f0;font-size:1rem}
button{padding:.7rem 1.2rem;border-radius:.5rem;border:0;background:#0284c7;color:#fff;font-weight:700;cursor:pointer}
button:hover{background:#0369a1}button:disabled{background:#475569;cursor:default}
button.ghost{background:#334155}
#panel{background:#1e293b;border-radius:.75rem;padding:1rem;margin-top:1rem;font-size:.85rem}
#panel h3{margin:.2rem 0 .6rem;color:#94a3b8;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
#panel pre{background:#0f172a;padding:.7rem;border-radius:.5rem;overflow-x:auto;color:#a5f3fc;white-space:pre-wrap}
.pills{display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0}
.pill{background:#334155;padding:.3rem .7rem;border-radius:1rem;font-size:.75rem;cursor:pointer}
.pill:hover{background:#475569}
.badge{display:inline-block;padding:.2rem .6rem;border-radius:1rem;font-size:.75rem;font-weight:600}
.badge.live{background:#166534;color:#bbf7d0}
</style></head><body>
<header><h1>&#129516; Debug Console — talk to Lina</h1><div><a href="/">&#8592; Dashboard</a> &nbsp; <span class="badge live">DEBUG MODE</span> &nbsp; <span id="voiceStatus" class="badge" style="background:#334155">voice: ...</span></div></header>
<main>
<div id="chat"></div>
<div class="pills">
<span class="pill" onclick="quick('Yeah, pretty good, thanks')">"Yeah, pretty good"</span>
<span class="pill" onclick="quick(&quot;I'm 67 years old&quot;)">"I'm 67"</span>
<span class="pill" onclick="quick('I am sixty seven')">"sixty seven" (words)</span>
<span class="pill" onclick="quick('yes')">"yes"</span>
<span class="pill" onclick="quick('I am 35')">"I am 35" (ineligible)</span>
<span class="pill" onclick="quick('Who is this?')">"Who is this?"</span>
<span class="pill" onclick="quick('I am busy right now')">"I'm busy"</span>
<span class="pill" onclick="quick('Take me off your list')">DNC test</span>
</div>
<div class="row">
<input id="text" placeholder="Type as the caller, e.g. I'm 67 years old — or use the mic" autocomplete="off"/>
<button id="send" onclick="send()">Send</button>
<button id="micBtn" class="ghost" onclick="toggleMic()" title="Record with microphone">&#127908; Mic</button>
<button id="voiceBtn" class="ghost" onclick="toggleVoice()" title="Toggle spoken replies">&#128266; Voice: on</button>
<button class="ghost" onclick="start(true)">&#8635; New call</button>
</div>
<div id="panel"><h3>Beat / state inspector</h3><pre id="dbg">starting...</pre></div>
</main>
<script>
let sid=null, voiceOn=true, voiceCfg={speak:false,mic:false}, voiceHintShown=false, audioUnlocked=false;
let mediaRecorder=null, micChunks=[], recording=false;
const chat=document.getElementById('chat'), dbg=document.getElementById('dbg'), input=document.getElementById('text');
input.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
document.addEventListener('click',()=>{audioUnlocked=true;},{once:true});
function bubble(cls,html){const d=document.createElement('div');d.className='msg '+cls;d.innerHTML=html;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;}
function quick(t){input.value=t;send();}
function toggleVoice(){voiceOn=!voiceOn;document.getElementById('voiceBtn').innerHTML=voiceOn?'&#128266; Voice: on':'&#128263; Voice: off';}
async function loadVoice(){
  try{
    const r=await fetch('/api/debug/voice');voiceCfg=await r.json();
    const el=document.getElementById('voiceStatus');
    if(voiceCfg.speak&&voiceCfg.mic){el.textContent='voice: full (talk + mic)';el.style.background='#166534';}
    else if(voiceCfg.speak){el.textContent='voice: replies spoken';el.style.background='#166534';}
    else{el.textContent='voice: text only (add API keys)';el.style.background='#334155';}
  }catch(e){}
}
async function speak(text){
  if(!voiceOn||!text||!voiceCfg.speak)return;
  try{
    const r=await fetch('/api/debug/speak',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    if(!r.ok)return;
    const blob=await r.blob();
    const audio=new Audio(URL.createObjectURL(blob));
    try{await audio.play();}
    catch(e){if(!voiceHintShown){voiceHintShown=true;bubble('sys','&#128266; click anywhere once to enable sound (browser autoplay policy), then continue chatting.');}}
  }catch(e){}
}
async function toggleMic(){
  if(recording){mediaRecorder.stop();return;}
  if(!voiceCfg.mic){bubble('sys','&#127908; mic needs STT keys — set STT_PROVIDER=deepgram and DEEPGRAM_API_KEY in .env, then restart. (Typing always works.)');return;}
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    micChunks=[];
    mediaRecorder=new MediaRecorder(stream);
    mediaRecorder.ondataavailable=e=>{if(e.data.size)micChunks.push(e.data);};
    mediaRecorder.onstop=async()=>{
      recording=false;document.getElementById('micBtn').innerHTML='&#127908; Mic';
      stream.getTracks().forEach(t=>t.stop());
      const blob=new Blob(micChunks,{type:mediaRecorder.mimeType||'audio/webm'});
      bubble('sys','&#127908; transcribing...');
      const fd=new FormData();fd.append('file',blob,'mic.webm');
      try{
        const r=await fetch('/api/debug/transcribe',{method:'POST',body:fd});
        const j=await r.json();
        if(j.error){bubble('sys','&#10060; '+escapeHtml(j.error));return;}
        if(j.text){input.value=j.text;send();}
        else bubble('sys','&#10060; heard nothing — try again, a little louder.');
      }catch(e){bubble('sys','&#10060; transcription failed: '+escapeHtml(e.message));}
    };
    mediaRecorder.start();recording=true;
    document.getElementById('micBtn').innerHTML='&#128308; Stop';
    bubble('sys','&#127908; recording... click Stop when done.');
  }catch(e){bubble('sys','&#10060; mic blocked — allow microphone access in the browser. (localhost is required for mic.)');}
}
async function start(fresh){
  if(fresh)chat.innerHTML='';
  bubble('sys','&#128259; starting new debug call...');
  const r=await fetch('/api/debug/start',{method:'POST'});const j=await r.json();sid=j.session_id;
  bubble('agent','<b>Lina:</b> '+escapeHtml(j.agent||''));
  speak(j.agent);
  dbg.textContent=JSON.stringify(j,null,2);
}
async function send(){
  const t=input.value.trim();if(!t||!sid)return;input.value='';
  bubble('caller','<b>You:</b> '+escapeHtml(t));
  const r=await fetch('/api/debug/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,text:t})});
  const j=await r.json();
  if(j.error){bubble('sys','&#10060; '+escapeHtml(j.error));return;}
  if(j.agent){bubble('agent','<b>Lina:</b> '+escapeHtml(j.agent));speak(j.agent);}
  if(j.ended)bubble('sys',(j.action==='transfer'?'&#128222; TRANSFER to closer queue':'&#128381; HANGUP')+' &nbsp; disposition=<b>'+escapeHtml(j.disposition||'')+'</b>');
  dbg.textContent=JSON.stringify(j,null,2);
}
function escapeHtml(s){return (s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
loadVoice();start(false);
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD)


@app.get("/debug", response_class=HTMLResponse)
def debug_console():
    return HTMLResponse(DEBUG_CONSOLE)


def run_ari():
    from agent.ari import ARI

    handler = build_stasis_handler(CONFIG, CALL_LOG)
    ARI(CONFIG).run_forever(CONFIG.ari_app, handler)
