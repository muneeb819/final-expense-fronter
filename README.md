# Jessica FE Fronter — Final Expense Voice Fronter

An AI **voice calling agent** that acts as a *fronter* for final-expense (burial/whole-life) leads. It answers the call, qualifies the prospect using a beat-based dialogue, and either **transfers qualified leads to a live closer** (VICIdial queue/agent) or politely logs the disposition and hangs up.

Built in Python, integrates with **VICIdial / Asterisk** via ARI, powered by OpenAI, Anthropic, or local Ollama.

## What it does
- Greets the caller and qualifies them in a **beat-based conversation** (greeting → hook → age check → transfer/close).
- **Eligibility:** age 45–80. A bare "yes" also counts as confirmation.
- If eligible: congratulates them on a **lower monthly premium** and transfers to a licensed agent.
- If not eligible or not interested: graceful close, logs disposition.
- Live dashboard of call results at `http://localhost:8000`.

## Architecture
```
caller → Asterisk → Stasis(jessica-fe-fronter)
                      → ARI client (answer/record/play/redirect)
                      → Conversation engine (LLM + beat-based state machine)
                      → TTS (ElevenLabs v3 Jessica / free neural fallback)
                      → STT (Deepgram)
                      → eligible? redirect to closer queue : hangup + log
                      → VICIdial non-agent API (disposition)
```

Core modules:
- `agent/config.py` – env-driven config (voice settings, thresholds, compliance)
- `agent/llm.py` – OpenAI / Anthropic / Ollama abstraction
- `agent/prompt.py` – Jessica persona + beat instructions (professional, calm, sales-appropriate)
- `agent/qualifier.py` – intent classifier + age extractor (45–80 window)
- `agent/conversation.py` – beat-based state machine (greeting → hook → age → transfer)
- `agent/stt.py`, `agent/tts.py` – speech providers (ElevenLabs v3 primary, console for testing)
- `agent/ari.py` – Asterisk ARI client (answer/play/record/redirect)
- `agent/call_session.py` – channel abstraction (ConsoleChannel / AriChannel)
- `agent/vicidial.py` – dialer disposition logging
- `server/app.py` – FastAPI dashboard + call log
- `scripts/make_demo_recording.py` – generates demo call with Edge neural voices (free)

## Quick start (no PBX / no keys)
```bash
pip install -r requirements.txt
cp .env.example .env   # leave providers on "console", LLM on "ollama" or set a key
python cli.py demo     # scripted smoke test using built-in mock LLM
python cli.py console  # interactive terminal role-play
python cli.py stream   # test ElevenLabs TTS (needs ELEVENLABS_API_KEY)
```

## Live deployment (VICIdial + Asterisk)
1. Enable ARI in `ari.conf` (`enabled = yes`, create `ari` user with read/write).
2. Route your inbound final-expense DID to the Stasis app — see `configs/extensions.conf.example`.
3. Create a `closers` queue (or set `CLOSER_EXTENSION`) in VICIdial for handoff.
4. Set `.env`: `LLM_PROVIDER`, keys, `STT_PROVIDER=deepgram`, `TTS_PROVIDER=elevenlabs`,
   `ARI_*` and `VICIDIAL_*`.
5. Run: `python cli.py serve` (starts web dashboard + ARI listener).

## Voice configuration (ElevenLabs v3)
| Setting | Value | Purpose |
|---------|-------|---------|
| `ELEVENLABS_VOICE` | `Jessica` | warm, young female voice (maps to ID `cgSgspJ2msm6clMCkdW9`) |
| `ELEVENLABS_MODEL` | `eleven_v3` | expressive v3 model |
| `ELEVENLABS_STABILITY` | `0.5` | balanced stability |
| `ELEVENLABS_SIMILARITY_BOOST` | `0.75` | consistent speaker |
| `ELEVENLABS_STYLE` | `0.3` | natural expressiveness |
| `ELEVENLABS_SPEAKER_BOOST` | `true` | enhanced clarity |

Free fallback: `python cli.py stream` uses Edge neural voices (`AriaNeural`) if no ElevenLabs key.

## Configuration highlights (env)
| Key | Purpose |
|-----|---------|
| `MIN_ELIGIBLE_AGE` / `MAX_ELIGIBLE_AGE` | eligibility window (default 45–80) |
| `POSITIVE_CONFIDENCE_THRESHOLD` | transfer threshold (default 0.85) |
| `CLOSER_QUEUE` / `CLOSER_EXTENSION` | where qualified leads go |
| `VICIDIAL_URL` / `VICIDIAL_USER` / `VICIDIAL_PASS` | disposition logging |
| `CALLING_HOUR_START` / `CALLING_HOUR_END` | compliance window (default 8–21) |

## Notes
- The live media path is **turn-based** (bot speaks, records caller until DTMF/timeout, then transcribes). For full-duplex low-latency audio, replace `agent/ari.py` with an ARI `externalMedia`/RTP bridge (see `agent/ari_streaming.py`).
- Map VICIdial status codes (`Q`, `NI`, `INC`, `QC`) to your status categories in Admin → Status Categories.