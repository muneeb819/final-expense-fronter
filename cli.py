"""CLI: run an interactive console session, a scripted smoke-test, or serve ARI + web."""
from __future__ import annotations

import json
import re
import sys

from agent.config import Config


def cmd_console():
    from agent.call_session import ConsoleChannel, run_session

    config = Config.load()
    channel = ConsoleChannel()
    run_session(channel, config)


def cmd_demo():
    from agent.call_session import ConsoleChannel, run_session

    config = Config.load()
    print("=== BEAT-BASED SMOKE TEST (scripted, no keys required for console) ===")
    channel = _ScriptedChannel([
        "Yeah, pretty good, thanks",
        "I'm 67 years old",
    ])
    result = run_session(channel, config, max_turns=8)
    print("\n=== RESULT ===")
    print(result)


def cmd_serve():
    import uvicorn

    from server.app import CONFIG, CALL_LOG, build_stasis_handler
    from agent.ari import ARI

    print(f"Starting web on {CONFIG.host}:{CONFIG.port} [tts={CONFIG.tts_provider}]")
    import threading

    def ari_thread():
        handler = build_stasis_handler(CONFIG, CALL_LOG)
        try:
            ARI(CONFIG).run_forever(CONFIG.ari_app, handler)
        except Exception as e:
            print(f"[ARI] not started: {e}")

    threading.Thread(target=ari_thread, daemon=True).start()
    uvicorn.run("server.app:app", host=CONFIG.host, port=CONFIG.port)


def cmd_stream():
    """Test ElevenLabs TTS streaming with Lina's voice."""
    from agent.config import Config as C
    from agent.tts import TTS

    config = C.load()
    if config.elevenlabs_api_key:
        config.tts_provider = "elevenlabs"
    print(f"=== STREAMING TTS TEST [{config.tts_provider}] voice={config.elevenlabs_voice} ===")
    tts = TTS(config)
    lines = [
        "Hello, this is Lina calling from Senior Resource Center. How are you doing today?",
        "I understand. We offer fixed-income coverage for final expenses. How young are you?",
        "That's great — you're eligible for a lower monthly premium. One moment, let me get you connected.",
    ]
    for line in lines:
        print(f"AGENT: {line}")
        audio = tts.synthesize(line)
        print(f"  [{len(audio)} bytes audio generated]")
    print("=== DONE ===")


class _ScriptedChannel:
    def __init__(self, lines):
        self.lines = list(lines)
        self.i = 0

    def say(self, text: str):
        print(f"AGENT: {text}")

    def listen(self) -> str:
        if self.i >= len(self.lines):
            return ""
        line = self.lines[self.i]
        self.i += 1
        print(f"CALLER: {line}")
        return line

    def transfer(self):
        print("[TRANSFER] -> closer queue")

    def hangup(self):
        print("[HANGUP]")


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "console"
    if cmd == "console":
        cmd_console()
    elif cmd == "demo":
        cmd_demo()
    elif cmd == "serve":
        cmd_serve()
    elif cmd == "stream":
        cmd_stream()
    else:
        print("Usage: python cli.py [console|demo|serve|stream]")
        sys.exit(1)


if __name__ == "__main__":
    main()
