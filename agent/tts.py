"""Text-to-speech providers with ElevenLabs v3 expressive streaming.

Console provider prints the line and returns empty bytes (no audio in test mode).
ElevenLabs provider supports both sync and streaming synthesis with audio tags.
"""
from __future__ import annotations

import json

from .config import Config


class TTS:
    def __init__(self, config: Config):
        self.config = config
        self.provider = config.tts_provider.lower()
        self._elevenlabs = None

    def synthesize(self, text: str) -> bytes:
        if self.provider == "console":
            print(f"AGENT: {text}")
            return b""
        if self.provider == "elevenlabs":
            return self._elevenlabs_sync(text)
        if self.provider == "deepgram":
            return self._deepgram(text)
        if self.provider == "openai":
            return self._openai(text)
        if self.provider == "azure":
            return self._azure(text)
        return b""

    def synthesize_stream(self, text: str):
        """Yield audio chunks for streaming playback."""
        if self.provider == "console":
            print(f"AGENT: {text}")
            return
        if self.provider == "elevenlabs":
            yield from self._elevenlabs_stream(text)
            return
        audio = self.synthesize(text)
        if audio:
            yield audio

    def _get_elevenlabs(self):
        if self._elevenlabs is None:
            from elevenlabs.client import ElevenLabs
            self._elevenlabs = ElevenLabs(api_key=self.config.elevenlabs_api_key)
        return self._elevenlabs

    def _elevenlabs_sync(self, text: str) -> bytes:
        from .voice_ids import resolve_voice_id

        client = self._get_elevenlabs()
        audio = client.text_to_speech.convert(
            voice_id=resolve_voice_id(self.config.elevenlabs_voice),
            text=text,
            model_id=self.config.elevenlabs_model,
            voice_settings={
                "stability": self.config.elevenlabs_stability,
                "similarity_boost": self.config.elevenlabs_similarity_boost,
                "style": self.config.elevenlabs_style,
                "use_speaker_boost": self.config.elevenlabs_speaker_boost,
            },
        )
        if hasattr(audio, "__iter__") and not isinstance(audio, (bytes, bytearray)):
            return b"".join(bytes(a) for a in audio)
        return audio

    def _elevenlabs_stream(self, text: str):
        from .voice_ids import resolve_voice_id

        client = self._get_elevenlabs()
        audio_stream = client.text_to_speech.convert(
            voice_id=resolve_voice_id(self.config.elevenlabs_voice),
            text=text,
            model_id=self.config.elevenlabs_model,
            voice_settings={
                "stability": self.config.elevenlabs_stability,
                "similarity_boost": self.config.elevenlabs_similarity_boost,
                "style": self.config.elevenlabs_style,
                "use_speaker_boost": self.config.elevenlabs_speaker_boost,
            },
        )
        for chunk in audio_stream:
            if chunk:
                yield chunk

    def _deepgram(self, text: str) -> bytes:
        import requests

        url = f"https://api.deepgram.com/v1/speak?model={self.config.tts_voice}&container=wav"
        r = requests.post(
            url,
            headers={"Authorization": f"Token {self.config.deepgram_api_key}"},
            json={"text": text},
            timeout=60,
        )
        r.raise_for_status()
        return r.content

    def _openai(self, text: str) -> bytes:
        from openai import OpenAI

        client = OpenAI(api_key=self.config.openai_tts_api_key)
        resp = client.audio.speech.create(
            model=self.config.openai_tts_model,
            voice=self.config.openai_tts_voice,
            response_format="wav",
            input=text,
        )
        return resp.content

    def _azure(self, text: str) -> bytes:
        import requests

        endpoint = (
            f"https://{self.config.azure_speech_region}.tts.speech.microsoft.com/"
            f"cognitiveservices/v1"
        )
        ssml = (
            "<speak version='1.0' xml:lang='en-US'>"
            f"<voice xml:lang='en-US' name='{self.config.azure_voice}'>{text}</voice>"
            "</speak>"
        )
        r = requests.post(
            endpoint,
            headers={
                "Ocp-Apim-Subscription-Key": self.config.azure_speech_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
            },
            data=ssml.encode("utf-8"),
            timeout=60,
        )
        r.raise_for_status()
        return r.content
