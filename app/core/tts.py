import io
import os
import time
import tempfile
from typing import Optional


class TextToSpeech:

    def synthesize(self, text: str, speed: float = 1.0,
                   language: str = "en", engine_override: str = None) -> dict:

        start_time = time.time()

        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text")

        if len(text) > 2000:
            text = text[:2000]

        engine = engine_override or "gtts"

        try:
            if engine == "gtts":
                audio_bytes, fmt = self._synthesize_gtts(text, language, speed)
            else:
                audio_bytes, fmt = self._synthesize_pyttsx3(text, speed)

            processing_ms = (time.time() - start_time) * 1000

            return {
                "audio_bytes": audio_bytes,
                "format": fmt,
                "processing_time_ms": round(processing_ms, 2),
                "engine_used": engine,
                "text_length": len(text),
            }

        except Exception as e:
            # Try pyttsx3 as fallback
            if engine == "gtts":
                try:
                    audio_bytes, fmt = self._synthesize_pyttsx3(text, speed)
                    processing_ms = (time.time() - start_time) * 1000
                    return {
                        "audio_bytes": audio_bytes,
                        "format": fmt,
                        "processing_time_ms": round(processing_ms, 2),
                        "engine_used": "pyttsx3_fallback",
                        "text_length": len(text),
                    }
                except Exception as e2:
                    raise RuntimeError(f"All TTS engines failed. gTTS: {e} | pyttsx3: {e2}")
            raise RuntimeError(f"TTS failed: {e}")

    def _synthesize_gtts(self, text: str, lang: str, speed: float) -> tuple:
        from gtts import gTTS
        slow = speed < 0.8
        tts = gTTS(text=text, lang=lang, slow=slow)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read(), "mp3"

    def _synthesize_pyttsx3(self, text: str, speed: float) -> tuple:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", int(175 * speed))
        engine.setProperty("volume", 1.0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes, "wav"
        finally:
            engine.stop()
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


_tts_instance = None

def get_tts() -> TextToSpeech:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TextToSpeech()
    return _tts_instance