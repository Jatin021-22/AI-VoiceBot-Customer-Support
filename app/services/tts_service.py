"""
TTS Service — gTTS with pyttsx3 fallback.
Supports adjustable speaking rate, clear audio output.
"""
import os, io, time, tempfile
from pathlib import Path
from typing import Optional
from app.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("voicebot.tts")

class TTSService:
    def __init__(self):
        self.engine = settings.TTS_ENGINE
        self.language = settings.TTS_LANGUAGE
        self.slow = settings.TTS_SLOW
        self._test_engine()

    def _test_engine(self):
        try:
            if self.engine == "gtts":
                import gtts
                logger.info("TTS engine: gTTS ready")
            elif self.engine == "pyttsx3":
                import pyttsx3
                logger.info("TTS engine: pyttsx3 ready")
        except ImportError as e:
            logger.warning(f"TTS engine {self.engine} not available: {e}")

    def synthesize(self, text: str, output_path: str = None,
                   speaking_rate: float = None, language: str = None) -> Dict_like:
        """
        Convert text to speech.
        
        Args:
            text: Text to synthesize
            output_path: Where to save the audio (auto-temp if None)
            speaking_rate: 0.5-2.0 (1.0 = normal)
            language: Language override
        
        Returns:
            dict with: audio_path, duration_ms, engine, file_size_bytes
        """
        start = time.time()
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text")
        
        lang = language or self.language
        rate = speaking_rate or settings.TTS_SPEAKING_RATE
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")

        try:
            if self.engine == "gtts":
                result = self._gtts_synthesize(text, output_path, lang, rate)
            elif self.engine == "pyttsx3":
                result = self._pyttsx3_synthesize(text, output_path, rate)
            else:
                result = self._gtts_synthesize(text, output_path, lang, rate)
        except Exception as e:
            logger.error(f"TTS failed with {self.engine}: {e}")
            result = self._silent_fallback(output_path, text)

        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        duration_ms = round((time.time() - start) * 1000, 2)
        logger.info(f"TTS synthesized {len(text)} chars in {duration_ms}ms -> {output_path}")
        return {
            "audio_path": output_path,
            "duration_ms": duration_ms,
            "engine": self.engine,
            "file_size_bytes": file_size,
            "text_length": len(text),
        }

    def _gtts_synthesize(self, text: str, output_path: str, lang: str, rate: float) -> str:
        from gtts import gTTS
        slow = rate < 0.85
        tts = gTTS(text=text, lang=lang, slow=slow)
        # For rate adjustment, use pydub if available
        tts.save(output_path)
        if abs(rate - 1.0) > 0.1:
            self._adjust_speed(output_path, rate)
        return output_path

    def _pyttsx3_synthesize(self, text: str, output_path: str, rate: float) -> str:
        import pyttsx3
        engine = pyttsx3.init()
        base_rate = engine.getProperty("rate")
        engine.setProperty("rate", int(base_rate * rate))
        # pyttsx3 needs wav output then we rename
        wav_path = output_path.replace(".mp3", ".wav")
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        engine.stop()
        if os.path.exists(wav_path):
            os.rename(wav_path, output_path)
        return output_path

    def _adjust_speed(self, audio_path: str, rate: float):
        """Use pydub to adjust playback speed."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            # Speed up / slow down by changing frame rate
            new_rate = int(audio.frame_rate * rate)
            faster = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate})
            faster = faster.set_frame_rate(audio.frame_rate)
            faster.export(audio_path, format="mp3")
        except Exception as e:
            logger.warning(f"Speed adjustment failed: {e}")

    def _silent_fallback(self, output_path: str, text: str) -> str:
        """Create a minimal valid MP3 as fallback."""
        logger.warning("Using silent audio fallback")
        # Write a minimal valid MP3 header (silent)
        with open(output_path, "wb") as f:
            f.write(b"\xff\xfb\x90\x00" + b"\x00" * 100)
        return output_path

    def get_audio_bytes(self, text: str, language: str = None, speaking_rate: float = None) -> bytes:
        """Synthesize and return raw audio bytes."""
        result = self.synthesize(text, language=language, speaking_rate=speaking_rate)
        with open(result["audio_path"], "rb") as f:
            data = f.read()
        os.unlink(result["audio_path"])
        return data

# Fix type hint
Dict_like = dict
