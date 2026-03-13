"""Audio validation and conversion utilities."""
import os, tempfile, wave
from pathlib import Path
from typing import Tuple
from app.utils.logger import setup_logger

logger = setup_logger("voicebot.audio_utils")

SUPPORTED_FORMATS = {"wav", "mp3", "m4a", "flac", "ogg", "webm"}

def validate_audio(file_path: str, max_duration_seconds: int = 60) -> Tuple[bool, str]:
    """Validate audio file format and duration."""
    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {file_path}"
    ext = path.suffix.lstrip(".").lower()
    if ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {ext}. Supported: {SUPPORTED_FORMATS}"
    if path.stat().st_size == 0:
        return False, "Audio file is empty"
    if ext == "wav":
        try:
            with wave.open(file_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                if duration > max_duration_seconds:
                    return False, f"Audio too long: {duration:.1f}s (max {max_duration_seconds}s)"
        except Exception as e:
            return False, f"Invalid WAV file: {e}"
    return True, "Valid"

def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """Convert audio to WAV format using pydub if available, else return as-is."""
    if input_path.lower().endswith(".wav"):
        return input_path
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        logger.info(f"Converted {input_path} -> {output_path}")
        return output_path
    except ImportError:
        logger.warning("pydub not installed; returning original path")
        return input_path
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return input_path

def get_audio_duration(file_path: str) -> float:
    """Get duration of WAV file in seconds."""
    try:
        with wave.open(file_path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0
