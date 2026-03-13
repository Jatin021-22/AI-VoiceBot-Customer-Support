import os
import time
import tempfile
import subprocess
import shutil
import numpy as np
import shutil
FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"

class WhisperASR:
    _model = None

    def __init__(self):
        self.model_name = "tiny"
        self._load_model()

    def _load_model(self):
        try:
            import whisper
            print(f"Loading Whisper model: {self.model_name}...")
            WhisperASR._model = whisper.load_model(self.model_name)
            print("Whisper model loaded successfully!")
        except Exception as e:
            print(f"Failed to load Whisper: {e}")
            WhisperASR._model = None

    def _convert_to_wav(self, input_path: str, output_path: str) -> bool:
        """Convert any audio file to 16kHz mono WAV using ffmpeg."""
        cmd = [
            FFMPEG_PATH, "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0

    def _load_audio_numpy(self, wav_path: str) -> np.ndarray:
        """
        Load WAV file to numpy array using ffmpeg directly.
        This bypasses Whisper's internal ffmpeg call.
        """
        cmd = [
            FFMPEG_PATH,
            "-nostdin",
            "-threads", "0",
            "-i", wav_path,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Audio load failed: {result.stderr.decode()[-200:]}")

        # Convert raw bytes to numpy float32 array
        audio = np.frombuffer(result.stdout, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        return audio

    def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.webm") -> dict:
        start_time = time.time()

        if not audio_bytes:
            return self._empty_result("Empty audio")

        if WhisperASR._model is None:
            return self._empty_result("Whisper not loaded")

        suffix = ".webm"
        if filename and "." in filename:
            suffix = "." + filename.split(".")[-1]

        input_tmp = None
        wav_tmp = None

        try:
            # Save incoming audio bytes to temp file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                input_tmp = f.name
            print(f"Saved: {input_tmp} ({len(audio_bytes)} bytes)")

            # Convert to WAV
            wav_tmp = input_tmp.replace(suffix, "_converted.wav")
            print("Converting to WAV...")
            if not self._convert_to_wav(input_tmp, wav_tmp):
                return self._empty_result("ffmpeg conversion failed")
            print(f"Converted: {wav_tmp}")

            # Save debug copy
            shutil.copy(wav_tmp, "last_recording.wav")

            # Load audio as numpy array using our ffmpeg
            print("Loading audio as numpy array...")
            audio = self._load_audio_numpy(wav_tmp)
            print(f"Audio loaded: {len(audio)} samples, duration={len(audio)/16000:.1f}s")

            if len(audio) < 1600:  # Less than 0.1 seconds
                return self._empty_result("Audio too short")

            # Transcribe using numpy array directly
            print("Transcribing with Whisper...")
            import whisper
            whisper_result = WhisperASR._model.transcribe(
                audio,
                fp16=False,
                task="transcribe",
                language="en",
                verbose=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.3,
                logprob_threshold=-1.0,
            )

            text = whisper_result.get("text", "").strip()
            segments = whisper_result.get("segments", [])

            if segments:
                log_probs = [s.get("avg_logprob", -0.5) for s in segments]
                avg_lp = sum(log_probs) / len(log_probs)
                confidence = min(1.0, max(0.0, avg_lp + 1.0))
            else:
                confidence = 0.5 if text else 0.0

            processing_ms = (time.time() - start_time) * 1000
            print(f"Transcribed: '{text}' | conf={confidence:.2f} | {processing_ms:.0f}ms")

            return {
                "text": text,
                "language": whisper_result.get("language", "en"),
                "confidence": round(confidence, 3),
                "segments": [
                    {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                    for s in segments
                ],
                "processing_time_ms": round(processing_ms, 2),
            }

        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result(str(e))

        finally:
            for path in [input_tmp, wav_tmp]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

    def _empty_result(self, reason: str = "") -> dict:
        print(f"Empty result: {reason}")
        return {
            "text": "",
            "language": "en",
            "confidence": 0.0,
            "segments": [],
            "processing_time_ms": 0,
            "warning": reason
        }


_asr_instance = None

def get_asr() -> WhisperASR:
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = WhisperASR()
    return _asr_instance