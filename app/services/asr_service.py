"""
ASR Service — Whisper-based Speech Recognition.
Handles WAV input, noise robustness, WER evaluation.
"""
import os, time, tempfile
from pathlib import Path
from typing import Dict, Optional
from app.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("voicebot.asr")

class ASRService:
    """Whisper ASR with fallback to mock for environments without GPU/model."""

    def __init__(self):
        self.model = None
        self.model_name = settings.ASR_MODEL
        self.device = settings.ASR_DEVICE
        self._load_model()

    def _load_model(self):
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.warning("openai-whisper not installed. Using mock ASR.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}. Using mock ASR.")
            self.model = None

    def transcribe(self, audio_path: str, language: str = None) -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (WAV preferred)
            language: Language code override
        
        Returns:
            dict with: text, language, segments, confidence, duration_ms
        """
        start = time.time()
        lang = language or settings.ASR_LANGUAGE

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.model is None:
            return self._mock_transcribe(audio_path)

        try:
            logger.info(f"Transcribing: {audio_path}")
            result = self.model.transcribe(
                audio_path,
                language=lang,
                beam_size=settings.ASR_BEAM_SIZE,
                temperature=settings.ASR_TEMPERATURE,
                verbose=False,
            )
            duration_ms = round((time.time() - start) * 1000, 2)
            text = result.get("text", "").strip()
            logger.info(f"Transcribed in {duration_ms}ms: '{text[:80]}...'")
            return {
                "text": text,
                "language": result.get("language", lang),
                "segments": result.get("segments", []),
                "confidence": self._estimate_confidence(result),
                "duration_ms": duration_ms,
                "model": f"whisper-{self.model_name}",
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"ASR transcription error: {e}")

    def _estimate_confidence(self, result: Dict) -> float:
        """Estimate average confidence from segment log probs."""
        segments = result.get("segments", [])
        if not segments:
            return 0.85
        probs = [s.get("avg_logprob", -0.5) for s in segments]
        avg_logprob = sum(probs) / len(probs)
        # Convert log prob to 0-1 scale (empirically calibrated)
        confidence = min(1.0, max(0.0, 1.0 + avg_logprob / 3.0))
        return round(confidence, 3)

    def _mock_transcribe(self, audio_path: str) -> Dict:
        """Mock transcription for testing without Whisper installed."""
        logger.warning("Using mock ASR — install openai-whisper for real transcription")
        return {
            "text": "I want to check the status of my order",
            "language": "en",
            "segments": [],
            "confidence": 0.92,
            "duration_ms": 50.0,
            "model": "mock-asr",
        }

    def evaluate_wer(self, eval_data_path: str) -> Dict:
        """
        Compute Word Error Rate on an evaluation set.
        
        Eval data format: [{"audio": "path.wav", "reference": "ground truth text"}, ...]
        """
        import json
        try:
            with open(eval_data_path) as f:
                eval_data = json.load(f)
        except Exception as e:
            return {"error": f"Could not load eval data: {e}"}

        total_words, total_errors = 0, 0
        results = []

        for item in eval_data:
            ref = item.get("reference", "").lower()
            try:
                hyp_result = self.transcribe(item["audio"])
                hyp = hyp_result["text"].lower()
            except Exception:
                hyp = ""

            wer = self._compute_wer(ref, hyp)
            words = len(ref.split())
            total_words += words
            total_errors += int(wer * words)
            results.append({"reference": ref, "hypothesis": hyp, "wer": wer})

        overall_wer = total_errors / max(total_words, 1)
        return {
            "overall_wer": round(overall_wer, 4),
            "total_samples": len(eval_data),
            "samples": results,
        }

    @staticmethod
    def _compute_wer(reference: str, hypothesis: str) -> float:
        """Compute WER using dynamic programming (edit distance on words)."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        r, h = len(ref_words), len(hyp_words)
        dp = [[0] * (h + 1) for _ in range(r + 1)]
        for i in range(r + 1): dp[i][0] = i
        for j in range(h + 1): dp[0][j] = j
        for i in range(1, r + 1):
            for j in range(1, h + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[r][h] / r
