"""
End-to-end VoiceBot Pipeline.
Orchestrates: Audio → ASR → Intent → Response → TTS → Audio
"""
import time
from typing import Dict, Optional
from app.utils.logger import setup_logger
from app.utils.audio_utils import validate_audio, convert_to_wav, get_audio_duration
from config.settings import settings

logger = setup_logger("voicebot.pipeline")


class VoiceBotPipeline:
    """Full pipeline: audio file in, audio bytes out."""

    def __init__(self, asr_service, intent_service, response_service, tts_service):
        self.asr = asr_service
        self.intent = intent_service
        self.response = response_service
        self.tts = tts_service

    def process(self, audio_path: str, language: str = None,
                speaking_rate: float = None, return_audio: bool = True) -> Dict:
        """
        Full pipeline execution.
        
        Args:
            audio_path: Path to input audio file
            language: ASR language override
            speaking_rate: TTS rate override
            return_audio: Whether to synthesize audio output
        
        Returns:
            Pipeline result with all stage outputs
        """
        pipeline_start = time.time()
        result = {
            "pipeline_version": "1.0.0",
            "stages": {},
            "success": False,
            "error": None,
        }

        # ── Stage 1: Validate Audio ─────────────────────────────────────
        try:
            valid, msg = validate_audio(audio_path, settings.MAX_AUDIO_DURATION_SECONDS)
            if not valid:
                result["error"] = f"Invalid audio: {msg}"
                return result
            wav_path = convert_to_wav(audio_path)
            result["stages"]["validation"] = {
                "status": "ok",
                "audio_duration_s": get_audio_duration(wav_path),
            }
        except Exception as e:
            result["error"] = f"Audio validation failed: {e}"
            logger.error(f"Pipeline validation error: {e}")
            return result

        # ── Stage 2: ASR ────────────────────────────────────────────────
        try:
            asr_result = self.asr.transcribe(wav_path, language=language)
            result["stages"]["asr"] = asr_result
            transcript = asr_result.get("text", "")
            if not transcript:
                result["error"] = "No speech detected in audio"
                return result
        except Exception as e:
            result["error"] = f"ASR failed: {e}"
            logger.error(f"Pipeline ASR error: {e}")
            return result

        # ── Stage 3: Intent Classification ─────────────────────────────
        try:
            intent_result = self.intent.predict_intent(transcript)
            result["stages"]["intent"] = intent_result
        except Exception as e:
            logger.error(f"Pipeline intent error: {e}")
            intent_result = {"intent": "unknown", "confidence": 0.0}
            result["stages"]["intent"] = intent_result

        # ── Stage 4: Response Generation ────────────────────────────────
        try:
            response_result = self.response.generate(
                intent=intent_result.get("intent", "unknown"),
                context={"transcript": transcript},
            )
            result["stages"]["response"] = response_result
        except Exception as e:
            logger.error(f"Pipeline response error: {e}")
            response_result = {"text": "I'm sorry, something went wrong. Please try again."}
            result["stages"]["response"] = response_result

        # ── Stage 5: TTS ─────────────────────────────────────────────────
        if return_audio:
            try:
                tts_result = self.tts.synthesize(
                    text=response_result.get("text", ""),
                    speaking_rate=speaking_rate,
                )
                result["stages"]["tts"] = tts_result
                result["audio_output_path"] = tts_result.get("audio_path")
            except Exception as e:
                logger.error(f"Pipeline TTS error: {e}")
                result["stages"]["tts"] = {"error": str(e)}

        # ── Summary ──────────────────────────────────────────────────────
        total_ms = round((time.time() - pipeline_start) * 1000, 2)
        result["success"] = True
        result["total_duration_ms"] = total_ms
        result["transcript"] = transcript
        result["intent"] = intent_result.get("intent")
        result["confidence"] = intent_result.get("confidence")
        result["response_text"] = response_result.get("text")

        logger.info(
            f"Pipeline complete in {total_ms}ms | "
            f"Intent: {result['intent']} ({result['confidence']:.2f}) | "
            f"Transcript: '{transcript[:60]}'"
        )
        return result
