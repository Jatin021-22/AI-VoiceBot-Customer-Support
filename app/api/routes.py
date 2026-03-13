"""
FastAPI route definitions for VoiceBot.
Endpoints: /transcribe, /predict-intent, /generate-response, /synthesize, /voicebot
"""
import os, json, tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from app.api.schemas import (
    TranscribeResponse, IntentRequest, IntentResponse,
    ResponseRequest, ResponseResult, SynthesizeRequest,
    PipelineResult, EvaluationResult,
)
from app.core.pipeline import VoiceBotPipeline
from app.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("voicebot.routes")
router = APIRouter()


def get_services(request: Request):
    return (
        request.app.state.asr,
        request.app.state.intent,
        request.app.state.response,
        request.app.state.tts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /transcribe — Audio → Text (ASR)
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/transcribe", response_model=TranscribeResponse, tags=["ASR"])
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    language: Optional[str] = Form("en"),
):
    """Convert uploaded audio to text using Whisper ASR."""
    asr, _, _, _ = get_services(request)
    
    # Save upload to temp file
    suffix = "." + (audio.filename.split(".")[-1] if audio.filename else "wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        if len(content) == 0:
            raise HTTPException(400, "Uploaded audio file is empty")
        if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = asr.transcribe(tmp_path, language=language)
        return TranscribeResponse(**{
            "text": result["text"],
            "language": result["language"],
            "confidence": result["confidence"],
            "duration_ms": result["duration_ms"],
            "model": result["model"],
            "segments": result.get("segments", []),
        })
    except Exception as e:
        logger.error(f"/transcribe error: {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict-intent — Text → Intent
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/predict-intent", response_model=IntentResponse, tags=["NLP"])
async def predict_intent(request: Request, body: IntentRequest):
    """Classify user intent from text with confidence scores."""
    _, intent_svc, _, _ = get_services(request)
    try:
        result = intent_svc.predict_intent(body.text)
        return IntentResponse(**result)
    except Exception as e:
        logger.error(f"/predict-intent error: {e}")
        raise HTTPException(500, f"Intent classification failed: {str(e)}")


@router.get("/intents", tags=["NLP"])
async def get_intents(request: Request):
    """List all supported intent labels."""
    _, intent_svc, _, _ = get_services(request)
    return {"intents": intent_svc.get_supported_intents(), "count": len(intent_svc.get_supported_intents())}


# ─────────────────────────────────────────────────────────────────────────────
# POST /generate-response — Intent → Text Response
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/generate-response", response_model=ResponseResult, tags=["NLP"])
async def generate_response(request: Request, body: ResponseRequest):
    """Generate a customer support response for a given intent."""
    _, _, response_svc, _ = get_services(request)
    try:
        result = response_svc.generate(intent=body.intent, context=body.context)
        return ResponseResult(**result)
    except Exception as e:
        logger.error(f"/generate-response error: {e}")
        raise HTTPException(500, f"Response generation failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# POST /synthesize — Text → Audio
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/synthesize", tags=["TTS"])
async def synthesize_speech(request: Request, body: SynthesizeRequest):
    """Convert text to speech audio (returns MP3 file)."""
    _, _, _, tts_svc = get_services(request)
    try:
        result = tts_svc.synthesize(
            text=body.text,
            language=body.language,
            speaking_rate=body.speaking_rate,
        )
        audio_path = result["audio_path"]
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename="response.mp3",
            headers={
                "X-TTS-Engine": result["engine"],
                "X-Duration-Ms": str(result["duration_ms"]),
            },
        )
    except Exception as e:
        logger.error(f"/synthesize error: {e}")
        raise HTTPException(500, f"TTS synthesis failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# POST /voicebot — UNIFIED: Audio → Audio
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/voicebot", tags=["Pipeline"])
async def voicebot_pipeline(
    request: Request,
    audio: UploadFile = File(..., description="User voice input"),
    language: Optional[str] = Form("en"),
    speaking_rate: Optional[float] = Form(1.0),
    return_json: Optional[bool] = Form(False),
):
    """
    Unified voice bot pipeline: audio input → audio output.
    
    Flow: Audio Upload → ASR → Intent Classification → Response Generation → TTS → MP3 Response
    
    Set `return_json=true` to get full pipeline metadata instead of audio.
    """
    asr, intent_svc, response_svc, tts_svc = get_services(request)
    pipeline = VoiceBotPipeline(asr, intent_svc, response_svc, tts_svc)

    suffix = "." + (audio.filename.split(".")[-1] if audio.filename else "wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        if not content:
            raise HTTPException(400, "Empty audio file")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.process(
            audio_path=tmp_path,
            language=language,
            speaking_rate=speaking_rate,
            return_audio=True,
        )

        if not result["success"]:
            raise HTTPException(422, result.get("error", "Pipeline failed"))

        if return_json:
            # Remove non-serializable paths for JSON response
            safe_result = {k: v for k, v in result.items() if k != "audio_output_path"}
            return JSONResponse(content=safe_result)

        audio_out = result.get("audio_output_path")
        if audio_out and os.path.exists(audio_out):
            return FileResponse(
                audio_out,
                media_type="audio/mpeg",
                filename="voicebot_response.mp3",
                headers={
                    "X-Transcript": result.get("transcript", "")[:200],
                    "X-Intent": result.get("intent", "unknown"),
                    "X-Confidence": str(result.get("confidence", 0)),
                    "X-Total-Duration-Ms": str(result.get("total_duration_ms", 0)),
                },
            )
        return JSONResponse(content={"error": "No audio output generated"}, status_code=500)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/voicebot error: {e}", exc_info=True)
        raise HTTPException(500, f"Pipeline error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# POST /evaluate — Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/evaluate/intent", tags=["Evaluation"])
async def evaluate_intent_model(request: Request, test_file: UploadFile = File(...)):
    """Evaluate intent classifier with labeled test data (JSON array)."""
    _, intent_svc, _, _ = get_services(request)
    try:
        content = await test_file.read()
        test_data = json.loads(content)
        metrics = intent_svc.evaluate(test_data)
        return metrics
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON test file")
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {str(e)}")


@router.post("/evaluate/wer", tags=["Evaluation"])
async def evaluate_wer(request: Request, eval_file: UploadFile = File(...)):
    """Evaluate ASR Word Error Rate with labeled audio set (JSON)."""
    asr, _, _, _ = get_services(request)
    try:
        content = await eval_file.read()
        eval_data = json.loads(content)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(eval_data, f)
            tmp_path = f.name
        result = asr.evaluate_wer(tmp_path)
        os.unlink(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(500, f"WER evaluation failed: {str(e)}")


@router.post("/train", tags=["Training"])
async def train_model(request: Request, training_file: UploadFile = File(...)):
    """Retrain intent classifier with new labeled data."""
    _, intent_svc, _, _ = get_services(request)
    try:
        content = await training_file.read()
        training_data = json.loads(content)
        result = intent_svc.retrain(training_data)
        return {"status": "trained", "result": result}
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")
