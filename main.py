import time
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Optional

_start_time = time.time()

app = FastAPI(
    title="VoiceBot Customer Support API",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────

class TextInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "I want to check my order status"}
        }


class ResponseInput(BaseModel):
    intent: str
    confidence: float = 0.9
    use_follow_up: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "intent": "order_status",
                "confidence": 0.95,
                "use_follow_up": False
            }
        }


class SynthesizeInput(BaseModel):
    text: str
    speed: float = 1.0
    language: str = "en"

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello! How can I help you today?",
                "speed": 1.0,
                "language": "en"
            }
        }


# ── Routes ───────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "VoiceBot Customer Support API",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "docs": "http://localhost:8000/docs",
    }


@app.get("/health")
async def health_check():
    components = {}
    try:
        from app.core.intent import get_classifier
        clf = get_classifier()
        components["intent_classifier"] = f"operational ({clf._model_type})"
    except Exception as e:
        components["intent_classifier"] = f"error: {str(e)[:60]}"

    try:
        from app.core.response import get_response_generator
        get_response_generator()
        components["response_generator"] = "operational"
    except Exception as e:
        components["response_generator"] = f"error: {str(e)[:60]}"

    components["tts"] = "operational (gTTS)"

    status = "healthy" if all("error" not in v for v in components.values()) else "degraded"
    return {"status": status, "version": "1.0.0", "components": components}


@app.get("/metrics")
async def get_metrics():
    try:
        from app.utils.metrics import metrics_collector
        return metrics_collector.get_summary()
    except Exception as e:
        return {"error": str(e)}


@app.get("/intents")
async def list_intents():
    try:
        from app.core.response import get_response_generator
        gen = get_response_generator()
        return {"intents": gen.get_all_intents()}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict-intent")
async def predict_intent(body: TextInput):
    """
    Classify intent from text input.
    """
    try:
        from app.core.intent import get_classifier
        classifier = get_classifier()
        result = classifier.predict(body.text)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate-response")
async def generate_response(body: ResponseInput):
    """
    Generate a customer support response for a given intent.
    """
    try:
        from app.core.response import get_response_generator
        generator = get_response_generator()
        result = generator.generate(
            intent=body.intent,
            confidence=body.confidence,
            use_follow_up=body.use_follow_up,
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/synthesize")
async def synthesize_speech(body: SynthesizeInput):
    """
    Convert text to speech and return audio file.
    """
    try:
        from app.core.tts import get_tts
        tts = get_tts()
        result = tts.synthesize(
            text=body.text,
            speed=body.speed,
            language=body.language,
        )
        fmt = result["format"]
        media_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"
        return Response(
            content=result["audio_bytes"],
            media_type=media_type,
            headers={
                "X-TTS-Engine": result["engine_used"],
                "Content-Disposition": f"attachment; filename=response.{fmt}",
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        from app.core.asr import get_asr
        audio_bytes = await audio.read()
        print(f"Received audio: {audio.filename}, size={len(audio_bytes)} bytes")
        asr = get_asr()
        result = asr.transcribe_bytes(audio_bytes, audio.filename or "audio.webm")
        print(f"Transcription result: {result}")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/voicebot")
async def voicebot_pipeline(audio: UploadFile = File(...), tts_speed: float = 1.0):
    """
    Full pipeline: Upload audio → Transcribe → Classify Intent → Generate Response → Return Audio.
    """
    stage_latencies = {}
    try:
        audio_bytes = await audio.read()

        # Stage 1: ASR
        t0 = time.time()
        from app.core.asr import get_asr
        asr = get_asr()
        asr_result = asr.transcribe_bytes(audio_bytes, audio.filename or "audio.wav")
        stage_latencies["asr_ms"] = round((time.time() - t0) * 1000, 2)
        transcribed_text = asr_result.get("text", "").strip()

        if not transcribed_text:
            transcribed_text = "I could not understand the audio."

        # Stage 2: Intent
        t1 = time.time()
        from app.core.intent import get_classifier
        classifier = get_classifier()
        intent_result = classifier.predict(transcribed_text)
        stage_latencies["intent_ms"] = round((time.time() - t1) * 1000, 2)

        # Stage 3: Response
        t2 = time.time()
        from app.core.response import get_response_generator
        generator = get_response_generator()
        response_result = generator.generate(
            intent=intent_result["intent"],
            confidence=intent_result["confidence"],
        )
        stage_latencies["response_ms"] = round((time.time() - t2) * 1000, 2)
        response_text = response_result["response_text"]

        # Stage 4: TTS
        t3 = time.time()
        from app.core.tts import get_tts
        tts = get_tts()
        tts_result = tts.synthesize(response_text, speed=tts_speed)
        stage_latencies["tts_ms"] = round((time.time() - t3) * 1000, 2)

        fmt = tts_result["format"]
        media_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"

        return Response(
            content=tts_result["audio_bytes"],
            media_type=media_type,
            headers={
                "X-Transcribed-Text": transcribed_text[:200],
                "X-Intent": intent_result["intent"],
                "X-Intent-Confidence": str(round(intent_result["confidence"], 3)),
                "X-Stage-Latencies": str(stage_latencies),
                "Content-Disposition": f"attachment; filename=response.{fmt}",
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)