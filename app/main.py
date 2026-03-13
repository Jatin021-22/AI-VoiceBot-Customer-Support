"""VoiceBot Customer Support System — Main FastAPI Application"""
import time, uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("voicebot.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("VoiceBot Customer Support System Starting...")
    logger.info(f"ASR Model: {settings.ASR_MODEL} | TTS: {settings.TTS_ENGINE}")
    
    from app.services.asr_service import ASRService
    from app.services.intent_service import IntentService
    from app.services.tts_service import TTSService
    from app.services.response_service import ResponseService
    
    app.state.asr = ASRService()
    app.state.intent = IntentService()
    app.state.tts = TTSService()
    app.state.response = ResponseService()
    logger.info("All services initialized. Ready.")
    yield
    logger.info("VoiceBot shutting down.")

app = FastAPI(
    title="VoiceBot Customer Support API",
    description="AI-Powered Voice Bot: ASR (Whisper) → Intent (BERT) → Response → TTS (gTTS)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def request_tracing(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    request.state.request_id = rid
    start = time.time()
    response = await call_next(request)
    ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = str(ms)
    logger.info(f"[{rid}] {request.method} {request.url.path} → {response.status_code} ({ms}ms)")
    return response

@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})

app.include_router(router, prefix="/api/v1")

@app.get("/", tags=["Health"])
def root():
    return {"service": "VoiceBot", "version": "1.0.0", "status": "running", "docs": "/docs"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "asr": settings.ASR_MODEL,
            "tts": settings.TTS_ENGINE, "intents": settings.NUM_INTENTS}
