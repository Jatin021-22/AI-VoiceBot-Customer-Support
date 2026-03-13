# 🎙️ VoiceBot — AI-Powered Customer Support Voice Bot

A production-ready, end-to-end voice bot system for customer support automation. Accepts voice input, understands intent, generates appropriate responses, and returns synthesized speech.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VoiceBot Pipeline                            │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────┐ │
│  │  Audio   │──▶│  ASR Layer   │──▶│  NLP Layer   │──▶│Response │ │
│  │  Input   │   │  (Whisper)   │   │  (BERT/LR)   │   │  Layer  │ │
│  │  (.wav)  │   │              │   │              │   │(Mapped) │ │
│  └──────────┘   └──────────────┘   └──────────────┘   └────┬────┘ │
│                       │                   │                 │      │
│                  Transcript          Intent +           Response    │
│                   + WER             Confidence           Text      │
│                                                            │        │
│  ┌──────────┐                                      ┌──────▼──────┐ │
│  │  Audio   │◀─────────────────────────────────────│ TTS Layer   │ │
│  │  Output  │                                      │  (gTTS)     │  │
│  │  (.mp3)  │                                      └─────────────┘  │
│  └──────────┘                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Structure
```
voicebot/
├── app/
│   ├── main.py                    # FastAPI app with lifespan & middleware
│   ├── api/
│   │   ├── routes.py              # All API endpoints
│   │   └── schemas.py             # Pydantic request/response models
│   ├── core/
│   │   └── pipeline.py            # End-to-end pipeline orchestrator
│   ├── models/
│   │   └── intent_classifier.py   # BERT + sklearn intent classifier
│   ├── services/
│   │   ├── asr_service.py         # Whisper ASR wrapper
│   │   ├── intent_service.py      # Intent classification service
│   │   ├── response_service.py    # Response generation service
│   │   └── tts_service.py         # gTTS / pyttsx3 TTS wrapper
│   └── utils/
│       ├── audio_utils.py         # Audio validation & conversion
│       └── logger.py              # Rotating file + console logger
├── config/
│   ├── settings.py                # Centralized config (no hard-coding)
│   └── responses.json             # Response template library (12 intents)
├── data/
│   ├── intents/training_data.json # 96 labeled training samples
│   └── audio_samples/             # Test WAV files
├── scripts/
│   ├── evaluate_model.py          # Full metrics + confusion matrix
│   └── generate_sample_audio.py   # Synthetic test audio generator
├── tests/
│   └── test_pipeline.py           # 20+ unit tests (pytest)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Model Choices & Justification

| Component | Choice | Justification |
|-----------|--------|---------------|
| **ASR** | OpenAI Whisper `base` | Production-grade, multilingual, noise-robust, runs on CPU |
| **Intent (primary)** | `distilbert-base-uncased` | 40% smaller than BERT, 60% faster, 97% accuracy retention |
| **Intent (fallback)** | TF-IDF + Logistic Regression | Zero-dependency fallback, interpretable, fast |
| **Response** | Intent-mapped templates | Deterministic, no hallucination, domain-constrained |
| **TTS** | gTTS (Google TTS) | Clear audio, multiple languages, adjustable rate via pydub |
| **TTS Fallback** | pyttsx3 | Fully offline, no API key needed |

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- `ffmpeg` installed (for audio conversion): `apt install ffmpeg` or `brew install ffmpeg`

### Installation
```bash
# Clone and setup
git clone <repo-url>
cd voicebot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env as needed
```

### Running the Server
```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Open API docs: http://localhost:8000/docs

### Generate Test Audio
```bash
python scripts/generate_sample_audio.py
```

### Run Evaluation
```bash
python scripts/evaluate_model.py
```

### Run Tests
```bash
pytest tests/ -v --tb=short --cov=app
```

---

## 📡 API Reference

### `POST /api/v1/transcribe`
Convert audio to text (ASR).
```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "audio=@data/audio_samples/test_order_status.wav" \
  -F "language=en"
```
**Response:**
```json
{
  "text": "Where is my order?",
  "language": "en",
  "confidence": 0.94,
  "duration_ms": 1230.5,
  "model": "whisper-base"
}
```

---

### `POST /api/v1/predict-intent`
Classify intent from text.
```bash
curl -X POST http://localhost:8000/api/v1/predict-intent \
  -H "Content-Type: application/json" \
  -d '{"text": "I need a refund for my order"}'
```
**Response:**
```json
{
  "intent": "refund_request",
  "confidence": 0.89,
  "all_scores": {
    "refund_request": 0.89,
    "order_status": 0.05,
    ...
  },
  "duration_ms": 45.2,
  "backend": "sklearn"
}
```

---

### `POST /api/v1/generate-response`
Generate customer support response.
```bash
curl -X POST http://localhost:8000/api/v1/generate-response \
  -H "Content-Type: application/json" \
  -d '{"intent": "refund_request"}'
```

---

### `POST /api/v1/synthesize`
Text to speech (returns MP3).
```bash
curl -X POST http://localhost:8000/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your refund is being processed.", "speaking_rate": 1.1}' \
  --output response.mp3
```

---

### `POST /api/v1/voicebot` ⭐ Unified Pipeline
Full voice-to-voice pipeline.
```bash
# Get audio response
curl -X POST http://localhost:8000/api/v1/voicebot \
  -F "audio=@data/audio_samples/test_refund.wav" \
  -F "speaking_rate=1.0" \
  --output bot_response.mp3

# Get JSON pipeline metadata
curl -X POST http://localhost:8000/api/v1/voicebot \
  -F "audio=@data/audio_samples/test_order_status.wav" \
  -F "return_json=true"
```
**JSON Response headers include:**
- `X-Transcript`: Recognized speech
- `X-Intent`: Predicted intent
- `X-Confidence`: Classification confidence
- `X-Total-Duration-Ms`: End-to-end latency

---

## 📊 Evaluation Metrics

### Intent Classification (on held-out 20% test set)

| Metric | Score |
|--------|-------|
| Accuracy | ~0.85+ |
| Precision | ~0.84+ |
| Recall | ~0.83+ |
| F1 Score | ~0.83+ |

*Exact values depend on sklearn/transformer backend. Run `python scripts/evaluate_model.py` for live metrics.*

### Supported Intents (12 classes)

| # | Intent | Example |
|---|--------|---------|
| 1 | `order_status` | "Where is my order?" |
| 2 | `order_cancellation` | "Cancel my order" |
| 3 | `refund_request` | "I need a refund" |
| 4 | `subscription_management` | "Change my plan" |
| 5 | `technical_support` | "App keeps crashing" |
| 6 | `billing_inquiry` | "Wrong charge on my account" |
| 7 | `account_management` | "Reset my password" |
| 8 | `product_inquiry` | "What features does Pro include?" |
| 9 | `shipping_delivery` | "When will it arrive?" |
| 10 | `complaint` | "This is unacceptable" |
| 11 | `general_inquiry` | "How can you help me?" |
| 12 | `unknown` | Fallback for unclear input |

### ASR (Word Error Rate)
- **Target WER:** < 10% on clean speech (Whisper `base`)
- **Whisper `base`** typical WER: 5–8% on English speech
- Run `POST /api/v1/evaluate/wer` with your labeled audio set

---

## ⚡ Performance

| Stage | Typical Latency |
|-------|----------------|
| ASR (Whisper base, CPU) | 0.8–2.5s |
| Intent Classification | 20–100ms |
| Response Generation | < 5ms |
| TTS Synthesis | 300–800ms |
| **End-to-end total** | **~1.5–4s** |

*GPU inference reduces ASR time by 5–10x.*

---

## 🔒 Error Handling
- Invalid/empty audio → HTTP 400 with descriptive message
- Unsupported format → HTTP 400
- File too large → HTTP 413
- Low-confidence intent → falls back to `unknown` intent
- TTS failure → silent audio fallback, error logged
- All errors logged with request ID for tracing

---

## 🧪 Testing
```bash
pytest tests/ -v                    # All tests
pytest tests/ -v -k "TestASR"       # ASR tests only
pytest tests/ -v -k "TestIntent"    # Intent tests only
pytest tests/ --cov=app --cov-report=html  # With coverage
```

---

## 📝 Configuration Reference (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL` | `base` | Whisper model size |
| `ASR_DEVICE` | `cpu` | `cpu` or `cuda` |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.5` | Min confidence for intent |
| `TTS_ENGINE` | `gtts` | `gtts` or `pyttsx3` |
| `TTS_SPEAKING_RATE` | `1.0` | 0.5–2.0 |
| `LOG_LEVEL` | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `MAX_UPLOAD_SIZE_MB` | `25` | Max audio upload size |
