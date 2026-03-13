"""Config-driven settings for VoiceBot. No hard-coded values in services."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    # ASR
    ASR_MODEL: str = "base"
    ASR_LANGUAGE: str = "en"
    ASR_DEVICE: str = "cpu"
    ASR_BEAM_SIZE: int = 5
    ASR_TEMPERATURE: float = 0.0
    MAX_AUDIO_DURATION_SECONDS: int = 60
    # Intent
    NLP_MODEL: str = "bert-base-uncased"
    INTENT_MODEL_PATH: str = str(BASE_DIR / "models" / "intent_classifier")
    INTENT_CONFIDENCE_THRESHOLD: float = 0.5
    NUM_INTENTS: int = 12
    MAX_SEQ_LENGTH: int = 128
    # Response
    RESPONSE_STRATEGY: str = "mapped"
    RESPONSE_CONFIG_PATH: str = str(BASE_DIR / "config" / "responses.json")
    # TTS
    TTS_ENGINE: str = "gtts"
    TTS_LANGUAGE: str = "en"
    TTS_SPEAKING_RATE: float = 1.0
    TTS_SLOW: bool = False
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_UPLOAD_SIZE_MB: int = 25
    # Paths
    DATA_DIR: str = str(BASE_DIR / "data")
    LOGS_DIR: str = str(BASE_DIR / "logs")
    MODELS_DIR: str = str(BASE_DIR / "models")
    AUDIO_SAMPLES_DIR: str = str(BASE_DIR / "data" / "audio_samples")
    EVAL_DATA_PATH: str = str(BASE_DIR / "data" / "eval_set.json")

settings = Settings()

for d in [settings.DATA_DIR, settings.LOGS_DIR, settings.MODELS_DIR,
          settings.AUDIO_SAMPLES_DIR, settings.INTENT_MODEL_PATH]:
    Path(d).mkdir(parents=True, exist_ok=True)
