"""Intent Classification Service — wraps IntentClassifier with training/loading logic."""
import os, json
from typing import Dict, List
from app.models.intent_classifier import IntentClassifier
from app.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("voicebot.intent")

class IntentService:
    def __init__(self):
        self.classifier = IntentClassifier()
        self._initialize()

    def _initialize(self):
        loaded = self.classifier.load()
        if not loaded:
            logger.info("No saved model found — training from built-in dataset")
            self._train_from_data()

    def _train_from_data(self):
        data_path = os.path.join(settings.DATA_DIR, "intents", "training_data.json")
        if not os.path.exists(data_path):
            logger.warning("Training data not found — using rule-based fallback")
            return
        with open(data_path) as f:
            training_data = json.load(f)
        logger.info(f"Training on {len(training_data)} samples")
        result = self.classifier.train(training_data)
        logger.info(f"Training complete: {result}")

    def predict_intent(self, text: str) -> Dict:
        if not text or not text.strip():
            return {
                "intent": "unknown",
                "confidence": 1.0,
                "all_scores": {},
                "error": "Empty input text",
            }
        result = self.classifier.predict(text)
        # Apply confidence threshold
        if result["confidence"] < settings.INTENT_CONFIDENCE_THRESHOLD:
            result["intent"] = "unknown"
        return result

    def get_supported_intents(self) -> List[str]:
        return self.classifier.intent_labels

    def evaluate(self, test_data: List[Dict]) -> Dict:
        return self.classifier.evaluate(test_data)

    def retrain(self, training_data: List[Dict]) -> Dict:
        result = self.classifier.train(training_data)
        return result
