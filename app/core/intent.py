import json
import pickle
import time
from pathlib import Path


INTENTS_FALLBACK = [
    "order_status", "order_cancellation", "refund_request",
    "subscription_management", "technical_support", "billing_inquiry",
    "account_management", "product_inquiry", "shipping_delivery",
    "complaint", "general_inquiry"
]


class IntentClassifier:
    _model = None
    _label_names = None
    _model_type = None

    def __init__(self):
        self.confidence_threshold = 0.1
        self._load_model()

    def _load_model(self):
        base_dir = Path(__file__).parent.parent.parent
        model_path = base_dir / "models" / "intent_classifier"
        sklearn_path = model_path / "sklearn_pipeline.pkl"
        mapping_path = model_path / "label_mappings.json"

        if sklearn_path.exists():
            with open(sklearn_path, "rb") as f:
                IntentClassifier._model = pickle.load(f)
            if mapping_path.exists():
                with open(mapping_path) as f:
                    mappings = json.load(f)
                IntentClassifier._label_names = mappings.get("intents", INTENTS_FALLBACK)
            else:
                IntentClassifier._label_names = INTENTS_FALLBACK
            IntentClassifier._model_type = "sklearn"
            print(f"Intent model loaded: {len(IntentClassifier._label_names)} intents")
        else:
            IntentClassifier._model_type = "keyword"
            IntentClassifier._label_names = INTENTS_FALLBACK
            print("No trained model found, using keyword fallback")

    def predict(self, text: str) -> dict:
        start = time.time()

        if not text or not text.strip():
            return {
                "intent": "general_inquiry",
                "confidence": 0.5,
                "all_scores": {},
                "model_type": self._model_type,
                "processing_time_ms": 0,
            }

        if IntentClassifier._model_type == "sklearn":
            return self._predict_sklearn(text, start)
        else:
            return self._predict_keyword(text, start)

    def _predict_sklearn(self, text: str, start: float) -> dict:
        import numpy as np
        probs = IntentClassifier._model.predict_proba([text])[0]
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        label_names = IntentClassifier._label_names
        intent = label_names[pred_id] if pred_id < len(label_names) else "general_inquiry"
        all_scores = {
            label_names[i]: round(float(p), 4)
            for i, p in enumerate(probs)
            if i < len(label_names)
        }
        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "model_type": "sklearn",
            "processing_time_ms": round((time.time() - start) * 1000, 2),
            "low_confidence": confidence < self.confidence_threshold,
        }

    def _predict_keyword(self, text: str, start: float) -> dict:
        text_lower = text.lower()
        keyword_map = {
            "order_status": ["order status", "track", "where is my order", "delivery status", "shipped"],
            "order_cancellation": ["cancel", "cancellation", "stop order", "don't want"],
            "refund_request": ["refund", "money back", "return", "reimbursement"],
            "subscription_management": ["subscription", "plan", "membership", "upgrade", "downgrade"],
            "technical_support": ["not working", "error", "bug", "crash", "technical", "broken"],
            "billing_inquiry": ["bill", "charge", "invoice", "payment", "receipt"],
            "account_management": ["password", "email", "account", "login", "username", "reset"],
            "product_inquiry": ["product", "price", "feature", "cost", "available"],
            "shipping_delivery": ["shipping", "delivery", "package", "courier", "dispatch"],
            "complaint": ["complaint", "unhappy", "terrible", "disappointed", "unacceptable"],
            "general_inquiry": ["help", "question", "information", "assistance"],
        }
        scores = {}
        for intent, keywords in keyword_map.items():
            scores[intent] = sum(1 for kw in keywords if kw in text_lower)

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        confidence = min(0.9, 0.5 + best_score * 0.1) if best_score > 0 else 0.4

        return {
            "intent": best_intent,
            "confidence": round(confidence, 4),
            "all_scores": scores,
            "model_type": "keyword",
            "processing_time_ms": round((time.time() - start) * 1000, 2),
            "low_confidence": confidence < self.confidence_threshold,
        }


_classifier_instance = None

def get_classifier() -> IntentClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance