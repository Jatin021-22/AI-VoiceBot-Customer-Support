"""
Intent Classification Model.
Fine-tuned DistilBERT or TF-IDF + Logistic Regression fallback.
"""
import os
import json
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

INTENT_LABELS = [
    "order_status", "order_cancellation", "refund_request",
    "subscription_management", "technical_support", "billing_inquiry",
    "account_management", "product_inquiry", "shipping_delivery",
    "complaint", "general_inquiry", "unknown",
]
LABEL2ID = {l: i for i, l in enumerate(INTENT_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(INTENT_LABELS)}

BASE_DIR = Path("C:/Users/tankr/Downloads/voicebot_complete/voicebot")
MODEL_PATH = BASE_DIR / "models" / "intent_classifier"
DATA_PATH = BASE_DIR / "data" / "intents" / "training_data.json"
MAX_SEQ_LENGTH = 128


class IntentClassifier:
    """
    Two-tier classifier:
    1. Transformer (DistilBERT) when available
    2. TF-IDF + Logistic Regression as lightweight fallback
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.lr_model = None
        self.backend = None
        self.label2id = LABEL2ID
        self.id2label = ID2LABEL
        self.intent_labels = INTENT_LABELS

    def train(self, training_data: List[Dict], save_path: str = None) -> Dict:
        """Train the intent classifier."""
        texts = [d["text"] for d in training_data]
        labels = [
            self.label2id.get(d["intent"], self.label2id["unknown"])
            for d in training_data
        ]
        try:
            return self._train_transformer(texts, labels, save_path)
        except Exception as e:
            print(f"Transformer training failed ({e}), falling back to sklearn")
            return self._train_sklearn(texts, labels, save_path)

    def _train_transformer(self, texts, labels, save_path):
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer
        )
        import torch
        from torch.utils.data import Dataset

        class IntentDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        print(f"Training DistilBERT on {len(texts)} samples...")
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_SEQ_LENGTH
        )
        dataset = IntentDataset(encodings, labels)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(INTENT_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        sp = save_path or str(MODEL_PATH)
        Path(sp).mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=sp,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=False,
            no_cuda=True,
            report_to="none",
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()

        tokenizer.save_pretrained(sp)
        model.save_pretrained(sp)
        self.model = model
        self.tokenizer = tokenizer
        self.backend = "transformer"
        print(f"Transformer model saved to {sp}")
        return {"backend": "transformer", "samples": len(texts), "epochs": 5}

    def _train_sklearn(self, texts, labels, save_path):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        print(f"Training TF-IDF + LR on {len(texts)} samples...")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), max_features=5000, sublinear_tf=True
            )),
            ("lr", LogisticRegression(
                max_iter=500, C=5.0, class_weight="balanced"
            )),
        ])
        pipeline.fit(texts, labels)
        self.lr_model = pipeline
        self.backend = "sklearn"

        sp = save_path or str(MODEL_PATH)
        Path(sp).mkdir(parents=True, exist_ok=True)
        model_file = os.path.join(sp, "sklearn_pipeline.pkl")

        with open(model_file, "wb") as f:
            pickle.dump(pipeline, f)

        # Save label mappings
        with open(os.path.join(sp, "label_mappings.json"), "w") as f:
            json.dump({
                "id2label": ID2LABEL,
                "label2id": LABEL2ID,
                "model_type": "sklearn",
                "intents": INTENT_LABELS,
            }, f, indent=2)

        print(f"Sklearn model saved to {model_file}")
        return {"backend": "sklearn", "samples": len(texts)}

    def load(self, model_path: str = None) -> bool:
        """Load a saved model."""
        mp = model_path or str(MODEL_PATH)
        sklearn_path = os.path.join(mp, "sklearn_pipeline.pkl")
        config_path = os.path.join(mp, "config.json")

        # Try transformer first
        if os.path.exists(config_path):
            try:
                from transformers import (
                    AutoTokenizer, AutoModelForSequenceClassification
                )
                print(f"Loading transformer model from {mp}")
                self.tokenizer = AutoTokenizer.from_pretrained(mp)
                self.model = AutoModelForSequenceClassification.from_pretrained(mp)
                self.model.eval()
                self.backend = "transformer"
                print("Transformer model loaded successfully")
                return True
            except Exception as e:
                print(f"Transformer load failed: {e}")

        # Try sklearn
        if os.path.exists(sklearn_path):
            try:
                with open(sklearn_path, "rb") as f:
                    self.lr_model = pickle.load(f)
                self.backend = "sklearn"
                print("Sklearn model loaded successfully")
                return True
            except Exception as e:
                print(f"Sklearn load failed: {e}")

        print("No model found, will use rule-based fallback")
        return False

    def predict(self, text: str) -> Dict:
        """Predict intent with confidence scores."""
        start = time.time()

        if not text or not text.strip():
            return self._unknown_result(time.time() - start)

        if self.backend == "transformer" and self.model:
            return self._predict_transformer(text, start)
        elif self.backend == "sklearn" and self.lr_model:
            return self._predict_sklearn(text, start)
        else:
            return self._rule_based_predict(text, start)

    def _predict_transformer(self, text: str, start: float) -> Dict:
        import torch
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, padding=True, max_length=MAX_SEQ_LENGTH
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        pred_id = int(np.argmax(probs))
        return {
            "intent": self.id2label[pred_id],
            "confidence": round(float(probs[pred_id]), 4),
            "all_scores": {
                self.id2label[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            },
            "duration_ms": round((time.time() - start) * 1000, 2),
            "backend": "transformer",
        }

    def _predict_sklearn(self, text: str, start: float) -> Dict:
        probs = self.lr_model.predict_proba([text])[0]
        classes = self.lr_model.classes_
        pred_id = int(np.argmax(probs))
        return {
            "intent": self.id2label[classes[pred_id]],
            "confidence": round(float(probs[pred_id]), 4),
            "all_scores": {
                self.id2label[classes[i]]: round(float(p), 4)
                for i, p in enumerate(probs)
            },
            "duration_ms": round((time.time() - start) * 1000, 2),
            "backend": "sklearn",
        }

    def _rule_based_predict(self, text: str, start: float) -> Dict:
        """Keyword-based fallback when no trained model exists."""
        text_lower = text.lower()
        rules = {
            "order_status": ["order", "track", "tracking", "shipped", "arrive", "package"],
            "order_cancellation": ["cancel", "cancellation", "stop order"],
            "refund_request": ["refund", "money back", "return", "reimburse"],
            "subscription_management": ["subscription", "plan", "upgrade", "downgrade", "renew"],
            "technical_support": ["error", "crash", "bug", "not working", "broken"],
            "billing_inquiry": ["bill", "charge", "invoice", "payment", "fee"],
            "account_management": ["account", "password", "email", "login", "username"],
            "product_inquiry": ["product", "feature", "specification", "stock"],
            "shipping_delivery": ["ship", "deliver", "shipping", "courier"],
            "complaint": ["complaint", "unhappy", "terrible", "disappointed"],
            "general_inquiry": ["help", "question", "assist", "support"],
        }
        best_intent, best_score = "unknown", 0
        for intent, keywords in rules.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_intent, best_score = intent, score

        confidence = min(0.9, 0.4 + best_score * 0.15) if best_score > 0 else 0.3
        return {
            "intent": best_intent,
            "confidence": round(confidence, 4),
            "all_scores": {best_intent: confidence},
            "duration_ms": round((time.time() - start) * 1000, 2),
            "backend": "rule_based",
        }

    def _unknown_result(self, elapsed: float) -> Dict:
        return {
            "intent": "unknown",
            "confidence": 1.0,
            "all_scores": {"unknown": 1.0},
            "duration_ms": round(elapsed * 1000, 2),
            "backend": "direct",
        }

    def evaluate(self, test_data: List[Dict]) -> Dict:
        """Compute accuracy, precision, recall, F1 and confusion matrix."""
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            accuracy_score, precision_recall_fscore_support
        )
        y_true, y_pred = [], []
        for item in test_data:
            pred = self.predict(item["text"])
            y_true.append(self.label2id.get(item["intent"], self.label2id["unknown"]))
            y_pred.append(self.label2id.get(pred["intent"], self.label2id["unknown"]))

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        report = classification_report(
            y_true, y_pred,
            labels=list(range(len(INTENT_LABELS))),
            target_names=INTENT_LABELS,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "accuracy": round(acc, 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "classification_report": report,
            "confusion_matrix": cm,
            "labels": INTENT_LABELS,
        }


# ── Standalone training script ────────────────────────────────────

def load_training_data(data_path: str) -> List[Dict]:
    with open(data_path, "r") as f:
        return json.load(f)


def main():
    print(f"Data path  : {DATA_PATH}")
    print(f"Model path : {MODEL_PATH}")
    print(f"Data exists: {DATA_PATH.exists()}")

    # Load data
    training_data = load_training_data(str(DATA_PATH))
    # Remove unknown intent for training
    training_data = [d for d in training_data if d.get("intent") != "unknown"]
    print(f"Training samples: {len(training_data)}")

    # Train
    classifier = IntentClassifier()
    result = classifier.train(training_data, save_path=str(MODEL_PATH))
    print(f"Training result: {result}")

    # Quick evaluation on training data
    metrics = classifier.evaluate(training_data)
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()