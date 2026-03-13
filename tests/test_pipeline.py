"""
Comprehensive tests for VoiceBot pipeline components.
Run: pytest tests/ -v --tb=short
"""
import json, os, sys, tempfile, wave, struct, math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Helpers ──────────────────────────────────────────────────────────────────
def create_test_wav(path: str, duration_seconds: float = 1.0, freq: int = 440):
    """Create a synthetic WAV file (sine wave) for testing."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            val = int(32767 * 0.3 * math.sin(2 * math.pi * freq * i / sample_rate))
            wf.writeframes(struct.pack("<h", val))
    return path


# ── ASR Tests ─────────────────────────────────────────────────────────────────
class TestASRService:
    def setup_method(self):
        from app.services.asr_service import ASRService
        self.asr = ASRService()

    def test_transcribe_returns_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            create_test_wav(f.name)
            result = self.asr.transcribe(f.name)
        assert isinstance(result, dict)
        assert "text" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        os.unlink(f.name)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.asr.transcribe("/nonexistent/audio.wav")

    def test_wer_computation(self):
        wer = self.asr._compute_wer("hello world", "hello world")
        assert wer == 0.0
        wer2 = self.asr._compute_wer("hello world", "")
        assert wer2 == 1.0
        wer3 = self.asr._compute_wer("the quick brown fox", "the quick brown fox")
        assert wer3 == 0.0

    def test_confidence_estimate(self):
        result = {"segments": [{"avg_logprob": -0.3}]}
        conf = self.asr._estimate_confidence(result)
        assert 0 <= conf <= 1


# ── Intent Tests ──────────────────────────────────────────────────────────────
class TestIntentClassifier:
    def setup_method(self):
        from app.models.intent_classifier import IntentClassifier
        self.clf = IntentClassifier()
        # Train with small dataset
        training_data = [
            {"text": "where is my order", "intent": "order_status"},
            {"text": "i want a refund", "intent": "refund_request"},
            {"text": "cancel my subscription", "intent": "subscription_management"},
            {"text": "i cant login", "intent": "technical_support"},
            {"text": "update payment method", "intent": "billing_inquiry"},
            {"text": "help me", "intent": "general_inquiry"},
            {"text": "track my package", "intent": "order_status"},
            {"text": "get my money back", "intent": "refund_request"},
            {"text": "change my plan", "intent": "subscription_management"},
            {"text": "app not working", "intent": "technical_support"},
        ]
        self.clf.train(training_data)

    def test_predict_returns_dict(self):
        result = self.clf.predict("where is my order?")
        assert isinstance(result, dict)
        assert "intent" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    def test_empty_input_returns_unknown(self):
        result = self.clf.predict("")
        assert result["intent"] == "unknown"

    def test_all_scores_sum_to_one(self):
        result = self.clf.predict("I need help with my bill")
        scores = result.get("all_scores", {})
        if scores:
            total = sum(scores.values())
            assert abs(total - 1.0) < 0.01

    def test_evaluate_metrics(self):
        test_data = [
            {"text": "where is my order", "intent": "order_status"},
            {"text": "i want a refund", "intent": "refund_request"},
        ]
        metrics = self.clf.evaluate(test_data)
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_predict_known_intents(self):
        test_cases = [
            ("track my order", ["order_status"]),
            ("I need a refund", ["refund_request"]),
            ("app keeps crashing", ["technical_support"]),
        ]
        for text, expected_intents in test_cases:
            result = self.clf.predict(text)
            assert result["intent"] in expected_intents or result["intent"] in [
                "unknown", "general_inquiry"
            ], f"Unexpected intent '{result['intent']}' for '{text}'"


# ── Response Tests ─────────────────────────────────────────────────────────────
class TestResponseService:
    def setup_method(self):
        from app.services.response_service import ResponseService
        self.svc = ResponseService()

    def test_generate_returns_text(self):
        result = self.svc.generate("order_status")
        assert isinstance(result, dict)
        assert "text" in result
        assert len(result["text"]) > 10

    def test_unknown_intent_handled(self):
        result = self.svc.generate("completely_unknown_intent_xyz")
        assert "text" in result
        assert result["text"]

    def test_all_intents_have_responses(self):
        intents = [
            "order_status", "order_cancellation", "refund_request",
            "subscription_management", "technical_support", "billing_inquiry",
            "account_management", "product_inquiry", "shipping_delivery",
            "complaint", "general_inquiry", "unknown",
        ]
        for intent in intents:
            result = self.svc.generate(intent)
            assert result["text"], f"No response for intent: {intent}"


# ── TTS Tests ─────────────────────────────────────────────────────────────────
class TestTTSService:
    def setup_method(self):
        from app.services.tts_service import TTSService
        self.tts = TTSService()

    def test_synthesize_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out_path = f.name
        result = self.tts.synthesize("Hello, how can I help you?", output_path=out_path)
        assert os.path.exists(out_path)
        assert result["file_size_bytes"] > 0
        os.unlink(out_path)

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            self.tts.synthesize("")

    def test_result_has_required_fields(self):
        result = self.tts.synthesize("Test synthesis")
        assert "audio_path" in result
        assert "duration_ms" in result
        assert "engine" in result
        if os.path.exists(result["audio_path"]):
            os.unlink(result["audio_path"])


# ── Audio Utils Tests ─────────────────────────────────────────────────────────
class TestAudioUtils:
    def test_validate_valid_wav(self):
        from app.utils.audio_utils import validate_audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            create_test_wav(f.name, duration_seconds=2.0)
            valid, msg = validate_audio(f.name)
        assert valid, msg
        os.unlink(f.name)

    def test_validate_missing_file(self):
        from app.utils.audio_utils import validate_audio
        valid, msg = validate_audio("/nonexistent.wav")
        assert not valid

    def test_validate_too_long(self):
        from app.utils.audio_utils import validate_audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            create_test_wav(f.name, duration_seconds=5.0)
            valid, msg = validate_audio(f.name, max_duration_seconds=2)
        assert not valid
        os.unlink(f.name)

    def test_get_duration(self):
        from app.utils.audio_utils import get_audio_duration
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            create_test_wav(f.name, duration_seconds=3.0)
            dur = get_audio_duration(f.name)
        assert abs(dur - 3.0) < 0.1
        os.unlink(f.name)


# ── WER Tests ─────────────────────────────────────────────────────────────────
class TestWER:
    def setup_method(self):
        from app.services.asr_service import ASRService
        self.asr = ASRService()

    @pytest.mark.parametrize("ref,hyp,expected", [
        ("hello world", "hello world", 0.0),
        ("hello world", "hello", 0.5),
        ("hello world", "goodbye world", 0.5),
        ("a b c d", "a b c d", 0.0),
        ("", "hello", 1.0),
    ])
    def test_wer_values(self, ref, hyp, expected):
        wer = self.asr._compute_wer(ref, hyp)
        assert abs(wer - expected) < 0.01, f"WER({ref!r}, {hyp!r}) = {wer}, expected {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
