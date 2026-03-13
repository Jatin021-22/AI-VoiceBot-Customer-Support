"""Response Generation Service — intent-mapped with random template selection."""
import json, random
from typing import Dict, Optional
from app.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("voicebot.response")

class ResponseService:
    def __init__(self):
        self.responses = {}
        self._load_responses()

    def _load_responses(self):
        try:
            with open(settings.RESPONSE_CONFIG_PATH) as f:
                self.responses = json.load(f)
            logger.info(f"Loaded response templates for {len(self.responses)} intents")
        except Exception as e:
            logger.error(f"Failed to load response config: {e}")
            self.responses = self._default_responses()

    def generate(self, intent: str, context: Dict = None) -> Dict:
        """Generate a response for the given intent."""
        intent_data = self.responses.get(intent) or self.responses.get("unknown")
        templates = intent_data.get("templates", [])
        template = random.choice(templates) if templates else "I'm here to help you."
        follow_up = intent_data.get("follow_up", "")
        response_text = template
        if follow_up and random.random() > 0.4:
            response_text = f"{template} {follow_up}"
        return {
            "text": response_text,
            "intent": intent,
            "template_used": template,
            "strategy": settings.RESPONSE_STRATEGY,
        }

    def _default_responses(self) -> Dict:
        return {
            "unknown": {
                "templates": ["I'm sorry, I didn't understand that. Could you rephrase?"],
                "follow_up": "I can help with orders, billing, subscriptions, and technical support.",
            }
        }
