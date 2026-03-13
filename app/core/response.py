import random
from pathlib import Path


RESPONSES = {
    "order_status": [
        "I can help you track your order. Please provide your order number and I'll look it up right away.",
        "Let me check your order status. Could you share your order ID or the email used for the purchase?",
        "I'd be happy to check on your order. Please provide your order number to get the latest update.",
    ],
    "order_cancellation": [
        "I can help you cancel your order. Orders can be cancelled within 24 hours of placement. Please share your order number.",
        "I'll assist with the cancellation. Could you provide your order ID so I can check if it's eligible?",
        "No problem, I can process the cancellation. Please share your order number and reason for cancellation.",
    ],
    "refund_request": [
        "I'm sorry to hear that. Refunds are processed within 5-7 business days. Please provide your order number to get started.",
        "I can help with your refund request. Could you share your order number and the reason for the refund?",
        "I'll initiate the refund process. Please provide your order ID and I'll get this resolved for you.",
    ],
    "subscription_management": [
        "I can help you manage your subscription. Are you looking to upgrade, downgrade, pause, or cancel?",
        "Let me assist with your subscription. Please describe what changes you'd like to make to your plan.",
        "I'll help you with your subscription settings. Could you tell me more about what you need?",
    ],
    "technical_support": [
        "I'm sorry you're experiencing technical issues. Could you describe the problem and any error messages you see?",
        "Let me help troubleshoot this. What device and browser are you using, and what exactly is happening?",
        "I can help resolve this technical issue. Please describe the problem in detail so I can assist you better.",
    ],
    "billing_inquiry": [
        "I can help clarify your billing. Could you tell me which charge or invoice you're asking about?",
        "Let me look into your billing details. Please share the transaction date or amount in question.",
        "I'll investigate this billing issue. Could you provide your account email and the charge details?",
    ],
    "account_management": [
        "I can help you update your account. What information would you like to change?",
        "Let me assist with your account. For security, I'll need to verify your identity first. Please confirm your email.",
        "I'll help you manage your account settings. What specifically would you like to update or change?",
    ],
    "product_inquiry": [
        "I'd be happy to tell you more about our products. Which product or feature are you interested in?",
        "Great question! Could you tell me which specific product or plan you'd like to know more about?",
        "I can provide detailed product information. What would you like to know — pricing, features, or availability?",
    ],
    "shipping_delivery": [
        "I can help with shipping information. Please share your order number and I'll get the latest delivery update.",
        "Let me check on your delivery. Could you provide your order number or tracking ID?",
        "I'll look into your shipment details. Please provide your order number for the latest shipping status.",
    ],
    "complaint": [
        "I sincerely apologize for your experience. Your feedback is important to us. Could you describe the issue so I can escalate it?",
        "I'm truly sorry to hear this. Please describe what happened and I'll make sure this gets resolved immediately.",
        "I apologize for the inconvenience. I want to make this right for you. Could you share more details about the problem?",
    ],
    "general_inquiry": [
        "I'm here to help! Could you please provide more details about your question or concern?",
        "Thank you for reaching out. To better assist you, could you elaborate on what you need help with?",
        "I'd be happy to assist. Could you be more specific about what you're looking for today?",
    ],
}

FOLLOW_UPS = {
    "order_status": "Is there anything else I can help you with regarding your order?",
    "order_cancellation": "Would you like me to process a refund to your original payment method?",
    "refund_request": "You'll receive a confirmation email once the refund is processed.",
    "subscription_management": "Would you like me to send you a summary of your current subscription?",
    "technical_support": "If the issue persists, I can escalate this to our technical team.",
    "billing_inquiry": "Would you like me to email you a detailed breakdown of the charges?",
    "account_management": "Changes to sensitive information may require email verification.",
    "product_inquiry": "Would you like me to connect you with a product specialist for more details?",
    "shipping_delivery": "You can also track your package on the courier's website using your tracking number.",
    "complaint": "Would you prefer a replacement, a refund, or store credit?",
    "general_inquiry": "Is there anything else I can help you with today?",
}


class ResponseGenerator:
    def generate(self, intent: str, confidence: float,
                 entities: dict = None, use_follow_up: bool = False) -> dict:

        if confidence < 0.1:
            return {
                "response_text": "I'm not entirely sure I understood your request. Could you please rephrase or provide more details?",
                "intent": intent,
                "intent_display_name": intent,
                "confidence": confidence,
                "follow_up": None,
                "is_fallback": True,
            }

        responses = RESPONSES.get(intent, RESPONSES["general_inquiry"])
        response_text = random.choice(responses)

        if entities:
            for key, value in entities.items():
                response_text = response_text.replace(f"{{{key}}}", str(value))

        follow_up = FOLLOW_UPS.get(intent) if use_follow_up else None

        return {
            "response_text": response_text,
            "intent": intent,
            "intent_display_name": intent.replace("_", " ").title(),
            "confidence": confidence,
            "follow_up": follow_up,
            "is_fallback": False,
        }

    def get_all_intents(self) -> dict:
        return {
            intent: {
                "display_name": intent.replace("_", " ").title(),
                "num_responses": len(responses),
            }
            for intent, responses in RESPONSES.items()
        }

    def get_error_response(self, error_type: str = "system_error") -> str:
        messages = {
            "audio_error": "I had trouble processing the audio. Please speak clearly and try again.",
            "system_error": "I'm experiencing technical difficulties. Please try again in a moment.",
            "low_confidence": "I'm not sure I understood that. Could you please rephrase?",
        }
        return messages.get(error_type, "I'm here to help. Could you clarify your request?")


_generator_instance = None

def get_response_generator() -> ResponseGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ResponseGenerator()
    return _generator_instance