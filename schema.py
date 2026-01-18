"""
Schema definitions for Conversation Intelligence pipeline.
These schemas are CONTRACTS and must not change casually.
"""

CI_SCHEMA_VERSION = "v1.0.0"

CI_REQUIRED_KEYS = {
    "emotion": ["label", "confidence"],
    "sentiment": ["label", "confidence", "overridden"],
    "intent": ["label", "confidence"],
    "priority": None,
    "customer_state": None,
    "rules_triggered": None,
    "rules_version": None,
}

FEATURE_MAPPING_KEYS = [
    "sentiment_positive",
    "sentiment_neutral",
    "sentiment_negative",
    "sentiment_confidence",
    "sentiment_overridden",

    "emotion_happy",
    "emotion_sad",
    "emotion_angry",
    "emotion_calm",
    "emotion_neutral",
    "emotion_confidence",

    "intent_inquiry",
    "intent_complaint",
    "intent_appreciation",
    "intent_feedback",
    "intent_request",
    "intent_confidence",

    "customer_at_risk",
    "customer_stable",
    "priority_high",

    "rule_count",
    "sarcasm_flag",
    "neutral_phrase_flag",
    "outcome_softened_flag",

    "strong_negative_signal",
    "emotional_distress",
    "passive_dissatisfaction",
]

AGGREGATION_KEYS = [
    "neg_ratio",
    "neutral_ratio",
    "positive_ratio",
    "sad_ratio",
    "angry_ratio",
    "complaint_ratio",
    "inquiry_ratio",
    "at_risk_ratio",

    "message_count",
    "sarcasm_count",
    "rule_trigger_count",

    "avg_sentiment_confidence",
    "avg_emotion_confidence",

    
    "strong_negative_ratio",

    "last_customer_at_risk",
    "last_sentiment_negative",
    "last_emotion_sad",
]
