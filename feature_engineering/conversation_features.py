"""
Conversation Intelligence → Churn Feature Mapping

This module converts Conversation Intelligence API outputs
into numeric, ML-ready churn features.

Design principles:
- Deterministic
- Explainable
- Model-agnostic
"""

from typing import Dict, Any


def ci_to_churn_features(ci: Dict[str, Any]) -> Dict[str, float]:
    features = {}

    # ----------------------
    # Sentiment features
    # ----------------------
    sentiment_label = ci["sentiment"]["label"].lower()
    sentiment_conf = ci["sentiment"]["confidence"]

    features["sentiment_positive"] = float(sentiment_label == "positive")
    features["sentiment_neutral"] = float(sentiment_label == "neutral")
    features["sentiment_negative"] = float(sentiment_label == "negative")
    features["sentiment_confidence"] = sentiment_conf
    features["sentiment_overridden"] = float(ci["sentiment"].get("overridden", False))

    # ----------------------
    # Emotion features
    # ----------------------
    emotion_label = ci["emotion"]["label"].lower()
    emotion_conf = ci["emotion"]["confidence"]

    for emotion in ["happy", "sad", "angry", "calm", "neutral"]:
        features[f"emotion_{emotion}"] = float(emotion_label == emotion)

    features["emotion_confidence"] = emotion_conf

    # ----------------------
    # Intent features
    # ----------------------
    intent_label = ci["intent"]["label"].lower()
    intent_conf = ci["intent"]["confidence"]

    for intent in ["inquiry", "complaint", "appreciation", "feedback", "request"]:
        features[f"intent_{intent}"] = float(intent_label == intent)

    features["intent_confidence"] = intent_conf

    # ----------------------
    # Customer state & priority
    # ----------------------
    customer_state = ci.get("customer_state", "").lower()
    priority = ci.get("priority", "").lower()

    features["customer_at_risk"] = float(customer_state == "at risk")
    features["customer_stable"] = float(customer_state == "stable")
    features["priority_high"] = float(priority == "high")

    # ----------------------
    # Rule-based signals
    # ----------------------
    rules = ci.get("rules_triggered", [])

    features["rule_count"] = float(len(rules))
    features["sarcasm_flag"] = float("sarcasm_override" in rules)
    features["neutral_phrase_flag"] = float("neutral_phrase" in rules)
    features["outcome_softened_flag"] = float("outcome_sad_neutral" in rules)

    # ----------------------
    # Composite churn indicators
    # ----------------------
    features["strong_negative_signal"] = float(
        sentiment_label == "negative" and sentiment_conf > 0.75
    )

    features["emotional_distress"] = float(
        emotion_label in ["sad", "angry"] and emotion_conf > 0.5
    )

    features["passive_dissatisfaction"] = float(
        features["sarcasm_flag"] == 1.0 and sentiment_label == "negative"
    )

    return features


from schema import FEATURE_MAPPING_KEYS


def validate_feature_mapping(features: dict) -> None:
    for key in FEATURE_MAPPING_KEYS:
        assert key in features, f"Missing feature: {key}"
