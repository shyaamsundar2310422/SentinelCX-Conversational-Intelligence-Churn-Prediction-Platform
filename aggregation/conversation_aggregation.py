"""
Conversation Feature Aggregation Layer

Aggregates message-level churn features into
customer-level features over a time window.
Safe for API usage (never crashes).
"""

from typing import List, Dict
import numpy as np

# -----------------------
# Defaults (CRITICAL)
# -----------------------

DEFAULT_AGG = {
    "neg_ratio": 0.0,
    "neutral_ratio": 0.0,
    "positive_ratio": 0.0,
    "sad_ratio": 0.0,
    "angry_ratio": 0.0,
    "complaint_ratio": 0.0,
    "inquiry_ratio": 0.0,
    "at_risk_ratio": 0.0,
    "message_count": 0,
    "sarcasm_count": 0,
    "rule_trigger_count": 0,
    "avg_sentiment_confidence": 0.0,
    "avg_emotion_confidence": 0.0,
    "strong_negative_ratio": 0.0,
    "last_customer_at_risk": 0.0,
    "last_sentiment_negative": 0.0,
    "last_emotion_sad": 0.0
}

# -----------------------
# Aggregation
# -----------------------

def aggregate_features(feature_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not feature_rows:
        return DEFAULT_AGG.copy()

    agg = DEFAULT_AGG.copy()
    keys = feature_rows[0].keys()

    # dict of lists (safe)
    values = {k: [row.get(k, 0.0) for row in feature_rows] for k in keys}

    # -----------------------
    # Ratios
    # -----------------------
    agg["neg_ratio"] = float(np.mean(values.get("sentiment_negative", [0])))
    agg["neutral_ratio"] = float(np.mean(values.get("sentiment_neutral", [0])))
    agg["positive_ratio"] = float(np.mean(values.get("sentiment_positive", [0])))

    agg["sad_ratio"] = float(np.mean(values.get("emotion_sad", [0])))
    agg["angry_ratio"] = float(np.mean(values.get("emotion_angry", [0])))

    agg["complaint_ratio"] = float(np.mean(values.get("intent_complaint", [0])))
    agg["inquiry_ratio"] = float(np.mean(values.get("intent_inquiry", [0])))

    agg["at_risk_ratio"] = float(np.mean(values.get("customer_at_risk", [0])))

    # -----------------------
    # Counts
    # -----------------------
    agg["message_count"] = len(feature_rows)
    agg["sarcasm_count"] = int(sum(values.get("sarcasm_flag", [0])))
    agg["rule_trigger_count"] = int(sum(values.get("rule_count", [0])))

    # -----------------------
    # Confidence trends
    # -----------------------
    agg["avg_sentiment_confidence"] = float(np.mean(values.get("sentiment_confidence", [0])))
    agg["avg_emotion_confidence"] = float(np.mean(values.get("emotion_confidence", [0])))

    # -----------------------
    # Composite
    # -----------------------
    agg["strong_negative_ratio"] = float(np.mean(values.get("strong_negative_signal", [0])))

    # -----------------------
    # Last known state
    # -----------------------
    agg["last_customer_at_risk"] = values.get("customer_at_risk", [0])[-1]
    agg["last_sentiment_negative"] = values.get("sentiment_negative", [0])[-1]
    agg["last_emotion_sad"] = values.get("emotion_sad", [0])[-1]

    return agg
