from aggregation.conversation_aggregation import aggregate_features

sample_features = [
    {
        "sentiment_positive": 0,
        "sentiment_neutral": 1,
        "sentiment_negative": 0,
        "sentiment_confidence": 0.88,
        "emotion_sad": 1,
        "emotion_angry": 0,
        "emotion_confidence": 0.55,
        "intent_complaint": 0,
        "intent_inquiry": 1,
        "customer_at_risk": 0,
        "sarcasm_flag": 0,
        "rule_count": 2,
        "emotional_distress": 1,
        "strong_negative_signal": 0
    },
    {
        "sentiment_positive": 0,
        "sentiment_neutral": 1,
        "sentiment_negative": 0,
        "sentiment_confidence": 0.82,
        "emotion_sad": 0,
        "emotion_angry": 0,
        "emotion_confidence": 0.40,
        "intent_complaint": 0,
        "intent_inquiry": 1,
        "customer_at_risk": 0,
        "sarcasm_flag": 0,
        "rule_count": 1,
        "emotional_distress": 0,
        "strong_negative_signal": 0
    }
]

agg = aggregate_features(sample_features)

for k, v in agg.items():
    print(f"{k}: {v}")
