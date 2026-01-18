from feature_engineering.conversation_features import ci_to_churn_features

sample_ci = {
    "emotion": {"label": "Sad", "confidence": 0.55},
    "sentiment": {
        "label": "Neutral",
        "confidence": 0.88,
        "overridden": True
    },
    "intent": {"label": "Inquiry", "confidence": 0.57},
    "priority": "Normal",
    "customer_state": "Stable",
    "rules_triggered": ["neutral_phrase", "outcome_sad_neutral"]
}

features = ci_to_churn_features(sample_ci)

for k, v in features.items():
    print(f"{k}: {v}")
