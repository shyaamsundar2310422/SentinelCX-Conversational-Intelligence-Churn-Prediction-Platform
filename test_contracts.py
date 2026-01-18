from feature_engineering.conversation_features import ci_to_churn_features
from aggregation.conversation_aggregation import aggregate_features
from validators import validate_ci_output
from schema import CI_SCHEMA_VERSION


# Sample frozen CI output
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
    "rules_triggered": ["neutral_phrase", "outcome_sad_neutral"],
    "rules_version": CI_SCHEMA_VERSION
}


# 1️⃣ Validate CI contract
validate_ci_output(sample_ci)

# 2️⃣ Validate feature mapping
features = ci_to_churn_features(sample_ci)
from feature_engineering.conversation_features import validate_feature_mapping
validate_feature_mapping(features)

# 3️⃣ Validate aggregation
agg = aggregate_features([features, features])
from aggregation.conversation_aggregation import validate_aggregation_output
validate_aggregation_output(agg)

print("✅ Phase 1 contracts validated successfully")
