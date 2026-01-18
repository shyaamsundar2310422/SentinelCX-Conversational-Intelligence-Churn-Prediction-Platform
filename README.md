# SentinelCX – Conversational Intelligence & Churn Prediction Platform

A rule-augmented conversation intelligence system that goes beyond traditional sentiment analysis by combining emotion detection, sentiment classification, intent recognition, and explicit reasoning rules to produce explainable, production-ready outputs. The system is designed to serve as the intelligence layer for downstream business applications such as churn prediction, escalation, and customer retention.

---

## Overview

Most sentiment analysis systems rely on a single model prediction, which often fails in real-world conversations involving sarcasm, mixed signals, or neutral phrasing around negative outcomes.

SentinelCX addresses this by separating emotion, sentiment, and intent as independent signals and reconciling them using a transparent rule engine. This produces outputs that are auditable, explainable, and reliable for production use.

---

## Key Features

- Emotion classification using transformer-based models (XLM-RoBERTa)
- Sentiment analysis with rule-based overrides
- Intent detection independent of sentiment
- Sarcasm, contradiction, and outcome handling
- Rule-driven confidence adjustment
- Explicit confidence breakdowns for every decision
- Business-level signals (priority, customer state)
- Structured JSON outputs for downstream systems
- Designed to plug directly into churn and retention models

---

## System Architecture

```
User Text
   ↓
Emotion Model
   ↓
Sentiment Model (raw)
   ↓
Intent Model
   ↓
Rule Engine (sarcasm, conflicts, outcomes)
   ↓
Confidence Reconciliation
   ↓
Final Decision + Business Signals
   ↓
Churn & Retention Intelligence
```

---

## Example

### Input

```json
{
  "text": "Amazing service — waited an hour for cold food."
}
```

### Output

```json
{
  "emotion": { "label": "Happy", "confidence": 0.79 },
  "sentiment": { "label": "Negative", "confidence": 0.84, "overridden": true },
  "intent": { "label": "Appreciation", "confidence": 0.96 },
  "customer_state": "At Risk",
  "rules_triggered": ["sarcasm_override"]
}
```

This example shows how surface emotion, true sentiment, and business risk can differ within a single message — a key limitation of traditional sentiment systems.

---

## Design Philosophy

- Models provide signals, not final truth
- Rules are explicit, minimal, and auditable
- Emotion is preserved and not force-corrected
- Sentiment overrides are transparent
- Confidence values are adjusted, not replaced

This hybrid design balances machine learning flexibility with production control and explainability.

---

## Project Structure

```
.
├── app.py
├── rules.yaml
├── schema.py
├── aggregation/
├── feature_engineering/
├── models/
│   ├── emotion_xlm_roberta_final/
│   ├── sentiment_xlm_roberta_final/
│   └── intent_xlm_roberta_final/
├── logs/
├── test_feature_mapping.py
├── test_aggregation.py
└── test_logs.py
```

Model weights are intentionally excluded to keep the repository lightweight and reproducible.

---

## Use Cases

- Customer support triage
- Conversation analytics
- Sarcasm detection
- Risk and escalation systems
- Churn prediction and early retention
- Explainable AI demonstrations

---

## License

MIT License
