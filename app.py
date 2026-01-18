import os
import json
from datetime import datetime
import yaml
import joblib
import pandas as pd
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

from feature_engineering.conversation_features import ci_to_churn_features
from aggregation.conversation_aggregation import aggregate_features

# ======================================================
# Config
# ======================================================

RULE_CONFIDENCE_DECAY = {
    "sarcasm_override": 0.15,
    "intent_sentiment_conflict": 0.12,
    "angry_neutral_override": 0.10,
    "complaint_neutral_override": 0.08,
    "outcome_override": 0.07,
    "neutral_phrase": 0.05,
}

DEFAULT_RULE_DECAY = 0.06
MIN_CONFIDENCE = 0.60

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rule_hits.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

# ======================================================
# App
# ======================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Conversation Intelligence + Churn API")

# ======================================================
# Load CI models
# ======================================================

emotion_tokenizer = XLMRobertaTokenizer.from_pretrained("models/emotion_xlm_roberta_final")
emotion_model = XLMRobertaForSequenceClassification.from_pretrained(
    "models/emotion_xlm_roberta_final"
).to(device).eval()

sentiment_tokenizer = XLMRobertaTokenizer.from_pretrained("models/sentiment_xlm_roberta_final")
sentiment_model = XLMRobertaForSequenceClassification.from_pretrained(
    "models/sentiment_xlm_roberta_final"
).to(device).eval()

intent_tokenizer = XLMRobertaTokenizer.from_pretrained("models/intent_xlm_roberta_final")
intent_model = XLMRobertaForSequenceClassification.from_pretrained(
    "models/intent_xlm_roberta_final"
).to(device).eval()

# ======================================================
# Load churn model
# ======================================================

churn_model = joblib.load("churn_model_real.joblib")

with open("churn_features.json") as f:
    CHURN_FEATURES = json.load(f)

# ======================================================
# Load rules
# ======================================================

with open("rules.yaml", "r", encoding="utf-8") as f:
    RULES_CONFIG = yaml.safe_load(f)

RULES_VERSION = RULES_CONFIG.get("rules_version", "unknown")
NEUTRAL_PHRASES = RULES_CONFIG.get("neutral_phrases", [])
OUTCOME_PHRASES = RULES_CONFIG.get("outcome_phrases", [])
SARCASM_PHRASES = RULES_CONFIG.get("sarcasm_phrases", [])
DYNAMIC_RULES = RULES_CONFIG.get("rules", [])

# ======================================================
# In-memory storage (demo)
# ======================================================

customer_logs = {}

# ======================================================
# Schemas
# ======================================================

class TextRequest(BaseModel):
    customer_id: str
    text: str

# ======================================================
# Helpers
# ======================================================

def log_rule_hit(payload: dict):
    payload["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_model(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_id = torch.argmax(probs).item()

    return {
        "label": model.config.id2label[pred_id],
        "confidence": min(float(probs[pred_id]), 0.99)
    }


def apply_consistency_rules(text, emotion, sentiment, intent):
    rules_triggered = []
    emotion_label = emotion["label"]
    sentiment_label = sentiment["label"]
    intent_label = intent["label"]
    text_lower = text.lower()

    if any(p in text_lower for p in NEUTRAL_PHRASES):
        sentiment_label = "Neutral"
        rules_triggered.append("neutral_phrase")

    if any(p in text_lower for p in OUTCOME_PHRASES) and emotion_label in ["Sad", "Calm"]:
        sentiment_label = "Neutral"
        rules_triggered.append("outcome_override")

    if sentiment_label == "Positive" and any(p in text_lower for p in SARCASM_PHRASES):
        sentiment_label = "Negative"
        rules_triggered.append("sarcasm_override")

    for rule in DYNAMIC_RULES:
        cond = rule.get("if", {})
        then = rule.get("then", {})

        if (
            (cond.get("emotion") in [None, emotion_label]) and
            (cond.get("sentiment") in [None, sentiment_label]) and
            (cond.get("intent") in [None, intent_label])
        ):
            sentiment_label = then.get("sentiment", sentiment_label)
            intent_label = then.get("intent", intent_label)
            rules_triggered.append(rule["name"])

    base_conf = sentiment["confidence"]
    total_decay = sum(RULE_CONFIDENCE_DECAY.get(r, DEFAULT_RULE_DECAY) for r in rules_triggered)
    final_conf = max(round(base_conf - total_decay, 2), MIN_CONFIDENCE)

    return {
        "emotion": emotion,
        "sentiment": {
            "label": sentiment_label,
            "confidence": final_conf,
            "overridden": len(rules_triggered) > 0
        },
        "intent": intent,
        "rules_triggered": rules_triggered,
        "rules_version": RULES_VERSION
    }

def score_churn(features: dict):
    full = {}

    for f in CHURN_FEATURES:
        full[f] = features.get(f, 0.0)   # default 0 if missing

    X = pd.DataFrame([full])
    return float(churn_model.predict_proba(X)[0, 1])

# ======================================================
# Routes
# ======================================================

@app.post("/analyze")
def analyze(req: TextRequest):
    emotion = run_model(emotion_tokenizer, emotion_model, req.text)
    sentiment = run_model(sentiment_tokenizer, sentiment_model, req.text)
    intent = run_model(intent_tokenizer, intent_model, req.text)

    ci_output = apply_consistency_rules(req.text, emotion, sentiment, intent)

    features = ci_to_churn_features(ci_output)

    customer_logs.setdefault(req.customer_id, []).append(features)

    log_rule_hit({
        "customer_id": req.customer_id,
        "text": req.text,
        "rules_triggered": ci_output["rules_triggered"]
    })

    return ci_output


@app.get("/churn/{customer_id}")
def churn_score(customer_id: str):
    if customer_id not in customer_logs:
        return {"error": "No messages found for customer"}

    agg = aggregate_features(customer_logs[customer_id])
    churn_prob = score_churn(agg)

    return {
        "customer_id": customer_id,
        "churn_risk": churn_prob,
        "risk_level": (
            "High" if churn_prob > 0.7 else
            "Medium" if churn_prob > 0.4 else
            "Low"
        )
    }
