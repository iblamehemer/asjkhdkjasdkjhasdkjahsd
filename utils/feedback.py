"""
utils/feedback.py
BrandSphere AI — Feedback Collection & Sentiment Analysis
"""

import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime


FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "..", "datasets", "cleaned", "feedback_data.csv")

FEEDBACK_COLUMNS = [
    "session_id", "timestamp", "module", "star_rating",
    "comment", "sentiment", "polarity_score"
]


def get_sentiment(text: str) -> tuple[str, float]:
    """Analyze sentiment using TextBlob if available, else simple rule-based."""
    try:
        from textblob import TextBlob
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
    except ImportError:
        positive_words = {"great", "excellent", "love", "perfect", "amazing", "awesome", "good", "helpful"}
        negative_words = {"bad", "poor", "terrible", "wrong", "hate", "useless", "disappointing"}
        words = set(text.lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        polarity = (pos - neg) / max(pos + neg, 1) * 0.5

    if polarity > 0.1:    return "positive", round(polarity, 3)
    elif polarity < -0.1: return "negative", round(polarity, 3)
    else:                 return "neutral",  round(polarity, 3)


def save_feedback(module: str, star_rating: int, comment: str, session_id: str = None) -> dict:
    """Save a feedback entry to local CSV storage."""
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]

    sentiment, polarity = get_sentiment(comment)
    record = {
        "session_id":    session_id,
        "timestamp":     datetime.now().isoformat(),
        "module":        module,
        "star_rating":   star_rating,
        "comment":       comment,
        "sentiment":     sentiment,
        "polarity_score": polarity,
    }

    # Load or create DataFrame
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
    else:
        df = pd.DataFrame(columns=FEEDBACK_COLUMNS)

    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    df.to_csv(FEEDBACK_FILE, index=False)
    return record


def load_feedback() -> pd.DataFrame:
    """Load all feedback records."""
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    # Return sample data if no real feedback yet
    np.random.seed(42)
    N = 50
    modules  = ["logo_studio", "content_hub", "campaign_studio", "aesthetics_engine"]
    comments = [
        "Great logo style suggestions!", "Taglines were very creative.",
        "Campaign predictions accurate.", "Color palette was perfect.",
        "Font recommendations helpful.", "Loved the multilingual feature.",
        "ROI estimate was a bit high.", "Animated GIF was awesome!",
        "Brand story could be longer.", "Very useful overall!"
    ]
    return pd.DataFrame({
        "session_id":    [f"sess_{i:03d}" for i in range(N)],
        "timestamp":     pd.date_range("2025-11-01", periods=N, freq="6h"),
        "module":        np.random.choice(modules, N),
        "star_rating":   np.random.choice([3, 4, 5], N, p=[0.2, 0.5, 0.3]),
        "comment":       np.random.choice(comments, N),
        "sentiment":     np.random.choice(["positive", "neutral"], N, p=[0.7, 0.3]),
        "polarity_score": np.random.uniform(0.0, 0.6, N),
    })


def get_feedback_summary() -> dict:
    """Return aggregated feedback stats for dashboard."""
    df = load_feedback()
    if len(df) == 0:
        return {"total": 0, "avg_rating": 0, "by_module": {}, "sentiment_counts": {}}

    return {
        "total":            len(df),
        "avg_rating":       round(df["star_rating"].mean(), 2),
        "by_module":        df.groupby("module")["star_rating"].mean().round(2).to_dict(),
        "sentiment_counts": df["sentiment"].value_counts().to_dict(),
        "rating_dist":      df["star_rating"].value_counts().sort_index().to_dict(),
        "recent_comments":  df.tail(5)["comment"].tolist(),
    }
