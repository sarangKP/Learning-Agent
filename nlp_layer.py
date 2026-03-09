"""
Layer 1 — NLP Signal Extraction

Outputs:
  sentiment_score  : float  -1.0 to +1.0   (VADER compound on last user message)
  repetition_score : float   0.0 to  1.0   (TF-IDF cosine sim between last two user msgs)
  confusion_score  : float   0.0 to  1.0   (cognitive confusion keyword detector)
  sadness_score    : float   0.0 to  1.0   (loneliness / sadness keyword detector)

Why four signals?
  VADER misses polite elderly phrasing. "I don't understand" and "I feel
  lonely" both score near zero on VADER. Keyword detectors catch them directly
  and prevent misclassification (e.g. loneliness being labelled as confusion).
"""

from __future__ import annotations
from typing import List, Tuple
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_analyser = SentimentIntensityAnalyzer()

# ── Confusion / frustration keyword patterns ─────────────────────────────────
_CONFUSION_PATTERNS = [
    # Direct confusion
    (r"\bdon'?t understand\b",                      0.7),
    (r"\bcan'?t follow\b",                          0.7),
    (r"\bnot making sense\b",                       0.7),
    (r"\bconfus(ed|ing)\b",                         0.6),
    (r"\bsay (it |that )?(again|once more|slowly)\b", 0.5),
    (r"\bmore simply\b",                            0.6),
    (r"\bsimpler\b",                                0.5),
    (r"\bwhat do you mean\b",                       0.5),
    (r"\bi don'?t (get|follow|know what)\b",        0.6),
    (r"\btoo complicated\b",                        0.7),
    (r"\bnothing (you say|helps|works|is helping)\b", 0.7),
    (r"\bhard to (understand|follow)\b",            0.6),
    (r"\bwhat (are you|did you) (say|mean|talking)\b", 0.5),
    # Frustration / repetition complaint
    (r"\balready (told|asked|said)\b",              0.8),
    (r"\byou never (remember|listen)\b",            0.8),
    (r"\bstop (repeating|saying)\b",                0.7),
    (r"\bkeep (asking|saying|repeating)\b",         0.7),
    (r"\bmaking me (upset|angry|frustrated)\b",     0.9),
    (r"\bvery upset\b",                             0.8),
]

# ── Sadness / loneliness keyword patterns ─────────────────────────────────────
# These are SEPARATE from confusion — sadness needs empathy, not clarity changes.
_SADNESS_PATTERNS = [
    (r"\blon(ely|eliness)\b",                       0.8),
    (r"\bmiss(ing)? (my|him|her|them|you)\b",       0.7),
    (r"\bnobody (calls?|visits?|comes?)\b",         0.8),
    (r"\bhadn'?t called\b",                         0.6),
    (r"\bhaven'?t (called|visited|come)\b",         0.6),
    (r"\bfeel(ing)? (sad|down|low|blue|empty)\b",   0.8),
    (r"\bno one (cares?|calls?|visits?)\b",         0.8),
    (r"\bmiss(ed)? (him|her|them|my)\b",            0.7),
    (r"\bwish (he|she|they) (was|were|would)\b",    0.6),
    (r"\bgriev(e|ing|ed)\b",                        0.8),
    (r"\bdepress(ed|ing)\b",                        0.8),
    (r"\bcry(ing)?\b",                              0.7),
    (r"\ball alone\b",                              0.9),
    (r"\bnobody (here|around|with me)\b",           0.8),
]

_CONFUSION_COMPILED = [(re.compile(p, re.IGNORECASE), w) for p, w in _CONFUSION_PATTERNS]
_SADNESS_COMPILED   = [(re.compile(p, re.IGNORECASE), w) for p, w in _SADNESS_PATTERNS]


def _keyword_score(text: str, patterns: list) -> float:
    total = 0.0
    for pattern, weight in patterns:
        if pattern.search(text):
            total += weight
    return min(1.0, total)


def confusion_keyword_score(text: str) -> float:
    return _keyword_score(text, _CONFUSION_COMPILED)


def sadness_keyword_score(text: str) -> float:
    return _keyword_score(text, _SADNESS_COMPILED)


def extract_signals(turns: list) -> Tuple[float, float, float, float]:
    """
    Returns (sentiment_score, repetition_score, confusion_score, sadness_score).
    """
    user_texts = [t.text for t in turns if t.role == "user"]

    if user_texts:
        sentiment_score = _analyser.polarity_scores(user_texts[-1])["compound"]
    else:
        sentiment_score = 0.0

    if len(user_texts) >= 2:
        try:
            vec = TfidfVectorizer(min_df=1, stop_words=None)
            tfidf = vec.fit_transform([user_texts[-2], user_texts[-1]])
            repetition_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except ValueError:
            repetition_score = 0.0
    else:
        repetition_score = 0.0

    last = user_texts[-1] if user_texts else ""
    confusion_score = confusion_keyword_score(last)
    sadness_score   = sadness_keyword_score(last)

    return sentiment_score, repetition_score, confusion_score, sadness_score