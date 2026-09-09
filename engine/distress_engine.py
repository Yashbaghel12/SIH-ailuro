"""
Distress Engine — base prototype
---------------------------------
Rule-based + lightweight keyword-sentiment scoring for the
AI-Powered Dynamic Mental Health Monitoring system (SIH 26094).

This is a deliberately dependency-free v0: no external NLP model
download needed, so it runs anywhere. Swap `score_text_sentiment`
for a real transformer/VADER model later without touching the
rest of the engine.
"""

from dataclasses import dataclass
from typing import Literal

RiskLevel = Literal["low", "medium", "high"]

# --- lightweight sentiment lexicon (v0 placeholder) -------------------
NEGATIVE_WORDS = {
    "scared", "afraid", "threat", "threatened", "unsafe", "hopeless",
    "alone", "isolated", "give up", "can't sleep", "cannot sleep",
    "anxious", "panic", "worthless", "no one helps", "exhausted",
    "trapped", "hiding", "followed", "intimidated",
}
POSITIVE_WORDS = {
    "better", "safe", "hopeful", "okay", "improving", "supported",
    "calm", "relieved", "progress", "sleeping better", "stronger",
}


def score_text_sentiment(text: str) -> float:
    """Returns a score in [-1, 1]. -1 = highly distressed language,
    +1 = positive/coping language. Naive keyword count, v0 only."""
    if not text:
        return 0.0
    lowered = text.lower()
    neg_hits = sum(1 for w in NEGATIVE_WORDS if w in lowered)
    pos_hits = sum(1 for w in POSITIVE_WORDS if w in lowered)
    total = neg_hits + pos_hits
    if total == 0:
        return 0.0
    return (pos_hits - neg_hits) / total


@dataclass
class CheckIn:
    mood_score: int          # 1 (very distressed) to 5 (very good)
    text: str                # free-text or transcribed voice note
    days_since_complaint: int
    missed_checkins: int      # consecutive missed check-ins
    threat_reported: bool = False


@dataclass
class DistressResult:
    score: float              # 0-100, higher = more distress
    risk_level: RiskLevel
    alert: bool
    reasons: list[str]


class DistressEngine:
    """Combines structured signals + text sentiment into one score.

    Weights are intentionally simple and tunable — this is a
    baseline to validate the pipeline end-to-end before swapping
    in a trained model.
    """

    def __init__(
        self,
        w_mood: float = 0.30,
        w_text: float = 0.30,
        w_delay: float = 0.15,
        w_missed: float = 0.15,
        threat_bonus: float = 10.0,
        medium_threshold: float = 30.0,
        high_threshold: float = 60.0,
        max_delay_days: int = 180,
        max_missed: int = 5,
    ):
        self.w_mood = w_mood
        self.w_text = w_text
        self.w_delay = w_delay
        self.w_missed = w_missed
        self.threat_bonus = threat_bonus
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.max_delay_days = max_delay_days
        self.max_missed = max_missed

    def score(self, c: CheckIn) -> DistressResult:
        reasons = []

        # mood: 1 (bad) -> 100, 5 (good) -> 0
        mood_component = ((5 - c.mood_score) / 4) * 100
        if c.mood_score <= 2:
            reasons.append("low self-reported mood")

        # text sentiment: -1 (bad) -> 100, +1 (good) -> 0
        sentiment = score_text_sentiment(c.text)
        text_component = ((1 - sentiment) / 2) * 100
        if sentiment < -0.3:
            reasons.append("distress language in check-in text")

        # days since complaint, capped and normalized
        delay_ratio = min(c.days_since_complaint / self.max_delay_days, 1.0)
        delay_component = delay_ratio * 100
        if delay_ratio > 0.6:
            reasons.append("prolonged case pendency")

        # missed check-ins, capped and normalized
        missed_ratio = min(c.missed_checkins / self.max_missed, 1.0)
        missed_component = missed_ratio * 100
        if c.missed_checkins >= 2:
            reasons.append("repeated missed check-ins")

        weighted = (
            self.w_mood * mood_component
            + self.w_text * text_component
            + self.w_delay * delay_component
            + self.w_missed * missed_component
        )

        if c.threat_reported:
            weighted += self.threat_bonus
            reasons.append("threat reported")

        final_score = min(weighted, 100.0)

        if final_score >= self.high_threshold or c.threat_reported:
            risk: RiskLevel = "high"
        elif final_score >= self.medium_threshold:
            risk = "medium"
        else:
            risk = "low"

        alert = risk == "high"

        return DistressResult(
            score=round(final_score, 1),
            risk_level=risk,
            alert=alert,
            reasons=reasons or ["no significant risk signals"],
        )


if __name__ == "__main__":
    engine = DistressEngine()

    sample_checkins = [
        CheckIn(mood_score=4, text="feeling a bit better this week",
                days_since_complaint=10, missed_checkins=0),
        CheckIn(mood_score=2, text="I'm scared, I feel followed and can't sleep",
                days_since_complaint=95, missed_checkins=2),
        CheckIn(mood_score=1, text="they threatened my family again",
                days_since_complaint=200, missed_checkins=3, threat_reported=True),
    ]

    for i, c in enumerate(sample_checkins, 1):
        result = engine.score(c)
        print(f"Check-in {i}: score={result.score} risk={result.risk_level} "
              f"alert={result.alert}")
        print(f"  reasons: {', '.join(result.reasons)}\n")