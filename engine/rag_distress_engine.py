"""
RAG + Gemma-based Distress Engine (v1)
---------------------------------------
Extends the base rule engine (distress_engine.py) with:
  1. A RAG layer — TF-IDF retrieval over a distress-assessment
     knowledge base — that grounds analysis in domain guidelines.
  2. A Gemma-based text understanding module for richer severity
     estimation than the v0 keyword scorer.

USES OLLAMA:
GemmaDistressAnalyzer calls a locally-running Ollama server
(http://localhost:11434) instead of downloading weights via
transformers/HuggingFace. Setup:
    ollama pull gemma3:1b
    (ollama serve runs automatically in the background after install)
If Ollama isn't running or reachable, it automatically falls back
to the v0 keyword scorer — no crash, just reduced text understanding.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal

import requests

from .distress_engine import (
    CheckIn, DistressResult, DistressEngine, score_text_sentiment,
)

RiskLevel = Literal["low", "medium", "high"]

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")


# ---------------------------------------------------------------------
# RAG retriever — TF-IDF for now (no model download needed).
# Swap for sentence-transformers embeddings + FAISS once you have
# HF access; the retrieve(query, k) interface stays the same.
# ---------------------------------------------------------------------
class GuidelineRetriever:
    def __init__(self, kb_path: str = KB_PATH):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(
            [item["text"] for item in self.kb]
        )

    def retrieve(self, query: str, k: int = 2) -> list[str]:
        if not query:
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [self.kb[i]["text"] for i in top_idx if sims[i] > 0]


# ---------------------------------------------------------------------
# Gemma-based text analyzer, with graceful fallback
# ---------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a mental-health triage assistant reviewing a victim's check-in.
Use the guidelines below only as context — never diagnose, only flag risk signals.

Guidelines:
{context}

Check-in text: "{text}"

Respond ONLY as JSON: {{"severity": 0-100, "indicators": [...], "needs_escalation": true/false}}
"""


class GemmaDistressAnalyzer:
    def __init__(self, model_name: str = OLLAMA_MODEL, ollama_url: str = OLLAMA_URL):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.available = self._check_ollama()

    def _check_ollama(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def _keyword_fallback(self, text: str) -> dict:
        sentiment = score_text_sentiment(text)
        severity = round(((1 - sentiment) / 2) * 100, 1)
        flagged = [w for w in {"scared", "threat", "unsafe", "alone", "afraid"}
                   if w in text.lower()]
        return {
            "severity": severity,
            "indicators": flagged or ["no strong keyword signal"],
            "needs_escalation": severity >= 60,
            "engine": "fallback-keyword-v0",
        }

    def analyze(self, text: str, context: list[str]) -> dict:
        if not self.available:
            return self._keyword_fallback(text)

        prompt = PROMPT_TEMPLATE.format(
            context="\n".join(f"- {c}" for c in context) or "none",
            text=text,
        )
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt,
                        "stream": False, "format": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            output = resp.json().get("response", "")
        except Exception:
            # Ollama was reachable at startup but failed mid-request
            # (e.g. model not pulled) — fall back rather than crash.
            return self._keyword_fallback(text)

        match = re.search(r"\{.*\}", output, re.DOTALL)
        try:
            result = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            result = {}
        result.setdefault("severity", 0)
        result.setdefault("indicators", [])
        result.setdefault("needs_escalation", False)
        # Defensive: force indicators to a flat list of strings —
        # small/quantized models sometimes return non-string items.
        result["indicators"] = [str(i) for i in result.get("indicators") or []]
        try:
            result["severity"] = float(result["severity"])
        except (TypeError, ValueError):
            result["severity"] = 0.0
        result["engine"] = f"gemma-ollama:{self.model_name}"
        return result


# ---------------------------------------------------------------------
# Combined engine: rule-based signals + RAG-grounded text severity
# ---------------------------------------------------------------------
@dataclass
class RagDistressResult(DistressResult):
    llm_indicators: list[str] = field(default_factory=list)
    retrieved_context: list[str] = field(default_factory=list)
    text_engine: str = "fallback-keyword-v0"


class RagDistressEngine:
    def __init__(self, w_llm_text: float = 0.5):
        self.base_engine = DistressEngine()
        self.retriever = GuidelineRetriever()
        self.analyzer = GemmaDistressAnalyzer()
        self.w_llm_text = w_llm_text  # blend vs. the base engine's own text score

    def score(self, c: CheckIn) -> RagDistressResult:
        context = self.retriever.retrieve(c.text, k=2)
        llm_result = self.analyzer.analyze(c.text, context)
        base_result = self.base_engine.score(c)

        blended_score = min(
            (1 - self.w_llm_text) * base_result.score
            + self.w_llm_text * llm_result["severity"],
            100.0,
        )

        if (blended_score >= self.base_engine.high_threshold
                or c.threat_reported or llm_result["needs_escalation"]):
            risk: RiskLevel = "high"
        elif blended_score >= self.base_engine.medium_threshold:
            risk = "medium"
        else:
            risk = "low"

        reasons = list(base_result.reasons)
        if llm_result["indicators"]:
            reasons.append(f"llm indicators: {', '.join(llm_result['indicators'])}")

        return RagDistressResult(
            score=round(blended_score, 1),
            risk_level=risk,
            alert=(risk == "high"),
            reasons=reasons,
            llm_indicators=llm_result["indicators"],
            retrieved_context=context,
            text_engine=llm_result["engine"],
        )


if __name__ == "__main__":
    engine = RagDistressEngine()

    samples = [
        CheckIn(mood_score=4, text="feeling a bit better this week",
                days_since_complaint=10, missed_checkins=0),
        CheckIn(mood_score=2, text="I'm scared, I feel followed and can't sleep",
                days_since_complaint=95, missed_checkins=2),
    ]

    for i, s in enumerate(samples, 1):
        r = engine.score(s)
        print(f"--- Check-in {i} ---")
        print(f"score={r.score} risk={r.risk_level} alert={r.alert}")
        print(f"text engine used: {r.text_engine}")
        print(f"retrieved context: {r.retrieved_context}")
        print(f"reasons: {r.reasons}\n")