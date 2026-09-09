# AI-powered distress monitoring system

**SIH Problem Statement 26094** — AI-Powered Dynamic Mental Health Monitoring
and Distress Prediction System for Victims of Atrocities.
Ministry of Social Justice and Empowerment (MoSJE).

Victims of atrocities often face prolonged psychological distress after
filing a complaint — threats, court delays, social ostracism, economic
hardship. This system lets victims check in regularly, scores distress
using a rule-based engine plus a RAG-grounded LLM analyzer, and flags
high-risk cases to counselors, NGOs, and legal aid.

## How it works

1. Victim submits a check-in (mood score, free text, days since complaint,
   missed check-ins, threat reports) via the app.
2. `engine/distress_engine.py` computes a rule-based distress score.
3. `engine/rag_distress_engine.py` retrieves relevant guideline snippets
   from `knowledge_base.json` and passes them + the check-in text to a
   Gemma-based analyzer for a second, richer severity estimate.
4. The two scores are blended into one risk level (low / medium / high).
   High risk or any threat report triggers an alert for case workers.

**Note:** the Gemma call requires GPU + internet access to download model
weights. If unavailable, `GemmaDistressAnalyzer` automatically falls back
to a lightweight keyword scorer — the pipeline still runs end-to-end,
just with reduced text understanding until Gemma is wired in.

## Structure

```
engine/
  distress_engine.py       — base rule-based scoring engine
  rag_distress_engine.py   — RAG retrieval + Gemma orchestration
  knowledge_base.json      — distress-assessment guideline snippets
  alerting.py               — case worker / authority notification (email or log fallback)
api/
  main.py                  — FastAPI wrapper (POST /checkin)
frontend/
  index.html               — victim-facing check-in UI
training/
  finetune_gemma_lora.py   — LoRA fine-tuning (needs GPU + real data)
  generate_finetune_dataset.py — builds a weak-labeled dataset from case metadata
  eval_accuracy.py / eval_finetuned.py — accuracy checks against a hand-labeled gold set
tests/
  test_distress_engine.py  — unit tests for the rule engine
```

## Design references

Two research systems informed this architecture:

- **CaiTI** (Nie et al., *ACM Trans. Comput. Healthcare*, 2026) — an
  LLM-based conversational screening agent that assigns a 3-level score
  (0/1/2) per dimension and escalates only the concerning ones for human
  follow-up. Their microbenchmarks also found large accuracy gaps between
  GPT-class and smaller/quantized (Llama) models on reasoning tasks —
  consistent with what we observed when fine-tuning a 1B Gemma on a small
  synthetic dataset (see `training/eval_finetuned.py` results).
- **ChatThero** (Wang et al., arXiv:2508.20996) — a multi-session,
  stressor-aware recovery-support agent. Its between-session "stressor"
  concept maps to our `days_since_complaint` / `missed_checkins` /
  `threat_reported` signals here.

Both papers are explicit that these systems must never operate
autonomously on high-risk or crisis cases — the AI flags, a licensed
human decides. This system follows the same principle: `engine/alerting.py`
hands off every high-risk check-in to a case worker rather than
resolving it in-model.

## Setup

```bash
pip install -r requirements.txt

# run the API (from repo root)
uvicorn api.main:app --reload --port 8000

# run tests
pytest tests/
```

Then POST to `http://localhost:8000/checkin`:

```json
{
  "mood_score": 2,
  "text": "I'm scared, I feel followed and can't sleep",
  "days_since_complaint": 95,
  "missed_checkins": 2,
  "threat_reported": false
}
```

## Important notes

- `training/sample_dataset.jsonl` contains **synthetic examples only** —
  never commit real or identifiable victim data to this repo.
- Lock down `allow_origins` in `api/main.py` and add authentication before
  connecting this to any real check-in data.
- No diagnostic claims are made anywhere in this system — it flags risk
  signals for human review, it does not diagnose.
- `engine/alerting.py` needs real SMTP credentials (`ALERT_SMTP_*` env vars)
  before it can actually email a case worker. Without them, alerts are
  safely logged to `engine/alerts_log.jsonl` instead — useful for demos,
  not a substitute for a real notification channel in production.