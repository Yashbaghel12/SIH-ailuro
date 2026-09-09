"""
LoRA fine-tuning - Gemma 3 1B for distress classification
----------------------------------------------------------
Key design decisions (v4 — direct classification):
  1. Training examples use the model's chat template so the prompt
     format exactly matches what eval_finetuned.py sends at inference.
  2. LoRA targets both attention AND MLP layers.
  3. Model outputs BOTH severity score AND a direct risk_level label.
     The eval script reads risk_level directly, bypassing the
     unreliable severity→threshold→bucket pipeline.
  4. Gold-labeled examples are oversampled 50× to dominate the
     classification signal, while rule-engine data provides diversity.
  5. Rule-engine severity scores are aggressively recalibrated so
     severity and risk_level are consistent in training data.
"""

import csv
import json
import random
from collections import Counter

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
# pyrefly: ignore [missing-import]
import torch

MODEL_NAME = "google/gemma-3-1b-it"
DATASET_PATH = "dataset.jsonl"
GOLD_CSV_PATH = "gold_set_filled.csv"

# --- LoRA config: wider target set + higher rank ----------------------
LORA_CONFIG = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Training hyperparameters ----------------------------------------
TRAINING_ARGS = {
    "output_dir": "./gemma-distress-lora",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 10,
    "learning_rate": 2e-5,
    "warmup_steps": 180,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "epoch",
    "bf16": True,
    "dataset_text_field": "text",
    "max_length": 512,
}

SYSTEM_PROMPT = (
    "You are a mental-health triage assistant "
    "reviewing a victim check-in. "
    "Respond ONLY with valid JSON. "
    "Do not include markdown, explanations, "
    "or additional text."
)

# --- Phrases for severity recalibration and indicator derivation ------
EXTREME_DISTRESS_PHRASES = [
    "don't know how much longer i can keep going",
    "don't know how much longer",
    "can't keep going",
    "can't take it anymore",
    "want to end it",
    "no point in living",
]

THREAT_PHRASES = [
    "following me",
    "keeps following",
    "threatened",
    "pressuring me to drop",
    "feel unsafe",
]


def severity_to_label(severity: float) -> str:
    """Bucket a severity score into low/medium/high."""
    if severity >= 60:
        return "high"
    elif severity >= 30:
        return "medium"
    return "low"


def derive_indicators_from_text(text: str) -> list[str]:
    """Derive risk indicators from check-in text for gold examples."""
    lowered = text.lower()
    indicators = []

    has_extreme = any(p in lowered for p in EXTREME_DISTRESS_PHRASES)
    moderate_phrases = [
        "feel hopeless", "exhausted by it all", "hard to stay hopeful",
        "finding it hard", "can't sleep",
    ]
    has_moderate = any(p in lowered for p in moderate_phrases)

    if has_extreme:
        indicators.extend([
            "low self-reported mood",
            "distress language in check-in text",
        ])
    elif has_moderate:
        indicators.append("low self-reported mood")

    if any(p in lowered for p in THREAT_PHRASES):
        indicators.append("threat reported")

    for phrase in [
        "days and my case is still dragging",
        "days and nothing seems to be moving",
    ]:
        if phrase in lowered:
            indicators.append("prolonged case pendency")
            break

    return indicators or ["no significant risk signals"]


def recalibrate_severity(records: list[dict]) -> list[dict]:
    """Aggressively recalibrate rule-engine severity to match human judgment.

    Key: recently-filed cases with only moderate distress get pulled
    to severity 15 (solidly "low"), matching how human labelers score them.
    """
    calibrated = []

    for rec in records:
        rec = dict(rec)  # copy
        text_lower = rec["text"].lower()
        severity = rec["severity"]

        has_extreme = any(p in text_lower for p in EXTREME_DISTRESS_PHRASES)
        has_threat = any(p in text_lower for p in THREAT_PHRASES)
        is_recent = text_lower.startswith("i filed my complaint recently")

        if is_recent and not has_extreme and not has_threat and severity > 25:
            # Recently filed + only moderate language → humans say LOW
            # Force to solidly low range
            rec["severity"] = 15.0
        elif has_extreme and severity < 60:
            # Extreme distress language → humans say HIGH
            # Push to high range
            rec["severity"] = max(severity, 70.0)

        calibrated.append(rec)

    return calibrated


def load_gold_examples(csv_path: str, oversample: int = 50) -> list[dict]:
    """Load human-labeled gold examples with calibrated severity.

    Gold examples are oversampled heavily to be the dominant
    classification signal in the training data.
    """
    # Map gold labels to severity values centered in each bucket
    severity_map = {"low": 15.0, "medium": 42.0, "high": 78.0}
    gold_records = []

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["your_risk_label"].strip().lower()
            if label not in ("low", "medium", "high"):
                continue

            text = row["text"]
            record = {
                "case_id": row["case_id"],
                "text": text,
                "severity": severity_map[label],
                "indicators": derive_indicators_from_text(text),
                "gold_label": label,  # human label for risk_level
            }

            for _ in range(oversample):
                gold_records.append(record)

    print(f"Gold examples: {len(gold_records)} "
          f"({len(gold_records) // oversample} unique × {oversample})")

    return gold_records


def balance_dataset(records: list[dict], seed: int = 42) -> list[dict]:
    """Oversample minority classes so all three buckets have equal count."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {"low": [], "medium": [], "high": []}

    for rec in records:
        label = rec.get("gold_label") or severity_to_label(rec["severity"])
        buckets[label].append(rec)

    max_count = max(len(v) for v in buckets.values())
    balanced = []

    for label, items in buckets.items():
        if not items:
            continue
        balanced.extend(items)
        deficit = max_count - len(items)
        if deficit > 0:
            extras = [rng.choice(items) for _ in range(deficit)]
            balanced.extend(extras)

    rng.shuffle(balanced)

    counts = Counter(
        rec.get("gold_label") or severity_to_label(rec["severity"])
        for rec in balanced
    )
    print(f"Balanced dataset: {dict(counts)} (total: {len(balanced)})")

    return balanced


def format_example(example: dict, tokenizer) -> dict:
    """Wrap each training example in the model's chat template.

    The JSON response now includes a 'risk_level' field so the model
    learns to classify directly, not just produce a severity score.
    """
    severity = max(0, min(100, round(float(example["severity"]), 1)))

    # Use gold_label if available, otherwise derive from severity
    risk_level = example.get("gold_label") or severity_to_label(severity)

    target = {
        "severity": severity,
        "risk_level": risk_level,
        "indicators": example["indicators"],
        "needs_escalation": risk_level == "high",
    }
    response_json = json.dumps(target)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f'Check-in text: "{example["text"]}"',
        },
        {
            "role": "assistant",
            "content": response_json,
        },
    ]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": full_text}


def main():
    # --- Load rule-engine dataset -------------------------------------
    raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    records = [dict(row) for row in raw_dataset]
    print(f"Rule-engine examples: {len(records)}")

    # --- Recalibrate rule-engine severity scores ----------------------
    records = recalibrate_severity(records)

    recal_counts = Counter(severity_to_label(r["severity"]) for r in records)
    print(f"After recalibration: {dict(recal_counts)}")

    # --- Load & inject gold-calibrated examples -----------------------
    gold_records = load_gold_examples(GOLD_CSV_PATH, oversample=50)
    records.extend(gold_records)

    # --- Balance across low/medium/high -------------------------------
    balanced_records = balance_dataset(records)

    # --- Load tokenizer -----------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Format with chat template ------------------------------------
    formatted = [format_example(rec, tokenizer) for rec in balanced_records]
    dataset = Dataset.from_list(formatted)

    print(f"\nSample training text (first 600 chars):")
    print(dataset[0]["text"][:600])
    print("...\n")

    # --- Load base model ----------------------------------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LORA_CONFIG
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # --- Train --------------------------------------------------------
    sft_config = SFTConfig(**TRAINING_ARGS)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model("./gemma-distress-lora-final")
    print("\nDone. Adapter saved to ./gemma-distress-lora-final")


if __name__ == "__main__":
    main()
