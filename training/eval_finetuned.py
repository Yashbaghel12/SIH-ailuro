"""
Evaluate fine-tuned Gemma 3 1B + LoRA.
Prints severity, prediction, gold label, and raw JSON.
"""

import argparse
import csv
import json

# pyrefly: ignore [missing-import]
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "google/gemma-3-1b-it"
ADAPTER_PATH = "./gemma-distress-lora-final"


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")

    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        print("WARNING: BF16 unsupported. Using FP16.")
        dtype = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.eval()

    return tokenizer, model


def build_prompt(tokenizer, text):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a mental-health triage assistant "
                "reviewing a victim check-in. "
                "Respond ONLY with valid JSON. "
                "Do not include markdown, explanations, "
                "or additional text."
            )
        },
        {
            "role": "user",
            "content": f'Check-in text: "{text}"'
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def extract_json(text):

    start = text.find("{")

    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):

        char = text[i]

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:

                candidate = text[start:i + 1]

                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None

    return None


def predict(tokenizer, model, text):

    prompt = build_prompt(tokenizer, text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    parsed = extract_json(raw_output)

    if parsed is None:
        return None, None, raw_output

    try:
        severity = float(
            parsed.get("severity", 0)
        )
    except (TypeError, ValueError):
        severity = 0.0

    severity = max(
        0.0,
        min(100.0, severity)
    )

    # Prefer the model's direct risk_level classification
    # over severity-based thresholding (much more accurate).
    risk_level = parsed.get("risk_level", "").strip().lower()

    if risk_level in ("low", "medium", "high"):
        risk = risk_level
    else:
        # Fallback to severity thresholds
        if severity >= 60:
            risk = "high"
        elif severity >= 30:
            risk = "medium"
        else:
            risk = "low"

    return risk, severity, parsed


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold",
        required=True
    )

    args = parser.parse_args()

    rows = []

    with open(
        args.gold,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            label = row[
                "your_risk_label"
            ].strip().lower()

            if label in ("low", "medium", "high"):

                rows.append(
                    (
                        row["case_id"],
                        row["text"],
                        label
                    )
                )

    print()
    print("=" * 100)
    print("GEMMA DISTRESS MODEL - SEVERITY DEBUG")
    print("=" * 100)

    tokenizer, model = load_model()

    correct = 0
    evaluated = 0

    for case_id, text, gold in rows:

        predicted, severity, parsed = predict(
            tokenizer,
            model,
            text
        )

        evaluated += 1

        if predicted == gold:
            correct += 1
            result = "MATCH"
        else:
            result = "X"

        print()
        print("-" * 100)

        print(f"CASE:       {case_id}")
        print(f"GOLD:       {gold}")
        print(f"SEVERITY:   {severity}")
        print(f"PREDICTED:  {predicted}")
        print(f"RESULT:     {result}")

        print()
        print("MODEL JSON:")

        if parsed is not None:
            print(
                json.dumps(
                    parsed,
                    indent=2,
                    ensure_ascii=False
                )
            )
        else:
            print("INVALID JSON")

    print()
    print("=" * 100)
    print("FINAL RESULT")
    print("=" * 100)

    print(
        f"Accuracy: {correct}/{evaluated} "
        f"= {100 * correct / evaluated:.1f}%"
    )

    print()
    print("THRESHOLDS")
    print("< 30       = LOW")
    print("30-59.9    = MEDIUM")
    print(">= 60      = HIGH")


if __name__ == "__main__":
    main()
