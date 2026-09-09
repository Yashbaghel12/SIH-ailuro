"""
Evaluate the distress engine risk classification against a
MANUALLY labeled gold set. Tests the FULL pipeline (rule engine
+ RAG + Gemma via Ollama).
"""

import argparse
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.distress_engine import CheckIn
from engine.rag_distress_engine import RagDistressEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()

    engine = RagDistressEngine()
    rows = []
    with open(args.gold, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["your_risk_label"].strip().lower()
            if label not in ("low", "medium", "high"):
                continue
            rows.append((row["case_id"], row["text"], label))

    if not rows:
        print("No labeled rows found.")
        return

    correct = 0
    print(f"{'case_id':<24} {'predicted':<10} {'yours':<10} {'text_engine':<22} match")
    for case_id, text, gold_label in rows:
        checkin = CheckIn(mood_score=3, text=text,
                           days_since_complaint=0, missed_checkins=0)
        result = engine.score(checkin)
        is_match = result.risk_level == gold_label
        correct += is_match
        print(f"{case_id:<24} {result.risk_level:<10} {gold_label:<10} "
              f"{result.text_engine:<22} {'MATCH' if is_match else 'X'}")

    print(f"\nAccuracy: {correct}/{len(rows)} = {100*correct/len(rows):.1f}%")


if __name__ == "__main__":
    main()
