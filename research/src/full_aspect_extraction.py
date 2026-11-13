from pathlib import Path
import json
import random
import time

from aspect_extraction import extract_aspects


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" /
    "train_aligned_reviews.jsonl"
)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "aspect_extraction_full.jsonl"

SUMMARY_FILE = OUTPUT_DIR / "aspect_extraction_full_summary.json"

PILOT_SIZE = 20
RANDOM_SEED = 42


# Evidence validation

def normalize_text(text):
    return " ".join(str(text).casefold().split())


def validate_evidence(aspects, review_text):
    results = []

    for aspect in aspects:
        item = dict(aspect)

        evidence = item.get("evidence", "")

        valid = (
            bool(evidence)
            and normalize_text(evidence)
            in normalize_text(review_text)
        )

        item["evidence_valid"] = valid

        results.append(item)

    return results


# Load aligned reviews

print("\n[1] Loading aligned reviews...")

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("Available reviews:", len(records))


# Select ALL training reviews

print("\n[2] Selecting all training reviews...")

selected = [
    record
    for record in records
    if str(record.get("text", "")).strip()
]

selected_users = {
    record["user_id"]
    for record in selected
}

PILOT_SIZE = len(selected)

print("Selected reviews:", len(selected))
print("Distinct users:", len(selected_users))


# Resume existing results

processed_ids = set()

if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                old = json.loads(line)

                if old.get("status") == "success":
                    processed_ids.add(old["train_row_id"])

print("Already processed:", len(processed_ids))


# Batch extraction

success_count = 0
error_count = 0

evidence_total = 0
evidence_valid_count = 0

start_time = time.time()

with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
    for index, record in enumerate(selected, start=1):
        row_id = record["train_row_id"]

        if row_id in processed_ids:
            print(f"\nSkipping existing ID: {row_id}")
            continue

        review_text = str(record["text"]).strip()

        print(
            f"\n[{index}/{PILOT_SIZE}] "
            f"Processing train ID: {row_id}"
        )

        try:
            task_start = time.time()

            result = extract_aspects(review_text)

            aspects = validate_evidence(
                result["aspects"],
                review_text
            )

            elapsed = time.time() - task_start

            valid_count = sum(
                item["evidence_valid"]
                for item in aspects
            )

            evidence_total += len(aspects)
            evidence_valid_count += valid_count

            output = {
                "status": "success",
                "train_row_id": row_id,
                "user_id": record["user_id"],
                "parent_asin": record["parent_asin"],
                "rating": record["rating_train"],
                "review_title": record.get("title", ""),
                "review_text": review_text,
                "aspects": aspects,
                "overall_sentiment": result[
                    "overall_sentiment"
                ],
                "evidence_valid_count": valid_count,
                "evidence_total": len(aspects),
                "processing_seconds": round(elapsed, 2)
            }

            fout.write(json.dumps(output, ensure_ascii=False) + "\n")

            fout.flush()

            success_count += 1

            print("Aspects extracted:", len(aspects))
            print(f"Valid evidence: {valid_count}/{len(aspects)}")
            print(f"Time: {elapsed:.2f}s")

        except Exception as e:
            error_count += 1

            print("ERROR:", str(e))


total_time = time.time() - start_time

summary = {
"total_selected_reviews": PILOT_SIZE,
    "selected_users": len(selected_users),
    "new_success": success_count,
    "errors": error_count,
    "total_evidence": evidence_total,
    "valid_evidence": evidence_valid_count,
    "evidence_validity_rate": (
        round(evidence_valid_count / evidence_total, 4)
        if evidence_total else None
    ),
    "total_processing_seconds": round(total_time, 2),
    "average_seconds_per_review": (
        round(total_time / success_count, 2)
        if success_count else None
    )
}

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 50)
print("FULL EXTRACTION COMPLETED")
print("=" * 50)

print(json.dumps(
    summary,
    indent=4,
    ensure_ascii=False
))

print("\nOutput:", OUTPUT_FILE)
print("Summary:", SUMMARY_FILE)
