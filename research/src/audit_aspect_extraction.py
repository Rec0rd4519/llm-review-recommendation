from pathlib import Path
import json
import csv
import re
import html
import unicodedata

from collections import Counter, defaultdict

# Paths

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "outputs" / "aspect_extraction_full.jsonl"
)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQUENCY_FILE = OUTPUT_DIR / "aspect_frequency.csv"

REVIEW_FILE = (
    OUTPUT_DIR / "evidence_review_required.jsonl"
)

UNICODE_FILE = (
    OUTPUT_DIR / "evidence_unicode_matches.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "aspect_audit_summary.json"
)


# Normalize text for evidence comparison

def normalize_match_text(text):

    text = html.unescape(str(text))

    # Convert HTML line breaks into spaces
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = unicodedata.normalize(
        "NFKC",
        text
    ).casefold()

    punctuation_map = str.maketrans({
        "’": "'",
        "‘": "'",
        "ʼ": "'",
        "＇": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        "‑": "-",
        "−": "-",
        "\u00a0": " "
    })

    text = text.translate(punctuation_map)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# Load extraction results

print("\n[1] Loading extraction results...")

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if line.strip():
            records.append(json.loads(line))

print("Total records:", len(records))


# Collect statistics

aspect_counts = Counter()
aspect_variants = defaultdict(Counter)

original_valid = 0
normalized_valid = 0

unicode_fixed = []
review_required = []

total_evidence = 0
empty_aspect_reviews = 0

users = set()
items = set()
row_ids = set()

for record in records:

    users.add(record["user_id"])
    items.add(record["parent_asin"])
    row_ids.add(record["train_row_id"])

    aspects = record.get("aspects", [])

    if not aspects:
        empty_aspect_reviews += 1

    review_text = normalize_match_text(
        record.get("review_text", "")
    )

    for aspect in aspects:

        total_evidence += 1

        name = str(
            aspect.get("aspect", "")
        ).strip()

        if name:

            key = name.casefold()

            aspect_counts[key] += 1
            aspect_variants[key][name] += 1

        evidence = str(
            aspect.get("evidence", "")
        ).strip()

        old_valid = bool(
            aspect.get("evidence_valid", False)
        )

        evidence_normalized = normalize_match_text(
            evidence
        )

        new_valid = (
            bool(evidence_normalized)
            and evidence_normalized in review_text
        )

        if old_valid:
            original_valid += 1

        if new_valid:
            normalized_valid += 1

        detail = {
            "train_row_id": record["train_row_id"],
            "user_id": record["user_id"],
            "parent_asin": record["parent_asin"],
            "aspect": name,
            "sentiment": aspect.get("sentiment"),
            "evidence": evidence,
            "review_text": record.get("review_text", "")
        }

        if not old_valid and new_valid:

            unicode_fixed.append(detail)

        elif not new_valid:

            review_required.append(detail)


# Export aspect frequency

print("\n[2] Exporting aspect frequency...")

with open(
    FREQUENCY_FILE,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "aspect_normalized",
        "frequency",
        "surface_forms"
    ])

    for name, count in aspect_counts.most_common():

        variants = aspect_variants[name]

        surface_forms = json.dumps(
            dict(variants),
            ensure_ascii=False
        )

        writer.writerow([
            name,
            count,
            surface_forms
        ])


# Export evidence requiring review

print("\n[3] Exporting evidence audit...")

def write_jsonl(file_path, data):

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:

        for item in data:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )


write_jsonl(
    REVIEW_FILE,
    review_required
)

write_jsonl(
    UNICODE_FILE,
    unicode_fixed
)


# Generate summary

summary = {

    "total_reviews": len(records),

    "unique_train_row_ids": len(row_ids),

    "unique_users": len(users),

    "unique_items": len(items),

    "empty_aspect_reviews": empty_aspect_reviews,

    "total_aspect_evidence_pairs": total_evidence,

    "unique_aspect_labels": len(aspect_counts),

    "original_valid_evidence": original_valid,

    "normalized_valid_evidence": normalized_valid,

    "unicode_normalization_recovered": len(
        unicode_fixed
    ),

    "evidence_requiring_review": len(
        review_required
    ),

    "normalized_match_rate": round(
        normalized_valid / total_evidence,
        4
    ) if total_evidence else 0
}

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# Display results

print("\n" + "=" * 55)
print("ASPECT AUDIT COMPLETED")
print("=" * 55)

print(json.dumps(
    summary,
    indent=4,
    ensure_ascii=False
))

print("\nGenerated files:")
print(FREQUENCY_FILE)
print(REVIEW_FILE)
print(UNICODE_FILE)
print(SUMMARY_FILE)
