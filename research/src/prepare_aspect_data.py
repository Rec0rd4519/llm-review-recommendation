from pathlib import Path
from collections import Counter
import json
import unicodedata
import html
import re


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "aspect_extraction_audited.jsonl"

SUMMARY_FILE = OUTPUT_DIR / "aspect_audit_summary.json"


def find_input_file(filename, required=True):
    candidates = [
        ROOT / "outputs" / filename,
        ROOT / "data" / "processed" / filename,
        ROOT / "data" / "raw" / filename,
        ROOT / filename
    ]

    for path in candidates:
        if path.exists():
            return path

    if required:
        raise FileNotFoundError(
            f"\nCannot find input file: {filename}\n"
            + "\n".join(str(p) for p in candidates)
        )

    return None


RAW_FILE = find_input_file("aspect_extraction_full.jsonl")

UNICODE_FILE = find_input_file("evidence_unicode_matches.jsonl")

REVIEW_FILE = find_input_file("evidence_review_required.jsonl")

AUDIT_SUMMARY_FILE = find_input_file(
    "aspect_audit_summary.json",
    required=False
)


# JSON / JSONL utilities

def load_records(path):
    records = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON:\n"
                    f"File: {path}\n"
                    f"Line: {line_number}\n"
                    f"Error: {e}"
                )

    return records


def write_jsonl(path, records):
    temp_file = path.with_suffix(path.suffix + ".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    temp_file.replace(path)


def write_json(path, data):
    temp_file = path.with_suffix(path.suffix + ".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    temp_file.replace(path)


# Text normalization

def normalize_basic(text):
    """
    Original validation:
    - Case-insensitive matching
    - Whitespace normalization
    """

    return " ".join(str(text).casefold().split())


def normalize_unicode(text):
    """
    Extended text normalization.

    Handles:
    - HTML entities
    - HTML line breaks
    - Unicode compatibility characters
    - Curly apostrophes
    - Different quotation marks
    - Different dash characters
    - Non-breaking spaces
    - Zero-width characters
    """

    # Decode HTML entities
    text = html.unescape(str(text))

    # Convert HTML line breaks into spaces
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Unicode normalization
    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # Normalize punctuation
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201B": "'",
        "\u02BC": "'",
        "\uFF07": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00A0": " ",
        "\u200B": "",
        "\u200C": "",
        "\u200D": "",
        "\uFEFF": ""
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize whitespace
    return " ".join(text.casefold().split())

def normalize_label(text):
    """
    Normalize aspect labels for consistent matching.
    """

    return " ".join(str(text).strip().casefold().split())

# Flexible audit record extraction

def flatten_audit_records(records):
    """
    Support both audit structures:

    A. Flat:
       {
           "train_row_id": 1,
           "aspect": "scent",
           "sentiment": "positive",
           "evidence": "pleasant fragrance"
       }

    B. Nested:
       {
           "train_row_id": 1,
           "aspects": [
               {
                   "aspect": "scent",
                   "sentiment": "positive",
                   "evidence": "pleasant fragrance"
               }
           ]
       }
    """

    flattened = []

    for record in records:
        if isinstance(
            record.get("aspects"),
            list
        ):
            for aspect in record["aspects"]:
                merged = {
                    **record,
                    **aspect
                }

                flattened.append(merged)

        elif isinstance(
            record.get("aspect"),
            dict
        ):
            merged = {
                **record,
                **record["aspect"]
            }

            flattened.append(merged)

        else:
            flattened.append(record)

    return flattened


def get_field(record, candidates):
    for field in candidates:
        value = record.get(field)

        if value is not None:
            return value

    return None


def get_row_id(record):
    value = get_field(
        record,
        [
            "train_row_id",
            "row_id",
            "train_id"
        ]
    )

    if value is None:
        raise ValueError(
            "Audit record missing train_row_id. "
            f"Available fields: {list(record.keys())}"
        )

    return str(value)


def get_aspect_name(record):
    value = get_field(
        record,
        [
            "aspect",
            "raw_aspect",
            "aspect_name",
            "aspect_label"
        ]
    )

    if value is None:
        raise ValueError(
            "Audit record missing aspect name. "
            f"Available fields: {list(record.keys())}"
        )

    return normalize_label(value)


def get_evidence(record):
    value = get_field(
        record,
        [
            "evidence",
            "original_evidence",
            "evidence_text"
        ]
    )

    if value is None:
        return ""

    return str(value)


# Aspect matching keys

def make_key(row_id, aspect, evidence):
    return (
        str(row_id),
        normalize_label(aspect),
        normalize_unicode(evidence)
    )


def make_source_key(row_id, aspect):
    return make_key(
        row_id,
        aspect.get("aspect", ""),
        aspect.get("evidence", "")
    )


def build_audit_counter(records):
    counter = Counter()

    for record in records:
        row_id = get_row_id(record)

        aspect = get_aspect_name(record)

        evidence = get_evidence(record)

        key = make_key(
            row_id,
            aspect,
            evidence
        )

        counter[key] += 1

    return counter


# Load source files

print("\n" + "=" * 60)
print("ASPECT DATA PREPARATION")
print("=" * 60)

print("\n[1] Loading source files...")

raw_records = load_records(RAW_FILE)

unicode_raw = load_records(UNICODE_FILE)

review_raw = load_records(REVIEW_FILE)

unicode_records = flatten_audit_records(unicode_raw)

review_records = flatten_audit_records(review_raw)

print("Original reviews:", len(raw_records))

print(
    "Unicode audit records:",
    len(unicode_records)
)

print(
    "Manual review records:",
    len(review_records)
)


# Check source review IDs

print("\n[2] Checking source review IDs...")

review_by_id = {}

for record in raw_records:
    if "train_row_id" not in record:
        raise ValueError("Original record missing train_row_id.")

    row_id = str(record["train_row_id"])

    if row_id in review_by_id:
        raise ValueError(f"Duplicate train_row_id: {row_id}")

    if record.get("status") not in (
        None,
        "success"
    ):
        raise ValueError(f"Unsuccessful extraction: {row_id}")

    review_by_id[row_id] = record


# Build audit indexes

print("\n[3] Building audit indexes...")

unicode_counter = build_audit_counter(unicode_records)

review_counter = build_audit_counter(review_records)

source_counter = Counter()

for record in raw_records:
    row_id = str(record["train_row_id"])

    for aspect in record.get("aspects", []):
        source_counter[
            make_source_key(row_id, aspect)
        ] += 1


# Audit categories must not overlap

overlap = (
    set(unicode_counter)
    & set(review_counter)
)

if overlap:
    raise ValueError(f"Conflicting audit categories: {len(overlap)}")


# Validate audit records

print("\n[4] Validating audit records...")


def validate_audit_records(records, label):
    for audit_record in records:
        row_id = get_row_id(audit_record)

        if row_id not in review_by_id:
            raise ValueError(f"{label}: Unknown train_row_id {row_id}")

        original = review_by_id[row_id]

        # Check user and product IDs

        for field in (
            "user_id",
            "parent_asin"
        ):
            if field in audit_record:
                if str(audit_record[field]) != str(original.get(field)):
                    raise ValueError(
                        f"{label}: {field} mismatch "
                        f"for train_row_id {row_id}"
                    )

        # Check review text when supplied

        audit_text = audit_record.get("review_text")

        original_text = original.get(
            "review_text",
            ""
        )

        if audit_text is not None:
            if (
                normalize_unicode(audit_text)
                != normalize_unicode(original_text)
            ):
                raise ValueError(
                    f"{label}: Review text mismatch "
                    f"for train_row_id {row_id}"
                )

        # Check evidence

        evidence = get_evidence(audit_record)

        if not evidence:
            raise ValueError(
                f"{label}: Empty evidence "
                f"for train_row_id {row_id}"
            )

        normalized_evidence = normalize_unicode(evidence)

        normalized_review = normalize_unicode(original_text)

        if label == "Unicode":
            if normalized_evidence not in normalized_review:
                raise ValueError(
                    f"Unicode evidence failed validation: "
                    f"{row_id}"
                )

        elif label == "Manual review":
            if normalized_evidence in normalized_review:
                raise ValueError(
                    f"Manual review evidence unexpectedly "
                    f"matches after normalization: {row_id}"
                )

    print(f"{label} audit validation: PASSED")

# Inspect a Unicode-normalization example.

target_id = "695"

print("\n[DEBUG] Inspecting train_row_id:", target_id)

found = False

for audit_record in unicode_records:
    if get_row_id(audit_record) == target_id:
        found = True

        original = review_by_id[target_id]

        evidence = get_evidence(audit_record)

        original_text = str(original.get("review_text", ""))

        normalized_evidence = normalize_unicode(evidence)

        normalized_review = normalize_unicode(original_text)

        print("\n--- Audit Record ---")

        print("Fields:", list(audit_record.keys()))

        print("Aspect:", get_aspect_name(audit_record))

        print("Evidence:", repr(evidence))

        print("\n--- Original Review ---")

        print("Review:", repr(original_text[:2000]))

        print("\n--- Normalization Check ---")

        print(
            "Normalized evidence:",
            repr(normalized_evidence)
        )

        print(
            "Normalized review:",
            repr(normalized_review[:2000])
        )

        print(
            "Evidence match:",
            normalized_evidence in normalized_review
        )

        break

if not found:
    print("Record 695 not found in Unicode audit records.")


validate_audit_records(
    unicode_records,
    "Unicode"
)

validate_audit_records(
    review_records,
    "Manual review"
)


# Check whether audit entries exist in source data

for label, counter in (
    ("Unicode", unicode_counter),
    ("Manual review", review_counter)
):
    for key, count in counter.items():
        if count > source_counter[key]:
            raise ValueError(
                f"{label} record does not match "
                f"original extraction:\n{key}"
            )


# Prepare audit counters

unicode_remaining = unicode_counter.copy()

review_remaining = review_counter.copy()


# Audit all aspect evidence

print("\n[5] Auditing all aspect evidence...")

basic_valid = 0

unicode_valid = 0

needs_review = 0

unlisted_unicode = 0

unlisted_unmatched = 0

original_flag_mismatches = 0

processed_records = []


def parse_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().casefold() in (
            "true",
            "1",
            "yes"
        )

    return None


for record in raw_records:
    row_id = str(record["train_row_id"])

    review_text = str(record.get("review_text", ""))

    if not review_text.strip():
        raise ValueError(f"Empty review text: {row_id}")

    basic_review = normalize_basic(review_text)

    unicode_review = normalize_unicode(review_text)

    updated_aspects = []

    for aspect in record.get("aspects", []):
        item = dict(aspect)

        aspect_name = item.get(
            "aspect",
            ""
        )

        evidence = str(item.get("evidence", ""))

        key = make_key(
            row_id,
            aspect_name,
            evidence
        )

        basic_evidence = normalize_basic(evidence)

        unicode_evidence = normalize_unicode(evidence)

        # Original validation

        basic_match = (
            bool(basic_evidence)
            and basic_evidence in basic_review
        )

        # Extended Unicode validation

        unicode_match = (
            bool(unicode_evidence)
            and unicode_evidence in unicode_review
        )

        # Check original evidence_valid flag

        old_flag = parse_bool(item.get("evidence_valid"))

        if old_flag is not None:
            if old_flag != basic_match:
                original_flag_mismatches += 1

        # Classification 1: Original valid

        if basic_match:
            if (
                unicode_remaining[key] > 0
                or review_remaining[key] > 0
            ):
                raise ValueError(
                    f"Audit category conflicts with "
                    f"original validation: {key}"
                )

            item["evidence_valid"] = True

            item["evidence_match_type"] = ("casefold_whitespace")

            item["evidence_review_required"] = False

            basic_valid += 1

        # Classification 2: Unicode recovered

        elif unicode_remaining[key] > 0:
            if not unicode_match:
                raise ValueError(f"Unicode audit match failed: {key}")

            unicode_remaining[key] -= 1

            item["evidence_valid"] = True

            item["evidence_match_type"] = ("unicode_normalized")

            item["evidence_review_required"] = False

            unicode_valid += 1

        # Classification 3: Needs manual review

        elif review_remaining[key] > 0:
            if unicode_match:
                raise ValueError(
                    f"Manual review classification "
                    f"conflicts with Unicode matching: {key}"
                )

            review_remaining[key] -= 1

            item["evidence_valid"] = False

            item["evidence_match_type"] = ("needs_review")

            item["evidence_review_required"] = True

            needs_review += 1

        # Classification 4: Unlisted Unicode match

        elif unicode_match:
            item["evidence_valid"] = True

            item["evidence_match_type"] = ("unicode_normalized_unlisted")

            item["evidence_review_required"] = False

            unicode_valid += 1

            unlisted_unicode += 1

        # Classification 5: Unmatched evidence

        else:
            item["evidence_valid"] = False

            item["evidence_match_type"] = ("needs_review_unlisted")

            item["evidence_review_required"] = True

            needs_review += 1

            unlisted_unmatched += 1

        updated_aspects.append(item)

    # Preserve the original nested structure

    record["aspects"] = updated_aspects

    record["evidence_total"] = len(updated_aspects)

    record["evidence_valid_count"] = sum(
        item["evidence_valid"]
        for item in updated_aspects
    )

    record["evidence_review_required_count"] = sum(
        item["evidence_review_required"]
        for item in updated_aspects
    )

    processed_records.append(record)


# Check that all audit records were consumed

print("\n[6] Checking audit record consistency...")


remaining_unicode = sum(unicode_remaining.values())

remaining_review = sum(review_remaining.values())

if remaining_unicode > 0:
    raise ValueError(
        f"{remaining_unicode} Unicode audit records "
        f"were not matched to original aspects."
    )

if remaining_review > 0:
    raise ValueError(
        f"{remaining_review} manual review records "
        f"were not matched to original aspects."
    )

if original_flag_mismatches > 0:
    raise ValueError(
        f"Original evidence_valid flag mismatches: "
        f"{original_flag_mismatches}"
    )

print("Audit record matching: PASSED")


# Calculate final statistics

print("\n[7] Calculating final statistics...")


total_reviews = len(processed_records)

total_evidence = sum(
    len(record.get("aspects", []))
    for record in processed_records
)

unique_row_ids = len({
    str(record["train_row_id"])
    for record in processed_records
})

unique_users = len({
    str(record["user_id"])
    for record in processed_records
})

unique_items = len({
    str(record["parent_asin"])
    for record in processed_records
})

empty_aspect_reviews = sum(
    not record.get("aspects")
    for record in processed_records
)

unique_aspect_labels = len({
    normalize_label(aspect.get("aspect", ""))
    for record in processed_records
    for aspect in record.get("aspects", [])
})

total_valid = basic_valid + unicode_valid

validity_rate = (
    total_valid / total_evidence
    if total_evidence
    else 0
)


summary = {
    "total_reviews": total_reviews,
    "unique_train_row_ids": unique_row_ids,
    "unique_users": unique_users,
    "unique_items": unique_items,
    "empty_aspect_reviews": empty_aspect_reviews,
    "total_aspect_evidence_pairs": total_evidence,
    "unique_aspect_labels": unique_aspect_labels,
    "original_valid_evidence": basic_valid,
    "unicode_normalization_recovered": unicode_valid,
    "normalized_valid_evidence": total_valid,
    "evidence_requiring_review": needs_review,
    "normalized_match_rate": round(
        validity_rate,
        4
    ),
    "unlisted_unicode_matches": unlisted_unicode,
    "unlisted_unmatched_evidence": unlisted_unmatched,
    "original_flag_mismatches": original_flag_mismatches
}


# Compare with previous audit summary

print("\n[8] Comparing previous audit summary...")


if AUDIT_SUMMARY_FILE is not None:
    with open(
        AUDIT_SUMMARY_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        expected = json.load(f)

    fields_to_check = [
        "total_reviews",
        "unique_train_row_ids",
        "unique_users",
        "unique_items",
        "empty_aspect_reviews",
        "total_aspect_evidence_pairs",
        "unique_aspect_labels",
        "original_valid_evidence",
        "unicode_normalization_recovered",
        "normalized_valid_evidence",
        "evidence_requiring_review",
        "normalized_match_rate"
    ]

    mismatches = {}

    for field in fields_to_check:
        if field not in expected:
            continue

        if summary[field] != expected[field]:
            mismatches[field] = {
                "expected": expected[field],
                "actual": summary[field]
            }

    if mismatches:
        print(
            json.dumps(
                mismatches,
                indent=4,
                ensure_ascii=False
            )
        )

        raise RuntimeError(
            "Audit statistics do not match "
            "the previous audit summary. "
            "Output files were not generated."
        )

    print("Previous audit summary: PASSED")

else:
    print(
        "Previous audit summary not found. "
        "Historical comparison skipped."
    )


# Final integrity checks

print("\n[9] Running final integrity checks...")


assert total_reviews == unique_row_ids, ("Review count and unique row ID count mismatch.")

assert total_evidence == (basic_valid + unicode_valid + needs_review), "Evidence count mismatch."

assert len(processed_records) == len(raw_records), ("Some original reviews were lost.")

for record in processed_records:
    assert len(record["aspects"]) == (record["evidence_total"])

    assert record["evidence_valid_count"] + (
        record["evidence_review_required_count"]
    ) == record["evidence_total"]


print("Review count check: PASSED")

print("Evidence count check: PASSED")

print("Data preservation check: PASSED")


# Save final data

print("\n[10] Saving audited data...")


write_jsonl(
    OUTPUT_FILE,
    processed_records
)

write_json(
    SUMMARY_FILE,
    summary
)


print("\n" + "=" * 60)

print("ASPECT DATA PREPARATION COMPLETED")

print("=" * 60)

print(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    )
)

print("\nGenerated files:")

print(
    "Audited data:",
    OUTPUT_FILE
)

print(
    "Audit summary:",
    SUMMARY_FILE
)

print("\n" + "=" * 60)
