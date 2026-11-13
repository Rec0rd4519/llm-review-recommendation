from pathlib import Path
from collections import Counter, defaultdict
import json
import csv


# Project paths

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"

INPUT_FILE = (
    DATA_DIR / "aspect_extraction_audited.jsonl"
)

TAXONOMY_FILE = (
    DATA_DIR / "aspect_taxonomy_v2.csv"
)

OUTPUT_REVIEW_FILE = (
    DATA_DIR / "profile_data.jsonl"
)

OUTPUT_ASPECT_FILE = (
    DATA_DIR / "profile_aspect_records.jsonl"
)

OUTPUT_SUMMARY_FILE = (
    DATA_DIR / "profile_data_summary.json"
)


# Utility functions

def normalize_label(value):

    return " ".join(
        str(value).strip().casefold().split()
    )


def parse_bool(value, field_name):

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        value = value.strip().casefold()

        if value in ("true", "1", "yes"):
            return True

        if value in ("false", "0", "no"):
            return False

    if isinstance(value, (int, float)):

        if value in (0, 1):
            return bool(value)

    raise ValueError(
        f"Invalid boolean value for {field_name}: {value}"
    )


def load_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as f:

        for line_number, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {e}"
                )

    return records


def load_taxonomy(path):

    taxonomy = {}

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required_fields = {

            "raw_aspect",
            "canonical_aspect",
            "parent_category",
            "mapping_status",
            "mapping_confidence"

        }

        missing = (
            required_fields - set(reader.fieldnames or [])
        )

        if missing:

            raise ValueError(
                f"Missing taxonomy columns: {missing}"
            )

        for row in reader:

            key = normalize_label(
                row["raw_aspect"]
            )

            if not key:

                raise ValueError(
                    "Empty raw_aspect in taxonomy."
                )

            if key in taxonomy:

                raise ValueError(
                    f"Duplicate taxonomy label: {key}"
                )

            taxonomy[key] = row

    return taxonomy


def write_jsonl(path, records):

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    temp_path.replace(path)


def write_json(path, data):

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    temp_path.replace(path)


# Validate input files

print("\n" + "=" * 60)

print("PROFILE DATA PREPARATION")

print("=" * 60)

for path in (
    INPUT_FILE,
    TAXONOMY_FILE
):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# Load source data

print("\n[1] Loading source data...")

reviews = load_jsonl(
    INPUT_FILE
)

taxonomy = load_taxonomy(
    TAXONOMY_FILE
)

print("Reviews loaded:", len(reviews))

print("Taxonomy labels loaded:", len(taxonomy))


# Initialize statistics

review_ids = set()

user_ids = set()

item_ids = set()

source_aspect_frequency = Counter()

sentiment_counts = Counter()

evidence_counts = Counter()

mapping_status_counts = Counter()

category_counts = Counter()

evidence_match_counts = Counter()

unmapped_labels = Counter()

taxonomy_label_status_counts = Counter(

    row["mapping_status"]

    for row in taxonomy.values()

)

taxonomy_approved_count = 0

reviews_without_aspects = 0

aspect_records = []

processed_reviews = []


# Process reviews and join taxonomy

print("\n[2] Joining reviews with taxonomy...")


for review_index, review in enumerate(reviews, 1):

    row_id = review.get("train_row_id")

    user_id = review.get("user_id")

    item_id = review.get("parent_asin")

    if row_id is None:

        raise ValueError(
            f"Missing train_row_id at record {review_index}"
        )

    if user_id is None or not str(user_id).strip():

        raise ValueError(
            f"Missing user_id: {row_id}"
        )

    if item_id is None or not str(item_id).strip():

        raise ValueError(
            f"Missing parent_asin: {row_id}"
        )

    if row_id in review_ids:

        raise ValueError(
            f"Duplicate train_row_id: {row_id}"
        )

    review_ids.add(row_id)

    user_ids.add(user_id)

    item_ids.add(item_id)

    if review.get("status") not in (None, "success"):

        raise ValueError(
            f"Unsuccessful extraction: {row_id}"
        )

    original_aspects = review.get(
        "aspects",
        []
    )

    if not original_aspects:

        reviews_without_aspects += 1

    enriched_aspects = []

    for aspect in original_aspects:

        raw_aspect = normalize_label(
            aspect.get("aspect", "")
        )

        if not raw_aspect:

            raise ValueError(
                f"Empty aspect label in review {row_id}"
            )

        evidence = str(
            aspect.get("evidence", "")
        )

        if not evidence.strip():

            raise ValueError(
                f"Empty evidence in review {row_id}"
            )

        sentiment = str(
            aspect.get("sentiment", "")
        ).strip().casefold()

        if sentiment not in (
            "positive",
            "negative",
            "neutral"
        ):

            raise ValueError(
                f"Invalid sentiment in review {row_id}: "
                f"{sentiment}"
            )

        evidence_valid = parse_bool(
            aspect.get("evidence_valid"),
            "evidence_valid"
        )

        evidence_review_required = parse_bool(
            aspect.get("evidence_review_required"),
            "evidence_review_required"
        )

        # Evidence validity and manual-review status
        # must be mutually exclusive.

        if evidence_valid == evidence_review_required:

            raise ValueError(
                f"Inconsistent evidence flags: {row_id}"
            )

        # Find taxonomy mapping

        tax = taxonomy.get(raw_aspect)

        if tax is None:

            candidate_canonical = raw_aspect

            parent_category = "Unassigned"

            mapping_status = "unmapped"

            mapping_confidence = "not_assessed"

            mapping_reason = "No taxonomy entry found."

            taxonomy_review_required = True

            review_decision = ""

            approved_canonical = ""

            approved_category = ""

        else:

            candidate_canonical = normalize_label(
                tax.get("canonical_aspect", raw_aspect)
            )

            parent_category = (
                tax.get("parent_category", "Unassigned")
                or "Unassigned"
            )

            mapping_status = (
                tax.get("mapping_status", "unreviewed")
            )

            mapping_confidence = (
                tax.get("mapping_confidence", "not_assessed")
            )

            mapping_reason = (
                tax.get("mapping_reason", "")
            )

            taxonomy_review_required = parse_bool(
                tax.get("review_required", "false"),
                "taxonomy.review_required"
            )

            review_decision = str(
                tax.get("review_decision", "")
            ).strip().casefold()

            approved_canonical = str(
                tax.get("approved_canonical_aspect", "")
            ).strip()

            approved_category = str(
                tax.get("approved_parent_category", "")
            ).strip()

        # Apply only explicitly approved mappings

        approved = review_decision in (
            "approved",
            "approve"
        )

        if approved:

            if (
                mapping_status == "candidate_alias"
                and not approved_canonical
            ):

                raise ValueError(
                    f"Approved alias lacks approved canonical aspect: "
                    f"{raw_aspect}"
                )

            profile_aspect = normalize_label(
                approved_canonical
                if approved_canonical
                else raw_aspect
            )

            profile_category = (
                approved_category
                if approved_category
                else parent_category
            )

            profile_mapping_status = "approved"

            taxonomy_approved_count += 1

        else:

            # Do not merge unapproved candidate aliases.

            profile_aspect = raw_aspect

            profile_category = (
                parent_category
                if parent_category != "Unassigned"
                else None
            )

            profile_mapping_status = (
                "raw_label_retained"
            )

        # Build enriched aspect

        enriched = dict(aspect)

        enriched.update({

            "raw_aspect": raw_aspect,

            "candidate_canonical_aspect": (
                candidate_canonical
            ),

            "parent_category": parent_category,

            "mapping_status": mapping_status,

            "mapping_confidence": mapping_confidence,

            "mapping_reason": mapping_reason,

            "taxonomy_review_required": (
                taxonomy_review_required
            ),

            "profile_aspect": profile_aspect,

            "profile_category": profile_category,

            "profile_mapping_status": (
                profile_mapping_status
            )

        })

        enriched_aspects.append(
            enriched
        )

        # Create one flat record per aspect

        flat_record = {

            "train_row_id": row_id,

            "user_id": user_id,

            "parent_asin": item_id,

            "rating": review.get("rating"),

            "review_title": review.get(
                "review_title", ""
            ),

            "overall_sentiment": review.get(
                "overall_sentiment", ""
            ),

            "raw_aspect": raw_aspect,

            "candidate_canonical_aspect": (
                candidate_canonical
            ),

            "profile_aspect": profile_aspect,

            "parent_category": parent_category,

            "profile_category": profile_category,

            "mapping_status": mapping_status,

            "mapping_confidence": mapping_confidence,

            "taxonomy_review_required": (
                taxonomy_review_required
            ),

            "profile_mapping_status": (
                profile_mapping_status
            ),

            "sentiment": sentiment,

            "evidence": evidence,

            "evidence_valid": evidence_valid,

            "evidence_match_type": aspect.get(
                "evidence_match_type", ""
            ),

            "evidence_review_required": (
                evidence_review_required
            )

        }

        aspect_records.append(
            flat_record
        )

        # Update statistics

        source_aspect_frequency[raw_aspect] += 1

        sentiment_counts[sentiment] += 1

        evidence_counts[
            "valid" if evidence_valid
            else "review_required"
        ] += 1

        mapping_status_counts[
            mapping_status
        ] += 1

        evidence_match_counts[
            aspect.get("evidence_match_type", "unknown")
        ] += 1

        if parent_category != "Unassigned":

            category_counts[parent_category] += 1

        if mapping_status == "unmapped":

            unmapped_labels[raw_aspect] += 1

    # Preserve original review fields

    processed_review = dict(review)

    processed_review["aspects"] = enriched_aspects

    processed_reviews.append(
        processed_review
    )

    if review_index % 500 == 0:

        print(
            f"Processed {review_index}/{len(reviews)} reviews..."
        )


# Validate taxonomy frequency consistency

print("\n[3] Validating taxonomy consistency...")


frequency_mismatches = {}

for label, count in source_aspect_frequency.items():

    if label not in taxonomy:

        continue

    expected = int(
        taxonomy[label]["frequency"]
    )

    if expected != count:

        frequency_mismatches[label] = {

            "source_count": count,

            "taxonomy_count": expected

        }


if frequency_mismatches:

    raise ValueError(
        "Taxonomy frequency mismatch:\n"
        + json.dumps(
            frequency_mismatches,
            indent=4,
            ensure_ascii=False
        )
    )


taxonomy_only_labels = sorted(
    set(taxonomy) - set(source_aspect_frequency)
)


print("Taxonomy frequency consistency: PASSED")


# Final integrity checks

print("\n[4] Running final integrity checks...")


total_reviews = len(
    processed_reviews
)

total_aspect_pairs = len(
    aspect_records
)

valid_evidence = evidence_counts["valid"]

review_required_evidence = evidence_counts[
    "review_required"
]


assert total_reviews == len(review_ids)

assert total_aspect_pairs == (
    valid_evidence + review_required_evidence
)

assert total_aspect_pairs == sum(
    source_aspect_frequency.values()
)

assert len(processed_reviews) == len(reviews)


for record in processed_reviews:

    assert len(record["aspects"]) == len(
        review_by_aspects := record.get("aspects", [])
    )

assert sum(mapping_status_counts.values()) == (
    total_aspect_pairs
)

print("Review preservation: PASSED")

print("Aspect-evidence preservation: PASSED")

print("Evidence status consistency: PASSED")

print("Mapping status consistency: PASSED")


# Build summary

print("\n[5] Building summary...")


summary = {

    "total_reviews": total_reviews,

    "unique_train_row_ids": len(review_ids),

    "unique_users": len(user_ids),

    "unique_items": len(item_ids),

    "reviews_without_aspects": reviews_without_aspects,

    "total_aspect_evidence_pairs": total_aspect_pairs,

    "unique_raw_aspects": len(source_aspect_frequency),

    "unique_profile_aspects": len({

        row["profile_aspect"]

        for row in aspect_records

    }),

    "valid_evidence_pairs": valid_evidence,

    "evidence_requiring_manual_review": (
        review_required_evidence
    ),

    "evidence_validity_rate": round(

        valid_evidence / total_aspect_pairs,

        4

    ) if total_aspect_pairs else 0,

    "sentiment_counts": dict(sentiment_counts),

    "evidence_match_type_counts": dict(
        evidence_match_counts
    ),

    "taxonomy_mapping_status_counts": dict(
        mapping_status_counts
    ),

    "taxonomy_label_status_counts": dict(
        taxonomy_label_status_counts
    ),

    "parent_category_counts": dict(
        category_counts
    ),

    "taxonomy_assigned_pairs": sum(
        category_counts.values()
    ),

    "taxonomy_unassigned_pairs": (
        total_aspect_pairs - sum(category_counts.values())
    ),

    "unmapped_aspect_pairs": sum(
        unmapped_labels.values()
    ),

    "unmapped_aspect_labels": dict(
        unmapped_labels
    ),

    "approved_mappings_applied": (
        taxonomy_approved_count
    ),

    "taxonomy_only_labels": len(
        taxonomy_only_labels
    ),

    "taxonomy_frequency_mismatches": len(
        frequency_mismatches
    ),

    "profile_mapping_policy": (

        "Use raw aspect labels for profile aggregation "
        "unless a taxonomy mapping has been explicitly approved. "
        "Candidate canonical mappings are retained separately."

    ),

    "evidence_policy": (

        "Only evidence_valid=True records should contribute "
        "to primary sentiment-based profile scores. "
        "Records requiring manual evidence review are retained "
        "for traceability but excluded from primary scoring."

    )

}


# Save outputs

print("\n[6] Saving profile preparation data...")


write_jsonl(
    OUTPUT_REVIEW_FILE,
    processed_reviews
)

write_jsonl(
    OUTPUT_ASPECT_FILE,
    aspect_records
)

write_json(
    OUTPUT_SUMMARY_FILE,
    summary
)


# Final report

print("\n" + "=" * 60)

print("PROFILE DATA PREPARATION COMPLETED")

print("=" * 60)

print(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    )
)

print("\nGenerated files:")

print("Review-level data:", OUTPUT_REVIEW_FILE)

print("Aspect-level data:", OUTPUT_ASPECT_FILE)

print("Summary:", OUTPUT_SUMMARY_FILE)

print("\n" + "=" * 60)
