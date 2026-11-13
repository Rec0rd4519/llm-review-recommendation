from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median
from math import isclose, isfinite
import json
import csv
from product_category_mapping import load_product_category_mapping

# Configuration

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

REVIEW_FILE = DATA_DIR / "profile_data.jsonl"

ASPECT_FILE = (
    DATA_DIR / "user_review_aspect_sentiment.jsonl"
)

PRODUCT_CATEGORY_MAPPING_FILE = (
    DATA_DIR / "product_category_mapping_v2_1_audit_merged_341.csv"
)

OUTPUT_PROFILE = DATA_DIR / "item_profiles.jsonl"

OUTPUT_CSV = (
    DATA_DIR / "item_profile_aspect_scores.csv"
)

OUTPUT_SUMMARY = (
    DATA_DIR / "item_profile_summary.json"
)


# Expected counts for the current dataset version.
# Update these only if the source dataset changes.

EXPECTED_REVIEWS = 2029
EXPECTED_ITEMS = 341
EXPECTED_NESTED_ASPECTS = 9255
EXPECTED_REVIEW_ASPECT_RECORDS = 9215

EXPECTED_VALID_EVIDENCE = 9173
EXPECTED_EXCLUDED_EVIDENCE = 82


SENTIMENT_VALUES = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0
}

VALID_POLARITIES = {
    "positive",
    "negative",
    "neutral",
    "mixed"
}


# Utility functions

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

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                )

    return records


def write_jsonl(path, records):

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp,
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

    temp.replace(path)


def write_json(path, data):

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    temp.replace(path)


def write_csv(path, rows):

    if not rows:

        raise ValueError(
            "No item aspect records to export."
        )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()

        writer.writerows(rows)

    temp.replace(path)


def to_float(value, field_name):

    if value is None or value == "":
        return None

    try:

        result = float(value)

    except (ValueError, TypeError):

        raise ValueError(
            f"Invalid numeric value in {field_name}: {value}"
        )

    if not isfinite(result):

        raise ValueError(
            f"Non-finite value in {field_name}: {value}"
        )

    return result


def to_int(value, field_name):

    try:

        result = int(value)

    except (ValueError, TypeError):

        raise ValueError(
            f"Invalid integer in {field_name}: {value}"
        )

    if result < 0:

        raise ValueError(
            f"Negative count in {field_name}: {value}"
        )

    return result


def to_bool(value, field_name):

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        value = value.strip().casefold()

        if value in ("true", "1", "yes"):
            return True

        if value in ("false", "0", "no"):
            return False

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    raise ValueError(
        f"Invalid boolean in {field_name}: {value}"
    )


def unique_strings(values):

    result = set()

    for value in values:

        if value is None:
            continue

        if isinstance(value, list):

            for element in value:

                if element is not None and str(element).strip():
                    result.add(str(element).strip())

        elif str(value).strip():

            result.add(str(value).strip())

    return sorted(result)


def rating_distribution(values):

    counts = Counter(
        format(value, "g")
        for value in values
    )

    return dict(
        sorted(
            counts.items(),
            key=lambda item: float(item[0])
        )
    )


def safe_mean(values):

    return mean(values) if values else None


# Load input files

print("\n" + "=" * 65)

print("ITEM PROFILE CONSTRUCTION")

print("=" * 65)


for path in (
    REVIEW_FILE,
    ASPECT_FILE,
    PRODUCT_CATEGORY_MAPPING_FILE
):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


print("\n[1] Loading source data...")


reviews = load_jsonl(REVIEW_FILE)

aspect_records = load_jsonl(ASPECT_FILE)

product_mapping = load_product_category_mapping(
    PRODUCT_CATEGORY_MAPPING_FILE
)

print("Original reviews:", len(reviews))

print("Review-aspect records:", len(aspect_records))


# Build review and item indexes

print("\n[2] Building review indexes...")


review_index = {}

item_reviews = defaultdict(list)

for review in reviews:

    review_id = review.get("train_row_id")

    user_id = review.get("user_id")

    item_id = review.get("parent_asin")

    if review_id is None:
        raise ValueError("Missing train_row_id.")

    if user_id is None:
        raise ValueError(f"Missing user_id: {review_id}")

    if item_id is None:
        raise ValueError(f"Missing parent_asin: {review_id}")

    if review_id in review_index:

        raise ValueError(
            f"Duplicate review ID: {review_id}"
        )

    rating = to_float(
        review.get("rating"),
        "review.rating"
    )

    review_index[review_id] = {

        "user_id": user_id,

        "parent_asin": item_id,

        "rating": rating

    }

    item_reviews[item_id].append(review)


review_count = len(review_index)

item_count = len(item_reviews)

# Validate product category mapping against review items

review_item_ids = {
    str(item_id).strip()
    for item_id in item_reviews
}

mapping_item_ids = set(product_mapping.keys())

missing_mapping = sorted(
    review_item_ids - mapping_item_ids
)

extra_mapping = sorted(
    mapping_item_ids - review_item_ids
)

if missing_mapping or extra_mapping:

    raise ValueError(
        "Product category mapping ASIN mismatch.\n"
        f"Missing mapping: {missing_mapping}\n"
        f"Extra mapping: {extra_mapping}"
    )

print(
    "Product category mapping ASIN consistency: PASSED"
)

print(
    "Mapped products:",
    len(product_mapping)
)

reviews_without_aspects = sum(
    not review.get("aspects")
    for review in reviews
)

nested_aspect_count = sum(
    len(review.get("aspects", []))
    for review in reviews
)


print("Unique reviews:", review_count)

print("Unique items:", item_count)

print("Reviews without aspects:", reviews_without_aspects)

print("Original nested aspect records:", nested_aspect_count)


# Validate source dataset

print("\n[3] Validating source dataset...")


if review_count != EXPECTED_REVIEWS:

    raise ValueError(
        f"Expected {EXPECTED_REVIEWS} reviews, "
        f"found {review_count}."
    )


if item_count != EXPECTED_ITEMS:

    raise ValueError(
        f"Expected {EXPECTED_ITEMS} items, "
        f"found {item_count}."
    )


if nested_aspect_count != EXPECTED_NESTED_ASPECTS:

    raise ValueError(
        f"Expected {EXPECTED_NESTED_ASPECTS} nested aspects, "
        f"found {nested_aspect_count}."
    )


print("Source dataset validation: PASSED")


# Validate review-aspect records

print("\n[4] Validating review-aspect records...")


required_fields = {

    "user_id",
    "train_row_id",
    "parent_asin",
    "rating",
    "profile_aspect",
    "review_aspect_sentiment_score",
    "review_aspect_polarity",
    "valid_evidence_count",
    "excluded_evidence_count",
    "sentiment_evidence_counts",
    "evidence_records"

}


seen_review_aspect_keys = set()

item_aspect_groups = defaultdict(list)

item_review_ids = defaultdict(set)

total_valid_evidence = 0

total_excluded_evidence = 0

valid_review_aspect_count = 0

excluded_review_aspect_count = 0

evidence_sentiment_counts = Counter()

review_polarity_counts = Counter()

review_aspect_review_ids = set()


for index, row in enumerate(aspect_records, 1):

    missing = required_fields - set(row.keys())

    if missing:

        raise ValueError(
            f"Missing fields at record {index}: {missing}"
        )

    review_id = row["train_row_id"]

    user_id = row["user_id"]

    item_id = row["parent_asin"]

    aspect = str(
        row["profile_aspect"]
    ).strip()

    if not aspect:

        raise ValueError(
            f"Empty profile_aspect at record {index}"
        )

    if review_id not in review_index:

        raise ValueError(
            f"Unknown review ID: {review_id}"
        )

    base = review_index[review_id]

    if base["user_id"] != user_id:

        raise ValueError(
            f"User ID mismatch: {review_id}"
        )

    if base["parent_asin"] != item_id:

        raise ValueError(
            f"Item ID mismatch: {review_id}"
        )

    row_rating = to_float(
        row.get("rating"),
        "aspect.rating"
    )

    if row_rating != base["rating"]:

        raise ValueError(
            f"Rating mismatch: {review_id}"
        )

    # One record per review and aspect.

    key = (
        review_id,
        aspect
    )

    if key in seen_review_aspect_keys:

        raise ValueError(
            f"Duplicate review-aspect key: {key}"
        )

    seen_review_aspect_keys.add(key)

    score = to_float(
        row["review_aspect_sentiment_score"],
        "review_aspect_sentiment_score"
    )

    if score is not None and not (-1.0 <= score <= 1.0):

        raise ValueError(
            f"Score outside [-1, 1]: {key}"
        )

    valid_count = to_int(
        row["valid_evidence_count"],
        "valid_evidence_count"
    )

    excluded_count = to_int(
        row["excluded_evidence_count"],
        "excluded_evidence_count"
    )

    evidence_records = row["evidence_records"]

    if not isinstance(evidence_records, list):

        raise ValueError(
            f"evidence_records must be a list: {key}"
        )

    if len(evidence_records) != valid_count + excluded_count:

        raise ValueError(
            f"Evidence count mismatch: {key}"
        )

    actual_valid = 0

    actual_excluded = 0

    actual_sentiments = Counter()

    for evidence in evidence_records:

        is_valid = to_bool(
            evidence.get("evidence_valid"),
            "evidence_valid"
        )

        sentiment = str(
            evidence.get("sentiment", "")
        ).strip().casefold()

        if sentiment not in SENTIMENT_VALUES:

            raise ValueError(
                f"Invalid evidence sentiment: {sentiment}"
            )

        if is_valid:

            actual_valid += 1

            actual_sentiments[sentiment] += 1

        else:

            actual_excluded += 1

    if actual_valid != valid_count:

        raise ValueError(
            f"Valid evidence mismatch: {key}"
        )

    if actual_excluded != excluded_count:

        raise ValueError(
            f"Excluded evidence mismatch: {key}"
        )

    declared_sentiments = {

        str(k): int(v)

        for k, v in row["sentiment_evidence_counts"].items()

        if int(v) != 0

    }

    if declared_sentiments != dict(actual_sentiments):

        raise ValueError(
            f"Sentiment evidence count mismatch: {key}"
        )

    polarity = row["review_aspect_polarity"]

    if score is None:

        if valid_count != 0 or excluded_count == 0:

            raise ValueError(
                f"Inconsistent excluded-only record: {key}"
            )

        if polarity != "excluded_review_required":

            raise ValueError(
                f"Unexpected polarity for excluded record: {key}"
            )

        excluded_review_aspect_count += 1

    else:

        if valid_count == 0:

            raise ValueError(
                f"Scored record has no valid evidence: {key}"
            )

        if polarity not in VALID_POLARITIES:

            raise ValueError(
                f"Invalid review aspect polarity: {key}"
            )

        pos = actual_sentiments["positive"]

        neg = actual_sentiments["negative"]

        neutral = actual_sentiments["neutral"]

        expected_score = (
            pos - neg
        ) / valid_count

        if not isclose(
            score,
            expected_score,
            abs_tol=1e-9
        ):

            raise ValueError(
                f"Sentiment score mismatch: {key}"
            )

        if pos > 0 and neg > 0:
            expected_polarity = "mixed"

        elif pos > 0:
            expected_polarity = "positive"

        elif neg > 0:
            expected_polarity = "negative"

        else:
            expected_polarity = "neutral"

        if polarity != expected_polarity:

            raise ValueError(
                f"Polarity mismatch: {key}"
            )

        valid_review_aspect_count += 1

        review_polarity_counts[polarity] += 1

        evidence_sentiment_counts.update(
            actual_sentiments
        )

    total_valid_evidence += valid_count

    total_excluded_evidence += excluded_count

    item_aspect_groups[
        (item_id, aspect)
    ].append(row)

    item_review_ids[item_id].add(review_id)

    review_aspect_review_ids.add(review_id)


# Final source consistency checks

print("\n[5] Checking data consistency...")


if len(aspect_records) != EXPECTED_REVIEW_ASPECT_RECORDS:

    raise ValueError(
        "Review-aspect record count mismatch."
    )


if total_valid_evidence != EXPECTED_VALID_EVIDENCE:

    raise ValueError(
        "Valid evidence count mismatch."
    )


if total_excluded_evidence != EXPECTED_EXCLUDED_EVIDENCE:

    raise ValueError(
        "Excluded evidence count mismatch."
    )


if set(item_reviews) != {
    row["parent_asin"]
    for row in aspect_records
}:

    raise ValueError(
        "Item IDs differ between source review and aspect files."
    )


if len(seen_review_aspect_keys) != len(aspect_records):

    raise ValueError(
        "Duplicate review-aspect records detected."
    )


print("Review ID consistency: PASSED")

print("User ID consistency: PASSED")

print("Item ID consistency: PASSED")

print("Rating consistency: PASSED")

print("Evidence consistency: PASSED")

print("Sentiment score consistency: PASSED")


# Aggregate review-aspect records by item and aspect

print("\n[6] Aggregating item-level aspects...")


item_aspect_profiles = []

item_aspect_profile_index = defaultdict(list)

support_tier_counts = Counter()

performance_direction_counts = Counter()

item_aspect_score_values = []


for (item_id, aspect), rows in sorted(
    item_aspect_groups.items()
):

    valid_rows = [

        row for row in rows

        if row["review_aspect_sentiment_score"] is not None

    ]

    excluded_rows = [

        row for row in rows

        if row["review_aspect_sentiment_score"] is None

    ]

    scores = [

        float(row["review_aspect_sentiment_score"])

        for row in valid_rows

    ]

    # One review-aspect score contributes one vote.

    performance_score = safe_mean(scores)

    if performance_score is None:

        performance_direction = "insufficient_evidence"

        performance_strength = None

        support_tier = "insufficient_evidence"

    else:

        performance_strength = abs(
            performance_score
        )

        if performance_score > 1e-12:

            performance_direction = "positive"

        elif performance_score < -1e-12:

            performance_direction = "negative"

        else:

            performance_direction = "neutral_or_balanced"

        if len(valid_rows) == 1:

            support_tier = "single_review_support"

        else:

            support_tier = "multi_review_support"

        item_aspect_score_values.append(
            performance_score
        )

    support_tier_counts[support_tier] += 1

    performance_direction_counts[
        performance_direction
    ] += 1

    # Review-level sentiment counts

    polarity_counts = Counter(
        row["review_aspect_polarity"]
        for row in valid_rows
    )

    evidence_counts = Counter()

    for row in valid_rows:

        evidence_counts.update(
            row["sentiment_evidence_counts"]
        )

    # Taxonomy information

    raw_aspects = unique_strings([
        row.get("raw_aspect_labels", [])
        for row in rows
    ])

    categories = unique_strings([
        row.get("parent_category")
        for row in rows
    ])

    if len(categories) == 1:

        parent_category = categories[0]

    elif len(categories) > 1:

        parent_category = "Multiple/Conflicting"

    else:

        parent_category = None

    mapping_statuses = unique_strings([
        row.get("mapping_statuses", [])
        for row in rows
    ])

    profile_mapping_statuses = unique_strings([
        row.get("profile_mapping_statuses", [])
        for row in rows
    ])

    # Rating context

    aspect_ratings = [

        float(row["rating"])

        for row in rows

        if row.get("rating") is not None

    ]

    # Evidence examples

    evidence_examples = []

    ordered_rows = valid_rows + excluded_rows

    for row in ordered_rows:

        for evidence in row["evidence_records"]:

            if len(evidence_examples) >= 5:
                break

            evidence_examples.append({

                "train_row_id": row["train_row_id"],

                "user_id": row["user_id"],

                "evidence": evidence["evidence"],

                "sentiment": evidence["sentiment"],

                "evidence_valid": evidence["evidence_valid"],

                "evidence_match_type": evidence.get(
                    "evidence_match_type",
                    ""
                )

            })

        if len(evidence_examples) >= 5:
            break

    # Conflict detection

    has_conflicting_reviews = (

        polarity_counts["positive"] > 0

        and polarity_counts["negative"] > 0

    ) or polarity_counts["mixed"] > 0

    # Build item-aspect profile

    profile = {

        "parent_asin": item_id,

        "profile_aspect": aspect,

        "parent_category": parent_category,

        "parent_categories": categories,

        "raw_aspect_labels": raw_aspects,

        "mapping_statuses": mapping_statuses,

        "profile_mapping_statuses": (
            profile_mapping_statuses
        ),

        "total_review_aspect_count": len(rows),

        "support_review_count": len(valid_rows),

        "excluded_only_review_count": len(excluded_rows),

        "valid_evidence_count": sum(
            row["valid_evidence_count"]
            for row in rows
        ),

        "excluded_evidence_count": sum(
            row["excluded_evidence_count"]
            for row in rows
        ),

        "evidence_validity_rate": round(

            sum(row["valid_evidence_count"] for row in rows)

            / max(

                1,

                sum(
                    row["valid_evidence_count"]
                    + row["excluded_evidence_count"]
                    for row in rows
                )

            ),

            4

        ),

        "positive_evidence_count": evidence_counts["positive"],

        "neutral_evidence_count": evidence_counts["neutral"],

        "negative_evidence_count": evidence_counts["negative"],

        "positive_review_count": polarity_counts["positive"],

        "neutral_review_count": polarity_counts["neutral"],

        "negative_review_count": polarity_counts["negative"],

        "mixed_review_count": polarity_counts["mixed"],

        "performance_score": (

            round(performance_score, 6)

            if performance_score is not None

            else None

        ),

        "performance_strength": (

            round(performance_strength, 6)

            if performance_strength is not None

            else None

        ),

        "performance_direction": performance_direction,

        "support_tier": support_tier,

        "has_conflicting_reviews": (
            has_conflicting_reviews
        ),

        "rating_count_for_aspect": len(aspect_ratings),

        "mean_overall_rating_for_aspect": (

            round(mean(aspect_ratings), 4)

            if aspect_ratings else None

        ),

        "source_review_ids": sorted(

            {row["train_row_id"] for row in rows},

            key=str

        ),

        "source_user_ids": unique_strings([

            row["user_id"]

            for row in rows

        ]),

        "evidence_examples": evidence_examples

    }

    item_aspect_profiles.append(profile)

    item_aspect_profile_index[item_id].append(
        profile
    )


# Build one complete profile per item

print("\n[7] Building complete item profiles...")


item_profiles = []

for item_id in sorted(item_reviews):

    source_reviews = item_reviews[item_id]

    # Retrieve manually reviewed product metadata
    product_metadata = product_mapping[item_id]

    aspect_profiles = item_aspect_profile_index.get(
        item_id,
        []
    )

    ratings = [

        float(review["rating"])

        for review in source_reviews

        if review.get("rating") is not None

    ]

    aspect_review_count = len(
        item_review_ids[item_id]
    )

    no_aspect_count = (
        len(source_reviews) - aspect_review_count
    )

    item_valid_evidence = sum(
        profile["valid_evidence_count"]
        for profile in aspect_profiles
    )

    item_excluded_evidence = sum(
        profile["excluded_evidence_count"]
        for profile in aspect_profiles
    )

    # Sort by the amount of review support.

    aspect_profiles = sorted(

        aspect_profiles,

        key=lambda x: (

            -x["support_review_count"],

            x["profile_aspect"]

        )

    )

    item_profile = {

        "parent_asin": item_id,
        # Product metadata from audited V2.1 mapping

        "product_category": product_metadata["product_category"],

        "product_type": product_metadata["product_type"],

        "scope_status": product_metadata["scope_status"],

        "audit_status": product_metadata["audit_status"],

        "final_mapping_resolution": product_metadata["mapping_resolution"],

        "audit_id": product_metadata["audit_id"],

        "audit_decision": product_metadata["audit_decision"],

        "dataset_review_count": len(source_reviews),

        "review_count_with_aspect_records": (
            aspect_review_count
        ),

        "review_count_without_aspect_records": (
            no_aspect_count
        ),

        "rating_count": len(ratings),

        "mean_overall_rating": (

            round(mean(ratings), 4)

            if ratings else None

        ),

        "median_overall_rating": (

            round(median(ratings), 4)

            if ratings else None

        ),

        "rating_distribution": rating_distribution(
            ratings
        ),

        "aspect_profile_count": len(aspect_profiles),

        "total_review_aspect_records": sum(

            profile["total_review_aspect_count"]

            for profile in aspect_profiles

        ),

        "valid_evidence_count": item_valid_evidence,

        "excluded_evidence_count": item_excluded_evidence,

        "source_review_ids": sorted(

            {

                review["train_row_id"]

                for review in source_reviews

            },

            key=str

        ),

        "source_user_ids": unique_strings([

            review["user_id"]

            for review in source_reviews

        ]),

        "aspects": aspect_profiles

    }

    item_profiles.append(
        item_profile
    )


# Integrity checks

print("\n[8] Running final integrity checks...")


assert len(item_profiles) == EXPECTED_ITEMS

assert len(item_aspect_profiles) == len(
    item_aspect_groups
)

assert sum(
    p["total_review_aspect_count"]
    for p in item_aspect_profiles
) == EXPECTED_REVIEW_ASPECT_RECORDS

assert sum(
    p["valid_evidence_count"]
    for p in item_aspect_profiles
) == EXPECTED_VALID_EVIDENCE

assert sum(
    p["excluded_evidence_count"]
    for p in item_aspect_profiles
) == EXPECTED_EXCLUDED_EVIDENCE

assert all(

    p["performance_score"] is None

    or -1.0 <= p["performance_score"] <= 1.0

    for p in item_aspect_profiles

)

assert len({

    (p["parent_asin"], p["profile_aspect"])

    for p in item_aspect_profiles

}) == len(item_aspect_profiles)


print("Item count validation: PASSED")

print("Review-aspect preservation: PASSED")

print("Evidence preservation: PASSED")

print("Unique item-aspect keys: PASSED")

print("Score range validation: PASSED")


# Build summary

print("\n[9] Generating summary...")


all_ratings = [

    float(review["rating"])

    for review in reviews

    if review.get("rating") is not None

]

# Product metadata statistics

product_scope_counts = Counter(
    metadata["scope_status"]
    for metadata in product_mapping.values()
)

product_audit_counts = Counter(
    metadata["audit_status"]
    for metadata in product_mapping.values()
)

product_category_counts = Counter(
    metadata["product_category"]
    for metadata in product_mapping.values()
    if (
        metadata["scope_status"] == "in_scope"
        and metadata["product_category"]
    )
)

items_with_product_category = sum(
    bool(metadata["product_category"])
    for metadata in product_mapping.values()
)

items_with_product_type = sum(
    bool(metadata["product_type"])
    for metadata in product_mapping.values()
)


summary = {

    "total_items": len(item_profiles),

    "product_metadata": {

        "mapping_source": (
            PRODUCT_CATEGORY_MAPPING_FILE.name
        ),

        "total_mapped_items": len(product_mapping),

        "items_with_product_category": (
            items_with_product_category
        ),

        "items_with_product_type": (
            items_with_product_type
        ),

        "scope_status_counts": dict(
            sorted(product_scope_counts.items())
        ),

        "audit_status_counts": dict(
            sorted(product_audit_counts.items())
        ),

        "product_category_counts": dict(
            sorted(product_category_counts.items())
        )

    },

    "total_reviews": len(reviews),

    "reviews_with_aspect_records": len(
        review_aspect_review_ids
    ),

    "reviews_without_aspects": reviews_without_aspects,

    "total_review_aspect_records": len(aspect_records),

    "unique_item_aspect_groups": len(
        item_aspect_profiles
    ),

    "valid_review_aspect_records": (
        valid_review_aspect_count
    ),

    "excluded_only_review_aspect_records": (
        excluded_review_aspect_count
    ),

    "total_valid_evidence": total_valid_evidence,

    "total_excluded_evidence": total_excluded_evidence,

    "evidence_validity_rate": round(

        total_valid_evidence

        / (total_valid_evidence + total_excluded_evidence),

        4

    ),

    "support_tier_counts": dict(
        support_tier_counts
    ),

    "performance_direction_counts": dict(
        performance_direction_counts
    ),

    "review_aspect_polarity_counts": dict(
        review_polarity_counts
    ),

    "evidence_sentiment_counts": dict(
        evidence_sentiment_counts
    ),

    "item_review_count_min": min(
        len(x) for x in item_reviews.values()
    ),

    "item_review_count_max": max(
        len(x) for x in item_reviews.values()
    ),

    "mean_overall_rating_across_dataset": (

        round(mean(all_ratings), 4)

        if all_ratings else None

    ),

    "item_aspect_score_statistics": {

        "count": len(item_aspect_score_values),

        "mean": (

            round(mean(item_aspect_score_values), 6)

            if item_aspect_score_values else None

        ),

        "median": (

            round(median(item_aspect_score_values), 6)

            if item_aspect_score_values else None

        ),

        "minimum": (

            min(item_aspect_score_values)

            if item_aspect_score_values else None

        ),

        "maximum": (

            max(item_aspect_score_values)

            if item_aspect_score_values else None

        )

    },

    "scoring_method": {

        "positive": 1.0,

        "neutral": 0.0,

        "negative": -1.0,

        "item_performance_score": (

            "Mean review-aspect sentiment score across "
            "distinct reviews for the same item and aspect."

        ),

        "performance_strength": (

            "Absolute value of item performance score."

        ),

        "aggregation_unit": (

            "One distinct review-aspect record contributes "
            "one score, regardless of the number of evidence spans."

        ),

        "evidence_policy": (

            "Only records with valid review-aspect scores "
            "contribute to primary performance scores. "
            "Excluded-only records remain in the output."

        ),

        "rating_policy": (

            "Overall ratings are calculated once per unique review, "
            "including reviews without extracted aspects."

        ),

        "taxonomy_policy": (

            "Use profile_aspect. Unapproved aliases remain separate."

        ),

        "support_tier_policy": (

            "Single-review, multi-review and insufficient-evidence "
            "are support-volume tiers, not statistical confidence."

        )

    },

    "limitations": [

        "Profiles are derived from the available dataset reviews, "
        "not the complete lifetime review history of each product.",

        "Thirty-eight reviews have no extracted aspects. "
        "They contribute to item rating context but not aspect scores.",

        "Support tiers describe the number of supporting reviews "
        "and should not be interpreted as statistical confidence.",

        "Sentiment labels are model-derived and are not independent "
        "ground-truth annotations.",

        "The profile integrates V2.1 product category/type mapping, "
        "scope status and audit metadata. The original review-derived "
        "aspect hierarchy (parent_category) remains a separate source "
        "and has not been overwritten.",

        "The metadata integration is limited to the V2.1 category, "
        "type, scope and audit fields; it does not incorporate all "
        "raw product metadata fields or full product descriptions."
    ]

}


# Export CSV

print("\n[10] Exporting aspect-level CSV...")


csv_rows = []

for profile in item_aspect_profiles:

    product_metadata = product_mapping[
        profile["parent_asin"]
    ]

    csv_rows.append({

        "parent_asin": profile["parent_asin"],

        "product_category": (
            product_metadata["product_category"] or ""
        ),

        "product_type": (
            product_metadata["product_type"] or ""
        ),

        "scope_status": (
            product_metadata["scope_status"]
        ),

        "audit_status": (
            product_metadata["audit_status"]
        ),

        "final_mapping_resolution": (
            product_metadata["mapping_resolution"] or ""
        ),

        "audit_id": (
            product_metadata["audit_id"] or ""
        ),

        "audit_decision": (
            product_metadata["audit_decision"] or ""
        ),

        "profile_aspect": profile["profile_aspect"],

        "parent_category": profile["parent_category"] or "",

        "parent_categories": json.dumps(
            profile["parent_categories"],
            ensure_ascii=False
        ),

        "total_review_aspect_count": (
            profile["total_review_aspect_count"]
        ),

        "support_review_count": (
            profile["support_review_count"]
        ),

        "excluded_only_review_count": (
            profile["excluded_only_review_count"]
        ),

        "valid_evidence_count": (
            profile["valid_evidence_count"]
        ),

        "excluded_evidence_count": (
            profile["excluded_evidence_count"]
        ),

        "positive_review_count": (
            profile["positive_review_count"]
        ),

        "neutral_review_count": (
            profile["neutral_review_count"]
        ),

        "negative_review_count": (
            profile["negative_review_count"]
        ),

        "mixed_review_count": (
            profile["mixed_review_count"]
        ),

        "positive_evidence_count": (
            profile["positive_evidence_count"]
        ),

        "neutral_evidence_count": (
            profile["neutral_evidence_count"]
        ),

        "negative_evidence_count": (
            profile["negative_evidence_count"]
        ),

        "performance_score": (
            profile["performance_score"]
        ),

        "performance_strength": (
            profile["performance_strength"]
        ),

        "performance_direction": (
            profile["performance_direction"]
        ),

        "support_tier": (
            profile["support_tier"]
        ),

        "mean_overall_rating_for_aspect": (
            profile["mean_overall_rating_for_aspect"]
        ),

        "has_conflicting_reviews": (
            profile["has_conflicting_reviews"]
        ),

        "mapping_statuses": json.dumps(
            profile["mapping_statuses"],
            ensure_ascii=False
        ),

        "profile_mapping_statuses": json.dumps(
            profile["profile_mapping_statuses"],
            ensure_ascii=False
        ),

        "source_review_ids": json.dumps(
            profile["source_review_ids"],
            ensure_ascii=False
        ),

        "source_user_ids": json.dumps(
            profile["source_user_ids"],
            ensure_ascii=False
        ),

        "evidence_examples": json.dumps(
            profile["evidence_examples"],
            ensure_ascii=False
        )

    })


# Save outputs

print("\n[11] Saving output files...")


write_jsonl(
    OUTPUT_PROFILE,
    item_profiles
)

write_csv(
    OUTPUT_CSV,
    csv_rows
)

write_json(
    OUTPUT_SUMMARY,
    summary
)


# Final report

print("\n" + "=" * 65)

print("ITEM PROFILE CONSTRUCTION COMPLETED")

print("=" * 65)

print(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    )
)

print("\nGenerated files:")

print("Item profiles:", OUTPUT_PROFILE)

print("Aspect scores:", OUTPUT_CSV)

print("Summary:", OUTPUT_SUMMARY)

print("=" * 65)
