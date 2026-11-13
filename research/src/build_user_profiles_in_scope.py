from pathlib import Path
from collections import Counter, defaultdict
import csv
import json


# Paths and configuration

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

REVIEW_FILE = DATA_DIR / "profile_data.jsonl"

ASPECT_FILE = DATA_DIR / "profile_aspect_records.jsonl"

PRODUCT_MAPPING_FILE = (
    DATA_DIR / "product_category_mapping_v2_1_audit_merged_341.csv"
)

OUTPUT_DIR = DATA_DIR

# Independent In-Scope outputs.
# Original all-product outputs remain unchanged.

REVIEW_ASPECT_OUTPUT = (
    OUTPUT_DIR / "user_review_aspect_sentiment_in_scope.jsonl"
)

USER_PROFILE_OUTPUT = (
    OUTPUT_DIR / "user_profiles_in_scope.jsonl"
)

USER_PROFILE_CSV = (
    OUTPUT_DIR / "user_profile_aspect_scores_in_scope.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "user_profile_summary_in_scope.json"
)


SENTIMENT_SCORE = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


# Utility functions

def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8-sig") as f:

        for line_number, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                )

    return records


def write_jsonl(path, records):

    temp = path.with_suffix(path.suffix + ".tmp")

    with open(temp, "w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    temp.replace(path)


def write_json(path, data):

    temp = path.with_suffix(path.suffix + ".tmp")

    with open(temp, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    temp.replace(path)


def write_csv(path, rows):

    if not rows:
        raise ValueError("No profile rows to save.")

    temp = path.with_suffix(path.suffix + ".tmp")

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


def to_bool(value, field):

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        value = value.strip().casefold()

        if value in ("true", "1", "yes"):
            return True

        if value in ("false", "0", "no"):
            return False

    raise ValueError(
        f"Invalid boolean in {field}: {value}"
    )


def mean(values):

    return (
        sum(values) / len(values)
        if values else None
    )


# Load and filter In-Scope data

print("\n" + "=" * 60)
print("IN-SCOPE USER PROFILE CONSTRUCTION")
print("=" * 60)

for path in (
    REVIEW_FILE,
    ASPECT_FILE,
    PRODUCT_MAPPING_FILE
):
    if not path.exists():
        raise FileNotFoundError(path)


# 1 Load original data

all_reviews = load_jsonl(REVIEW_FILE)
all_aspect_rows = load_jsonl(ASPECT_FILE)

print("\n[1] Loading original data")
print("Original reviews:", len(all_reviews))
print("Original aspect records:", len(all_aspect_rows))


# 2 Load V2.1 product scope mapping

product_scope = {}

with open(
    PRODUCT_MAPPING_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        asin = row["parent_asin"].strip()
        scope = row["final_scope_status"].strip()

        if not asin:
            raise ValueError("Empty parent_asin in mapping.")

        if asin in product_scope:
            raise ValueError(f"Duplicate mapping ASIN: {asin}")

        if scope not in {
            "in_scope",
            "mixed_bundle",
            "out_of_scope",
            "needs_review"
        }:
            raise ValueError(
                f"Invalid scope status for {asin}: {scope}"
            )

        product_scope[asin] = scope


# 3 Preserve all original user IDs

all_user_ids = {
    review["user_id"]
    for review in all_reviews
}


# 4 Validate original review IDs and ASIN mapping

all_review_by_id = {}

for review in all_reviews:

    review_id = review["train_row_id"]
    asin = review["parent_asin"]

    if review_id in all_review_by_id:
        raise ValueError(
            f"Duplicate review ID: {review_id}"
        )

    if asin not in product_scope:
        raise ValueError(
            f"Missing product mapping: {asin}"
        )

    all_review_by_id[review_id] = review


# 5 Filter In-Scope reviews

reviews = [
    review
    for review in all_reviews
    if product_scope[review["parent_asin"]] == "in_scope"
]

in_scope_review_ids = {
    review["train_row_id"]
    for review in reviews
}


# 6 Filter In-Scope aspect records

aspect_rows = []

for row in all_aspect_rows:

    review_id = row["train_row_id"]

    source_review = all_review_by_id.get(review_id)

    if source_review is None:
        raise ValueError(
            f"Orphan aspect record: {review_id}"
        )

    if row["parent_asin"] != source_review["parent_asin"]:
        raise ValueError(
            f"Aspect ASIN mismatch: {review_id}"
        )

    if row["user_id"] != source_review["user_id"]:
        raise ValueError(
            f"Aspect user mismatch: {review_id}"
        )

    if review_id in in_scope_review_ids:
        aspect_rows.append(row)


# 7 Validate filtered dataset

users_with_in_scope_reviews = {
    review["user_id"]
    for review in reviews
}

if len(product_scope) != 341:
    raise ValueError(
        f"Expected 341 mapped products, found {len(product_scope)}"
    )

if len(all_reviews) != 2029:
    raise ValueError(
        f"Expected 2029 original reviews, found {len(all_reviews)}"
    )

if len(reviews) != 1792:
    raise ValueError(
        f"Expected 1792 In-Scope reviews, found {len(reviews)}"
    )

if len(aspect_rows) != 8161:
    raise ValueError(
        f"Expected 8161 In-Scope aspect records, found {len(aspect_rows)}"
    )

if len(all_user_ids) != 253:
    raise ValueError(
        f"Expected 253 original users, found {len(all_user_ids)}"
    )

if len(users_with_in_scope_reviews) != 251:
    raise ValueError(
        "Unexpected number of users with In-Scope reviews."
    )


print("\n[2] In-Scope filtering results")

print("Mapped products:", len(product_scope))
print("Original users:", len(all_user_ids))
print("In-Scope reviews:", len(reviews))
print("In-Scope aspect records:", len(aspect_rows))
print(
    "Users with In-Scope reviews:",
    len(users_with_in_scope_reviews)
)
print(
    "Users without In-Scope reviews:",
    len(all_user_ids - users_with_in_scope_reviews)
)

print("In-Scope filtering validation: PASSED")


# Build review and user indexes

print("\n[2] Validating IDs")

review_by_id = {}

# Preserve all original users, including those with
# no In-Scope review history.

user_meta = {
    user_id: {
        "review_ids": set(),
        "reviews_with_aspects": set(),
        "ratings": []
    }
    for user_id in all_user_ids
}

nested_counts = Counter()

for review in reviews:

    review_id = review.get("train_row_id")
    user_id = review.get("user_id")

    if review_id is None or user_id is None:
        raise ValueError("Missing review or user ID.")

    if review_id in review_by_id:
        raise ValueError(f"Duplicate review ID: {review_id}")

    review_by_id[review_id] = review

    nested_counts[review_id] = len(
        review.get("aspects", [])
    )

    meta = user_meta.setdefault(
        user_id,
        {
            "review_ids": set(),
            "reviews_with_aspects": set(),
            "ratings": []
        }
    )

    meta["review_ids"].add(review_id)

    if review.get("aspects"):
        meta["reviews_with_aspects"].add(review_id)

    if review.get("rating") is not None:
        meta["ratings"].append(float(review["rating"]))


# Group evidence by user, review and aspect

print("\n[3] Grouping review-level aspects")

review_aspect_groups = defaultdict(list)
flat_counts = Counter()

for row in aspect_rows:

    review_id = row["train_row_id"]

    if review_id not in review_by_id:
        raise ValueError(
            f"Unknown review ID: {review_id}"
        )

    source = review_by_id[review_id]

    if row["user_id"] != source["user_id"]:
        raise ValueError(
            f"User ID mismatch: {review_id}"
        )

    if row["parent_asin"] != source["parent_asin"]:
        raise ValueError(
            f"Item ID mismatch: {review_id}"
        )

    aspect = str(
        row.get("profile_aspect", "")
    ).strip()

    sentiment = str(
        row.get("sentiment", "")
    ).strip().casefold()

    if not aspect:
        raise ValueError(
            f"Empty aspect in review {review_id}"
        )

    if sentiment not in SENTIMENT_SCORE:
        raise ValueError(
            f"Invalid sentiment: {sentiment}"
        )

    valid = to_bool(
        row["evidence_valid"],
        "evidence_valid"
    )

    needs_review = to_bool(
        row["evidence_review_required"],
        "evidence_review_required"
    )

    if valid == needs_review:
        raise ValueError(
            f"Inconsistent evidence flags: {review_id}"
        )

    key = (
        row["user_id"],
        review_id,
        aspect
    )

    review_aspect_groups[key].append(row)

    flat_counts[review_id] += 1


# Verify nested and flat data consistency.

if sum(nested_counts.values()) != len(aspect_rows):
    raise ValueError(
        "Nested and flat aspect counts do not match."
    )

for review_id, count in nested_counts.items():

    if flat_counts[review_id] != count:

        raise ValueError(
            f"Aspect count mismatch for review {review_id}"
        )


print("Review ID consistency: PASSED")
print("Aspect count consistency: PASSED")


# Calculate one sentiment score per review and aspect

print("\n[4] Calculating review-level sentiment")

review_aspect_records = []

duplicate_groups = 0
duplicate_extra_rows = 0
mixed_groups = 0

for (user_id, review_id, aspect), rows in review_aspect_groups.items():

    if len(rows) > 1:

        duplicate_groups += 1
        duplicate_extra_rows += len(rows) - 1

    valid_rows = [
        row for row in rows
        if to_bool(row["evidence_valid"], "evidence_valid")
        and not to_bool(
            row["evidence_review_required"],
            "evidence_review_required"
        )
    ]

    excluded_rows = [
        row for row in rows
        if not to_bool(row["evidence_valid"], "evidence_valid")
    ]

    sentiment_counts = Counter(
        row["sentiment"] for row in valid_rows
    )

    if valid_rows:

        scores = [
            SENTIMENT_SCORE[row["sentiment"]]
            for row in valid_rows
        ]

        score = mean(scores)

        has_positive = sentiment_counts["positive"] > 0
        has_negative = sentiment_counts["negative"] > 0

        if has_positive and has_negative:

            polarity = "mixed"
            mixed_groups += 1

        elif has_positive:
            polarity = "positive"

        elif has_negative:
            polarity = "negative"

        else:
            polarity = "neutral"

    else:

        score = None
        polarity = "excluded_review_required"

    first = rows[0]

    categories = sorted({
        str(row["profile_category"])
        for row in rows
        if row.get("profile_category")
    })

    evidence_records = [

        {
            "evidence": row["evidence"],
            "sentiment": row["sentiment"],
            "evidence_valid": to_bool(
                row["evidence_valid"],
                "evidence_valid"
            ),
            "evidence_match_type": row.get(
                "evidence_match_type", ""
            )
        }

        for row in rows
    ]

    review_aspect_records.append({

        "user_id": user_id,

        "train_row_id": review_id,

        "parent_asin": first["parent_asin"],

        "rating": first.get("rating"),

        "profile_aspect": aspect,

        "raw_aspect_labels": sorted({
            str(row.get("raw_aspect", ""))
            for row in rows
        }),

        "parent_category": (
            categories[0]
            if len(categories) == 1
            else (
                "Multiple/Conflicting"
                if categories else None
            )
        ),

        "mapping_statuses": sorted({
            str(row.get("mapping_status", "unknown"))
            for row in rows
        }),

        "profile_mapping_statuses": sorted({
            str(row.get("profile_mapping_status", "unknown"))
            for row in rows
        }),

        "valid_evidence_count": len(valid_rows),

        "excluded_evidence_count": len(excluded_rows),

        "sentiment_evidence_counts": dict(
            sentiment_counts
        ),

        "review_aspect_sentiment_score": score,

        "review_aspect_polarity": polarity,

        "evidence_records": evidence_records

    })


# Aggregate review-level scores into user profiles

print("\n[5] Building user-aspect profiles")

user_aspect_groups = defaultdict(list)

for record in review_aspect_records:

    key = (
        record["user_id"],
        record["profile_aspect"]
    )

    user_aspect_groups[key].append(record)


user_aspect_profiles = []

for (user_id, aspect), records in user_aspect_groups.items():

    valid_records = [
        record for record in records
        if record["review_aspect_sentiment_score"] is not None
    ]

    excluded_records = [
        record for record in records
        if record["review_aspect_sentiment_score"] is None
    ]

    scores = [
        record["review_aspect_sentiment_score"]
        for record in valid_records
    ]

    preference_score = mean(scores)

    if preference_score is None:

        direction = "insufficient_evidence"
        strength = None

    elif preference_score > 0:

        direction = "positive"
        strength = abs(preference_score)

    elif preference_score < 0:

        direction = "negative"
        strength = abs(preference_score)

    else:

        direction = "neutral"
        strength = 0.0

    polarity_counts = Counter(
        record["review_aspect_polarity"]
        for record in valid_records
    )

    evidence_sentiments = Counter()

    for record in valid_records:

        evidence_sentiments.update(
            record["sentiment_evidence_counts"]
        )

    categories = sorted({
        record["parent_category"]
        for record in records
        if record["parent_category"] is not None
    })

    evidence_examples = []

    for record in valid_records:

        for evidence in record["evidence_records"]:

            if evidence["evidence_valid"]:

                evidence_examples.append({

                    "train_row_id": record["train_row_id"],

                    "parent_asin": record["parent_asin"],

                    "evidence": evidence["evidence"],

                    "sentiment": evidence["sentiment"]

                })

            if len(evidence_examples) >= 5:
                break

        if len(evidence_examples) >= 5:
            break

    user_aspect_profiles.append({

        "user_id": user_id,

        "profile_aspect": aspect,

        "parent_category": (
            categories[0]
            if len(categories) == 1
            else (
                "Multiple/Conflicting"
                if categories else None
            )
        ),

        "mapping_statuses": sorted({
            status
            for record in records
            for status in record["mapping_statuses"]
        }),

        "profile_mapping_statuses": sorted({
            status
            for record in records
            for status in record["profile_mapping_statuses"]
        }),

        "support_review_count": len(valid_records),

        "excluded_only_review_count": len(excluded_records),

        "total_review_aspect_groups": len(records),

        "valid_evidence_count": sum(
            record["valid_evidence_count"]
            for record in records
        ),

        "excluded_evidence_count": sum(
            record["excluded_evidence_count"]
            for record in records
        ),

        "positive_evidence_count": evidence_sentiments["positive"],

        "neutral_evidence_count": evidence_sentiments["neutral"],

        "negative_evidence_count": evidence_sentiments["negative"],

        "positive_review_count": polarity_counts["positive"],

        "neutral_review_count": polarity_counts["neutral"],

        "negative_review_count": polarity_counts["negative"],

        "mixed_review_count": polarity_counts["mixed"],

        "preference_score": (
            round(preference_score, 6)
            if preference_score is not None else None
        ),

        "preference_strength": (
            round(strength, 6)
            if strength is not None else None
        ),

        "preference_direction": direction,

        "has_conflicting_reviews": (
            polarity_counts["positive"] > 0
            and polarity_counts["negative"] > 0
        ) or polarity_counts["mixed"] > 0,

        "source_review_ids": sorted({
            record["train_row_id"]
            for record in records
        }),

        "source_item_ids": sorted({
            record["parent_asin"]
            for record in records
        }),

        "evidence_examples": evidence_examples

    })


user_aspect_profiles.sort(
    key=lambda row: (
        row["user_id"],
        -row["support_review_count"],
        row["profile_aspect"]
    )
)


# Assemble one profile per user

print("\n[6] Assembling user profiles")

profiles_by_user = defaultdict(list)

for profile in user_aspect_profiles:

    profiles_by_user[profile["user_id"]].append(profile)


user_profiles = []

for user_id, meta in user_meta.items():

    user_aspects = profiles_by_user.get(
        user_id,
        []
    )

    user_profiles.append({

        "user_id": user_id,

        "review_count": len(meta["review_ids"]),

        "reviews_with_aspects": len(
            meta["reviews_with_aspects"]
        ),

        "mean_overall_rating": (
            round(mean(meta["ratings"]), 4)
            if meta["ratings"] else None
        ),

        "aspect_profile_count": len(user_aspects),

        "valid_aspect_profile_count": sum(
            1 for aspect in user_aspects
            if aspect["support_review_count"] > 0
        ),

        "aspects": user_aspects

    })


user_profiles.sort(
    key=lambda row: row["user_id"]
)


# Calculate summary statistics

print("\n[7] Running integrity checks")

profile_direction_counts = Counter(
    row["preference_direction"]
    for row in user_aspect_profiles
)

valid_review_aspect_counts = Counter(
    row["review_aspect_polarity"]
    for row in review_aspect_records
    if row["review_aspect_sentiment_score"] is not None
)

valid_evidence_sentiments = Counter(
    row["sentiment"]
    for row in aspect_rows
    if to_bool(row["evidence_valid"], "evidence_valid")
)

single_support = sum(
    row["support_review_count"] == 1
    for row in user_aspect_profiles
)

multi_support = sum(
    row["support_review_count"] >= 2
    for row in user_aspect_profiles
)

zero_support = sum(
    row["support_review_count"] == 0
    for row in user_aspect_profiles
)

assert len(user_profiles) == len(user_meta)

assert len(review_aspect_records) == len(review_aspect_groups)

assert sum(
    row["valid_evidence_count"]
    + row["excluded_evidence_count"]
    for row in review_aspect_records
) == len(aspect_rows)

assert sum(
    profile["aspect_profile_count"]
    for profile in user_profiles
) == len(user_aspect_profiles)


summary = {

    "total_users": len(user_profiles),

    "total_reviews": len(reviews),

    "reviews_without_aspects": sum(
        not review.get("aspects")
        for review in reviews
    ),

    "users_with_aspect_profiles": sum(
        profile["aspect_profile_count"] > 0
        for profile in user_profiles
    ),

    "total_aspect_evidence_records": len(aspect_rows),

    "unique_review_aspect_groups": len(
        review_aspect_records
    ),

    "duplicate_review_aspect_groups": duplicate_groups,

    "extra_evidence_rows_within_duplicate_groups": (
        duplicate_extra_rows
    ),

    "review_aspect_groups_with_valid_evidence": sum(
        row["review_aspect_sentiment_score"] is not None
        for row in review_aspect_records
    ),

    "review_aspect_groups_excluded_only": sum(
        row["review_aspect_sentiment_score"] is None
        for row in review_aspect_records
    ),

    "mixed_review_aspect_groups": mixed_groups,

    "user_aspect_profile_count": len(
        user_aspect_profiles
    ),

    "single_review_support_profiles": single_support,

    "multi_review_support_profiles": multi_support,

    "zero_valid_support_profiles": zero_support,

    "profile_direction_counts": dict(
        profile_direction_counts
    ),

    "review_aspect_polarity_counts": dict(
        valid_review_aspect_counts
    ),

    "valid_evidence_sentiment_counts": dict(
        valid_evidence_sentiments
    ),

    "scoring_method": {

        "positive": 1.0,

        "neutral": 0.0,

        "negative": -1.0,

        "review_aspect_score": (
            "Mean sentiment score across valid evidence "
            "within one review and one aspect."
        ),

        "user_preference_score": (
            "Mean review-aspect score across distinct reviews "
            "for the same user and aspect."
        ),

        "preference_strength": (
            "Absolute value of the user preference score."
        ),

        "evidence_policy": (
            "Only evidence_valid=True records contribute to "
            "primary scoring. Review-required evidence is retained "
            "but excluded from scoring."
        ),

        "duplicate_policy": (
            "Multiple evidence records for the same aspect within "
            "one review are aggregated before user-level scoring."
        ),

        "taxonomy_policy": (
            "Use profile_aspect. Unapproved aliases remain separate."
        ),

        "rating_policy": (
            "Overall review ratings are retained as context and "
            "are not used as aspect-level sentiment scores."
        ),

        "temporal_policy": (
            "No recency weighting because review timestamps "
            "are not available in the prepared data."
        )

    }

}


# Save outputs

print("\n[8] Saving files")

write_jsonl(
    REVIEW_ASPECT_OUTPUT,
    review_aspect_records
)

write_jsonl(
    USER_PROFILE_OUTPUT,
    user_profiles
)

csv_rows = []

for row in user_aspect_profiles:

    csv_rows.append({

        "user_id": row["user_id"],

        "profile_aspect": row["profile_aspect"],

        "parent_category": row["parent_category"] or "",

        "support_review_count": row["support_review_count"],

        "excluded_only_review_count": row["excluded_only_review_count"],

        "valid_evidence_count": row["valid_evidence_count"],

        "excluded_evidence_count": row["excluded_evidence_count"],

        "positive_evidence_count": row["positive_evidence_count"],

        "neutral_evidence_count": row["neutral_evidence_count"],

        "negative_evidence_count": row["negative_evidence_count"],

        "positive_review_count": row["positive_review_count"],

        "neutral_review_count": row["neutral_review_count"],

        "negative_review_count": row["negative_review_count"],

        "mixed_review_count": row["mixed_review_count"],

        "preference_score": row["preference_score"],

        "preference_strength": row["preference_strength"],

        "preference_direction": row["preference_direction"],

        "has_conflicting_reviews": row["has_conflicting_reviews"],

        "mapping_statuses": json.dumps(
            row["mapping_statuses"],
            ensure_ascii=False
        ),

        "profile_mapping_statuses": json.dumps(
            row["profile_mapping_statuses"],
            ensure_ascii=False
        ),

        "source_review_ids": json.dumps(
            row["source_review_ids"],
            ensure_ascii=False
        ),

        "source_item_ids": json.dumps(
            row["source_item_ids"],
            ensure_ascii=False
        ),

        "evidence_examples": json.dumps(
            row["evidence_examples"],
            ensure_ascii=False
        )

    })


write_csv(
    USER_PROFILE_CSV,
    csv_rows
)

write_json(
    SUMMARY_OUTPUT,
    summary
)


# Final report

print("\n" + "=" * 60)

print("USER PROFILE CONSTRUCTION COMPLETED")

print("=" * 60)

print(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    )
)

print("\nGenerated files:")

print(REVIEW_ASPECT_OUTPUT)
print(USER_PROFILE_OUTPUT)
print(USER_PROFILE_CSV)
print(SUMMARY_OUTPUT)

print("=" * 60)
