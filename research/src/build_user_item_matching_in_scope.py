from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import statistics


# Configuration

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# Input files

# Keep the complete training history for interaction validation
# and previously reviewed item exclusion.

REVIEW_FILE = DATA_DIR / "profile_data.jsonl"

# Use the newly constructed In-Scope user profiles.

USER_FILE = (
    DATA_DIR / "user_profile_aspect_scores_in_scope.csv"
)

# Keep the complete 341-item profile dataset.
# Candidate filtering will be handled separately.

ITEM_FILE = (
    DATA_DIR / "item_profile_aspect_scores.csv"
)

# V2.1 product category and scope mapping.

PRODUCT_MAPPING_FILE = (
    DATA_DIR / "product_category_mapping_v2_1_audit_merged_341.csv"
)


# Independent output files

PAIR_OUTPUT = (
    DATA_DIR / "user_item_matching_scores_in_scope.csv"
)

RECOMMENDATION_OUTPUT = (
    DATA_DIR / "user_item_recommendations_in_scope.jsonl"
)

SUMMARY_OUTPUT = (
    DATA_DIR / "user_item_matching_summary_in_scope.json"
)


# Matching parameters

MIN_SHARED_ASPECTS = 2
TOP_K = 10
SHRINKAGE_STRENGTH = 2.0
ALLOW_SINGLE_ASPECT_FALLBACK = True

# Dataset integrity expectations

EXPECTED_USERS = 253
EXPECTED_ITEMS = 341
EXPECTED_TRAIN_REVIEWS = 2029
EXPECTED_IN_SCOPE_ITEMS = 297


# Utility functions

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}, line {line_no}: {exc}")
    return rows


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to export.")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def num(value, field, allow_none=False):
    if value is None or value == "":
        if allow_none:
            return None
        raise ValueError(f"Missing numeric value: {field}")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value for {field}: {value}")
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value for {field}: {value}")
    return value


def count(value, field):
    value = num(value, field)
    if value < 0 or not value.is_integer():
        raise ValueError(f"Invalid non-negative integer for {field}: {value}")
    return int(value)


def rounded(value, digits=6):
    return round(value, digits) if value is not None else None


# Load files and training interactions

print("\n" + "=" * 64)
print("IN-SCOPE USER-ITEM MATCHING")
print("=" * 64)

# 1 Check required files

for path in (
    REVIEW_FILE,
    USER_FILE,
    ITEM_FILE,
    PRODUCT_MAPPING_FILE
):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# 2 Load source files

reviews = load_jsonl(REVIEW_FILE)
user_rows = read_csv(USER_FILE)
item_rows = read_csv(ITEM_FILE)
mapping_rows = read_csv(PRODUCT_MAPPING_FILE)

print("\n[1] Loaded files")
print("Training reviews:", len(reviews))
print("User-aspect rows:", len(user_rows))
print("Item-aspect rows:", len(item_rows))
print("Product mapping rows:", len(mapping_rows))


# 3 Validate training review count

if len(reviews) != EXPECTED_TRAIN_REVIEWS:
    raise ValueError(
        f"Expected {EXPECTED_TRAIN_REVIEWS} reviews, "
        f"got {len(reviews)}."
    )


# 4 Build product metadata mapping

product_mapping = {}

for row in mapping_rows:

    iid = row["parent_asin"].strip()
    scope = row["final_scope_status"].strip()

    if not iid:
        raise ValueError("Empty parent_asin in product mapping.")

    if iid in product_mapping:
        raise ValueError(
            f"Duplicate product mapping ASIN: {iid}"
        )

    if scope not in {
        "in_scope",
        "mixed_bundle",
        "out_of_scope",
        "needs_review"
    }:
        raise ValueError(
            f"Invalid scope status for {iid}: {scope}"
        )

    product_mapping[iid] = row


if len(product_mapping) != EXPECTED_ITEMS:
    raise ValueError(
        f"Expected {EXPECTED_ITEMS} mapped items, "
        f"got {len(product_mapping)}."
    )


# 5 Validate product scope distribution

expected_scope_counts = {
    "in_scope": 297,
    "mixed_bundle": 13,
    "out_of_scope": 18,
    "needs_review": 13
}

actual_scope_counts = {
    scope: sum(
        row["final_scope_status"].strip() == scope
        for row in product_mapping.values()
    )
    for scope in expected_scope_counts
}

if actual_scope_counts != expected_scope_counts:
    raise ValueError(
        "Unexpected product scope distribution: "
        f"{actual_scope_counts}"
    )


# 6 Build complete training interaction history

user_history = defaultdict(set)
item_ratings = defaultdict(list)
review_ids = set()

for row in reviews:

    rid = row.get("train_row_id")
    uid = row.get("user_id")
    iid = row.get("parent_asin")

    if rid is None or uid is None or iid is None:
        raise ValueError(
            "A training review is missing "
            "train_row_id, user_id or parent_asin."
        )

    if rid in review_ids:
        raise ValueError(
            f"Duplicate train_row_id: {rid}"
        )

    review_ids.add(rid)

    user_history[uid].add(iid)

    rating = num(
        row.get("rating"),
        f"rating in review {rid}",
        allow_none=True
    )

    if rating is not None:
        item_ratings[iid].append(rating)


# 7 Separate complete users and item sets

history_pairs = sum(
    len(x) for x in user_history.values()
)

# All 253 users come from complete training history.
users = sorted(user_history.keys())

# Users represented in the In-Scope aspect CSV.
profile_users = {
    row["user_id"]
    for row in user_rows
}

# All 341 products remain available for integrity checks.
all_items = sorted({
    row["parent_asin"]
    for row in item_rows
})

history_items = {
    iid
    for values in user_history.values()
    for iid in values
}

# Only 297 products are eligible recommendation candidates.
candidate_items = sorted(
    iid
    for iid, row in product_mapping.items()
    if row["final_scope_status"].strip() == "in_scope"
)

candidate_item_set = set(candidate_items)

# Build validated metadata for In-Scope candidates

candidate_metadata = {}

for iid in candidate_items:

    row = product_mapping[iid]

    metadata = {
        "product_category": (
            row.get("final_product_category") or ""
        ).strip(),

        "product_type": (
            row.get("final_product_type") or ""
        ).strip(),

        "scope_status": (
            row.get("final_scope_status") or ""
        ).strip(),

        "audit_status": (
            row.get("audit_status") or ""
        ).strip(),

        "mapping_resolution": (
            row.get("final_mapping_resolution") or ""
        ).strip(),

        "audit_id": (
            row.get("audit_id") or ""
        ).strip(),

        "audit_decision": (
            row.get("audit_decision") or ""
        ).strip()
    }

    if metadata["scope_status"] != "in_scope":
        raise ValueError(
            f"Non-In-Scope item in candidate metadata: {iid}"
        )

    if not metadata["product_category"]:
        raise ValueError(
            f"Missing product category for In-Scope item: {iid}"
        )

    candidate_metadata[iid] = metadata


if len(candidate_metadata) != EXPECTED_IN_SCOPE_ITEMS:
    raise ValueError(
        f"Expected {EXPECTED_IN_SCOPE_ITEMS} metadata records, "
        f"got {len(candidate_metadata)}."
    )

print("Candidate product metadata validation: PASSED")


# 8 Validate complete dataset coverage

if len(users) != EXPECTED_USERS:
    raise ValueError(
        f"Expected {EXPECTED_USERS} users, got {len(users)}."
    )

if len(all_items) != EXPECTED_ITEMS:
    raise ValueError(
        f"Expected {EXPECTED_ITEMS} items, got {len(all_items)}."
    )

if len(candidate_items) != EXPECTED_IN_SCOPE_ITEMS:
    raise ValueError(
        f"Expected {EXPECTED_IN_SCOPE_ITEMS} In-Scope items, "
        f"got {len(candidate_items)}."
    )

if set(product_mapping) != set(all_items):
    raise ValueError(
        "Product mapping ASINs do not match Item Profile ASINs."
    )

if history_items != set(all_items):
    raise ValueError(
        "Training-history items do not match Item Profile items."
    )

if not profile_users.issubset(set(users)):
    raise ValueError(
        "User profile contains users absent from training history."
    )

if not candidate_item_set.issubset(set(all_items)):
    raise ValueError(
        "In-Scope candidates contain unknown items."
    )


# Only In-Scope products are eligible for recommendation.
# All 341 products remain available for integrity validation.

items = candidate_items

print("Active recommendation candidate items:", len(items))

if len(items) != EXPECTED_IN_SCOPE_ITEMS:
    raise ValueError(
        f"Expected {EXPECTED_IN_SCOPE_ITEMS} candidates, "
        f"got {len(items)}."
    )


# 9 Print validation results

print("\n[2] Interaction history")

print("Users in complete training history:", len(users))
print("Users with In-Scope aspect profiles:", len(profile_users))
print("Users without aspect profiles:", len(set(users) - profile_users))

print("All Item Profile products:", len(all_items))
print("Eligible In-Scope candidates:", len(candidate_items))

print("Unique user-item history pairs:", history_pairs)

print("Product mapping validation: PASSED")
print("Complete user coverage validation: PASSED")
print("Complete item coverage validation: PASSED")
print("In-Scope candidate count validation: PASSED")


# Index valid user and item aspect profiles

print("\n[3] Building valid aspect indexes")

user_aspects = defaultdict(dict)
item_aspects = defaultdict(dict)

for row in user_rows:
    uid, aspect = row["user_id"], row["profile_aspect"].strip()
    score = num(row.get("preference_score"), "user preference_score", allow_none=True)
    n = count(row.get("support_review_count"), "user support_review_count")
    if not aspect:
        raise ValueError("Empty user profile_aspect.")
    if score is None:
        continue
    if not -1 <= score <= 1 or n < 1:
        raise ValueError(f"Invalid user aspect score/support: {uid}, {aspect}")
    if aspect in user_aspects[uid]:
        raise ValueError(f"Duplicate user-aspect key: {uid}, {aspect}")
    user_aspects[uid][aspect] = {
        "historical_sentiment_score": score,
        "historical_sentiment_direction": row.get("preference_direction", ""),
        "support_review_count": n,
        "has_conflicting_reviews": row.get("has_conflicting_reviews", "").lower() == "true",
        "positive_review_count": count(row.get("positive_review_count", 0), "user positive count"),
        "negative_review_count": count(row.get("negative_review_count", 0), "user negative count"),
        "neutral_review_count": count(row.get("neutral_review_count", 0), "user neutral count"),
        "mixed_review_count": count(row.get("mixed_review_count", 0), "user mixed count")
    }

for row in item_rows:
    iid, aspect = row["parent_asin"], row["profile_aspect"].strip()
    score = num(row.get("performance_score"), "item performance_score", allow_none=True)
    n = count(row.get("support_review_count"), "item support_review_count")
    if not aspect:
        raise ValueError("Empty item profile_aspect.")
    if score is None:
        continue
    if not -1 <= score <= 1 or n < 1:
        raise ValueError(f"Invalid item aspect score/support: {iid}, {aspect}")
    if aspect in item_aspects[iid]:
        raise ValueError(f"Duplicate item-aspect key: {iid}, {aspect}")
    item_aspects[iid][aspect] = {
        "performance_score": score,
        "support_review_count": n,
        "support_tier": row.get("support_tier", ""),
        "performance_direction": row.get("performance_direction", ""),
        "has_conflicting_reviews": row.get("has_conflicting_reviews", "").lower() == "true",
        "positive_review_count": count(row.get("positive_review_count", 0), "item positive count"),
        "negative_review_count": count(row.get("negative_review_count", 0), "item negative count"),
        "neutral_review_count": count(row.get("neutral_review_count", 0), "item neutral count"),
        "mixed_review_count": count(row.get("mixed_review_count", 0), "item mixed count")
    }

user_vocab = {r["profile_aspect"] for r in user_rows}
item_vocab = {r["profile_aspect"] for r in item_rows}
shared_vocab = user_vocab & item_vocab

print("Valid user-aspect profiles:", sum(map(len, user_aspects.values())))
print("Valid item-aspect profiles:", sum(map(len, item_aspects.values())))
print("Shared global aspect labels:", len(shared_vocab))


# Calculate leave-one-item-out aspect priors

print("\n[4] Calculating smoothed item quality")

# Pooled review-level mean for each aspect, weighted by distinct
# supporting reviews. For each item, its own score is removed
# from the prior to avoid using itself as its own reference.

aspect_sum = defaultdict(float)
aspect_n = defaultdict(int)

# Calculate aspect priors using In-Scope items only.

for iid in candidate_items:

    aspects = item_aspects.get(iid, {})

    for aspect, row in aspects.items():

        n = row["support_review_count"]

        aspect_sum[aspect] += (
            row["performance_score"] * n
        )

        aspect_n[aspect] += n


def smooth_item_score(iid, aspect, row):
    n = row["support_review_count"]
    score = row["performance_score"]
    other_n = aspect_n[aspect] - n
    other_sum = aspect_sum[aspect] - score * n
    prior = other_sum / other_n if other_n > 0 else 0.0
    adjusted = (n * score + SHRINKAGE_STRENGTH * prior) / (n + SHRINKAGE_STRENGTH)
    return adjusted, prior


# Score unseen user-item pairs

print("\n[5] Scoring candidate pairs")

all_unseen_pairs = 0
pairs_any_overlap = 0
pairs_min_overlap = 0
pairs_by_user = defaultdict(list)

for uid in users:
    u_map = user_aspects.get(uid, {})
    if not u_map:
        continue

    for iid in items:
        # Exclude every item already reviewed in the training data.
        if iid in user_history[uid]:
            continue

        all_unseen_pairs += 1
        i_map = item_aspects.get(iid, {})
        common = sorted(set(u_map) & set(i_map))
        if not common:
            continue

        pairs_any_overlap += 1
        if len(common) >= MIN_SHARED_ASPECTS:
            pairs_min_overlap += 1

        details = []
        weighted_total = 0.0
        weight_total = 0.0

        for aspect in common:
            u = u_map[aspect]
            i = i_map[aspect]

            # Review frequency is an aspect-attention proxy.
            # User historical sentiment sign is NOT treated as explicit taste.
            weight = math.log1p(u["support_review_count"])
            adjusted, prior = smooth_item_score(iid, aspect, i)

            weighted_total += weight * adjusted
            weight_total += weight

            details.append({
                "profile_aspect": aspect,
                "user_support_review_count": u["support_review_count"],
                "user_historical_sentiment_score": rounded(u["historical_sentiment_score"]),
                "user_historical_sentiment_direction": u["historical_sentiment_direction"],
                "user_historical_sentiment_is_conflicted": u["has_conflicting_reviews"],
                "item_performance_score": rounded(i["performance_score"]),
                "item_prior_mean_excluding_current_item": rounded(prior),
                "item_adjusted_performance_score": rounded(adjusted),
                "item_support_review_count": i["support_review_count"],
                "item_support_tier": i["support_tier"],
                "item_performance_direction": i["performance_direction"],
                "item_has_conflicting_reviews": i["has_conflicting_reviews"],
                "user_aspect_weight": round(weight, 6)
            })

        if weight_total == 0:
            continue

        score = weighted_total / weight_total
        if not -1 <= score <= 1:
            raise ValueError(f"Match score out of range: {uid}, {iid}")

        for d in details:
            d["normalized_aspect_weight"] = round(d["user_aspect_weight"] / weight_total, 6)
            d["weighted_quality_contribution"] = round(
                d["normalized_aspect_weight"] * d["item_adjusted_performance_score"], 6
            )

        item_rating_values = item_ratings.get(iid, [])
        pairs_by_user[uid].append({
            "user_id": uid,
            "parent_asin": iid,
            "match_score": score,
            "matched_aspect_count": len(common),
            "user_valid_aspect_count": len(u_map),
            "item_valid_aspect_count": len(i_map),
            "user_aspect_coverage": len(common) / len(u_map),
            "item_aspect_coverage": len(common) / len(i_map) if i_map else 0.0,
            "mean_item_rating_context": (
                statistics.mean(item_rating_values) if item_rating_values else None
            ),
            "item_rating_count_context": len(item_rating_values),
            "matched_aspects": details,
            "low_coverage_fallback": False
        })


# Apply overlap threshold and fallback

print("\n[6] Applying minimum-overlap threshold")

selected = []
fallback_users = []
candidate_counts = {}

for uid in users:
    all_candidates = pairs_by_user.get(uid, [])
    eligible = [
        row for row in all_candidates
        if row["matched_aspect_count"] >= MIN_SHARED_ASPECTS
    ]

    # If a user has no 2-aspect match, retain 1-aspect matches
    # as an explicitly marked low-coverage fallback.
    if not eligible and ALLOW_SINGLE_ASPECT_FALLBACK:
        eligible = [
            row for row in all_candidates
            if row["matched_aspect_count"] >= 1
        ]
        if eligible:
            fallback_users.append(uid)
            for row in eligible:
                row["low_coverage_fallback"] = True

    candidate_counts[uid] = len(eligible)
    selected.extend(eligible)

selected.sort(key=lambda x: (
    x["user_id"], -x["match_score"], -x["matched_aspect_count"],
    -x["user_aspect_coverage"], x["parent_asin"]
))


# Build top-K recommendations

print("\n[7] Building top-K recommendation lists")

recommendations = []

for uid in users:
    rows = [x for x in selected if x["user_id"] == uid]
    rows.sort(key=lambda x: (
        -x["match_score"], -x["matched_aspect_count"],
        -x["user_aspect_coverage"], x["parent_asin"]
    ))

    top = []
    for rank, row in enumerate(rows[:TOP_K], 1):
        top.append({
            "rank": rank,
            "parent_asin": row["parent_asin"],
            "product_category": candidate_metadata[
                row["parent_asin"]
            ]["product_category"],

            "product_type": candidate_metadata[
                row["parent_asin"]
            ]["product_type"],

            "scope_status": candidate_metadata[
                row["parent_asin"]
            ]["scope_status"],

            "audit_status": candidate_metadata[
                row["parent_asin"]
            ]["audit_status"],

            "mapping_resolution": candidate_metadata[
                row["parent_asin"]
            ]["mapping_resolution"],

            "audit_id": candidate_metadata[
                row["parent_asin"]
            ]["audit_id"],

            "audit_decision": candidate_metadata[
                row["parent_asin"]
            ]["audit_decision"],
            "match_score": round(row["match_score"], 6),
            "matched_aspect_count": row["matched_aspect_count"],
            "user_aspect_coverage": round(row["user_aspect_coverage"], 4),
            "item_aspect_coverage": round(row["item_aspect_coverage"], 4),
            "mean_item_rating_context": rounded(row["mean_item_rating_context"], 4),
            "item_rating_count_context": row["item_rating_count_context"],
            "low_coverage_fallback": row["low_coverage_fallback"],
            "matched_aspects": row["matched_aspects"]
        })

    recommendations.append({
        "user_id": uid,
        "ranking_method": "Historical aspect-attention weighted item-quality baseline",
        "candidate_count": len(rows),
        "returned_recommendation_count": len(top),
        "top_k": TOP_K,
        "recommendations": top
    })


# Export pair scores, recommendations and summary

print("\n[8] Exporting results")

pair_rows = []
for row in selected:
    pair_rows.append({
        "user_id": row["user_id"],
        "parent_asin": row["parent_asin"],
        "match_score": round(row["match_score"], 6),
    "product_category": candidate_metadata[
        row["parent_asin"]
    ]["product_category"],

    "product_type": candidate_metadata[
        row["parent_asin"]
    ]["product_type"],

    "scope_status": candidate_metadata[
        row["parent_asin"]
    ]["scope_status"],

    "audit_status": candidate_metadata[
        row["parent_asin"]
    ]["audit_status"],

    "mapping_resolution": candidate_metadata[
        row["parent_asin"]
    ]["mapping_resolution"],

    "audit_id": candidate_metadata[
        row["parent_asin"]
    ]["audit_id"],

    "audit_decision": candidate_metadata[
        row["parent_asin"]
    ]["audit_decision"],
        "matched_aspect_count": row["matched_aspect_count"],
        "user_valid_aspect_count": row["user_valid_aspect_count"],
        "item_valid_aspect_count": row["item_valid_aspect_count"],
        "user_aspect_coverage": round(row["user_aspect_coverage"], 6),
        "item_aspect_coverage": round(row["item_aspect_coverage"], 6),
        "mean_item_rating_context": rounded(row["mean_item_rating_context"], 4),
        "item_rating_count_context": row["item_rating_count_context"],
        "low_coverage_fallback": row["low_coverage_fallback"],
        "matched_aspects": json.dumps(row["matched_aspects"], ensure_ascii=False)
    })

pair_keys = [(x["user_id"], x["parent_asin"]) for x in selected]
# Validate every selected recommendation against
# the approved In-Scope product mapping.

for row in selected:

    iid = row["parent_asin"]

    if iid not in candidate_item_set:
        raise ValueError(
            f"Non-In-Scope product in recommendation: {iid}"
        )

    metadata = candidate_metadata.get(iid)

    if metadata is None:
        raise ValueError(
            f"Missing product metadata: {iid}"
        )

    if metadata["scope_status"] != "in_scope":
        raise ValueError(
            f"Invalid recommendation scope: {iid}"
        )

    if not metadata["product_category"]:
        raise ValueError(
            f"Missing product category: {iid}"
        )

print("Recommendation product metadata validation: PASSED")
if len(pair_keys) != len(set(pair_keys)):
    raise ValueError("Duplicate user-item candidate pairs detected.")
if len(recommendations) != len(users):
    raise ValueError("Recommendation output does not cover every user.")
if any(not (-1 <= x["match_score"] <= 1) for x in selected):
    raise ValueError("At least one match score is outside [-1, 1].")

print("Unique user-item pairs: PASSED")
print("Previously reviewed item exclusion: PASSED")
print("Match score range: PASSED")
print("Recommendation output coverage: PASSED")

score_values = [x["match_score"] for x in selected]
counts = list(candidate_counts.values())

summary = {
    "total_users": len(users),

    # Complete product inventory and eligible candidates.
    "total_items": len(all_items),
    "candidate_items": len(candidate_items),
    "training_review_records": len(reviews),
    "unique_user_item_history_pairs": history_pairs,
    "global_user_aspect_vocabulary": len(user_vocab),
    "global_item_aspect_vocabulary": len(item_vocab),
    "shared_global_aspect_vocabulary": len(shared_vocab),
    "valid_user_aspect_profiles": sum(map(len, user_aspects.values())),
    "valid_item_aspect_profiles": sum(map(len, item_aspects.values())),
    "valid_candidate_item_aspect_profiles": sum(
    len(item_aspects.get(iid, {}))
    for iid in candidate_items
      ),
    "candidate_pairs_after_history_exclusion_for_profiled_users": all_unseen_pairs,
    "candidate_pairs_with_any_shared_valid_aspect": pairs_any_overlap,
    "candidate_pairs_meeting_min_shared_aspects": pairs_min_overlap,
    "scored_candidate_pairs_after_fallback": len(selected),
    "minimum_shared_aspects": MIN_SHARED_ASPECTS,
    "single_aspect_fallback_enabled": ALLOW_SINGLE_ASPECT_FALLBACK,
    "fallback_user_count": len(fallback_users),
    "fallback_users": fallback_users,
    "users_with_at_least_one_recommendation": sum(x > 0 for x in counts),
    "users_with_at_least_top_k_candidates": sum(x >= TOP_K for x in counts),
    "users_with_fewer_than_top_k_candidates": sum(0 < x < TOP_K for x in counts),
    "candidate_count_per_user": {
        "minimum": min(counts) if counts else 0,
        "median": statistics.median(counts) if counts else 0,
        "mean": round(statistics.mean(counts), 4) if counts else 0,
        "maximum": max(counts) if counts else 0
    },
    "match_score_statistics": {
        "count": len(score_values),
        "minimum": min(score_values) if score_values else None,
        "maximum": max(score_values) if score_values else None,
        "mean": round(statistics.mean(score_values), 6) if score_values else None,
        "median": round(statistics.median(score_values), 6) if score_values else None
    },
    "users_with_valid_aspect_profiles": sum(
    bool(user_aspects.get(uid)) for uid in users
        ),

    "users_without_valid_aspect_profiles": sum(
    not user_aspects.get(uid) for uid in users
        ),
    "scoring_method": {
        "user_aspect_weight": "log(1 + distinct valid historical review count)",
        "user_sentiment_policy": (
            "Historical sentiment is retained for explanation; its sign is not "
            "treated as explicit user preference."
        ),
        "item_quality": "Review-derived item performance_score.",
        "item_score_smoothing": (
            "(n * observed_item_score + lambda * leave-one-item-out_aspect_prior) "
            "/ (n + lambda)"
        ),
        "shrinkage_lambda": SHRINKAGE_STRENGTH,
        "pair_score": (
            "Weighted mean of smoothed item aspect quality over shared valid aspects."
        ),
        "coverage": "User and item aspect coverage ratios are reported separately.",
        "history_exclusion": (
            "Items already reviewed by the user in profile_data.jsonl are excluded."
        )
    },
    "limitations": [
        "This is a history-informed aspect-attention weighted item-quality baseline, "
        "not a validated model of explicit user preference.",
        "Historical sentiment may indicate satisfaction or dissatisfaction; "
        "its sign is not treated as desired product quality.",
        "Review frequency is an attention proxy, not proof of preference importance.",
        "Item score shrinkage is a heuristic and requires sensitivity analysis.",
        "Current Intent has not yet been integrated. "
        "Product category, type and audit metadata are attached "
        "for traceability but are not used in the current matching score."
    ]
}

write_csv(PAIR_OUTPUT, pair_rows)
write_jsonl(RECOMMENDATION_OUTPUT, recommendations)
write_json(SUMMARY_OUTPUT, summary)

print("\n" + "=" * 64)
print("USER-ITEM MATCHING COMPLETED")
print("=" * 64)
print(json.dumps(summary, indent=4, ensure_ascii=False))
print("\nGenerated files:")
print(PAIR_OUTPUT)
print(RECOMMENDATION_OUTPUT)
print(SUMMARY_OUTPUT)
print("=" * 64)
