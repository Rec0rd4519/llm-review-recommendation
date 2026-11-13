from pathlib import Path
import pandas as pd
import json


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

train_file = (
    RAW / "benchmark" / "5core" /
    "last_out_w_his" / "All_Beauty.train.csv"
)

review_file = PROCESSED / "train_interaction_reviews.jsonl"

output_file = PROCESSED / "train_aligned_reviews.jsonl"

summary_file = PROCESSED / "alignment_summary.json"


print("[1] Loading data...")

train = pd.read_csv(train_file)

reviews = pd.read_json(
    review_file,
    lines=True
)

print("Train records:", len(train))
print("Review records:", len(reviews))


# Normalize matching fields

print("\n[2] Normalizing fields...")

for df in [train, reviews]:
    for col in ["user_id", "parent_asin"]:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )

    df["rating_key"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

# Preserve original timestamps
train = train.rename(columns={"timestamp": "train_timestamp_raw"})

reviews = reviews.rename(columns={"timestamp": "review_timestamp_raw"})

train["timestamp_key"] = pd.to_numeric(
    train["train_timestamp_raw"],
    errors="coerce"
).round().astype("Int64")

# Review timestamps require conversion
reviews["timestamp_key"] = (
    pd.to_numeric(
        reviews["review_timestamp_raw"],
        errors="coerce"
    ) * 0.001
).round().astype("Int64")


# Remove exact duplicate reviews

print("\n[3] Removing exact duplicates...")

identity_cols = [
    "user_id",
    "parent_asin",
    "asin",
    "rating",
    "review_timestamp_raw",
    "title",
    "text"
]

before = len(reviews)

reviews = reviews.drop_duplicates(subset=identity_cols).copy()

print("Before:", before)
print("After:", len(reviews))
print("Removed:", before - len(reviews))


# Prepare unique training IDs

train = train.copy()

train["train_row_id"] = range(len(train))

keys = [
    "user_id",
    "parent_asin",
    "rating_key",
    "timestamp_key"
]

# Check uniqueness before merging
if train.duplicated(keys).any():
    raise ValueError("Duplicate matching keys found in Train.")

if reviews.duplicated(keys).any():
    raise ValueError("Ambiguous review matching keys found.")


# Exact one-to-one alignment

print("\n[4] Matching reviews...")

aligned = train.merge(
    reviews,
    on=keys,
    how="left",
    validate="one_to_one",
    indicator=True,
    suffixes=("_train", "_review")
)

matched = aligned[
    aligned["_merge"] == "both"
].copy()

unmatched = aligned[
    aligned["_merge"] != "both"
].copy()

print("Train rows:", len(train))
print("Matched rows:", len(matched))
print("Unmatched rows:", len(unmatched))


# Validate alignment

if len(matched) != len(train):
    print("\nERROR: Not all rows matched.")

    print(
        unmatched[
            ["train_row_id", "user_id", "parent_asin"]
        ].head(10)
    )

    raise ValueError("Alignment validation failed.")

if matched["train_row_id"].duplicated().any():
    raise ValueError("Duplicate train_row_id detected.")

if len(matched) != 2029:
    raise ValueError(f"Expected 2029 matches, got {len(matched)}.")


# Save aligned dataset

print("\n[5] Saving results...")

matched = matched.drop(columns=["_merge"])

matched.to_json(
    output_file,
    orient="records",
    lines=True,
    force_ascii=False
)

summary = {
    "dataset": "Amazon Reviews 2023 - All_Beauty",
    "train_records": len(train),
    "matched_records": len(matched),
    "unmatched_records": len(unmatched),
    "unique_train_row_ids": int(matched["train_row_id"].nunique()),
    "matching_keys": keys,
    "timestamp_conversion": "review_timestamp * 0.001",
    "validation": "passed"
}

with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 50)
print("ALIGNMENT COMPLETED")
print("=" * 50)

print(json.dumps(
    summary,
    indent=4,
    ensure_ascii=False
))

print("\nOutput:", output_file)
