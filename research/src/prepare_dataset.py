from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

# Project paths

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

review_file = (
    RAW / "raw" / "review_categories" / "All_Beauty.jsonl"
)

meta_file = (
    RAW / "raw" / "meta_categories" / "meta_All_Beauty.jsonl"
)

benchmark_dir = (
    RAW / "benchmark" / "5core" / "last_out_w_his"
)


# Load recommendation benchmark

print("\n[1] Loading benchmark datasets...")

train = pd.read_csv(
    benchmark_dir / "All_Beauty.train.csv"
)

valid = pd.read_csv(
    benchmark_dir / "All_Beauty.valid.csv"
)

test = pd.read_csv(
    benchmark_dir / "All_Beauty.test.csv"
)


def normalize_ids(df):

    df = df.copy()

    for col in ["user_id", "parent_asin"]:
        df[col] = df[col].astype("string").str.strip()

    return df.dropna(
        subset=["user_id", "parent_asin"]
    )


train = normalize_ids(train)
valid = normalize_ids(valid)
test = normalize_ids(test)

# Only training interactions are used to build
# historical user profiles and review-based item profiles.

train_pairs = set(
    zip(train["user_id"], train["parent_asin"])
)

train_users = set(train["user_id"])

# Full catalog includes items in all three splits.
catalog_items = set(
    pd.concat([
        train["parent_asin"],
        valid["parent_asin"],
        test["parent_asin"]
    ]).dropna()
)

print("Train rows:", len(train))
print("Train users:", len(train_users))
print("Train interaction pairs:", len(train_pairs))
print("Catalog items:", len(catalog_items))


# Match historical review records

print("\n[2] Matching historical reviews...")

matched_pairs = set()
matched_review_count = 0

review_output = (
    PROCESSED / "train_interaction_reviews.jsonl"
)

with open(review_file, "r", encoding="utf-8") as fin, \
     open(review_output, "w", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="Reading reviews"):

        record = json.loads(line)

        user_id = record.get("user_id")
        item_id = record.get("parent_asin")

        if not user_id or not item_id:
            continue

        pair = (user_id, item_id)

        if pair in train_pairs:

            fout.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            matched_pairs.add(pair)
            matched_review_count += 1


# Match product metadata

print("\n[3] Matching product metadata...")

matched_items = set()
metadata_count = 0

metadata_output = (
    PROCESSED / "catalog_metadata.jsonl"
)

with open(meta_file, "r", encoding="utf-8") as fin, \
     open(metadata_output, "w", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="Reading metadata"):

        record = json.loads(line)

        item_id = record.get("parent_asin")

        if item_id in catalog_items:

            fout.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            matched_items.add(item_id)
            metadata_count += 1


# Generate data summary

summary = {
    "dataset": "Amazon Reviews 2023 - All_Beauty",

    "benchmark": {
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "train_users": len(train_users),
        "train_unique_pairs": len(train_pairs),
        "catalog_items": len(catalog_items)
    },

    "review_matching": {
        "matched_review_records": matched_review_count,
        "matched_train_pairs": len(matched_pairs),
        "unmatched_train_pairs": len(
            train_pairs - matched_pairs
        )
    },

    "metadata_matching": {
        "matched_metadata_records": metadata_count,
        "matched_catalog_items": len(matched_items),
        "unmatched_catalog_items": len(
            catalog_items - matched_items
        )
    }
}

summary_file = PROCESSED / "data_summary.json"

with open(summary_file, "w", encoding="utf-8") as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# Display results

print("\n" + "=" * 55)
print("DATA PREPARATION COMPLETED")
print("=" * 55)

print(json.dumps(
    summary,
    indent=4,
    ensure_ascii=False
))

print("\nGenerated files:")
print(review_output)
print(metadata_output)
print(summary_file)
