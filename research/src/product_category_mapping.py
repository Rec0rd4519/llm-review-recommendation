import csv
from pathlib import Path
from collections import Counter


# Product Category Mapping Configuration

ALLOWED_CATEGORIES = {
    "Bath & Body Care",
    "Beauty Supplements & Wellness",
    "Beauty Tools & Accessories",
    "Fragrance & Aromatherapy",
    "Grooming & Shaving",
    "Hair Care & Styling",
    "Makeup & Cosmetics",
    "Nail Care",
    "Oral Care",
    "Personal Hygiene",
    "Skin Care"
}

VALID_SCOPE_STATUSES = {
    "in_scope",
    "mixed_bundle",
    "out_of_scope",
    "needs_review"
}

VALID_AUDIT_STATUSES = {
    "decision_recorded",
    "not_audited"
}

EXPECTED_SCOPE_COUNTS = {
    "in_scope": 297,
    "mixed_bundle": 13,
    "out_of_scope": 18,
    "needs_review": 13
}

EXPECTED_AUDIT_COUNTS = {
    "decision_recorded": 150,
    "not_audited": 191
}

EXPECTED_ITEMS = 341


# Load and Validate Product Category Mapping

def load_product_category_mapping(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Category mapping file not found: {csv_path}")

    required_fields = {
        "parent_asin",
        "final_scope_status",
        "final_product_category",
        "final_product_type",
        "audit_status",
        "final_mapping_resolution",
        "audit_id",
        "audit_decision"
    }

    mapping = {}

    scope_counts = Counter()
    audit_counts = Counter()

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(f)

        missing_fields = required_fields - set(reader.fieldnames or [])

        if missing_fields:
            raise ValueError(f"Missing CSV columns: {missing_fields}")

        for line_number, row in enumerate(reader, start=2):
            asin = (row.get("parent_asin") or "").strip()

            if not asin:
                raise ValueError(f"Empty parent_asin at line {line_number}")

            if asin in mapping:
                raise ValueError(f"Duplicate ASIN: {asin}")

            scope_status = (row.get("final_scope_status") or "").strip()

            product_category = (row.get("final_product_category") or "").strip()

            product_type = (row.get("final_product_type") or "").strip()

            audit_status = (row.get("audit_status") or "").strip()

            if scope_status not in VALID_SCOPE_STATUSES:
                raise ValueError(
                    f"Invalid scope status for {asin}: "
                    f"{scope_status}"
                )

            if audit_status not in VALID_AUDIT_STATUSES:
                raise ValueError(
                    f"Invalid audit status for {asin}: "
                    f"{audit_status}"
                )

            # In-scope products must have one valid category.
            if scope_status == "in_scope":
                if product_category not in ALLOWED_CATEGORIES:
                    raise ValueError(
                        f"Invalid or missing category for {asin}: "
                        f"{product_category}"
                    )

            # Non-single-category products must not have
            # a confirmed single top-level category.
            else:
                if product_category:
                    raise ValueError(
                        f"Unexpected single category for {asin}: "
                        f"{product_category}"
                    )

            mapping[asin] = {
                "product_category": (product_category or None),
                "product_type": (product_type or None),
                "scope_status": scope_status,
                "audit_status": audit_status,
                "mapping_resolution": (row["final_mapping_resolution"].strip()),
                "audit_id": (row["audit_id"].strip() or None),
                "audit_decision": (row["audit_decision"].strip() or None)
            }

            scope_counts[scope_status] += 1
            audit_counts[audit_status] += 1

    # Dataset-level validation

    if len(mapping) != EXPECTED_ITEMS:
        raise ValueError(
            f"Expected {EXPECTED_ITEMS} items, "
            f"found {len(mapping)}"
        )

    if dict(scope_counts) != EXPECTED_SCOPE_COUNTS:
        raise ValueError(f"Scope distribution mismatch: {dict(scope_counts)}")

    if dict(audit_counts) != EXPECTED_AUDIT_COUNTS:
        raise ValueError(f"Audit distribution mismatch: {dict(audit_counts)}")

    print("\nProduct category mapping validation: PASSED")

    print("Total products:", len(mapping))

    print("Scope distribution:")
    for status, count in sorted(scope_counts.items()):
        print(f"  {status}: {count}")

    print("\nAudit distribution:")
    for status, count in sorted(audit_counts.items()):
        print(f"  {status}: {count}")

    print("\nAll ASINs are unique.")
    print("All in-scope categories are valid.")

    return mapping
