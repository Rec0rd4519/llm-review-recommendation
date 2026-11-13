# Architecture

The app runs a Python HTTP server on `127.0.0.1`. The browser sends requests to the same server. Data loads from `data/demo` by default; set `RECOMMENDER_DATA_DIR` before starting the process to use another prepared dataset.

## Profiles

In the original research pipeline, a local LLM extracted attributes, sentiment, and supporting quotes from training reviews. Evidence checks and attribute-label mappings prepared those records for aggregation. User profiles group records by user and attribute; item profiles group them by product and attribute. Review-level aggregation prevents multiple quotes from giving one review extra weight. Unapproved attribute aliases remain separate.

Historical matching weights an attribute by `log(1 + supporting review count)`. It combines these attention weights with smoothed item-quality scores. Historical sentiment remains available for explanation; its sign alone does not specify the attribute value a user wants today.

The public demo uses constructed profiles and scores over fictional reviews. Historical extraction, evidence-checking, and profile-building code is available under [research/](../research/README.md). Manual taxonomy and category-audit inputs are not bundled; the portable app consumes prepared files.

## Current request

Saved examples load directly. For a new request, Ollama produces a draft intent. The source policy builds a constrained output schema from that draft and the request text, then asks for one regeneration. The result must pass schema, evidence, and contract checks.

The policy handles a finite set of English expressions. Unsupported non-strict comparisons remain unresolved and can trigger a clarification request. The implementation retains the adopted Run14 v1.1 policy; its prompts are in `prompts/`.

## Recommendations

Product-type rules select candidates, and previously reviewed products are removed. Hard requirements need explicit catalog support. Contradicted requirements exclude a product; missing support puts it in the unverified list.

The app combines normalized history scores with current preference evidence using weights of 0.4 and 0.6. Unknown or conflicting preference evidence scores 0.5. Aligned higher-priority preferences receive weight 2; other preferences receive weight 1. Without ordinary preferences, the app uses history alone.

BM25 retrieves up to three related review snippets for each displayed product. Its index contains eligible training-review evidence with traceable review IDs. Retrieval adds explanation fields without changing product scores.

## Offline comparison

The experiment compares H, I, 0.4H + 0.6I, and 0.4H + 0.6I_Q. I interprets all attribute-aligned snippets; I_Q selects the five highest positive-scoring BM25 snippets from the same set. Both use the same interpretation rules and catalog fields.

All four strategies share product-type filtering and hard-constraint checks. Thus, history-only refers to the scoring component within a query-filtered pool. See the [experiment report](../reports/offline_ranking_comparison.md) for denominators and results.
