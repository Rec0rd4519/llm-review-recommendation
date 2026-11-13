# Architecture

The Python server binds to `127.0.0.1` and serves both the browser interface and API. Data loads from `data/demo` by default. Set `RECOMMENDER_DATA_DIR` before starting the process to use another prepared dataset; the interface labels the loaded dataset.

## Profiles

The research pipeline extracts attributes, sentiment, and supporting text from training reviews with a local LLM. Evidence checks and approved label mappings prepare the records for aggregation by user and item. Multiple quotes from one review are combined before aggregation, so each review contributes one score per attribute. Unapproved aliases remain separate.

Historical matching weights attributes by `log(1 + supporting review count)` and combines those weights with smoothed item-quality scores. Historical sentiment describes satisfaction with past products; its sign alone does not specify the attribute value wanted now.

[Research code](../research/README.md) covers extraction and aggregation. The app reads prepared files. The public demo uses constructed profiles over fictional reviews; manual taxonomy and category-audit inputs are not bundled.

## Intent extraction

Saved examples load directly. For a new request, Ollama produces a draft intent. The source policy derives a constrained output schema from the draft and request, then asks for one regeneration. Schema, source-evidence, and contract checks validate the result.

The policy implements a finite set of English expressions. Unsupported non-strict priority comparisons remain unresolved and can require clarification. The prompts are in `prompts/`; the policy version is Run14 v1.1.

## Recommendations

Product-type rules select candidates, and previously reviewed products are removed. Catalog contradictions exclude a candidate. Missing support for a hard requirement puts it in the unverified list.

The score combines normalized history and current evidence with weights of 0.4 and 0.6. Unknown or conflicting preference evidence scores 0.5. An aligned higher-priority preference receives weight 2; other preferences receive weight 1. Requests without ordinary preferences use history alone.

BM25 adds up to three query-related snippets to each displayed product after ranking. Snippets retain review IDs for inspection. Their lexical relevance does not establish requirement satisfaction.

## Offline comparison

The experiment compares H, I, 0.4H + 0.6I, and 0.4H + 0.6I_Q. I interprets all attribute-aligned evidence; I_Q selects up to five positive-scoring BM25 snippets from the same set. Interpretation rules and catalog fields are shared.

All strategies use the same query-filtered candidate pool and hard-constraint gate. History-only therefore refers to scoring within that pool. The [report](../reports/offline_ranking_comparison.md) gives the evaluation protocol and results.
