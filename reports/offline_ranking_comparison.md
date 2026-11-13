# Offline ranking comparison

This experiment compares four ranking strategies using 253 user histories and 72 simulated development queries. It examines changes in ranking when current requirements and query-selected evidence are added to historical matching.

## Setup

| Strategy | Score |
|---|---|
| History-only | H: normalized historical aspect-attention/item-quality score. Missing pairs use the item-rating fallback. |
| Intent-only | I: current preferences interpreted over all attribute-aligned review evidence and catalog fields. Unknown or conflicting evidence scores 0.5. |
| Fixed-weight | 0.4H + 0.6I. |
| Query-conditioned | 0.4H + 0.6I_Q. I_Q uses the five highest positive-scoring BM25 snippets from the same aligned evidence set. |

Without ordinary preferences, both hybrid strategies use H. Preference weights follow the app's rule: 2 for aligned higher-priority attributes and 1 otherwise. BM25 uses k1 = 1.2 and b = 0.75.

All strategies share product-type recall, hard-constraint checks, and exclusion of previously reviewed products. History-only therefore describes scoring within a query-filtered candidate pool. I and I_Q use the same 8,050-snippet training corpus, interpreter, and catalog fields; only snippet selection differs. The app's own preference scoring uses stored profile examples, while this comparison uses the full aligned corpus for I.

The Cartesian pairing produces 18,216 combinations. Histories and queries are reused across combinations; the combinations are artificial shopping scenarios. No parameters were tuned after reviewing results.

| Outcome | Combinations |
|---|---:|
| Rankable | 12,639 |
| All candidates previously reviewed | 11 |
| No candidate with verified hard requirements | 3,036 |
| No matching product type | 2,277 |
| Comparison needs clarification | 253 |
| Total | 18,216 |

The primary subset requires at least one ordinary preference and five eligible unseen products. It contains **4,299 combinations across 17 queries**. Selection depends on candidate availability, not scores. The coverage table also retains outcomes outside this subset.

## Ranking results

| Strategy | Top-5 set overlap with history | Identical Top-5 ordering | Internal rule support at 5 | Candidate score-tie fraction |
|---|---:|---:|---:|---:|
| History-only | 100.00% | 100.00% | 9.98% | 9.79% |
| Intent-only | 33.17% | 0.00% | 30.16% | 84.91% |
| Fixed-weight | 76.97% | 41.52% | 30.15% | 6.70% |
| Query-conditioned | 79.68% | 41.59% | 27.73% | 7.46% |

Top-5 overlap is the intersection size divided by five, averaged over eligible combinations. Identical ordering requires all five positions to match. The tie fraction is 1 minus the number of distinct scores divided by the candidate count.

Internal rule support evaluates ordinary preferences using the shared full-corpus interpreter, averaging within a combination before averaging across combinations. Since scoring uses those same rules, this metric is a diagnostic rather than independent relevance or Precision@5.

Intent-only changes the ranking substantially but produces many tied scores. Fixed-weight retains more of the historical ordering. Fixed-weight and query-conditioned Top-5 sets overlap by **97.10%** on average. BM25 selection changes the aggregate intent score in only **4 of 318 item-query feature pairs** containing ordinary preferences.

One example is the query “I would like a shampoo with a mild scent and thick lather.” For product B085J24382, the full corpus yields I = 0.75; BM25 selection misses an interpretable mild-scent snippet and yields I_Q = 0.5. Query-item relevance labels were unavailable, so the lower internal support rate cannot establish true relevance quality.

## Held-out interactions

The benchmark has 2,029 training, 253 validation, and 253 test interactions. Profiles use training data. Test ranking excludes each user's supplied history, including the known validation interaction; validation reviews do not update profiles. Training/test user-item separation was checked. The supplied split was used without reconstructing strict global event-time feature snapshots.

There are no paired current queries for the test interactions, so this part evaluates history-only ranking. Each observed test interaction supplies one target, regardless of its rating.

| Scope | Users | HR@5 | HR@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| All test users | 253 | 1.58% | 3.16% | 0.0121 | 0.0065 |
| Target within the catalog scope | 160 | 2.50% | 5.00% | 0.0192 | 0.0102 |

The 93 out-of-scope targets contribute zero hits to the all-user results. Among in-scope targets, four appear in the Top-5 and eight in the Top-10.

For one target, HR@K records a hit in the first K positions, NDCG is 1/log2(rank+1) for a hit and zero otherwise, and MRR uses 1/rank. See [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) for ranked-retrieval evaluation.

## Timing and verification

Evidence-feature computation across the 72 queries took 18.364 ms for I and 16.961 ms for I_Q in the recorded run. These timers exclude index construction, history loading, candidate filtering, model inference, HTTP handling, and file writes. Warm score/sort timing is reported separately in the aggregate JSON. These small-corpus timings measure feature computation, not end-to-end latency.

An earlier run included trace-file writing in the conditioned timer. The final run moved writes outside that timer. Rankings, coverage, primary metrics, and holdout results matched between runs. Verification reconstructed all 50,556 strategy ranking records from their saved components. The comparison made no new LLM calls and collected no new relevance labels.

## Interpretation and reproduction

The strategies produce different rankings. The available diagnostics do not independently establish an accuracy gain from BM25-conditioned scoring. Histories and simulated development queries are reused, rule-based diagnostics share the scoring logic, and no significance tests were conducted. The app retains fixed-weight scoring; query-conditioned ranking remains experimental.

Aggregate results are included in [JSON](original_research_summary.json) and [CSV](original_research_strategy_summary.csv). The original run also retained its protocol, query snapshots, candidate outcomes, per-user rankings, selected evidence, holdout records, and source hashes. Detailed artifacts are retained outside this repository.

See [reproduction instructions](../docs/REPRODUCIBILITY.md) for the prepared research inputs. The public fictional demo uses a different dataset.
