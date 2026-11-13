# BM25 review retrieval

The app retrieves up to three query-related review snippets for each displayed product. Each hit includes its review ID, BM25 score, and matched terms. Retrieval runs after ranking, so these fields leave scores and constraint checks unchanged.

## Try it

Expand **BM25: Query-related review snippets** in a product card, or run:

```powershell
python src/bm25_review_retrieval.py --query "absorbs quickly non-greasy" --item-id DEMO-ITEM-004 --top-k 3
```

Omit `--item-id` to search the whole corpus. Repeat it to select several products. `--output` saves a new JSON file. This command uses the fictional demo. The corpus counts below refer to the research dataset.

## Corpus

The original index contains **8,050 evidence snippets from 297 in-scope products**. Inputs are extracted attribute-evidence records, the source reviews, audited scope mappings, and training interaction pairs. Validation and test review texts are excluded.

Of 9,255 source pairs, filtering removes 82 invalid or review-required records, 1,091 out-of-scope records, five records failing the substring provenance check, and one tokenless record. Deduplicating the remaining 8,076 pairs merges 26 repetitions.

Deduplication uses product, review ID, and normalized evidence text. Attribute and sentiment labels remain metadata. Both positive and negative snippets are indexed. Source matching normalizes Unicode, case, whitespace, and a declared set of punctuation characters.

## Scoring

The implementation uses k1 = 1.2 and b = 0.75:

```text
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(q,d) = sum over unique query terms:
    idf(t) * tf(t,d) * (k1 + 1)
    / (tf(t,d) + k1 * (1 - b + b * length(d) / average_length))
```

IDF and average document length use the full eligible corpus, including when results are filtered to one product. Tokenization retains negation and uses a small morphology map, such as absorbs/absorption to absorb. Document IDs break ties; a query with no matching terms returns no hits.

The formula follows the positive-IDF form documented by [Apache Lucene](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/search/similarities/BM25Similarity.html). Tokenization differs from Lucene's analyzers.

## Checks and use in ranking

The original integration passed 57 numerical, provenance, and integration checks. Six scenarios matched the previous engine after removing the added retrieval fields. Detailed records are retained with the research run. The repository tests cover the demo corpus and app integration.

BM25 ranks lexical matches, including negative reviews. Preference interpretation and hard-constraint verification are separate steps. The [offline comparison](offline_ranking_comparison.md) tests BM25-selected evidence as an experimental ranking component; the interactive app keeps fixed-weight ranking.
