# Research pipeline

The scripts in `src/` process reviews and build the profiles used for historical matching. The interactive app and four-strategy comparison are in the repository's top-level `src/` directory.

## Dataset

| Quantity | Count |
|---|---:|
| Training reviews processed | 2,029 |
| User records | 253 |
| Users with in-scope aspect profiles | 250 |
| Reviews in the in-scope user-profile dataset | 1,792 |
| Item profiles before scope filtering | 341 |
| In-scope products | 297 |
| Extracted attribute–sentiment–evidence records | 9,255 |

The [output examples](examples/research_snapshot.json) contain profile aggregates and three recommendation cases for the first user in the research app. That user has six in-scope reviews and 23 attribute profiles. User identifiers and review text are omitted; up to six attributes are shown per profile.

For this user, the moisturizer request recalls 38 unseen candidates and displays five recommendations. The sulfate-free shampoo request displays four recommendations with catalog support and five unverified candidates. A non-strict priority comparison requests clarification. Intent parsing uses saved development outputs; ranking and retrieval use the prepared research data. These selected examples illustrate the app and are separate from the evaluation sample.

## Processing stages

| Stage | Source |
|---|---|
| Select and align training reviews and metadata | [prepare_dataset.py](src/prepare_dataset.py), [align_reviews.py](src/align_reviews.py) |
| Extract attributes, sentiment, and verbatim evidence | [aspect_extraction.py](src/aspect_extraction.py), [full_aspect_extraction.py](src/full_aspect_extraction.py) |
| Check evidence and assemble audited records | [audit_aspect_extraction.py](src/audit_aspect_extraction.py), [prepare_aspect_data.py](src/prepare_aspect_data.py) |
| Join attribute labels to approved mappings | [prepare_profile_data.py](src/prepare_profile_data.py) |
| Aggregate user attributes | [build_user_profiles.py](src/build_user_profiles.py), [build_user_profiles_in_scope.py](src/build_user_profiles_in_scope.py) |
| Aggregate item performance | [build_item_profiles.py](src/build_item_profiles.py) |
| Score items using historical attribute attention | [build_user_item_matching_in_scope.py](src/build_user_item_matching_in_scope.py) |
| Validate product categories and scope | [product_category_mapping.py](src/product_category_mapping.py) |

[Source hashes](source_manifest.json) identify the original and published files. AST checks cover every Python statement and string.

## Research scope

The proposal concerns attribute representations, coordination between history and current requirements, and evaluation of recommendations and explanations. This implementation uses LLM extraction, statistical aggregation, fixed fusion weights, finite attribute rules, and catalog-based hard-constraint checks. Unapproved aliases remain separate, leaving systematic attribute alignment unresolved.

The [offline comparison](../reports/offline_ranking_comparison.md) evaluates four strategies over 253 histories and 72 simulated development queries. Its primary subset has 4,299 artificial combinations. Independent query-item relevance and explanation-faithfulness evaluation remain future work.

## Inputs and execution

The scripts expect source inputs in `research/data/raw`, prepared files in `research/data/processed`, and extraction outputs in `research/outputs`. All three directories are ignored by Git. Each script defines its required filenames near the top. Approved taxonomy and category-mapping inputs are also required; the synthetic demo files do not replace them.

Additional dependencies are listed in `research/requirements.txt`. Several scripts begin processing when executed, and full extraction calls Ollama. Their syntax and source equivalence have been checked; the full workflow has not been rerun from raw inputs for this release.

The app can load an existing prepared dataset through `RECOMMENDER_DATA_DIR`. See [reproduction instructions](../docs/REPRODUCIBILITY.md).

The data source is [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/). Its maintainers discuss usage and licensing [here](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1). This directory includes code, aggregates, and selected scores; original review text and manual audit workbooks are not distributed.
