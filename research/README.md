# Research implementation

This directory contains the historical review-processing and profile-building code used by the research prototype. The portable app is in `../src`. The two share the same research design; the small synthetic app dataset is provided for quick installation.

## Original research data

| Quantity | Count |
|---|---:|
| Training reviews processed | 2,029 |
| User records | 253 |
| Users with in-scope aspect profiles | 250 |
| Reviews contributing to the in-scope user-profile dataset | 1,792 |
| Item profiles before scope filtering | 341 |
| In-scope products used for recommendation | 297 |
| Extracted attribute–sentiment–evidence records | 9,255 |

The [research snapshot](examples/research_snapshot.json) contains selected real profile aggregates and three saved recommendation examples. The example user has six in-scope reviews and 23 attribute profiles. Its identity and review text are omitted. The snapshot records the sampling choice; it is an illustration, not an evaluation sample.

For that user, the moisturizer query recalls 38 unseen candidates and displays five recommendations. The sulfate-free shampoo query displays four recommendations with catalog support and five unverified candidates. A non-strict nail-polish priority comparison requests clarification. The selected intents are cached development outputs; ranking and retrieval were executed against the prepared research data.

## Read the implementation

| Stage | Source |
|---|---|
| Select training reviews and catalog metadata; align review records | [prepare_dataset.py](src/prepare_dataset.py), [align_reviews.py](src/align_reviews.py) |
| Extract attributes, sentiment, and verbatim evidence with a local LLM | [aspect_extraction.py](src/aspect_extraction.py), [full_aspect_extraction.py](src/full_aspect_extraction.py) |
| Check evidence and prepare audited extraction records | [audit_aspect_extraction.py](src/audit_aspect_extraction.py), [prepare_aspect_data.py](src/prepare_aspect_data.py) |
| Apply approved label mappings and prepare profile inputs | [prepare_profile_data.py](src/prepare_profile_data.py) |
| Aggregate review-level and user-level attributes | [build_user_profiles.py](src/build_user_profiles.py), [build_user_profiles_in_scope.py](src/build_user_profiles_in_scope.py) |
| Aggregate item attribute performance | [build_item_profiles.py](src/build_item_profiles.py) |
| Build historical attention-weighted item scores | [build_user_item_matching_in_scope.py](src/build_user_item_matching_in_scope.py) |
| Validate audited product categories and scope | [product_category_mapping.py](src/product_category_mapping.py) |

Section formatting has been tidied. [Source hashes and AST checks](source_manifest.json) record that the published Python statements and strings match the original scripts.

## Connection to the proposed research

The prototype investigates three linked problems: constructing user and item attribute representations, coordinating historical attention with explicit current requirements, and evaluating recommendation outcomes and explanations.

The implemented representation uses LLM extraction and statistical aggregation. Unapproved attribute aliases remain separate, so systematic alignment is still an open problem. Coordination currently uses fixed weights, finite attribute rules, and explicit hard-constraint checks. BM25-based evidence selection is compared separately in the offline experiment. These implementations provide baselines for studying more general coordination mechanisms.

The [offline report](../reports/offline_ranking_comparison.md) compares four strategies over 253 histories and 72 simulated development queries. The primary subset has 4,299 artificial combinations. Internal support metrics share the scoring rules; independent query-item relevance and explanation-faithfulness evaluation remain future work.

## Running and data access

The portable app can load the prepared research inputs through `RECOMMENDER_DATA_DIR`; see [reproduction instructions](../docs/REPRODUCIBILITY.md). The repository does not distribute the original review corpus or manual audit workbooks.

The scripts here retain the original layout: `research/data/raw` for source inputs, `research/data/processed` for prepared files, and `research/outputs` for extraction and audit outputs. These directories are ignored by Git. Required filenames are defined near the beginning of each script. The historical workflow also depends on approved taxonomy and category-mapping files; those inputs are not replaced by the synthetic demo files.

`research/requirements.txt` lists the additional imports. This publication checks source equivalence and syntax; it has not rerun the full historical preprocessing pipeline from raw inputs. Some scripts write outputs at top level and full extraction calls Ollama, so inspect their input paths before executing them.

The source is [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/). The maintainers discuss its usage and licensing in [this thread](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1). The snapshot here publishes aggregates and selected score outputs, without redistributing review text.
