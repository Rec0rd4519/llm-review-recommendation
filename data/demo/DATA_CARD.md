# Demo data

The demo contains 3 fictional users, 12 products, and 12 synthetic reviews. Product names, catalog details, review texts, and user identifiers were written for this dataset.

Its 24 aspect-evidence records were assigned manually. Profiles were constructed from those records. History weights use `log(1 + support count)`, item sentiment is +1 or -1, and match scores average shared attributes using those weights. These fixtures exercise the app; they are not measurements from a new LLM extraction run.

The training CSV identifies demo review pairs. The mapping filename retains `341` for loader compatibility, although it contains 12 rows.

The three saved intents use project-authored queries and adopted Run14 interpretations. They cover ordinary preferences, a sulfate-free requirement, and a priority comparison needing clarification. The app labels them as saved examples.

Research results use a separate dataset. See [reproduction instructions](../../docs/REPRODUCIBILITY.md).
