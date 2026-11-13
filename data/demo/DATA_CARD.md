# Demo data

The dataset contains 3 fictional users, 12 products, and 12 synthetic reviews. Product names, catalog details, review texts, and user identifiers were written for the demo. No real customer identifiers or Amazon review text is included.

The 24 aspect-evidence records were assigned manually. User and item profiles were constructed from them, rather than measured in a new LLM extraction run. Demo history weights use `log(1 + support count)`; item sentiment is +1 or -1; match scores average shared aspects using those weights. This is a simplified fixture for exercising the app.

The training CSV identifies demo review pairs. The mapping filename retains `341` for loader compatibility, although the file has 12 rows.

The three saved intent examples come from project-authored queries and adopted Run14 interpretations. They cover ordinary preferences, a sulfate-free requirement, and a priority comparison needing clarification. The interface labels them as cached examples.

This dataset demonstrates app behavior. The reported research experiment uses separate data described in [reproduction instructions](../../docs/REPRODUCIBILITY.md).
