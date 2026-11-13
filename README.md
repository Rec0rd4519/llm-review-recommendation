# LLM-Assisted Review-Based Personalized Recommendation

This prototype combines review-based user and item profiles with a shopper's current request. It was built to test the data flow and ranking choices in a research proposal.

The research pipeline processed **2,029 training reviews from 253 users**, built **341 item profiles**, and used **297 in-scope products** for recommendation. The app's quick-start dataset contains three fictional users so it can run without the original review corpus.

## Research code and results

- [Research pipeline](research/README.md): LLM extraction, evidence checks, user and item profiles, and historical scoring.
- [Research output examples](research/examples/research_snapshot.json): profile statistics and three recommendation cases. User identifiers and review text are omitted.
- [Offline comparison](reports/offline_ranking_comparison.md): four ranking strategies, results, and evaluation limits.

Original reviews and manual audit workbooks are not bundled. The [reproduction guide](docs/REPRODUCIBILITY.md) lists the inputs needed to rerun the research experiment.

## Run the demo

Use Python 3.11 or later. From the project directory:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src/prototype_app.py
```

On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

Open **http://127.0.0.1:8765**. If the port is busy, add `--port 8766` to the last command. Select `DEMO-USER-001`, keep the moisturizer request, and click **Parse request**, then **Recommend**. Expand a product card to inspect its evidence, or export the result as JSON.

The demo has **3 fictional users, 12 products, and 12 synthetic reviews**. Three requests use saved interpretations and need no model server: a non-greasy moisturizer, a sulfate-free shampoo, and a nail-polish priority comparison that needs clarification. See the [data card](data/demo/DATA_CARD.md).

## Parse a new request

Install and start [Ollama](https://ollama.com/), then download the model:

```sh
ollama pull qwen3.5:9b
```

Enter an English request and enable live parsing in the app. Extraction uses a draft and one constrained regeneration, with at most two model calls. Request and response traces go to `outputs/prototype_sessions/`.

`RECOMMENDER_MODEL` selects another compatible model. `RECOMMENDER_MODEL_DIGEST` can require a specific installed digest.

## Ranking and retrieval

The app scores products with **0.4 × history + 0.6 × current preference evidence**. It excludes reviewed products and checks hard requirements against catalog facts. Products with unverified requirements appear separately. Without ordinary preferences, scoring uses history alone.

BM25 retrieves related review snippets for each displayed product without changing its score. The offline experiment separately tests ranking with BM25-selected evidence. [Architecture](docs/ARCHITECTURE.md) describes both paths.

```powershell
.venv\Scripts\python.exe src/bm25_review_retrieval.py --query "quick absorption non-greasy" --top-k 3
.venv\Scripts\python.exe scripts/demo_comparison.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Experiment summary

The comparison pairs 253 histories with 72 simulated development queries. Its primary subset has 4,299 combinations. Fixed-weight and query-conditioned Top-5 sets overlap by 97.10% on average.

Query-item relevance labels were unavailable, so the comparison measures ranking behavior. Held-out interaction metrics are reported for history-only ranking. The [experiment report](reports/offline_ranking_comparison.md) gives definitions and denominators; the [BM25 report](reports/bm25_retrieval.md) describes the evidence corpus and retrieval checks.

## Files

```text
research/     review-processing code and research output examples
src/          app, intent extraction, retrieval, and ranking
prompts/      extraction and correction prompts
data/demo/    fictional data and saved example intents
data/schemas/ intent JSON schema
scripts/      demo comparison and local data import
tests/        checks that run without a model server
reports/      results and verification records
docs/         architecture and reproduction instructions
```

[Verification records](reports/portability_verification.json) cover extraction-contract parity, experiment summaries, and tests in a relocated copy. Runner tests use mocked model responses. The language rules cover a limited set of English requests. A software license has [not yet been selected](LICENSE_SELECTION.md).
