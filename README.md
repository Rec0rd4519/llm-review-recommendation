# Review Compass

A local recommendation prototype that combines a user's review history with their current shopping request. Results include the review snippets and catalog details used to score each product.

The project uses a local LLM for intent extraction, attribute-level profiles for historical matching, and BM25 for review retrieval. A separate experiment compares four ranking strategies.

## Try the demo

Python 3.11 or later is required. In the project directory, run:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src/prototype_app.py
```

On macOS or Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

Open **http://127.0.0.1:8765**. If the port is busy, add `--port 8766` to the last command.

Select `DEMO-USER-001`, keep the default moisturizer request, and click **Parse request**, then **Recommend**. Expand a product card to inspect its evidence, or export the result as JSON.

The bundled dataset has **3 fictional users, 12 products, and 12 synthetic reviews**. Its three example requests use saved interpretations and work without a running model:

- A fast-absorbing, non-greasy moisturizer.
- A sulfate-free shampoo, with verified and unverified candidates shown separately.
- A nail-polish request whose priority comparison needs clarification.

## Parse a new request

Install [Ollama](https://ollama.com/) and download the model:

```sh
ollama pull qwen3.5:9b
```

Start Ollama, enter an English request, and enable live parsing in the app. Extraction uses a draft followed by one constrained regeneration, with at most two model calls. Traces are saved under `outputs/prototype_sessions/`.

The default model is `qwen3.5:9b`. Set `RECOMMENDER_MODEL` to use another compatible model, or `RECOMMENDER_MODEL_DIGEST` to require a specific installed digest.

## How ranking works

The app scores products with **0.4 × history + 0.6 × current preference evidence**. It excludes previously reviewed products and checks hard requirements against catalog facts. Candidates with unverified hard requirements appear in a separate list. If the request has no ordinary preferences, scoring uses history alone.

BM25 retrieves related review snippets for the displayed products. It adds evidence to the explanation; it leaves the app's ranking unchanged. The separate experiment also tests using BM25-selected evidence in the ranking itself.

```powershell
.venv\Scripts\python.exe src/bm25_review_retrieval.py --query "quick absorption non-greasy" --top-k 3
.venv\Scripts\python.exe scripts/demo_comparison.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See [architecture](docs/ARCHITECTURE.md) for profile construction and scoring details.

## Offline experiment

The original research dataset contains 253 user histories. An experiment paired them with 72 simulated development queries and compared history-only, intent-only, fixed-weight, and query-conditioned ranking. The primary subset contains 4,299 combinations.

Fixed-weight and query-conditioned rankings had 97.10% mean Top-5 set overlap. Query-item relevance labels were unavailable, so these results describe ranking behavior rather than an accuracy improvement. Held-out interaction metrics are reported separately for history-only ranking.

- [Experiment report and results](reports/offline_ranking_comparison.md)
- [BM25 implementation](reports/bm25_retrieval.md)
- [Data requirements and reproduction](docs/REPRODUCIBILITY.md)

The research data is separate from the public demo. This repository includes aggregate results, not the original review corpus or private run logs.

## Project layout

```text
src/          app, intent extraction, retrieval, and ranking
prompts/      extraction and correction prompts
data/demo/    fictional demo data and saved example intents
data/schemas/ intent JSON schema
scripts/      demo comparison and research-data import
tests/        checks that run without a model server
reports/      experiment results and verification records
docs/         architecture, reproduction, and upload instructions
```

The demo data is described in its [data card](data/demo/DATA_CARD.md). [Portability checks](reports/portability_verification.json) cover the saved extraction contracts, original experiment summaries, and tests in a relocated copy. Model calls in the runner test are mocked.

This prototype supports a limited set of English requests and attribute rules. Scores and retrieved snippets need human interpretation. A software license has [not yet been selected](LICENSE_SELECTION.md).
