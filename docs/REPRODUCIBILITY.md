# Reproducing the results

## Demo

The README commands use fictional data bundled with the repository. Saved requests and tests run without a model server. `scripts/demo_comparison.py` illustrates the four ranking rules on this dataset; it does not reproduce the research metrics.

## Research experiment

The comparison requires prepared profiles, audited mappings, train/validation/test splits, and 72 adopted Run14 intent records. It expects 253 users and 72 queries and checks those counts. The source data and manual audit inputs are not distributed.

With an existing prepared project and permission to use its data, run:

```powershell
python scripts/import_research_data.py --source-project "PATH_TO_ORIGINAL_PREPARED_PROJECT"
$env:RECOMMENDER_DATA_DIR = (Resolve-Path data/local).Path
python src/run_offline_ranking_experiment_v1.py --output-dir outputs/research_repeat01
```

The importer copies the required files into ignored `data/local`. It stops if that directory already exists. Use a new output directory for each experiment. The comparison reads saved intents and makes no new LLM calls.

On macOS/Linux, set the path with `export RECOMMENDER_DATA_DIR="$(pwd)/data/local"`. Restart the app after changing the variable. Unset it to return to the demo. In PyCharm, put the variable in the app's Run configuration; a terminal setting affects only that terminal's child processes.

The [research scripts](../research/README.md) show how reviews were extracted and aggregated. Rebuilding the prepared files also requires the source data and approved taxonomy and category mappings. Those prerequisites are separate from rerunning the ranking comparison.

## Published results

`reports/original_research_summary.json` and `reports/original_research_strategy_summary.csv` retain the original aggregate results. Per-user rankings, review traces, and audit files remain outside the repository. Original-run hashes identify that run's files; they do not describe the refactored app. Timings depend on the runtime environment.

The study reuses simulated development queries and pairs them artificially with histories. Rule-support metrics share the scoring logic. Held-out interactions lack paired current queries. The [report](../reports/offline_ranking_comparison.md) explains how these limits affect interpretation.
