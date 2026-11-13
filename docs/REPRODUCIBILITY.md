# Reproducing the results

## Public demo

The README commands run on the bundled fictional data. Saved example requests and tests need no model server. `scripts/demo_comparison.py` illustrates the four ranking rules on this small dataset; it will not reproduce the research metrics.

## Research experiment

The original experiment requires prepared profiles, audited mappings, the train/validation/test splits, and 72 adopted Run14 intent records. These inputs are not distributed. The runner expects 253 users and 72 queries and stops if the counts differ.

If you have the original prepared project and permission to use its data:

```powershell
python scripts/import_research_data.py --source-project "PATH_TO_ORIGINAL_PREPARED_PROJECT"
$env:RECOMMENDER_DATA_DIR = (Resolve-Path data/local).Path
python src/run_offline_ranking_experiment_v1.py --output-dir outputs/research_repeat01
```

The importer copies the required files to ignored `data/local` and refuses to overwrite an existing import. Use a new output directory for each experiment. The comparison uses saved intents and makes no new LLM calls.

On macOS/Linux, set the data path with `export RECOMMENDER_DATA_DIR="$(pwd)/data/local"`. Start a new app process after changing the variable; unset it to return to the demo.

This release supports rerunning the comparison from prepared inputs. Rebuilding those inputs from raw reviews also requires the original extraction and manual review stages.

## Result files

`reports/original_research_summary.json` and `reports/original_research_strategy_summary.csv` contain the original aggregate results. Per-user rankings, review traces, and historical audit files remain outside this repository. Hashes in original research artifacts refer to that run's files, not refactored public source. Timings depend on the runtime environment.

The study reused simulated development queries and paired them artificially with histories. Its rule-support metrics reuse the scoring rules. Held-out interactions have no paired current queries. These limits prevent interpreting the results as independent query-aware recommendation accuracy.
