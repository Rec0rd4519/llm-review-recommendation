"""Copy prepared research inputs into the local data directory."""
from pathlib import Path
import argparse, shutil
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-project',required=True,type=Path)
args=parser.parse_args()
source=args.source_project.resolve(); target=ROOT/'data/local'
names=['profile_data.jsonl','profile_aspect_records.jsonl','item_profiles.jsonl','catalog_metadata.jsonl','user_profiles_in_scope.jsonl','product_category_mapping_v2_1_audit_merged_341.csv','user_item_matching_scores_in_scope.csv']
pairs=[(source/'data/processed'/n,target/n) for n in names]
pairs += [(source/'data/raw/benchmark/5core/last_out_w_his'/f'All_Beauty.{s}.csv',target/'benchmark'/f'All_Beauty.{s}.csv') for s in ('train','valid','test')]
pairs += [(source/'outputs/current_intent_beauty_development_run14/intent_extraction_results.jsonl',target/'intent_results.jsonl')]
missing=[str(a) for a,b in pairs if not a.is_file()]
if missing:raise SystemExit('Missing required source files: '+', '.join(missing))
if target.exists():raise SystemExit('Preserve existing data/local. Choose a clean distribution directory for another import.')
for a,b in pairs:
    b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
print('Imported prepared research files into ignored data/local. Set RECOMMENDER_DATA_DIR to its absolute path before starting a new Python process.')
