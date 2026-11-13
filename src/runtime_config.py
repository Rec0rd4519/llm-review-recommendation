"""Repository-relative configuration; demo data is the default."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=Path(os.environ.get('RECOMMENDER_DATA_DIR',str(ROOT/'data/demo'))).expanduser().resolve()
IS_DEMO=DATA_DIR==(ROOT/'data/demo').resolve()
MODEL=os.environ.get('RECOMMENDER_MODEL','qwen3.5:9b')
EXPECTED_MODEL_DIGEST=os.environ.get('RECOMMENDER_MODEL_DIGEST','')
TRAIN_FILE=DATA_DIR/'benchmark/All_Beauty.train.csv'
