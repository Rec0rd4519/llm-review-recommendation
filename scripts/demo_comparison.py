"""Illustrative ranking on fictional data, not the reported research experiment."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prototype_engine import PrototypeEngine,hard_fact
from offline_ranking_strategies import ExperimentFeatures,STRATEGIES,combine_scores,rank_items,global_bm25_scores
engine=PrototypeEngine(); feature=ExperimentFeatures(engine)
query='I need a moisturizer that absorbs quickly and feels non-greasy.'
intent=engine.extract(query)['intent']; user='DEMO-USER-001'
weights=feature.preference_weights(intent);bm=global_bm25_scores(feature.index,query)
pool,_=engine.recall(intent);scores={}
for iid in pool:
    if iid in engine.seen[user]:continue
    h,_=feature.history(user,iid)
    i=feature.aggregate([feature.preference(p,iid) for p in intent['preferences']],weights)
    q=feature.aggregate([feature.preference(p,iid,bm) for p in intent['preferences']],weights)
    scores[iid]=combine_scores(h,i,q,bool(weights))
print(json.dumps({'data':'fictional synthetic demonstration only','query':query,'rankings':{s:rank_items(scores,s)[:5] for s in STRATEGIES},'scores':scores},indent=2))
