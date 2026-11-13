"""Compare four ranking rules on reused development queries.

The primary comparison pairs histories and simulated queries.
A separate history-only evaluation uses observed held-out interactions."""
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from prototype_engine import PrototypeEngine,ROOT,DATA,C,hard_fact
from offline_ranking_strategies import (ExperimentFeatures,STRATEGIES,RETRIEVAL_K,
    combine_scores,rank_items,overlap_at_k,single_target_metrics,global_bm25_scores)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)
def append(file,value):
    file.write(json.dumps(value,ensure_ascii=False)+'\n')
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
def normalized_stats(checks):
    counts=Counter(c['status'] for c in checks)
    n=len(checks)
    return {'support':counts['supporting_hint']/n if n else 0,
        'resolved':(counts['supporting_hint']+counts['contrary_hint'])/n if n else 0,
        'conflict':counts['conflicting_evidence']/n if n else 0}
def add_metrics(container,values):
    container['cases']+=1
    for name,value in values.items():
        container[name]+=value
def averages(container):
    n=container['cases']
    return {'cases':n,**{k:v/n for k,v in container.items() if k!='cases'}} if n else {'cases':0}


def self_checks():
    assert combine_scores(.8,.2,.9,True)=={'history_only':.8,'intent_only':.2,'fixed_weight':.44000000000000006,'query_conditioned':.8600000000000001}
    assert combine_scores(.8,.5,.5,False)['fixed_weight']==.8
    assert overlap_at_k(['A','B'],['B','C'],5)==.5
    assert single_target_metrics(['A','B'],'B',5)['ndcg']==1/math.log2(3)
    assert single_target_metrics(['A'],'B',5)=={'hr':0.,'ndcg':0.,'mrr':0.}
    assert rank_items({'B':{'intent_only':.5},'A':{'intent_only':.5}},'intent_only')==['A','B']


def run(out):
    self_checks()
    from runtime_config import IS_DEMO
    if IS_DEMO:
        raise SystemExit("This research protocol requires the original prepared research inputs. See docs/REPRODUCIBILITY.md; use scripts/demo_comparison.py for synthetic data.")
    if out.exists():
        raise FileExistsError(f'Preserve prior experiment: {out}')
    engine=PrototypeEngine();feature=ExperimentFeatures(engine)
    users=sorted(engine.users)
    records=sorted(engine.cached.values(),key=lambda r:r['intent_id'])
    assert len(users)==253 and len(records)==72
    out.mkdir(parents=True)
    benchmark=DATA/'benchmark'
    sources=[Path(__file__),ROOT/'src/offline_ranking_strategies.py',ROOT/'src/prototype_engine.py',
        ROOT/'src/bm25_review_retrieval.py',DATA/'user_item_matching_scores_in_scope.csv',
        DATA/'user_profiles_in_scope.jsonl',DATA/'item_profiles.jsonl',DATA/'catalog_metadata.jsonl',
        DATA/'intent_results.jsonl']
    sources += [benchmark/f'All_Beauty.{split}.csv' for split in ('train','valid','test')]
    hashes={str(p):sha(p) for p in sources}
    hashes.update({str(DATA/('benchmark' if name=='All_Beauty.train.csv' else '')/name):digest for name,digest in engine.review_bm25_manifest['source_files_sha256'].items()})
    protocol={'status':'fixed_before_score_generation','version':'ranking_experiment_v1.1_timing_instrumentation_corrected',
        'strategies':{
            'history_only':'H: normalized existing historical attention/item-quality baseline; missing pair uses existing rating fallback.',
            'intent_only':'I: rule-interpreted explicit preferences over ALL attribute-aligned evidence in the same8050-snippet training corpus plus catalog facts; unknown/conflict=.5.',
            'fixed_weight':'.4H+.6I; if no ordinary preferences, H.',
            'query_conditioned':'.4H+.6I_Q; I_Q uses BM25 top5 positive-score snippets per item/preference FROM THE SAME aligned evidence set, same interpreter/catalog; no preferences =>H.'},
        'query_text':'Exact original current query; unique normalized terms; global BM25 IDF statistics.',
        'weights':'Ordinary preference1; higher priority endpoint aligned as in prototype2. No inferred new desired values.',
        'shared_candidate_pool':'Audited in_scope297, same current product-type recall and metadata hard-constraint gate, exclude training-reviewed items. This is history-only SCORING within a query-filtered pool.',
        'primary_cases':'All253 histories x all72 reused simulated queries=18216 artificial combinations; no claim these are real paired user tasks.',
        'primary_metric_subset':'Has ordinary preferences AND at least5 eligible unseen items; selection does not depend on scores/evidence support.',
        'primary_metrics':['Top5/10 overlap with history','Top5 list identity with history','full-corpus rule support/resolution/conflict rates@5 (internal diagnostic only)','mean historical baseline@5 (internal)','score tie fraction'],
        'timing':'Record evidence feature build per unique query separately from warm feature-cached score/sort time per combination. Shared loading/filtering excluded; not end-to-end latency.',
        'secondary':'History-only ranking of all in_scope unseen items for each benchmark test user; exclude all supplied history including validation interaction. Training-only profiles; no validation-profile update.',
        'secondary_metrics':'Single observed heldout interaction per test user:HR@5/10,NDCG@5/10,MRR@10. Report all253 and in_scope-target subset separately. Observed interaction is not a positive preference rating label.',
        'not_computed':'No intent-aware NDCG/Recall or claimed relevance accuracy without query-item labels. No parameter tuning, significance claims, real-user generalization or accuracy uplift.',
        'difference_from_live_prototype':'Live I uses stored profile examples; experimental I uses all aligned evidence from the BM25 corpus to keep evidence access comparable. App remains unchanged.',
        'fixed_parameters':{'history_weight':.4,'intent_weight':.6,'bm25_k1':1.2,'bm25_b':.75,'retrieval_top_k':RETRIEVAL_K},
        'sources_sha256':hashes,'users':users,'query_ids':[r['intent_id'] for r in records],
        'new_model_calls':0,'new_relevance_labels':0}
    save(out/'protocol.json',protocol)
    save(out/'query_intents_snapshot.json',[{'intent_id':r['intent_id'],'query':r['original_query'],'status':r['status'],'intent':r['intent']} for r in records])
    save(out/'bm25_index_manifest.json',engine.review_bm25_manifest)
    # Check scores and tie-breaking against the retriever.
    for q in [r['original_query'] for r in records[:5]]:
        raw=global_bm25_scores(feature.index,q)
        expected=sorted(raw,key=lambda i:(-raw[i],feature.index.documents[i]['document_id']))[:5]
        actual=feature.index.search(q,5)
        assert [feature.index.documents[i]['document_id'] for i in expected]==[h['document_id'] for h in actual]
        assert all(abs(raw[i]-h['bm25_score'])<1e-8 for i,h in zip(expected,actual))
    sums={name:defaultdict(float) for name in STRATEGIES}
    query_metrics=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    state_counts=Counter();pairwise=defaultdict(float);pairwise_n=0
    timing=defaultdict(float);query_summaries=[];feature_diffs=Counter()
    ranking_files={name:(out/f'{name}_rankings.jsonl').open('x',encoding='utf-8') for name in STRATEGIES}
    try:
      with (out/'all_combination_outcomes.jsonl').open('x',encoding='utf-8') as cases_file, (out/'query_item_evidence_features.jsonl').open('x',encoding='utf-8') as feature_file:
        for number,row in enumerate(records,1):
            intent=row['intent'];qid=row['intent_id'];prefs=intent['preferences'];weights=feature.preference_weights(intent)
            blocked=any(u['reason'].startswith('unsupported_') for u in intent['unresolved_requirements'])
            pool,_=engine.recall(intent)
            eligible=[];rejected=Counter()
            if not blocked:
                for iid in pool:
                    hard=[hard_fact(h,engine.metadata.get(iid,{})) for h in intent['hard_constraints']]
                    if any(h['status']=='violates' for h in hard):
                        rejected['violates']+=1
                    elif any(h['status']=='unknown' for h in hard):
                        rejected['unknown']+=1
                    else:
                        eligible.append(iid)
            features={}
            started=time.perf_counter()
            for iid in eligible:
                checks=[feature.preference(p,iid) for p in prefs]
                features[iid]={'I':feature.aggregate(checks,weights),'checks_I':checks}
            i_ms=(time.perf_counter()-started)*1000
            started=time.perf_counter()
            retrieval=global_bm25_scores(feature.index,row['original_query']) if prefs and eligible else {}
            for iid in eligible:
                checks=[feature.preference(p,iid,retrieval) for p in prefs]
                features[iid].update(I_Q=feature.aggregate(checks,weights),checks_Q=checks)
                if prefs:
                    feature_diffs['item_query_preference_cases']+=1
                    feature_diffs['changed_intent_score']+=features[iid]['I']!=features[iid]['I_Q']
            q_ms=(time.perf_counter()-started)*1000
            for iid in eligible:
                append(feature_file,{'query_id':qid,'parent_asin':iid,**features[iid]})
            timing['intent_feature_ms']+=i_ms;timing['conditioned_feature_ms']+=q_ms
            query_counts=Counter()
            for user in users:
                candidates=[iid for iid in eligible if iid not in engine.seen[user]]
                status='needs_clarification' if blocked else ('no_catalog_match' if not pool else ('no_verified_hard_candidates' if not eligible else ('all_candidates_seen' if not candidates else 'rankable')))
                state_counts[status]+=1;query_counts[status]+=1
                primary=status=='rankable' and bool(prefs) and len(candidates)>=5
                append(cases_file,{'query_id':qid,'user_id':user,'status':status,'catalog_candidate_count':len(pool),
                    'hard_gate_excluded':dict(rejected),'eligible_unseen_item_ids':candidates,'primary_metric_eligible':primary})
                if status!='rankable':
                    continue
                histories={iid:feature.history(user,iid) for iid in candidates}
                scores={iid:combine_scores(histories[iid][0],features[iid]['I'],features[iid]['I_Q'],bool(prefs)) for iid in candidates}
                rankings={}
                for strategy in STRATEGIES:
                    tick=time.perf_counter();ranked=rank_items(scores,strategy);elapsed=(time.perf_counter()-tick)*1000
                    timing[strategy+'_warm_rank_ms']+=elapsed
                    rankings[strategy]=ranked
                    append(ranking_files[strategy],{'query_id':qid,'user_id':user,'strategy':strategy,
                        'candidate_count':len(candidates),'primary_metric_eligible':primary,
                        'top10':[{'parent_asin':iid,'score':scores[iid][strategy],
                                  'H':histories[iid][0],'history_fallback':histories[iid][1],
                                  'I':features[iid]['I'],'I_Q':features[iid]['I_Q']} for iid in ranked[:10]]})
                assert all(set(r)==set(candidates) for r in rankings.values())
                if primary:
                    for strategy,ranked in rankings.items():
                        selected=ranked[:5]
                        checks=[c for iid in selected for c in features[iid]['checks_I']]
                        hints=normalized_stats(checks)
                        values={'top5_overlap_with_history':overlap_at_k(rankings['history_only'],ranked,5),
                            'top10_overlap_with_history':overlap_at_k(rankings['history_only'],ranked,10),
                            'top5_identical_to_history':float(rankings['history_only'][:5]==selected),
                            'internal_rule_support_at5':hints['support'],'internal_rule_resolved_at5':hints['resolved'],
                            'internal_rule_conflict_at5':hints['conflict'],
                            'mean_internal_history_quality_at5':statistics.mean(histories[iid][0] for iid in selected),
                            'candidate_score_tie_fraction':1-len({round(scores[iid][strategy],12) for iid in candidates})/len(candidates)}
                        add_metrics(sums[strategy],values);add_metrics(query_metrics[qid][strategy],values)
                    pairwise_n+=1
                    for a,b in combinations(STRATEGIES,2):
                        pairwise[a+'__'+b]+=overlap_at_k(rankings[a],rankings[b],5)
            query_summaries.append({'query_id':qid,'query':row['original_query'],'preference_count':len(prefs),
                'catalog_candidate_count':len(pool),'hard_verified_candidate_count':len(eligible),
                'hard_gate_excluded':dict(rejected),'combination_status_counts':dict(query_counts),
                'intent_feature_build_ms':i_ms,'conditioned_feature_build_ms':q_ms})
            print(f'[{number}/72] {qid}: pool={len(pool)}, hard_eligible={len(eligible)}, states={dict(query_counts)}',flush=True)
    finally:
        for f in ranking_files.values():
            f.close()
    primary={name:averages(sums[name]) for name in STRATEGIES}
    assert sum(state_counts.values())==253*72
    assert len({m['cases'] for m in primary.values()})==1
    macro={name:{} for name in STRATEGIES}
    for name in STRATEGIES:
        qmeans=[averages(per[name]) for per in query_metrics.values()]
        macro[name]={'unique_queries':len(qmeans),**{metric:statistics.mean(q[metric] for q in qmeans) for metric in primary[name] if metric!='cases'}}
    # Evaluate history-only scoring on observed holdout interactions.
    # Keep target text, titles, ratings, and held-out reviews out of intent inputs.
    train=read_csv(benchmark/'All_Beauty.train.csv');valid=read_csv(benchmark/'All_Beauty.valid.csv');test=read_csv(benchmark/'All_Beauty.test.csv')
    train_pairs={(r['user_id'],r['parent_asin']) for r in train}
    assert len(test)==253 and not train_pairs&{(r['user_id'],r['parent_asin']) for r in test}
    assert not train_pairs&{(r['user_id'],r['parent_asin']) for r in valid}
    training_reviews=[]
    with (DATA/'profile_data.jsonl').open(encoding='utf-8-sig') as f:
        training_reviews=[json.loads(s) for s in f if s.strip()]
    assert all((r['user_id'],r['parent_asin']) in train_pairs for r in training_reviews)
    holdout_sums={k:defaultdict(float) for k in ('all253','target_in_scope')};holdout_rows=[]
    for r in test:
        user=r['user_id'];target=r['parent_asin'];history=set(r['history'].split())|engine.seen[user]
        candidates=[iid for iid in engine.mapping if iid not in history]
        ranked=sorted(candidates,key=lambda iid:(-feature.history(user,iid)[0],iid))
        metrics={}
        for k in (5,10):
            m=single_target_metrics(ranked,target,k)
            metrics.update({f'HR@{k}':m['hr'],f'NDCG@{k}':m['ndcg']})
        metrics['MRR@10']=single_target_metrics(ranked,target,10)['mrr']
        add_metrics(holdout_sums['all253'],metrics)
        if target in engine.mapping:
            add_metrics(holdout_sums['target_in_scope'],metrics)
        holdout_rows.append({'user_id':user,'target':target,'target_in_scope':target in engine.mapping,
            'target_in_supplied_history':target in history,'candidate_count':len(candidates),
            'target_rank':ranked.index(target)+1 if target in ranked else None,'metrics':metrics,'top10':ranked[:10]})
    sources_unchanged=all(sha(Path(p))==h for p,h in hashes.items())
    assert sources_unchanged
    ranking_cases=state_counts['rankable']
    result={'status':'completed_preliminary_offline_ranking_comparison',
        'unique_queries':72,'user_histories':253,'artificial_combinations':253*72,
        'combination_status_counts':dict(state_counts),'primary_per_strategy':primary,
        'query_macro_primary_metrics':macro,'pairwise_top5_overlap':{k:v/pairwise_n for k,v in pairwise.items()},
        'feature_score_differences':dict(feature_diffs),
        'timing':{'shared_history_table_loading_and_index_build_excluded':True,
            'intent_features_total_ms_over72_queries':timing['intent_feature_ms'],
            'conditioned_features_total_ms_over72_queries':timing['conditioned_feature_ms'],
            'mean_warm_rank_ms_per_rankable_combination':{s:timing[s+'_warm_rank_ms']/ranking_cases for s in STRATEGIES}},
        'secondary_history_only_observed_holdout':{k:averages(v) for k,v in holdout_sums.items()},
        'holdout_target_outside_scope':sum(not r['target_in_scope'] for r in holdout_rows),
        'holdout_target_in_supplied_history':sum(r['target_in_supplied_history'] for r in holdout_rows),
        'sources_unchanged':sources_unchanged,'new_model_calls':0,'new_query_item_relevance_labels':0,
        'conclusion_limits':protocol['not_computed'],
        'primary_metric_reference':'Internal rule rates are diagnostic and partly optimized by the compared methods; not independent relevance evaluation.',
        'holdout_metric_reference':'https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html'}
    save(out/'summary.json',result);save(out/'query_coverage_and_timing.json',query_summaries)
    save(out/'per_query_primary_metrics.json',{q:{s:averages(v) for s,v in models.items()} for q,models in query_metrics.items()})
    save(out/'history_only_holdout_records.json',holdout_rows)
    with (out/'strategy_summary.csv').open('x',encoding='utf-8-sig',newline='') as f:
        fields=['strategy']+list(primary['history_only'])
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for strategy in STRATEGIES:
            writer.writerow({'strategy':strategy,**primary[strategy]})
    print(json.dumps({'status':result['status'],'primary_metrics':primary,'holdout':result['secondary_history_only_observed_holdout'],'new_model_calls':0},ensure_ascii=False))
    return result


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/offline_ranking_experiment_v1')
    args=parser.parse_args()
    run(args.output_dir)
