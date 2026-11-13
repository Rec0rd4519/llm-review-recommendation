"""Scoring rules for the four-strategy offline comparison.

Intent-only uses all aligned snippets; query-conditioned uses BM25-selected
snippets from the same corpus. Both share the interpreter and catalog facts."""
from collections import defaultdict
import math
from bm25_review_retrieval import tokenize
from prototype_engine import ASPECT_ALIASES,key,preference_signal

STRATEGIES=('history_only','intent_only','fixed_weight','query_conditioned')
HISTORY_WEIGHT=.4
INTENT_WEIGHT=.6
RETRIEVAL_K=5


def combine_scores(history,intent,conditioned,has_preferences):
    return {'history_only':history,'intent_only':intent,
        'fixed_weight':HISTORY_WEIGHT*history+INTENT_WEIGHT*intent if has_preferences else history,
        'query_conditioned':HISTORY_WEIGHT*history+INTENT_WEIGHT*conditioned if has_preferences else history}


def rank_items(scores,strategy):
    if strategy not in STRATEGIES:
        raise ValueError('Unknown strategy.')
    return sorted(scores,key=lambda iid:(-scores[iid][strategy],iid))


def overlap_at_k(left,right,k):
    size=min(k,len(left),len(right))
    return len(set(left[:size])&set(right[:size]))/size if size else None


def single_target_metrics(ranking,target,k):
    rank=ranking.index(target)+1 if target in ranking else None
    hit=rank is not None and rank<=k
    return {'hr':float(hit),'ndcg':1/math.log2(rank+1) if hit else 0.,'mrr':1/rank if hit else 0.}


def global_bm25_scores(index,query):
    """Untruncated scores with the already frozen BM25 implementation's formula."""
    scores=defaultdict(float)
    for term in sorted(set(tokenize(query))):
        posting=index.postings.get(term,{})
        if not posting:
            continue
        idf=math.log1p((index.n-len(posting)+.5)/(len(posting)+.5))
        for i,tf in posting.items():
            norm=1-index.b+index.b*index.lengths[i]/index.avgdl
            scores[i]+=idf*tf*(index.k1+1)/(tf+index.k1*norm)
    return scores


def classify(signs):
    signs=set(s for s in signs if s is not None)
    if signs=={1}:
        return 1.,'supporting_hint'
    if signs=={-1}:
        return 0.,'contrary_hint'
    return .5,'conflicting_evidence' if len(signs)==2 else 'not_verified'


class ExperimentFeatures:
    def __init__(self,engine):
        self.engine=engine
        self.index=engine.review_bm25
        self.item_docs=defaultdict(list)
        for i,d in enumerate(self.index.documents):
            self.item_docs[d['parent_asin']].append(i)
        self.baseline={}
        # Read the precomputed history baseline once, not once per query/user.
        import csv
        from prototype_engine import DATA
        with (DATA/'user_item_matching_scores_in_scope.csv').open(encoding='utf-8-sig',newline='') as file:
            for r in csv.DictReader(file):
                if r['match_score']:
                    self.baseline[(r['user_id'],r['parent_asin'])]=float(r['match_score'])

    def history(self,user,item):
        value=self.baseline.get((user,item))
        fallback=value is None
        if fallback:
            value=((self.engine.profiles[item].get('mean_overall_rating') or 3)-3)/2
        return max(0,min(1,(value+1)/2)),fallback

    def preference_weights(self,intent):
        result=[]
        for p in intent['preferences']:
            names=ASPECT_ALIASES.get(p['normalized_aspect'],[key(p['normalized_aspect'])])
            high=any(any(name in key(endpoint) or key(endpoint) in name for name in names)
                     for r in intent['priority_relations'] for endpoint in r['higher_priority_aspects'])
            result.append(2 if high else 1)
        return result

    def catalog_sources(self,item):
        details=self.engine.metadata.get(item,{}).get('details') or {}
        return [{'source':'catalog.details.'+field,'evidence':str(details[field]),
                 'match_context':str(details[field])+' scent' if field=='Scent' else field+': '+str(details[field])}
                for field in ('Scent','Finish Type','Product Benefits','Special Feature') if details.get(field)]

    def preference(self,p,item,scores=None):
        aliases=ASPECT_ALIASES.get(p['normalized_aspect'],[key(p['normalized_aspect'])])
        aligned=[i for i in self.item_docs[item] if any(key(a) in aliases for a in self.index.documents[i]['profile_aspects'])]
        selected=aligned
        if scores is not None:
            selected=sorted((i for i in aligned if scores.get(i,0)>0),
                key=lambda i:(-scores[i],self.index.documents[i]['document_id']))[:RETRIEVAL_K]
        signs=[];signal_trace=[]
        for i in selected:
            d=self.index.documents[i];sign=preference_signal(p,d['evidence']);signs.append(sign)
            if sign is not None:
                signal_trace.append({'document_id':d['document_id'],'signal':sign})
        for s in self.catalog_sources(item):
            sign=preference_signal(p,s['match_context']);signs.append(sign)
            if sign is not None:
                signal_trace.append({'source':s['source'],'evidence':s['evidence'],'signal':sign})
        value,status=classify(signs)
        return {'score':value,'status':status,'aligned_review_documents':len(aligned),
                'selected_review_documents':len(selected),
                'selected_document_ids':[self.index.documents[i]['document_id'] for i in selected],
                'interpretable_signal_trace':signal_trace}

    def aggregate(self,checks,weights):
        return sum(c['score']*w for c,w in zip(checks,weights))/sum(weights) if weights else .5
