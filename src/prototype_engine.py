"""Product recall, evidence checks, and history/intent scoring for the local app."""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import csv
import json
import math
import re
import threading
import uuid
from jsonschema import Draft202012Validator
import intent_runner as frozen_runner
import runtime_config
from bm25_review_retrieval import load_training_index,tokenize

ROOT=Path(__file__).resolve().parents[1]
DATA=runtime_config.DATA_DIR
C=frozen_runner.policy
LOCK=threading.Lock()


def jsonl(path):
    with path.open(encoding='utf-8-sig') as file:
        return [json.loads(s) for s in file if s.strip()]


def key(value):
    return re.sub(r'[^a-z0-9]+',' ',str(value).lower()).strip()


# An explicit catalog type is retained as a subtype; title lookup is a recall
# aid, not evidence that ingredient/quantity/suitability requirements are met.
FAMILIES=[
 ('cleanser',r'face wash|facial cleanser|face cleanser|facial cleansing',r'facial cleanser|facial cleansing|cleansers and toners|face wash',{'Skin Care'}),
 ('moisturizer',r'face cream|facial cream|moisturizer|facial moisturizer',r'moisturiz|facial cream|skin cream',{'Skin Care'}),
 ('body lotion',r'body lotion|body butter|body cream',r'body lotion|body moisturizer|body butter|body cream',{'Bath & Body Care','Skin Care'}),
 ('shampoo',r'shampoo',r'shampoo',{'Hair Care & Styling'}),
 ('conditioner',r'conditioner',r'conditioner',{'Hair Care & Styling'}),
 ('hairbrush',r'hairbrush|hair brush',r'hair brush|hairbrush|brush and comb',{'Hair Care & Styling','Beauty Tools & Accessories'}),
 ('nail polish',r'nail polish',r'nail polish',{'Nail Care'}),
 ('mascara',r'mascara',r'mascara',{'Makeup & Cosmetics'}),
 ('brow gel',r'brow gel|eyebrow gel',r'brow gel|eyebrow gel|eyebrow makeup',{'Makeup & Cosmetics'}),
 ('lip balm',r'lip balm',r'lip balm|lip care',{'Skin Care','Makeup & Cosmetics'}),
 ('serum',r'facial serum|face serum|skin serum',r'serum|ampoule|essence',{'Skin Care'}),
 ('soap',r'hand soap|hand wash',r'hand soap|hand wash|hand and body bar soap',{'Personal Hygiene','Bath & Body Care'}),
 ('brush',r'brush|brush set',r'brush',None),
]
ASPECT_ALIASES={
 'absorption_speed':['absorption speed','absorbency','absorption'],
 'drying_speed':['drying speed','drying time'],
 'greasiness':['greasiness','greasy residue','oiliness','texture'],
 'stickiness':['stickiness','texture','residue'],
 'scent_intensity':['scent intensity','fragrance intensity','scent','fragrance','smell'],
 'scent_duration':['scent longevity','scent duration','fragrance longevity','scent','fragrance'],
 'scent_character':['scent','fragrance','smell'],
 'finish_type':['finish','finish type','matte finish','appearance'],
 'softness':['softness','bristle softness','brush texture','texture'],
 'weight':['weight','lightweight'],
 'wear_duration':['longevity','wear duration','lasting power','durability'],
 'removal_ease':['ease of removal','removal ease','rinsing ease','removal'],
 'foaming_ability':['lather','foaming','foaming ability'],
}


def bounded(pattern,text):
    return bool(re.search(r'(?<!\w)(?:'+pattern+r')(?!\w)',text,re.I))


def polarity(text,yes,no):
    # Remove explicitly negated/absent expressions before positive matching.
    negative=bool(re.search(no,text,re.I)) if no else False
    stripped=re.sub(no,' ',text,flags=re.I) if no else text
    positive=bool(re.search(yes,stripped,re.I))
    if positive and negative:
        return None
    return 1 if positive else (-1 if negative else None)


def preference_signal(preference,text):
    """A lexical hint from a cited quote, never a guaranteed product fact."""
    a=preference['normalized_aspect'];v=key(preference['desired_value'])
    target=None;observed=None
    if a in ('absorption_speed','drying_speed'):
        target=1 if v in ('fast','quick','quickly','rapid','rapidly') else (-1 if v in ('slow','slowly') else None)
        # Restrict speed evidence to the relevant process, not shipping speed.
        if not re.search(r'absorb|absorption' if a=='absorption_speed' else r'dry|dries|drying',text,re.I):
            return None
        observed=polarity(text,r'\b(?:fast|quick\w*|rapid\w*)\b',r'\b(?:slow\w*|not (?:very )?(?:fast|quick)|does not (?:absorb|dry) (?:fast|quickly))\b')
    elif a in ('greasiness','stickiness'):
        term='greasy' if a=='greasiness' else 'sticky'
        if term not in v and v not in ('present','absent'):
            return None
        target=-1 if v.startswith(('non ','not ')) or v in ('absent',) else 1
        observed=polarity(text,rf'\b{term}\b',rf'\b(?:non[- ]{term}|not(?: at all)? {term}|isn.t {term}|without (?:any )?{term})\b')
    elif a=='scent_intensity':
        if not re.search(r'scent|fragrance|smell',text,re.I):
            return None
        target=1 if v in ('strong','intense','powerful','overpowering') else (-1 if v in ('light','faint','mild','subtle','weak') else None)
        observed=polarity(text,r'\b(?:strong|intense|powerful|overpowering)\b',r'\b(?:light|faint|mild|subtle|weak|not (?:too |very )?strong|not overpowering)\b')
    elif a=='weight':
        target=-1 if v in ('light','lightweight','light weight') else (1 if v in ('heavy',) else None)
        observed=polarity(text,r'\bheavy\b',r'\b(?:lightweight|light weight|not heavy)\b')
    elif a=='softness':
        target=1 if v=='soft' else (-1 if v in ('hard','stiff') else None)
        observed=polarity(text,r'\bsoft\b',r'\b(?:hard|stiff|not soft)\b')
    elif a=='scent_character' and v=='clean':
        return (1 if preference['direction']=='prefer' else -1) if re.search(r'\bclean\s+(?:scent|fragrance|smell)\b',text,re.I) else None
    elif a=='finish_type' and v in ('matte','satin','sheer','glossy','shimmery'):
        if not re.search(r'finish|matte|satin|sheer|glossy|shimmer',text,re.I):
            return None
        if re.search(r'\bnot\s+'+re.escape(v)+r'\b',text,re.I):
            return -1 if preference['direction']=='prefer' else 1
        return (1 if preference['direction']=='prefer' else -1) if bounded(re.escape(v),text) else None
    if target is None or observed is None:
        return None
    signal=1 if target==observed else -1
    return -signal if preference['direction']=='avoid' else signal


def hard_fact(constraint,metadata):
    """Only narrow, explicit catalog claims/fields; unknown is not a pass."""
    operator=constraint['operator']; value=constraint['value']
    details=metadata.get('details') or {}
    numeric_keys={
        'speed settings':['Number of Speeds','Speed Settings'],
        'brushes':['Number of Brushes'],'replacement heads':['Number of Replacement Heads'],
    }
    if operator in ('equals','at_least','at_most','less_than','greater_than') and isinstance(value,(float,int)) and not isinstance(value,bool):
        for field in numeric_keys.get(key(constraint.get('unit')),[]):
            raw=str(details.get(field,''))
            if re.fullmatch(r'\s*\d+(?:\.\d+)?\s*',raw):
                n=float(raw)
                ok={'equals':n==value,'at_least':n>=value,'at_most':n<=value,'less_than':n<value,'greater_than':n>value}[operator]
                return {'status':'meets' if ok else 'violates','source':'catalog.details.'+field,'evidence':raw}
    if operator in ('must_include','must_exclude') and isinstance(value,str):
        term=key(value)
        aliases={'sulfates':r'sulfates?','alcohol':r'alcohol','fragrance':r'fragrance','parabens':r'parabens?'}
        pattern=aliases.get(term,re.escape(term))
        # Exclusion needs a positive "free/without" assertion; a missing
        # ingredient in a partial ingredient list cannot establish absence.
        statements=[('catalog.title',metadata.get('title',''))]
        statements += [('catalog.features',str(x)) for x in (metadata.get('features') or [])]
        statements += [('catalog.description',str(x)) for x in (metadata.get('description') or [])]
        absent=[];present=[]
        for source,text in statements:
            m=re.search(r'\b(?:'+pattern+r')[- ]free\b|\b(?:free (?:of|from)|without|no)\s+(?:'+pattern+r')\b',text,re.I)
            if m:
                absent.append({'source':source,'evidence':text})
        for field in ('Ingredients','Active Ingredients'):
            raw=str(details.get(field,''))
            if bounded(pattern,raw):
                present.append({'source':'catalog.details.'+field,'evidence':raw})
        if absent and present:
            return {'status':'unknown','reason':'Conflicting catalog ingredient claims.'}
        if absent or present:
            evidence=(absent or present)[0]
            exists=bool(present)
            ok=exists if operator=='must_include' else not exists
            return {'status':'meets' if ok else 'violates',**evidence}
    return {'status':'unknown','reason':'No supported explicit catalog fact for this condition.'}


class PrototypeEngine:
    def __init__(self):
        self.baseline={'candidate':'Run14-derived portable source policy','data_mode':'synthetic_demo' if runtime_config.IS_DEMO else 'external_research_data'}
        frozen_runner.prepare()
        self.users={r['user_id']:r for r in jsonl(DATA/'user_profiles_in_scope.jsonl')}
        self.profiles={r['parent_asin']:r for r in jsonl(DATA/'item_profiles.jsonl')}
        self.metadata={r['parent_asin']:r for r in jsonl(DATA/'catalog_metadata.jsonl')}
        with (DATA/'product_category_mapping_v2_1_audit_merged_341.csv').open(encoding='utf-8-sig',newline='') as file:
            self.mapping={r['parent_asin']:r for r in csv.DictReader(file) if r['final_scope_status']=='in_scope'}
        self.seen=defaultdict(set)
        for r in jsonl(DATA/'profile_data.jsonl'):
            self.seen[r['user_id']].add(r['parent_asin'])
        self.cached={r['original_query']:r for r in jsonl(DATA/'intent_results.jsonl')}
        self.review_bm25,self.review_bm25_manifest=load_training_index(ROOT)

    def bootstrap(self):
        examples=[r for r in self.cached.values() if r['intent_id'] in
            ('INTENT-BEAUTY-002','INTENT-BEAUTY-003','INTENT-BEAUTY-004','INTENT-BEAUTY-005',
             'INTENT-BEAUTY-007','NEW-BEAUTY-009','NEW-BEAUTY-016')]
        return {'baseline':C.POLICY_VERSION,
                'data_mode':'synthetic_demo' if runtime_config.IS_DEMO else 'external_research_data',
                'users':[{'user_id':u,'review_count':r['review_count']} for u,r in self.users.items()],
                'categories':sorted({m['final_product_category'] for m in self.mapping.values()}),
                'examples':[{'query':r['original_query'],'id':r['intent_id']} for r in examples],
                'in_scope_item_count':len(self.mapping)}

    def extract(self,query,allow_live=False):
        query=query.strip()
        if not query or len(query)>2000:
            raise ValueError('Enter an English purchasing request of 1 to 2000 characters.')
        if re.search(r'[\u4e00-\u9fff]',query):
            raise ValueError('This policy supports English requests. Use English or select an example.')
        if query in self.cached:
            row=self.cached[query]
            return {'intent':row['intent'],'status':row['status'],'provenance':'bundled_demo_intent_fixture' if runtime_config.IS_DEMO else 'cached_imported_research_result',
                    'model_calls':0,'query':query,'trace_directory':'intent_results.jsonl'}
        if not allow_live:
            raise ValueError('No cached interpretation exists. Enable live parsing for up to two local model calls.')
        with LOCK:
            item={'intent_id':'PROTO-'+uuid.uuid4().hex[:12],'source_type':'user_search_query','language':'en','query':query}
            _,protected=C.initialize()
            protected[str(Path(frozen_runner.__file__))]=C.sha(Path(frozen_runner.__file__))
            out=ROOT/'outputs/prototype_sessions'/item['intent_id']
            client=frozen_runner.ollama.Client(timeout=10.0)
            start=C.frozen.BASE.model_metadata(client)
            if C.frozen.BASE.EXPECTED_MODEL_DIGEST and start['installed_model']['digest']!=C.frozen.BASE.EXPECTED_MODEL_DIGEST:
                raise RuntimeError('The installed model digest does not match RECOMMENDER_MODEL_DIGEST.')
            frozen_runner.run([item],protected,out,start,client,'prototype_single_query_run14_v1_1')
            row=jsonl(out/'intent_extraction_results.jsonl')[0]
            if row['intent'] is None:
                raise ValueError('The request could not be parsed reliably. Rephrase it. Trace: '+str(out))
            return {'intent':row['intent'],'status':row['status'],'provenance':'live_run14_v1_1',
                    'model_calls':C.load(out/'intent_extraction_summary.json')['attempted_model_calls'],
                    'query':query,'trace_directory':str(out)}

    @lru_cache(maxsize=8)
    def history_scores(self,user):
        scores={}
        with (DATA/'user_item_matching_scores_in_scope.csv').open(encoding='utf-8-sig',newline='') as file:
            for r in csv.DictReader(file):
                if r['user_id']==user and r['match_score']:
                    scores[r['parent_asin']]=r
        return scores

    def recall(self,intent,category=None):
        product=intent['task']['product_type']
        if product['status']!='explicit' or not product['normalized_value']:
            return [],'Specify the product type before requesting recommendations.'
        noun=key(product['normalized_value'])
        family=next((f for f in FAMILIES if re.fullmatch(f[1],noun)),None)
        found=[]
        for iid,m in self.mapping.items():
            cat=m['final_product_category']
            if category and cat!=category:
                continue
            text=m['final_product_type']+' '+m['title']
            if family:
                if family[3] and cat not in family[3]:
                    continue
                matched=bool(re.search(family[2],text,re.I))
                if family[0]=='nail polish' and re.search(r'removal|remover|curing lamp',m['final_product_type'],re.I):
                    matched=False
                if family[0]=='shampoo' and re.search(r'shampoo brush|scalp massager|shampoo cap|empty.{0,20}bottle|dispenser',text,re.I):
                    matched=False
            else:
                matched=all(bounded(re.escape(t),key(text)) for t in noun.split())
            if matched:
                found.append(iid)
        return found,None if found else 'No matching product type in the catalog. Change the request or category filter.'

    def preference_check(self,preference,iid):
        profile=self.profiles[iid]
        aliases=ASPECT_ALIASES.get(preference['normalized_aspect'],[key(preference['normalized_aspect'])])
        sources=[]
        for aspect in profile['aspects']:
            if key(aspect['profile_aspect']) in aliases:
                sources += [{'source':'review','review_id':e.get('train_row_id'),'evidence':e['evidence']} for e in aspect.get('evidence_examples',[]) if e.get('evidence_valid') is True]
        meta=self.metadata.get(iid,{})
        for field in ('Scent','Finish Type','Product Benefits','Special Feature'):
            if (meta.get('details') or {}).get(field):
                # Retain dimension context (e.g. "clean scent") for matching.
                value=str(meta['details'][field])
                text=(value+' scent') if field=='Scent' else (field+': '+value)
                sources.append({'source':'catalog.details.'+field,'evidence':value,'match_context':text})
        hits=[]
        for source in sources:
            sign=preference_signal(preference,source.get('match_context',source['evidence']))
            if sign is not None:
                hits.append({**source,'signal':sign})
        signs={h['signal'] for h in hits}
        status='conflicting_evidence' if len(signs)>1 else ('supporting_hint' if signs=={1} else ('contrary_hint' if signs=={-1} else 'not_verified'))
        score=1 if signs=={1} else (0 if signs=={-1} else .5)
        return {'requirement':preference['raw_expression'],'aspect':preference['normalized_aspect'],
                'desired_value':preference['desired_value'],'direction':preference['direction'],
                'status':status,'score':score,'evidence':hits[:3]}

    def recommend(self,user_id,intent,category=None,top_k=5):
        if user_id not in self.users:
            raise ValueError('Select an existing user.')
        if not 1<=top_k<=10:
            raise ValueError('top_k must be between 1 and 10.')
        Draft202012Validator(C.frozen.SCHEMA).validate(intent)
        C.frozen.BASE.validate_evidence(intent,intent['source']['original_text'])
        unresolved=intent['unresolved_requirements']
        result={'user_id':user_id,'query':intent['source']['original_text'],'intent':intent,
                'mode':'prototype_current_intent_plus_existing_history_baseline',
                'scoring_policy':'0.4 normalized historical baseline +0.6 current evidence hints; missing/conflicting hints neutral. Prototype heuristic, not evaluated accuracy.',
                'clarifications':[u['raw_expression']+': '+u['reason'] for u in unresolved],
                'unverified_context':[],
                'recommendations':[],'unverified_candidates':[],'excluded_hard_constraint_count':0}
        if any(intent['task'][field]['status']!='not_stated' for field in ('recipient','use_case')):
            result['unverified_context'].append('Product suitability for the stated recipient or use case has not been verified.')
        for relation in intent['priority_relations']:
            def aligned(endpoints):
                return any(any(name in key(endpoint) or key(endpoint) in name
                    for name in ASPECT_ALIASES.get(p['normalized_aspect'],[key(p['normalized_aspect'])]))
                    for endpoint in endpoints for p in intent['preferences'])
            if not (len(intent['preferences'])>=2 and aligned(relation['higher_priority_aspects']) and aligned(relation['lower_priority_aspects'])):
                result['unverified_context'].append('Some priority relations could not be applied to ranking and are retained for confirmation.')
                break
        if any(u['reason'].startswith('unsupported_') for u in unresolved):
            return {**result,'status':'needs_clarification','message':'The schema cannot express this comparison. Clarify it before recommendations.','candidate_count':0}
        pool,message=self.recall(intent,category)
        pool=[iid for iid in pool if iid not in self.seen[user_id]]
        scores=self.history_scores(user_id)
        ranked=[];unknown=[]
        for iid in pool:
            m=self.mapping[iid]; meta=self.metadata.get(iid,{})
            checks=[{'requirement':h['raw_expression'],**hard_fact(h,meta)} for h in intent['hard_constraints']]
            if any(h['status']=='violates' for h in checks):
                result['excluded_hard_constraint_count']+=1
                continue
            pref_checks=[self.preference_check(p,iid) for p in intent['preferences']]
            # Double the weight of preferences aligned with a higher-priority endpoint.
            weights=[]
            for preference in intent['preferences']:
                names=ASPECT_ALIASES.get(preference['normalized_aspect'],[key(preference['normalized_aspect'])])
                weight=1
                for relation in intent['priority_relations']:
                    if any(any(name in key(endpoint) or key(endpoint) in name for name in names) for endpoint in relation['higher_priority_aspects']):
                        weight=2
                weights.append(weight)
            current=sum(x['score']*weight for x,weight in zip(pref_checks,weights))/sum(weights) if weights else None
            row=scores.get(iid)
            baseline=float(row['match_score']) if row else ((self.profiles[iid].get('mean_overall_rating') or 3)-3)/2
            norm=max(0,min(1,(baseline+1)/2))
            final=.4*norm+.6*current if current is not None else norm
            matched=json.loads(row['matched_aspects']) if row else []
            history=[]
            for a in sorted(matched,key=lambda a:-a['normalized_aspect_weight'])[:2]:
                aspect=next((x for x in self.profiles[iid]['aspects'] if x['profile_aspect']==a['profile_aspect']),{})
                history.append({'aspect':a['profile_aspect'],'history_attention_review_count':a['user_support_review_count'],
                    'item_evidence':[e for e in aspect.get('evidence_examples',[]) if e.get('evidence_valid') is True][:2]})
            candidate={'parent_asin':iid,'title':m['title'],'product_category':m['final_product_category'],
                'product_type':m['final_product_type'],'score':round(final,6),'history_baseline_score':round(baseline,6),
                'history_fallback':row is None,'current_evidence_hint_score':current,
                'preference_checks':pref_checks,'hard_constraint_checks':checks,
                'history_reasons':history,'hard_constraints_verified':bool(checks) and all(h['status']=='meets' for h in checks)}
            if any(h['status']=='unknown' for h in checks):
                unknown.append(candidate)
            else:
                ranked.append(candidate)
        ranked.sort(key=lambda r:(-r['score'],r['parent_asin']))
        unknown.sort(key=lambda r:(-r['score'],r['parent_asin']))
        for i,r in enumerate(ranked[:top_k],1):
            r['rank']=i
        output={**result,'candidate_count':len(pool),'recommendations':ranked[:top_k],
                'unverified_candidates':unknown[:top_k],
                'status':'ok' if ranked else ('hard_constraints_unverified' if unknown else 'no_candidates'),
                'message':message or ('Hard conditions could not be verified. These candidates are for inspection and are not verified recommendations.' if unknown and not ranked else '')}
        # Attach retrieval hits after ranking so scores stay unchanged.
        for candidate in output['recommendations']+output['unverified_candidates']:
            candidate['bm25_review_hits']=self.review_bm25.search(
                intent['source']['original_text'],top_k=3,item_ids={candidate['parent_asin']})
        output['bm25_retrieval']={'algorithm':'BM25','query':intent['source']['original_text'],
            'query_terms':sorted(set(tokenize(intent['source']['original_text']))),
            'corpus_documents':self.review_bm25.n,'k1':self.review_bm25.k1,'b':self.review_bm25.b,
            'top_k_per_item':3,'affects_product_ranking':False,'scope':'traceable_in_scope_training_evidence',
            'note':'Related review snippets; lexical relevance does not establish preference or hard-constraint satisfaction.'}
        return output
