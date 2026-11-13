"""BM25 retrieval over traceable training-review snippets.

Item filters retain corpus-wide IDF and length statistics."""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import unicodedata

from runtime_config import ROOT,DATA_DIR,TRAIN_FILE
STOP=set('a an the i me my we our you your it its this that these those am is are was were be been being have has had do does did for of to in on at by with from as and or but would should could can want wants need needs looking look prefer prefers like product products'.split())
NORMAL={'absorbs':'absorb','absorbed':'absorb','absorbing':'absorb','absorption':'absorb',
        'quickly':'quick','rapidly':'rapid','dries':'dry','drying':'dry','dried':'dry',
        'feels':'feel','feeling':'feel','scents':'scent','fragrances':'fragrance'}


def normalize(text):
    text=unicodedata.normalize('NFKC',str(text)).translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"','–':'-','—':'-','−':'-'}))
    return ' '.join(text.casefold().split())


def tokenize(text):
    text=normalize(text)
    text=re.sub(r"\b(?:isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|couldn't|won't|wouldn't)\b",'not',text)
    return [NORMAL.get(t,t) for t in re.findall(r'[a-z0-9]+',text) if t not in STOP]


class BM25Index:
    def __init__(self,documents,k1=1.2,b=.75):
        if not math.isfinite(k1) or k1<=0 or not math.isfinite(b) or not 0<=b<=1:
            raise ValueError('BM25 requires finite k1>0 and0<=b<=1.')
        self.documents=list(documents)
        self.k1,self.b=k1,b
        self.lengths=[];self.postings=defaultdict(dict)
        ids=set()
        for i,doc in enumerate(self.documents):
            if doc['document_id'] in ids:
                raise ValueError('Duplicate document ID.')
            ids.add(doc['document_id'])
            terms=Counter(tokenize(doc['evidence']))
            self.lengths.append(sum(terms.values()))
            for term,tf in terms.items():
                self.postings[term][i]=tf
        self.n=len(self.documents)
        self.avgdl=sum(self.lengths)/self.n if self.n else 0

    def search(self,query,top_k=5,item_ids=None):
        if not isinstance(top_k,int) or isinstance(top_k,bool) or not 1<=top_k<=100:
            raise ValueError('top_k must be an integer between1 and100.')
        allowed=None if item_ids is None else set(item_ids)
        scores=defaultdict(float);matched=defaultdict(set)
        for term in sorted(set(tokenize(query))):
            posting=self.postings.get(term,{})
            if not posting:
                continue
            df=len(posting)
            idf=math.log1p((self.n-df+.5)/(df+.5))
            for i,tf in posting.items():
                if allowed is not None and self.documents[i]['parent_asin'] not in allowed:
                    continue
                norm=1-self.b+self.b*self.lengths[i]/self.avgdl
                scores[i]+=idf*tf*(self.k1+1)/(tf+self.k1*norm)
                matched[i].add(term)
        ranked=sorted(scores,key=lambda i:(-scores[i],self.documents[i]['document_id']))[:top_k]
        return [{**self.documents[i],'bm25_score':round(scores[i],8),'matched_terms':sorted(matched[i]),'rank':rank}
                for rank,i in enumerate(ranked,1) if scores[i]>0]


def load_training_index(root=ROOT):
    data=DATA_DIR if root==ROOT else root/'data/demo'
    source_paths=[data/'profile_data.jsonl',data/'profile_aspect_records.jsonl',
                  data/'product_category_mapping_v2_1_audit_merged_341.csv',
                  TRAIN_FILE if root==ROOT else data/'benchmark/All_Beauty.train.csv']
    with source_paths[2].open(encoding='utf-8-sig',newline='') as file:
        scope={r['parent_asin'] for r in csv.DictReader(file) if r['final_scope_status']=='in_scope'}
    with source_paths[3].open(encoding='utf-8-sig',newline='') as file:
        train_pairs={(r['user_id'],r['parent_asin']) for r in csv.DictReader(file)}
    reviews={}
    with source_paths[0].open(encoding='utf-8-sig') as file:
        for line in file:
            if line.strip():
                r=json.loads(line);rid=r['train_row_id']
                if rid in reviews:
                    raise ValueError('Duplicate training row ID.')
                reviews[rid]=r
    documents={};counts=Counter()
    with source_paths[1].open(encoding='utf-8-sig') as file:
        for line in file:
            if not line.strip():
                continue
            r=json.loads(line);counts['source_pairs']+=1
            if r.get('evidence_valid') is not True or r.get('evidence_review_required') is True:
                counts['excluded_invalid_or_review_required']+=1;continue
            if r['parent_asin'] not in scope:
                counts['excluded_outside_in_scope']+=1;continue
            review=reviews.get(r['train_row_id'])
            if not review or (r['user_id'],r['parent_asin']) not in train_pairs or any(review[k]!=r[k] for k in ('user_id','parent_asin')):
                counts['excluded_source_identity_mismatch']+=1;continue
            evidence=r['evidence']
            if not normalize(evidence) or normalize(evidence) not in normalize(review['review_text']):
                counts['excluded_untraceable']+=1;continue
            if not tokenize(evidence):
                counts['excluded_no_tokens']+=1;continue
            counts['eligible_pairs']+=1
            identity=(r['parent_asin'],r['train_row_id'],normalize(evidence))
            if identity not in documents:
                identifier='SNIP-'+hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
                documents[identity]={'document_id':identifier,'parent_asin':r['parent_asin'],
                    'review_id':r['train_row_id'],'evidence':evidence,'source':'training_review_evidence',
                    'source_file':'profile_data.jsonl','profile_aspects':[],
                    'sentiment_labels':[],'source_match':'normalized_substring'}
            doc=documents[identity]
            for field,value in [('profile_aspects',r['profile_aspect']),('sentiment_labels',r['sentiment'])]:
                if value not in doc[field]:
                    doc[field].append(value)
    index=BM25Index(sorted(documents.values(),key=lambda d:d['document_id']))
    manifest={'algorithm':'BM25','k1':index.k1,'b':index.b,'document_count':index.n,
        'item_count':len({d['parent_asin'] for d in index.documents}),'average_document_length':index.avgdl,
        'counts':dict(counts),'deduplicated_pairs':counts['eligible_pairs']-index.n,
        'source_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'tokenization':'NFKC casefold, punctuation split, declared stopwords and small morphology map; negation retained; unique query terms.',
        'index_text':'Only verbatim evidence snippets; product titles and aspect labels are metadata, not indexed text.',
        'scope':'in_scope training evidence only; no validation/test review texts, no sentiment filtering.',
        'score_meaning':'Lexical relevance, not preference satisfaction, semantic support, probability or product ranking.',
        'formula_reference':'https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/search/similarities/BM25Similarity.html'}
    return index,manifest


if __name__=='__main__':
    import argparse
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query',required=True)
    parser.add_argument('--item-id',action='append')
    parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    index,manifest=load_training_index()
    result={'query':args.query,'query_terms':sorted(set(tokenize(args.query))),
            'index':manifest,'hits':index.search(args.query,args.top_k,args.item_id)}
    if args.output:
        with args.output.open('x',encoding='utf-8') as file:
            json.dump(result,file,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))
