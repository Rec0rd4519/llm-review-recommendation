"""Checks for the public fictional corpus; no model server or private runs needed."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prototype_engine import PrototypeEngine,C
from jsonschema import Draft202012Validator
from offline_ranking_strategies import ExperimentFeatures,combine_scores,rank_items,global_bm25_scores
import intent_runner,intent_runtime

class PortableDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.engine=PrototypeEngine()

    def test_cached_demo_no_model_needed(self):
        b=self.engine.bootstrap()
        self.assertEqual((len(b['users']),b['in_scope_item_count'],len(b['examples'])),(3,12,3))
        for row in self.engine.cached.values():
            result=self.engine.extract(row['original_query'])
            self.assertEqual(result['model_calls'],0)
            self.assertEqual(result['provenance'],'bundled_demo_intent_fixture')
            Draft202012Validator(C.frozen.SCHEMA).validate(result['intent'])
            C.frozen.BASE.validate_evidence(result['intent'],row['original_query'])

    def test_recommendations_exclude_seen_and_cite_retrieval(self):
        intent=self.engine.cached['I need a moisturizer that absorbs quickly and feels non-greasy.']['intent']
        result=self.engine.recommend('DEMO-USER-001',intent)
        self.assertEqual(len(result['recommendations']),5)
        self.assertFalse({r['parent_asin'] for r in result['recommendations']}&self.engine.seen['DEMO-USER-001'])
        self.assertTrue(all(r['bm25_review_hits'] for r in result['recommendations']))
        self.assertFalse(result['bm25_retrieval']['affects_product_ranking'])

    def test_hard_constraint_unknown_and_violated_are_separate(self):
        intent=self.engine.cached['I need a shampoo that must be sulfate-free.']['intent']
        result=self.engine.recommend('DEMO-USER-001',intent)
        self.assertEqual([r['parent_asin'] for r in result['recommendations']],['DEMO-ITEM-010'])
        self.assertEqual([r['parent_asin'] for r in result['unverified_candidates']],['DEMO-ITEM-012'])
        self.assertEqual(result['excluded_hard_constraint_count'],1)

    def test_non_strict_comparison_requests_clarification(self):
        row=next(r for r in self.engine.cached.values() if r['status']=='needs_clarification')
        result=self.engine.recommend('DEMO-USER-001',row['intent'])
        self.assertEqual(result['status'],'needs_clarification')
        self.assertEqual(result['recommendations'],[])

    def test_bm25_trace_and_monotonicity(self):
        self.assertEqual(self.engine.review_bm25.n,24)
        from prototype_engine import DATA,jsonl
        reviews={r['train_row_id']:r for r in jsonl(DATA/'profile_data.jsonl')}
        for doc in self.engine.review_bm25.documents:
            self.assertIn(doc['evidence'],reviews[doc['review_id']]['review_text'])
        hits=self.engine.review_bm25.search('quickly absorbs non-greasy',5)
        self.assertTrue(hits)
        self.assertEqual([h['bm25_score'] for h in hits],sorted([h['bm25_score'] for h in hits],reverse=True))

    def test_ranking_components_and_deterministic_ties(self):
        self.assertAlmostEqual(combine_scores(.8,.2,.9,True)['query_conditioned'],.86)
        self.assertEqual(combine_scores(.8,.2,.9,False)['fixed_weight'],.8)
        self.assertEqual(rank_items({'B':{'history_only':.5},'A':{'history_only':.5}},'history_only'),['A','B'])
        f=ExperimentFeatures(self.engine);raw=global_bm25_scores(f.index,'absorbs quickly')
        hits=f.index.search('absorbs quickly',5)
        lookup={d['document_id']:i for i,d in enumerate(f.index.documents)}
        for hit in hits:self.assertAlmostEqual(hit['bm25_score'],raw[lookup[hit['document_id']]],places=7)

    def test_two_stage_runner_contract_and_logs_without_network(self):
        row=self.engine.cached['I need a moisturizer that absorbs quickly and feels non-greasy.']
        item={'intent_id':row['intent_id'],'query':row['original_query'],'language':'en','source_type':'user_search_query'}
        def fake_chat(**request):
            value=copy.deepcopy(row['intent'])
            return SimpleNamespace(model_dump=lambda **kwargs:{'done':True,'done_reason':'stop','message':{'content':json.dumps(value)}})
        metadata={'installed_model':{'digest':C.frozen.BASE.EXPECTED_MODEL_DIGEST or 'mock-only'}}
        with tempfile.TemporaryDirectory(prefix='review-compass-test-') as temp:
            output=Path(temp)/'run'
            _,protected=intent_runner.prepare()
            with patch.object(intent_runtime.ollama,'chat',side_effect=fake_chat) as calls,patch.object(C.frozen.BASE,'model_metadata',return_value=metadata):
                intent_runner.run([item],protected,output,metadata,None,'mocked_portability_test')
            self.assertEqual(calls.call_count,2)
            result=json.loads((output/'intent_extraction_results.jsonl').read_text().splitlines()[0])
            self.assertEqual(result['status'],'success',result['validation_errors'])
            summary=json.loads((output/'intent_extraction_summary.json').read_text())
            self.assertEqual(summary['attempted_model_calls'],2)
            self.assertTrue(summary['protected_files_unchanged'])

if __name__=='__main__':unittest.main()
