"""Run draft extraction and constrained regeneration, saving both requests and responses."""
from importlib.metadata import version
from pathlib import Path
import json
import sys
import ollama
import source_policy as policy
def prepare():return policy.initialize()
def run(items, protected, out, start, client, mode):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Preserve existing output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    manifest = {'status':'started','started_at_utc':policy.now(),'mode':mode,
        'candidate':policy.POLICY_VERSION,'model':policy.MODEL,'options':policy.OPTIONS,'think':False,
        'planned_queries':len(items),'maximum_model_calls':len(items)*2,
        'files_sha256':protected,'python_version':sys.version,'ollama_python_version':version('ollama'),
        'protocol':'Unchanged Run09 full-schema draft, declared source/draft contract, one actual constrained regeneration. No reference answers or historical predictions are model inputs.',
        'limitations':'Reused development data and finite English rules. Technical checks do not establish independent semantic accuracy.',
        'counting_policy':'Retain every outcome. Needs-clarification deliveries are valid JSON but are not complete extractions.'}
    policy.write(out/'run_manifest.json',manifest)
    policy.write(out/'model_metadata_start.json',start)
    policy.write(out/'input_queries_snapshot.json',items)
    policy.write(out/'final_schema_snapshot.json',policy.frozen.SCHEMA)
    (out/'system_prompt.txt').write_text(policy.frozen.BASE.SYSTEM_PROMPT,encoding='utf-8')
    (out/'correction_prompt.txt').write_text(policy.CORRECTION_PROMPT,encoding='utf-8')
    counters = {'calls':0,'responses':0,'last_raw':''}
    results, interrupted = [], False
    for i,item in enumerate(items,1):
        print(f'[{i}/{len(items)}] {item["intent_id"]}',flush=True)
        draft = final = trace = None
        errors, schema_valid, evidence_valid, acceptance_valid = [], False, False, False
        counters['last_raw'] = ''
        try:
            request = policy.frozen.BASE.make_request(item['query'],item['intent_id'],item['source_type'],item['language'])
            raw = policy.engineering.generate(item,'draft',request,out,counters)
            draft = policy.frozen.BASE.parse_intent(json.dumps(raw,ensure_ascii=False),item['query'],item['intent_id'],item['source_type'],item['language'])
            contract,variants,trace = policy.build_contract(draft,item)
            policy.append(out/'correction_contracts.jsonl',{'intent_id':item['intent_id'],
                'schema':contract,'schema_sha256':policy.jhash(contract),'trace':trace})
            request = policy.engineering.request(policy.CORRECTION_PROMPT,
                {'original_query':item['query'],'fresh_draft':draft,'declared_source_policy':trace},contract)
            final = policy.engineering.generate(item,'regenerate',request,out,counters)
            policy.verify_final(final,contract,variants,item)
            schema_valid = evidence_valid = acceptance_valid = True
            status = 'needs_clarification' if trace['requires_clarification'] else 'success'
        except KeyboardInterrupt:
            interrupted, status = True, 'interrupted'
            errors.append('Interrupted; no retries or fallback answer.')
        except Exception as exc:
            status = 'failed'
            errors.append(f'{type(exc).__name__}: {exc}')
        delivered = status in ('success','needs_clarification')
        record = {'intent_id':item['intent_id'],'original_query':item['query'],'model':policy.MODEL,
            'status':status,'intent':final if delivered else None,'fresh_draft':draft,
            'source_policy_trace':trace,'validation':{'schema_valid':schema_valid,'evidence_valid':evidence_valid},
            'acceptance_valid':acceptance_valid,'extraction_complete':status=='success',
            'requires_clarification':status=='needs_clarification','validation_errors':errors,
            'raw_model_response':counters['last_raw'],'generation_protocol':'fresh_draft_declared_source_contract_regeneration'}
        policy.append(out/'intent_extraction_results.jsonl',record)
        results.append(record)
        print(f'  status={status}; errors={errors}',flush=True)
        if interrupted:
            break
    try:
        end = policy.frozen.BASE.model_metadata(client)
    except Exception as exc:
        end = {'error':str(exc)}
    policy.write(out/'model_metadata_end.json',end)
    unchanged = all(Path(p).is_file() and policy.sha(Path(p))==h for p,h in protected.items())
    start_digest = start.get('installed_model',{}).get('digest')
    end_digest = end.get('installed_model',{}).get('digest')
    digest_ok = bool(start_digest) and end_digest == start_digest and (not policy.frozen.BASE.EXPECTED_MODEL_DIGEST or end_digest == policy.frozen.BASE.EXPECTED_MODEL_DIGEST)
    summary = {'candidate':policy.POLICY_VERSION,'mode':mode,'model':policy.MODEL,
        'total_queries':len(results),'planned_queries':len(items),
        'successful_extractions':sum(r['status']=='success' for r in results),
        'needs_clarification_count':sum(r['status']=='needs_clarification' for r in results),
        'delivered_count':sum(r['intent'] is not None for r in results),
        'failed_extractions':sum(r['status'] in ('failed','interrupted') for r in results),
        'schema_valid_count':sum(r['validation']['schema_valid'] for r in results),
        'evidence_valid_count':sum(r['validation']['evidence_valid'] for r in results),
        'acceptance_valid_count':sum(r['acceptance_valid'] for r in results),
        'attempted_model_calls':counters['calls'],'recorded_model_responses':counters['responses'],
        'maximum_model_calls':len(items)*2,'interrupted':interrupted,'timestamp':policy.now(),
        'protected_files_unchanged':unchanged,'model_digest_unchanged':digest_ok,
        'evaluation_note':'Reused development/regression inputs. Complete references exclude the predeclared weak-priority boundary; keep boundary outcomes separately. Technical validity is not independent semantic accuracy.'}
    policy.write(out/'intent_extraction_summary.json',summary)
    manifest.update(status='interrupted' if interrupted else 'completed',completed_at_utc=policy.now(),
                    protected_files_unchanged=unchanged,model_digest_unchanged=digest_ok)
    policy.write(out/'run_completion.json',manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not unchanged or not digest_ok:
        raise RuntimeError('Protected file or model digest changed.')
    return 130 if interrupted else (1 if summary['failed_extractions'] else 0)
