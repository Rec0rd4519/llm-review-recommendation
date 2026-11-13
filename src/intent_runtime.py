"""Shared helpers for schema validation, evidence checks, and Ollama requests."""
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import re
import ollama
from jsonschema import Draft202012Validator
from runtime_config import ROOT,MODEL,EXPECTED_MODEL_DIGEST
MODEL_NAME=MODEL
MODEL_OPTIONS={'temperature':0,'num_ctx':4096}
OPTIONS=MODEL_OPTIONS
SCHEMA=json.loads((ROOT/'data/schemas/current_intent.schema.json').read_text(encoding='utf-8-sig'))
Draft202012Validator.check_schema(SCHEMA)
SYSTEM_PROMPT=(ROOT/'prompts/current_intent_system.txt').read_text(encoding='utf-8')
CATEGORIES=('preferences','hard_constraints','priority_relations','unresolved_requirements')
TASKS=('product_type','recipient','use_case')
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def jhash(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def append(p,v):
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n');f.flush()

def parse(raw):
    def unique(pairs):
        obj={}
        for k,v in pairs:
            if k in obj:raise ValueError(f'Duplicate JSON key: {k}')
            obj[k]=v
        return obj
    return json.loads(raw,object_pairs_hook=unique)

def source_spans(query):
    words=list(re.finditer(r"\w+(?:[-’']\w+)*",query,re.UNICODE))
    values={query}
    for i,a in enumerate(words):
        for b in words[i:]:values.add(query[a.start():b.end()])
    return sorted(values,key=lambda s:(len(s),s))

def validate_evidence(intent, original_query):

    errors = []

    def check_span(label, value):

        if value is None:
            return

        if value not in original_query:
            errors.append(
                f"{label}: evidence is not an exact "
                "substring of the original query."
            )

    # Task fields

    for field_name, field in intent["task"].items():

        if field["status"] == "not_stated":

            if any(
                field[key] is not None
                for key in (
                    "raw_text",
                    "normalized_value",
                    "evidence"
                )
            ):
                errors.append(
                    f"task.{field_name}: "
                    "not_stated fields must be null."
                )

        else:

            check_span(
                f"task.{field_name}.raw_text",
                field["raw_text"]
            )

            check_span(
                f"task.{field_name}.evidence",
                field["evidence"]
            )

            if not field["evidence"]:
                errors.append(
                    f"task.{field_name}: "
                    "missing evidence."
                )

    # Preferences

    for index, item in enumerate(intent["preferences"]):

        for field_name in (
            "raw_expression",
            "aspect_raw",
            "evidence"
        ):
            check_span(
                f"preferences[{index}].{field_name}",
                item[field_name]
            )

    # Hard constraints

    for index, item in enumerate(intent["hard_constraints"]):

        for field_name in (
            "raw_expression",
            "aspect_raw",
            "evidence"
        ):
            check_span(
                f"hard_constraints[{index}].{field_name}",
                item[field_name]
            )

    # Priority relations

    for index, item in enumerate(intent["priority_relations"]):

        check_span(
            f"priority_relations[{index}].evidence",
            item["evidence"]
        )

    # Unresolved requirements

    for index, item in enumerate(
        intent["unresolved_requirements"]
    ):

        check_span(
            f"unresolved_requirements[{index}].raw_expression",
            item["raw_expression"]
        )

        check_span(
            f"unresolved_requirements[{index}].evidence",
            item["evidence"]
        )

    if errors:
        raise ValueError(
            "Evidence validation failed:\n"
            + "\n".join(errors)
        )

def parse_intent(
    raw_content,
    query,
    intent_id,
    source_type,
    language
):

    intent = json.loads(raw_content)

    # System-controlled metadata

    intent["schema_version"] = "1.0"
    intent["intent_id"] = intent_id

    intent["source"] = {
        "source_type": source_type,
        "original_text": query,
        "language": language
    }

    intent["extraction_policy"] = {
        "only_explicit_current_input": True,
        "historical_profile_used": False,
        "system_generated_query_used": False
    }

    return intent

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def make_request(query, intent_id, source_type, language):
    payload = {'intent_id': intent_id, 'source_type': source_type, 'language': language, 'original_text': query}
    return {'model': MODEL_NAME, 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                                           {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
            'format': SCHEMA, 'options': dict(MODEL_OPTIONS), 'think': False}

def model_metadata(client):
    installed = client.list().model_dump(mode='json')
    matches = [x for x in installed.get('models', []) if x.get('model', x.get('name')) == MODEL_NAME]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one installed {MODEL_NAME}; found {len(matches)}. No automatic model download.')
    value = {'checked_at_utc': utc_now(), 'installed_model': matches[0],
             'show': client.show(MODEL_NAME).model_dump(mode='json')}
    try:
        value['loaded_model'] = [x for x in client.ps().model_dump(mode='json').get('models', [])
                                 if x.get('model', x.get('name')) == MODEL_NAME]
    except Exception as exc:
        value['loaded_model_lookup_error'] = f'{type(exc).__name__}: {exc}'
    return value

def obj(properties):
    return {'type': 'object', 'additionalProperties': False, 'properties': properties, 'required': list(properties)}

def text_key(value):
    return ' '.join(value.casefold().split())

def overlaps(query, a, b):
    # Locate overlapping source spans.
    return any(max(x, y) < min(x+len(a), y+len(b))
               for x in (m.start() for m in re.finditer(re.escape(a), query))
               for y in (m.start() for m in re.finditer(re.escape(b), query)))

def request(prompt, payload, schema):
    return {'model': MODEL, 'options': dict(OPTIONS), 'think': False, 'format': schema,
            'messages': [{'role': 'system', 'content': prompt},
                         {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}

def generate(item, stage, req, out, counters):
    rid = item['intent_id'] + '-' + stage
    append(out / 'model_requests.jsonl', {'request_id': rid, 'intent_id': item['intent_id'], 'stage': stage,
                                        'request': req, 'request_sha256': jhash(req)})
    counters['calls'] += 1
    print(f'  {stage}: call {counters["calls"]}', flush=True)
    response = ollama.chat(**req).model_dump(mode='json')
    append(out / 'model_responses.jsonl', {'request_id': rid, 'intent_id': item['intent_id'], 'stage': stage, 'response': response})
    counters['responses'] += 1
    counters['last_raw'] = response.get('message', {}).get('content') or ''
    if response.get('done') is not True or response.get('done_reason') != 'stop':
        raise ValueError(f'Incomplete {stage}: {response.get("done_reason")}')
    value = frozen.parse(counters['last_raw'])
    Draft202012Validator(req['format']).validate(value)
    return value

def inside_comparison(unit, comparisons):
    expression = unit.get('raw_expression', '')
    if not expression:
        return False
    for comp in comparisons:
        if not re.search(r'(?<!\w)'+re.escape(expression)+r'(?!\w)', comp['expression']):
            continue
        value = unit.get('desired_value', unit.get('value'))
        evidence = unit.get('evidence', '')
        # Keep independently stated values even when the attribute also appears
        # in a comparison. Require evidence outside the comparison clause.
        if isinstance(value, str) and value and value in evidence and value not in comp['expression']:
            continue
        precise_separate_evidence = (expression in evidence and evidence != comp['expression']
            and not any(c['expression'] in evidence for c in comparisons)
            and not any(evidence in c['expression'] for c in comparisons))
        if precise_separate_evidence and re.search(r"\b(?:prefer|like|want|need|avoid|dislike|must|required)\b|don't", evidence, re.I):
            continue
        return True
    return False

def matches_descriptor(unit, descriptor):
    word = engineering.text_key(descriptor['raw_expression'])
    aspect = engineering.text_key(unit.get('aspect_raw', ''))
    expression = engineering.text_key(unit.get('raw_expression', ''))
    return word == expression or bool(aspect and re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)', aspect))

def clearly_identified(unit):
    # Conservative removal of duplicate unresolved targets only for explicitly
    # named sensory/process attributes and a stated value, not arbitrary labels.
    aspect = unit.get('aspect_raw', '')
    value = unit.get('desired_value')
    patterns = {
        'scent_intensity': r'\b(?:scent|scents|fragrance|smell)\b',
        'finish_type': r'\bfinish\b', 'absorption_speed': r'\b(?:absorb|absorbs|absorption)\b',
        'drying_speed': r'\b(?:dry|dries|drying)\b', 'residue_presence': r'\bresidue\b',
    }
    pattern = patterns.get(unit.get('normalized_aspect'))
    return bool(pattern and value is not None and re.search(pattern, aspect, re.I))

def verify_final(final, contract, variants, item):
    Draft202012Validator(frozen.SCHEMA).validate(final)
    Draft202012Validator(contract).validate(final)
    frozen.BASE.validate_evidence(final, item['query'])
    for category, schemas in variants.items():
        # Enforce each target once; anyOf alone would allow duplicating one
        # allowed unit and omitting a distinct one of the same category.
        unused = set(range(len(schemas)))
        for unit in final[category]:
            match = next((i for i in unused if not list(Draft202012Validator(schemas[i]).iter_errors(unit))), None)
            if match is None:
                raise ValueError(f'Unit conservation failed: {category}')
            unused.remove(match)
        if unused:
            raise ValueError(f'Missing target unit: {category}')

base=SimpleNamespace(SCHEMA=SCHEMA,SYSTEM_PROMPT=SYSTEM_PROMPT,MODEL_NAME=MODEL,
    EXPECTED_MODEL_DIGEST=EXPECTED_MODEL_DIGEST,validate_evidence=validate_evidence,
    parse_intent=parse_intent,make_request=make_request,model_metadata=model_metadata)
frozen=SimpleNamespace(SCHEMA=SCHEMA,BASE=base,parse=parse,source_spans=source_spans)
engineering=SimpleNamespace(CATEGORIES=CATEGORIES,TASKS=TASKS,obj=obj,text_key=text_key,
    overlaps=overlaps,request=request,generate=generate)
AMBIGUOUS_MODIFIERS=('professional-quality', 'high-end', 'luxurious', 'premium', 'gentle', 'balanced', 'clean')
EXPLICIT_DIMENSIONS=re.compile('^(?:price|pricing|cost|texture|pH|ph|scent|fragrance|smell|finish|color|colour|packaging|ingredient|formula|moisture|hydration|quality|performance)\\b',re.I)
BASE_CORRECTION_PROMPT="Return the complete current-intent JSON under the supplied source-derived contract. The original query is the only evidence source. This is an explicitly logged constrained regeneration, not a reference-answer lookup.\nThe contract preserves unaffected facts from this fresh full-schema draft, and implements declared generic English source policies for literal importance comparisons, unqualified ambiguous product modifiers, explicit count entities and source validity. No historical prediction or reference annotation is supplied.\nCopy original raw phrases; never insert normalized labels in raw fields. For an unconstrained aspect_raw choose a relevant phrase from inside that unit's original raw_expression. Preserve the stated values and directions. Emit each contract unit once. Comparison endpoints are verbatim phrases, never standalone preferences. Unclear descriptor hypotheses are conservatively not enumerated; they do not become facts. All non-null evidence is intentionally the complete original current query so request, negation, force, recipient relation and timing remain available. Output JSON only."
previous=SimpleNamespace(engineering=engineering,frozen=frozen,MODEL=MODEL,OPTIONS=OPTIONS,
    sha=sha,jhash=jhash,now=now,write=write,append=append,load=load,
    AMBIGUOUS_MODIFIERS=AMBIGUOUS_MODIFIERS,EXPLICIT_DIMENSIONS=EXPLICIT_DIMENSIONS,
    CORRECTION_PROMPT=BASE_CORRECTION_PROMPT,inside_comparison=inside_comparison,
    matches_descriptor=matches_descriptor,clearly_identified=clearly_identified,verify_final=verify_final)
