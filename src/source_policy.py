"""Build output contracts from the current query and draft intent.

The finite English rules cover comparisons, counts, and ambiguous modifiers.
Non-strict priority relations remain unresolved when the schema cannot express them."""
from __future__ import annotations
import copy
import re
from pathlib import Path
from jsonschema import Draft202012Validator
from intent_runtime import previous

engineering, frozen = previous.engineering, previous.frozen
MODEL, OPTIONS = previous.MODEL, previous.OPTIONS
sha, jhash, now, write, append, load = previous.sha, previous.jhash, previous.now, previous.write, previous.append, previous.load
ROOT = Path(__file__).resolve().parents[1]
POLICY_VERSION = 'Run14 English source policy v1.1'
AMBIGUOUS_MODIFIERS = previous.AMBIGUOUS_MODIFIERS + ('elegant', 'refined')
CORRECTION_PROMPT = previous.CORRECTION_PROMPT + '''
The new trace also declares source-grounded strict comparisons, unsupported
non-strict/equality comparisons, complete counting noun phrases, explicit
absolute priority checks, attribute dimensions and product/attribute boundaries.
Copy an unsupported relation's raw clause and exact reason into unresolved
requirements. It needs clarification: never strengthen <= or >= into >, reverse
a relation, drop it, or claim that it is an unclear sensory adjective.
'''
WEAK_REASON = 'unsupported_non_strict_priority_in_schema1_equality_allowed'
EQUAL_REASON = 'unsupported_equal_priority_in_schema1'
STRICT_PATTERNS = [
    (r'(?P<low>.+?)\s+(?:is|are)\s+(?:secondary to|less important(?: to me)? than)\s+(?P<high>.+)', 'strict'),
    (r'(?P<high>.+?)\s+(?:takes?|take)\s+precedence\s+over\s+(?P<low>.+)', 'strict'),
    (r'I\s+put\s+(?P<high>.+?)\s+ahead of\s+(?P<low>.+)', 'strict'),
    (r'(?P<high>.+?)\s+(?:is|are)\s+more important(?: to me)? than\s+(?P<low>.+)', 'strict'),
    (r'(?P<high>.+?)\s+matters?\s+more(?: to me)? than\s+(?P<low>.+)', 'strict'),
    (r'(?P<low>.+?)\s+matters?\s+less(?: to me)? than\s+(?P<high>.+)', 'strict'),
    (r'(?P<high>.+?)\s+ranks?\s+above\s+(?P<low>.+)', 'strict'),
    (r'(?P<low>.+?)\s+ranks?\s+below\s+(?P<high>.+)', 'strict'),
]
WEAK_PATTERNS = [
    (r'(?P<left>.+?)\s+(?:is|are)\s+(?:no more|not more) important(?: to me)? than\s+(?P<right>.+)', '<='),
    (r'(?P<left>.+?)\s+(?:is|are)\s+(?:no less|not less) important(?: to me)? than\s+(?P<right>.+)', '>='),
    (r'(?P<left>.+?)\s+(?:is|are)\s+at least as important(?: to me)? as\s+(?P<right>.+)', '>='),
    (r'(?P<left>.+?)\s+(?:is|are)\s+at most as important(?: to me)? as\s+(?P<right>.+)', '<='),
    (r'(?P<left>.+?)\s+(?:is|are)\s+(?:equally important|as important)(?: to me)? as\s+(?P<right>.+)', '='),
]
ABSOLUTE = {
    'explicit_high': r'\b(?:top|main|highest|high|primary) priority\b|\b(?:very|extremely) important\b',
    'explicit_medium': r'\b(?:medium|moderate|moderately) (?:priority|important|importance)\b',
    'explicit_low': r'\b(?:low|lowest|minor) priority\b|\b(?:slightly important|of little importance)\b',
}
COUNT = re.compile(r'\b(?:exactly|at least|at most|fewer than|more than|no more than|no fewer than|under|over)\s+(?P<number>\d+(?:\.\d+)?)\s+(?P<tail>[A-Za-z][A-Za-z -]*)', re.I)
COUNT_STOP = re.compile(r'\s+(?:is|are|was|were|would|should|must|can|could|will|and|but|or|to|for|in|of|with|without|if|when|that|which|because|so)\b', re.I)
SEPARABLE = re.compile(r'^(?:soft|hard|foaming|non-foaming|scented|unscented|fragrant|sticky|non-sticky|greasy|non-greasy|lightweight|heavy|gentle|elegant|refined|premium)$', re.I)


def clauses(query):
    # Preserve literal text/case. Independent stance clauses are split, but
    # coordinated endpoints are outside the single-endpoint grammar.
    for m in re.finditer(r'[^,;.!?]+', query):
        for piece in re.split(r'\s+(?:and|but)\s+(?=(?:I|we)\s)', m.group()):
            text = re.sub(r'^(?:and|but)\s+', '', piece.strip(), flags=re.I)
            if text:
                yield text


def comparisons(query):
    strict, weak = [], []
    for clause in clauses(query):
        matched = False
        for pattern, operator in WEAK_PATTERNS:
            m = re.fullmatch(pattern, clause, re.I)
            if m:
                weak.append({'expression': clause, 'left':m['left'].strip(),
                             'right':m['right'].strip(), 'operator':operator,
                             'reason':EQUAL_REASON if operator == '=' else WEAK_REASON})
                matched = True
                break
        if matched:
            continue
        if re.search(r"\b(?:not|never|no|equal|equally)\b|don't|doesn't|at least|at most", clause, re.I):
            continue
        for pattern, _ in STRICT_PATTERNS:
            m = re.fullmatch(pattern, clause, re.I)
            if m:
                high, low = m['high'].strip(), m['low'].strip()
                # Lists and nested comparisons need a richer parser.
                if re.search(r'\b(?:and|or|than|ranks|matters)\b', high+' '+low, re.I):
                    break
                strict.append({'expression':clause,'higher':high,'lower':low})
                break
    return strict, weak


def modifiers(query, product):
    result = []
    if not product or product not in query:
        return result
    positions = [m.start() for m in re.finditer(re.escape(product), query, re.I)]
    for term in AMBIGUOUS_MODIFIERS:
        for m in re.finditer(r'(?<!\w)'+re.escape(term)+r'(?!\w)', query, re.I):
            tail = query[m.end():].lstrip()
            if previous.EXPLICIT_DIMENSIONS.match(tail):
                continue
            if term == 'clean' and re.match(r'(?:the|this|that|your|my|it|off|out)\b', tail, re.I):
                continue
            if any(m.end() <= pos <= m.end()+65 and not re.search(r'[,;.!?]|\b(?:with|using|after|before|while|to|for|on|off)\b', query[m.end():pos], re.I) for pos in positions):
                result.append({'raw_expression':m.group(), 'source_start':m.start(), 'source_end':m.end(), 'term':term})
    return sorted(result, key=lambda x:x['source_start'])


def counting_entities(query):
    found = []
    for m in COUNT.finditer(query):
        phrase = COUNT_STOP.split(m['tail'], maxsplit=1)[0].strip(' -')
        if not phrase or len(phrase.split()) > 6:
            continue
        found.append({'expression':query[m.start():m.start('tail')+len(phrase)],
                      'value':float(m['number']), 'unit':phrase})
    return found


def build_contract(draft, item):
    """Build the correction contract and record changes to the draft."""
    q = item['query']
    Draft202012Validator(frozen.SCHEMA).validate(draft)
    if draft['source']['original_text'] != q or draft['intent_id'] != item['intent_id']:
        raise ValueError('Draft identity/source differs from current input.')
    d = copy.deepcopy(draft)
    strict, weak = comparisons(q)
    spans = strict + [{'expression':w['expression']} for w in weak]
    changes = []
    def note(path, rule, before, after):
        if before != after:
            changes.append({'path':path,'rule':rule,'before':copy.deepcopy(before),'after':copy.deepcopy(after)})

    # Split a product modifier only when the normalized noun matches the suffix
    # and every prefix word is in the separable-modifier list.
    product = d['task']['product_type']
    raw = product['raw_text'] or ''
    noun = (product['normalized_value'] or '').replace('_',' ')
    if noun and raw.casefold().endswith(' '+noun.casefold()):
        prefix = raw[:-len(noun)].strip()
        if prefix and all(SEPARABLE.fullmatch(word) for word in prefix.split()):
            after = raw[-len(noun):]
            note('task.product_type.raw_text','separate_independent_modifier',raw,after)
            product['raw_text'] = after
    descriptor_list = modifiers(q, product['raw_text'])
    units = counting_entities(q)
    for category in ('preferences','hard_constraints','unresolved_requirements'):
        kept = []
        for i, original in enumerate(d[category]):
            path = f'{category}[{i}]'
            if previous.inside_comparison(original, spans):
                note(path,'comparison_clause_not_independent_requirement',original,None)
                continue
            if any(previous.matches_descriptor(original, desc) for desc in descriptor_list):
                note(path,'unqualified_modifier_remains_unresolved',original,None)
                continue
            u = copy.deepcopy(original)
            if category != 'unresolved_requirements':
                text = u['raw_expression']+' '+u['aspect_raw']
                aspect = None
                if re.search(r'\b(?:scent|fragrance|smell)\b', text, re.I):
                    if re.search(r'\b(?:lingering|lasting|persistent|persistence|duration)\b', text, re.I):
                        aspect = 'scent_duration'
                    elif re.search(r'\bclean\s+(?:scent|fragrance|smell)\b', text, re.I):
                        aspect = 'scent_character'
                if re.search(r'(?<!\w)(?:non-sticky|sticky|stickiness)(?!\w)', text, re.I):
                    aspect = 'stickiness'
                if re.search(r'(?<!\w)soft(?!\w)', text, re.I) and re.fullmatch(r'(?:\w+_)?texture|softness', u['normalized_aspect']):
                    aspect = 'softness'
                if aspect:
                    note(path+'.normalized_aspect','explicit_source_dimension',u['normalized_aspect'],aspect)
                    u['normalized_aspect'] = aspect
                if category == 'preferences' and u['priority_level'] != 'unspecified':
                    relevant = [s for s in clauses(q) if u['raw_expression'] in s or u['aspect_raw'] in s]
                    pattern = ABSOLUTE.get(u['priority_level'],'(?!)')
                    if not any(re.search(pattern,s,re.I) for s in relevant):
                        note(path+'.priority_level','absolute_priority_requires_own_source',u['priority_level'],'unspecified')
                        u['priority_level'] = 'unspecified'
                if category == 'hard_constraints' and u['operator'] in ('equals','at_least','at_most','less_than','greater_than'):
                    matches = [v for v in units if v['value'] == u['value'] and engineering.overlaps(q,u['raw_expression'],v['expression'])]
                    if len(matches) == 1:
                        note(path+'.unit','complete_counting_noun_phrase',u['unit'],matches[0]['unit'])
                        u['unit'] = matches[0]['unit']
            kept.append(u)
        d[category] = kept
    recognized = len(strict)+len(weak)
    absolute = any(re.search(p,q,re.I) for p in ABSOLUTE.values())
    if d['priority_relations'] and not recognized and not absolute:
        raise ValueError('Unsupported comparison grammar; literal endpoints alone do not prove meaning. Needs clarification.')
    if len(d['priority_relations']) > recognized and (recognized or not absolute or re.search(r'\b(?:than|ranks|outweighs?|above|below)\b', q, re.I)):
        raise ValueError('Mixed comparison association is unsupported; no silent relation loss.')
    relations = [{'relation':'more_important_than','higher_priority_aspects':[c['higher']],
                  'lower_priority_aspects':[c['lower']],'evidence':q} for c in strict]
    note('priority_relations','source_comparison_direction_and_strictness',d['priority_relations'],relations)
    d['priority_relations'] = relations
    for desc in descriptor_list:
        d['unresolved_requirements'].append({'raw_expression':desc['raw_expression'],
            'reason':'ambiguous_attribute','possible_interpretations':[],'evidence':q})
    for w in weak:
        d['unresolved_requirements'].append({'raw_expression':w['expression'],
            'reason':w['reason'],'possible_interpretations':[],'evidence':q})
    # Deduplicate on both source text and reason.
    unique = {}
    for u in d['unresolved_requirements']:
        unique[(u['raw_expression'],u['reason'])] = u
    d['unresolved_requirements'] = list(unique.values())
    # Retain the earlier source-grounded entity absence and clear-attribute
    # deduplication policies, but compile here so the old comparison parser
    # cannot reintroduce a relation rejected by this version's strictness gate.
    for u in d['preferences']:
        if u['normalized_aspect'] == 'ingredient' and str(u['desired_value']).casefold() in ('none','absent'):
            entity = re.sub(r'[^\w]+','_',u['aspect_raw'].casefold()).strip('_')
            if not entity or u['aspect_raw'] not in q:
                raise ValueError('Preferred absence lacks a grounded entity.')
            u['normalized_aspect'], u['desired_value'] = entity+'_presence', 'absent'
    d['unresolved_requirements'] = [u for u in d['unresolved_requirements']
        if u['reason'].startswith('unsupported_') or not any(previous.clearly_identified(p) and p['aspect_raw'] and
            (p['aspect_raw'] in u['raw_expression'] or u['raw_expression'] in p['aspect_raw'])
            for p in d['preferences'])]
    for name, field in d['task'].items():
        if field['status'] == 'not_stated':
            if any(field[k] is not None for k in ('raw_text','normalized_value','evidence')):
                raise ValueError(f'Task consistency cannot be guessed: {name}')
        else:
            if not field['raw_text'] or field['raw_text'] not in q:
                raise ValueError(f'Task source cannot be guessed: {name}')
            field['evidence'] = q
    contract = copy.deepcopy(frozen.SCHEMA)
    for name in ('schema_version','intent_id','source','extraction_policy','task'):
        contract['properties'][name] = {'const':d[name]}
    variants = {}
    for category in engineering.CATEGORIES:
        choices = []
        for index, u in enumerate(d[category]):
            if 'raw_expression' in u and (not u['raw_expression'] or u['raw_expression'] not in q):
                raise ValueError(f'Unanchored expression cannot be guessed: {category}')
            if 'aspect_raw' in u and (not u['aspect_raw'] or u['aspect_raw'] not in q):
                alias = (u['aspect_raw'] or '').replace('_',' ')
                literal = alias if alias and alias in u['raw_expression'] else u['raw_expression']
                note(f'{category}[{index}].aspect_raw','copy_meaningful_original_attribute_phrase',u['aspect_raw'],literal)
                u['aspect_raw'] = literal
            if category == 'unresolved_requirements':
                u['possible_interpretations'] = []
            u['evidence'] = q
            choices.append(engineering.obj({k:{'const':v} for k,v in u.items()}))
        variants[category] = choices
        contract['properties'][category] = {'type':'array','minItems':len(choices),'maxItems':len(choices),
            'items':{'anyOf':choices} if choices else {}}
    Draft202012Validator.check_schema(contract)
    trace = {'semantic_targets':{k:d[k] for k in engineering.CATEGORIES},
             'task_target':d['task'], 'evidence_policy':'complete_current_query',
             'hypothesis_policy':'conservative_non_enumeration'}
    trace.update(policy_version=POLICY_VERSION, source_comparisons_v3=strict,
                 unsupported_schema_relations=weak, complete_counting_entities=units,
                 source_modifiers_v3=descriptor_list, declared_preprocessing_changes=changes,
                 requires_clarification=bool(weak),
                 limitations='Finite English templates and descriptor vocabulary; unchanged draft meanings outside declared source rules can remain wrong.')
    return contract, variants, trace


def verify_final(final, contract, variants, item):
    previous.verify_final(final,contract,variants,item)


def initialize():
    protected={str(p):sha(p) for p in [Path(__file__),ROOT/'src/intent_runtime.py',
        ROOT/'data/schemas/current_intent.schema.json',ROOT/'prompts/current_intent_system.txt',
        ROOT/'prompts/current_intent_correction.txt']}
    return [],protected
