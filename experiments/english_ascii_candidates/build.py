"""Generate an English-ASCII candidate mask without changing tokenizer/model IDs.

This is a CHARACTER-SET filter, not an English language classifier. It retains
all non-special IDs decoding to printable ASCII or TAB/LF/CR, including aliases.
It does not use a dictionary, a probability score, KV values, or target labels.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent/'kv_guided_search_gemma4_int8/model'
OLD_POOL=ROOT.parent/'ciphertext_hybrid_l2_long/results/observed/candidate_pool.json'
ALLOWED_WHITESPACE='\t\n\r'


def allowed(text):
    return bool(text) and all(32 <= ord(c) <= 126 or c in ALLOWED_WHITESPACE for c in text)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def build():
    from transformers import AutoTokenizer
    output=ROOT/'candidate_pool.json'
    if output.exists():
        raise FileExistsError('Preserve the candidate pool before rebuilding')
    source_hash=digest(MODEL/'tokenizer.json')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    special=set(tokenizer.all_special_ids)
    ids=[];counts=collections.Counter();examples={}
    feature_counts=collections.Counter()
    for tid in range(len(tokenizer)):
        text=tokenizer.decode([tid],clean_up_tokenization_spaces=False)
        if tid in special:
            category='excluded_special'
        elif not text:
            category='excluded_empty_decoding'
        elif not text.isascii():
            category='excluded_non_ascii_decoding'
        elif not allowed(text):
            category='excluded_other_ascii_control'
        else:
            category='retained'
            ids.append(tid)
            if re.fullmatch(r'[A-Za-z0-9+/=]+',text):
                feature_counts['base64_only']+=1
            if text.startswith(' '):
                feature_counts['leading_space']+=1
            if any(c in text for c in ' \t\n\r'):
                feature_counts['contains_whitespace']+=1
            if any(32<=ord(c)<=126 and not c.isalnum() and c!=' ' for c in text):
                feature_counts['contains_ascii_punctuation_or_symbol']+=1
        counts[category]+=1
        if len(examples.setdefault(category,[]))<12:
            examples[category].append(dict(token_id=tid,text=text))
    assert len(ids)==len(set(ids)) and sum(counts.values())==len(tokenizer)
    dump(output,dict(token_ids=ids,count=len(ids),full_vocabulary=len(tokenizer),
        name='english_ascii',
        rule='All non-special IDs whose standalone decoded text is nonempty and contains only U+0020..U+007E or TAB/LF/CR',
        character_set='A-Z, a-z, digits, ASCII punctuation/symbols, space, TAB, LF, CR',
        language_classifier=False,canonical_roundtrip_required=False,
        source_tokenizer_sha256=source_hash,
        model_revision='03dcf209f3f549b4075e7191e77cf69b3d48e1b2',
        token_ids_renumbered=False,model_weights_or_tokenizer_modified=False,
        selected_using_target_tokens=False,embedding_dimensions=1536))
    # Candidate pool is fixed before coverage checks. Checks never insert IDs.
    pool=set(ids)
    previous=json.loads(OLD_POOL.read_text())
    assert set(previous['token_ids'])<=pool
    probes=[
        'Hello, world!\nHow should I save this encrypted message?',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=',
        'file_01.txt: key="abc-123"; value=(42) [ok] {done} #tag @home $5 100%',
        'https://example.com/path?q=a+b&n=10\\backup\tcolumn\r\nnext line',
    ]
    coverage=[]
    for text in probes:
        encoded=tokenizer.encode(text,add_special_tokens=False)
        missing=sorted(set(encoded)-pool)
        assert not missing, (text,missing)
        coverage.append(dict(text=text,token_count=len(encoded),missing_token_ids=missing))
    character_count=0
    for text in [chr(i) for i in range(32,127)]+list(ALLOWED_WHITESPACE):
        encoded=tokenizer.encode(text,add_special_tokens=False)
        assert encoded and set(encoded)<=pool, (repr(text),encoded)
        character_count+=1
    rejected_probes=['中文','Привет','ภาษาไทย','こんにちは','مرحبا','é','“hello”','🙂']
    for text in rejected_probes:
        assert not allowed(text)
        assert any(tid not in pool for tid in tokenizer.encode(text,add_special_tokens=False))
    assert digest(MODEL/'tokenizer.json')==source_hash
    # Verify every retained row against the predicate independently of its category.
    assert all(tid not in special and allowed(tokenizer.decode([tid],clean_up_tokenization_spaces=False)) for tid in ids)
    dump(ROOT/'audit.json',dict(counts=dict(counts),overlapping_retained_features=dict(feature_counts),
        examples=examples,old_base64_candidates=previous['count'],
        all_old_base64_candidates_retained=True,
        embedding_matrix_shape=[len(ids),1536],
        existing_canonical_ascii_mask='Earlier mathematical-analysis pool requires an extra encode/decode round trip; this pool deliberately keeps aliases instead',
        limitations=[
            'ASCII spelling is not English language identification; Latin-script words from other languages can remain.',
            'Non-ASCII English typography such as smart quotes and em dash is outside this explicitly ASCII scope.',
            'Single-ID decoding is a lexical mask; coverage of a target with normalization or boundary-spanning tokens must be verified.',
            'This changes only candidate eligibility; it does not prune or retrain the model.',
            'Previous 53,150-candidate experiments remain unchanged; their timing does not apply to the larger pool.']))
    dump(ROOT/'validation.json',dict(passed=True,candidates=len(ids),unique_ids=True,
        all_retained_rows_rechecked=True,all_printable_ascii_and_tab_lf_cr_characters_covered=character_count,
        english_and_code_probes=coverage,non_ascii_rejection_probes=rejected_probes,
        all_previous_base64_candidates_retained=True,tokenizer_unchanged=True,
        candidate_pool_sha256=digest(output),builder_sha256=digest(Path(__file__))))
    print(json.dumps(dict(count=len(ids),counts=dict(counts),features=dict(feature_counts),
        shape=[len(ids),1536],validation_passed=True,path=str(output)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    build()
