"""Full original embedding-row index: no token eligibility filters whatsoever."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent/'kv_guided_search_gemma4_int8/model'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    from transformers import AutoTokenizer
    output=ROOT/'candidate_pool.json'
    if output.exists():
        raise FileExistsError('Preserve the existing pool before rebuilding')
    config=json.loads((MODEL/'config.json').read_text())['text_config']
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    n=config['vocab_size']
    assert n==len(tokenizer)==262144
    ids=list(range(n))
    special=sorted(tokenizer.all_special_ids)
    assert set(special)<=set(ids)
    # The IDs are original embedding-row numbers, with no renumbering.
    result=dict(token_ids=ids,count=n,full_vocabulary=n,
        name='full_original_vocabulary',rule='Every original embedding row, token IDs 0 through vocab_size-1',
        character_filter=None,language_filter=None,canonical_roundtrip_required=False,
        special_tokens_excluded=False,special_token_count=len(special),
        special_token_ids=special,selected_using_target_tokens=False,
        embedding_matrix_shape=[n,config['hidden_size']],
        model_revision='03dcf209f3f549b4075e7191e77cf69b3d48e1b2',
        tokenizer_sha256=digest(MODEL/'tokenizer.json'),
        config_sha256=digest(MODEL/'config.json'),
        model_weights_or_tokenizer_modified=False,
        scope='Candidate indices only. All real KV evaluations still use the original quantized model.',
        previous_measurements='Prior 53,150-candidate and ASCII-only results are separate restricted-pool experiments; accuracy and timing have not been rerun for this pool.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    check=json.loads(output.read_text())
    assert check['token_ids']==list(range(n))
    validation=dict(passed=True,all_original_rows_present=True,
        original_ids_preserved=True,all_special_ids_included=True,
        no_language_or_alphabet_or_probability_filter=True,
        count=n,candidate_pool_sha256=digest(output),
        builder_sha256=digest(Path(__file__)),
        model_inference_run=False)
    (ROOT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(dict(count=n,shape=result['embedding_matrix_shape'],
        special_ids_included=len(special),validation_passed=True,
        file=str(output)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    build()
