"""Immutable context profiles and per-profile acquisition directories."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROFILES = HERE / 'profiles'


def load(name):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name):
        raise ValueError('Profile name: 1–64 letters, numbers, _ or -')
    return json.loads((PROFILES / name / 'job.json').read_text())


def listing():
    result = [dict(name='fixed991', kind='image', description='Original fixed 991-token image collection')]
    for p in sorted(PROFILES.glob('*/job.json')):
        result.append(json.loads(p.read_text()))
    return result


def create(name, messages, suffix='这个图片的 password 是什么？', limit=10000):
    from tokenizers import Tokenizer
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name) or name == 'fixed991':
        raise ValueError('Use a new name containing letters, numbers, _ or -')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must contain chat turns ending with user text before the image')
    if not isinstance(limit, int) or not 1 <= limit <= 10000:
        raise ValueError('limit must be 1–10000')
    text = '<bos>'
    expected = 'user'
    for i, msg in enumerate(messages):
        role, content = msg.get('role'), msg.get('content')
        if role != expected or not isinstance(content, str):
            raise ValueError('Turns must alternate user/assistant and finish with user')
        if any(x in content for x in ('<bos>', '<|turn>', '<turn|>', '<|image>', '<|image|>')):
            raise ValueError('Use plain text; template markers are added automatically')
        text += '<|turn>' + ('model' if role == 'assistant' else role) + '\n' + content
        if i < len(messages) - 1:
            text += '<turn|>\n'
        expected = 'assistant' if role == 'user' else 'user'
    if messages[-1]['role'] != 'user':
        raise ValueError('Last turn must be user text before the image')
    text += '\n\n\n<|image>'
    tokpath = ROOT / 'kvidence/experiments/kv_guided_search_gemma4_int8/model/tokenizer.json'
    tok = Tokenizer.from_file(str(tokpath))
    ids = tok.encode(text, add_special_tokens=False).ids
    if ids[0] != 2 or ids[-1] != 255999:
        raise ValueError('Unexpected native template token IDs')
    if len(ids) > 2000:
        raise ValueError('Initial acquisition profiles support at most 2000 prefix tokens')
    folder = PROFILES / name
    folder.mkdir(parents=True, exist_ok=False)
    prefix = dict(schema=1, prefix_text=text, prefix_ids=ids, prefix_tokens=len(ids),
        prefix_sha256=hashlib.sha256(text.encode()).hexdigest(),
        tokenizer_sha256=hashlib.sha256(tokpath.read_bytes()).hexdigest(),
        messages=messages, suffix_question=suffix, image_begin_id=255999,
        target_used_to_construct_prefix=False)
    (folder / 'prefix.json').write_text(json.dumps(prefix, ensure_ascii=False, indent=2))
    job = dict(name=name, kind='image', limit=limit, prefix_tokens=len(ids),
        prefix=str(folder / 'prefix.json'), out=str(HERE / 'data' / ('profile_' + name)),
        manifest=str(ROOT / 'kvidence/datasets/fixed_prefix_mixed_10000/train_manifest.json'))
    (folder / 'job.json').write_text(json.dumps(job, ensure_ascii=False, indent=2))
    return job
