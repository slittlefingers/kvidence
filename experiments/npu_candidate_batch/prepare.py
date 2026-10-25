"""Read-only extraction of compiled signatures and an existing NPU KV snapshot."""
import argparse
import gzip
import hashlib
import json
import mmap
import random
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]


class FlatReader:
    def __init__(self, data):
        self.data = data

    def scalar_at(self, pos, fmt='<I'):
        return struct.unpack_from(fmt, self.data, pos)[0]

    def field(self, table, index):
        vt = table - self.scalar_at(table, '<i')
        entry = 4 + 2 * index
        offset = self.scalar_at(vt + entry, '<H') if entry < self.scalar_at(vt, '<H') else 0
        return table + offset if offset else 0

    def ref(self, pos):
        return pos + self.scalar_at(pos) if pos else 0

    def table(self, table, index):
        return self.ref(self.field(table, index))

    def scalar(self, table, index, fmt='<I'):
        pos = self.field(table, index)
        return self.scalar_at(pos, fmt) if pos else 0

    def string(self, table, index):
        pos = self.table(table, index)
        return self.data[pos + 4:pos + 4 + self.scalar_at(pos)].decode() if pos else ''

    def vector(self, table, index, fmt='<i', tables=False):
        pos = self.table(table, index)
        if not pos:
            return []
        count = self.scalar_at(pos)
        assert count < 1000000
        width = struct.calcsize(fmt)
        return [self.ref(pos + 4 + 4 * i) if tables else self.scalar_at(pos + 4 + width * i, fmt)
                for i in range(count)]


def inspect(path):
    result = {}
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        r = FlatReader(data)
        assert data[:8] == b'LITERTLM'
        root = 32 + r.scalar_at(32)
        for section in r.vector(r.table(root, 1), 0, tables=True):
            kind = r.scalar(section, 3, '<B')
            info = {r.string(x, 0): r.string(r.table(x, 2), 0)
                    for x in r.vector(section, 0, tables=True) if r.scalar(x, 1, '<B') == 9}
            key = info.get('model_type', 'unknown')
            begin = r.scalar(section, 1, '<Q')
            end = r.scalar(section, 2, '<Q')
            if kind == 7:
                result.setdefault(key, {})['external_weights'] = [begin, end - begin]
            if kind != 3:
                continue
            assert data[begin + 4:begin + 8] == b'TFL3'
            model = begin + r.scalar_at(begin)
            subgraphs = r.vector(model, 2, tables=True)
            signatures = {}
            for index, sig in enumerate(r.vector(model, 7, tables=True)):
                tensors = r.vector(subgraphs[r.scalar(sig, 4)], 0, tables=True)
                item = {'index': index}
                for slot, label in [(0, 'inputs'), (1, 'outputs')]:
                    entries = []
                    for tensor_map in r.vector(sig, slot, tables=True):
                        t = tensors[r.scalar(tensor_map, 1)]
                        q = r.table(t, 4)
                        entries.append(dict(name=r.string(tensor_map, 0), shape=r.vector(t, 0),
                            shape_signature=r.vector(t, 7), type=r.scalar(t, 1, '<B'),
                            scale=r.vector(q, 2, '<f') if q else [],
                            zero_point=r.vector(q, 3, '<q') if q else []))
                    item[label] = entries
                signatures[r.string(sig, 2)] = item
            result.setdefault(key, {}).update(offset=begin, length=end-begin, signatures=signatures)
    return result


def prepare(output, candidate_count=4, seed=20261019):
    output.mkdir(parents=True, exist_ok=False)
    model = WORKSPACE / 'annoymous/models/gemma-4-E2B-it_Google_Tensor_G5.litertlm'
    source = ROOT.parents[1] / 'datasets/shadow_chat/encrypted_chat_100/capture/0001/all_layers.gkv.gz'
    manifest = inspect(model)
    with gzip.open(source, 'rb') as f:
        header = f.read(16)
        assert header[:8] == b'GHOSTKV1'
        layout = json.loads(f.read(struct.unpack('<Q', header[8:])[0]))
        with (output / 'kv_payload.bin').open('wb') as out:
            while block := f.read(1024 * 1024):
                out.write(block)
    vocab = json.loads((ROOT.parent / 'kv_guided_search_gemma4_int8/model/tokenizer.json').read_text())['model']['vocab']
    if isinstance(vocab, list):
        vocab = {item[0]: i for i, item in enumerate(vocab)}
    words = ['the', 'a', 'cat', 'dog']
    candidates = [dict(text=w, id=vocab[w]) for w in words]
    if candidate_count == 128:
        used = {c['id'] for c in candidates}
        reverse = {i: word for word, i in vocab.items()}
        extra = random.Random(seed).sample(sorted(set(vocab.values()) - used), 124)
        candidates.extend(dict(text=reverse[i], id=i) for i in extra)
    job = dict(model_sections=manifest, snapshot=layout, candidates=candidates,
        source_capture=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        prefix_positions=[0, 32, 128, 600] if candidate_count == 4 else [32, 600],
        note='Captured NPU KV as shared prefix; no generated-token recovery is performed.')
    if candidate_count == 128:
        job.update(candidate_seed=seed, candidate_sampling='First four named tokens; remaining 124 distinct IDs sampled from original tokenizer model vocabulary, no language filtering.')
    (output / 'probe.json').write_text(json.dumps(job, indent=2)+'\n')
    print('Candidates:', candidates)
    for key, value in manifest.items():
        print(key, value.get('length'), 'external_weights=', value.get('external_weights'),
              'signatures=', list(value.get('signatures', {})))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'inputs')
    p.add_argument('--candidate-count', type=int, choices=(4, 128), default=4)
    p.add_argument('--seed', type=int, default=20261019)
    args = p.parse_args()
    prepare(args.output, args.candidate_count, args.seed)
