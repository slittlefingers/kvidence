"""Compare the first image to the separately captured real chat application KV."""
import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    root = parser.parse_args().data.resolve()
    index = json.loads((root/'index.json').read_text())
    row = index[0]
    image = row['images'][0]
    workspace = Path(__file__).resolve().parents[4]
    source = workspace/'kvidence/datasets/shadow_chat/image_chat_100'
    capture = source/'capture'/image['image_id']/'all_layers.gkv.gz'
    with gzip.open(capture, 'rb') as f:
        header = f.read(16)
        layout = json.loads(f.read(struct.unpack('<Q',header[8:])[0]))
        payload = f.read()
    parts = []
    for part in ('k','v'):
        buf = next(x for x in layout['buffers'] if x['name']==f'kv_cache_{part}_4')
        data = np.ndarray(buf['shape'],dtype=np.int8,buffer=payload,
            offset=buf['payload_offset'],strides=tuple(buf['strides']))
        parts.append(np.moveaxis(data,buf['sequence_axis'],0).reshape(buf['capacity'],-1))
    expected = np.concatenate(parts,axis=1)
    start = image['start']; end = start+image['rows']
    with np.load(root/row['file'],allow_pickle=False) as d:
        actual = d['kv_codes'][start:end]
    delta = actual.astype(np.float64)-expected[start:end]
    different = np.flatnonzero(np.any(delta != 0,axis=1))
    report = dict(image_id=image['image_id'],positions=len(actual),layer_index=4,
        reference=str(capture),exact=bool(np.array_equal(actual,expected[start:end])),
        rmse_lsb=float(np.sqrt(np.mean(delta**2))),max_abs_lsb=float(np.max(np.abs(delta))),
        matching_dimensions=float(np.mean(delta==0)),
        first_different_absolute_position=int(start+different[0]) if len(different) else None,
        scope='first image with same public prefix only; not proof for every long multi-image context')
    (root/'chat_capture_control.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
