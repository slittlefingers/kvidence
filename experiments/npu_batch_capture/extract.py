"""Split a captured .gkv.gz into original per-buffer bytes without dequantizing."""
import argparse
import gzip
import json
from pathlib import Path
import struct


def extract(capture,output,layers=None):
    with gzip.open(capture,'rb') as f:
        header=f.read(16)
        if len(header)!=16 or header[:8]!=b'GHOSTKV1':raise ValueError('Invalid GHOSTKV1 file')
        length=struct.unpack('<Q',header[8:])[0]
        if not 0<length<16*1024*1024:raise ValueError('Invalid layout length')
        layout=json.loads(f.read(length))
        output.mkdir(parents=True,exist_ok=False)
        (output/'manifest.json').write_text(json.dumps(layout,indent=2)+'\n')
        offset=0;files=[]
        for i,b in enumerate(layout['buffers']):
            if b['payload_offset']!=offset:raise ValueError('Unexpected payload layout')
            wanted=layers is None or b['layer'] in layers
            filename=f"buffer-{i:03d}-layer-{int(b['layer'])}.bin"
            out=(output/filename).open('wb') if wanted else None
            remaining=b['allocated_bytes']
            if remaining<0:raise ValueError('Negative allocation size')
            try:
                while remaining:
                    block=f.read(min(1024*1024,remaining))
                    if not block:raise ValueError('Truncated KV payload')
                    if out:out.write(block)
                    remaining-=len(block)
            finally:
                if out:out.close()
            if wanted:files.append(dict(file=filename,**b))
            offset+=b['allocated_bytes']
        if f.read(1):raise ValueError('Unexpected bytes after KV payload')
        (output/'files.json').write_text(json.dumps(files,indent=2)+'\n')
    return files


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--layers',type=int,nargs='+',help='Raw runtime layer IDs, as listed in layout.json')
    args=p.parse_args()
    files=extract(args.capture,args.output,args.layers)
    print(f'Extracted {len(files)} buffers; see files.json for shapes/strides/dtypes.')
