"""Validate the extracted Gemma 3 embedding table against native execution."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.config import MODEL_CONFIGS,load_model_config
from kvrecon.common import digest,dump
from kvrecon.npu import NpuBackend


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);profile=load_model_config(MODEL_CONFIGS['gemma3-1b'])
    sections=json.loads(profile['manifest'].read_text())['model_sections']
    table=np.load(profile['embeddings'],mmap_mode='r');ids=[0,1,2,1437,9307,21029,236746,262143]
    errors=[]
    with NpuBackend(sections,5,a.output/'device',digest(profile['model']),install=False,device_model=profile['device_model']) as b:
        for tid in ids:
            actual=np.frombuffer(b.rpc('embedding',id=tid),dtype='<f4')
            np.testing.assert_allclose(actual,table[tid],rtol=1e-6,atol=1e-5)
            errors.append(float(np.max(np.abs(actual-table[tid]))))
    dump(a.output/'validation.json',dict(passed=True,token_ids=ids,maximum_absolute_error=max(errors),
        embeddings_sha256=digest(profile['embeddings']),note='Sampled extraction validation against native embedder; not an exhaustive numerical comparison'))
    print('Gemma 3 native embedding check passed:',max(errors),flush=True)


if __name__=='__main__':main()
