"""Reachability control: target produced entirely by the phone's own graph."""
import json
from pathlib import Path
import numpy as np
from device_adapter_test import HERE,fit_slot,unit
from kvrecon.common import dump,EXPERIMENTS,digest
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard

def main():
    import torch
    private_guard()
    meta=json.loads((HERE/'input/metadata.json').read_text())
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    out=HERE/'device_synthetic_control';out.mkdir(exist_ok=False)
    dump(out/'protocol.json',dict(target='synthetic native-generated; no original image or captured KV',
        target_seed=999,search_seed=17,steps=30,
        source_sha256=digest(HERE/'device_adapter_test.py'),
        purpose='Does this finite-difference search find a known reachable native target?'))
    with NpuBackend(sections,4,out/'device',meta['model_sha256']) as b:
        b.reset(meta['public_prefix'])
        truth=unit(np.random.default_rng(999).standard_normal((1,768)).astype(np.float32))
        target=b.values(b.query_vision(truth))[0]
        slot=out/'position_5';slot.mkdir()
        u,r=fit_slot(b,target,np.random.default_rng(17),slot)
        z=b.project_vision(u[None]);kv=b.values(b.commit_embeddings(z))[0]
        r['commit_l2']=float(np.linalg.norm(kv-target));r['passes_commit']=r['commit_l2']<4
        dump(slot/'result.json',r)
        print('Synthetic native control:',json.dumps(r),flush=True)

if __name__=='__main__':main()
