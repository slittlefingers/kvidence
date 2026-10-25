"""Same APK, public prefix, fixed candidates: file/socket equivalence and timing."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from kvrecon.common import dump,digest
from kvrecon.config import DEFAULT_MODEL_CONFIG,load_model_config
from kvrecon.npu import NpuBackend
from kvrecon.backend import CandidateBatchBackend


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    profile=load_model_config(DEFAULT_MODEL_CONFIG)
    sections=json.loads(profile['manifest'].read_text())['model_sections']
    prefix_path=ROOT/'experiments/stored_short0001_layer5/input/prefix.json'
    prefix=json.loads(prefix_path.read_text())
    ids=np.random.default_rng(20261019).choice(262144,145,replace=False).tolist()
    dump(args.output/'protocol.json',dict(prefix=prefix,prefix_sha256=digest(prefix_path),candidate_ids=ids,
        note='Public prefix only; no target or private evaluation labels; same APK for both transports'))
    records={};reference=None
    for transport in ('file','socket'):
        output=args.output/transport;arrays={}
        with NpuBackend(sections,4,output,digest(profile['model']),transport=transport) as backend:
            backend.reset(prefix)
            arrays['ranking']=backend.probabilities()
            raw=np.frombuffer(backend.rpc('logits'),dtype='<f4')
            np.testing.assert_array_equal(arrays['ranking'],np.argsort(-raw,kind='stable'))
            for i in range(3):arrays[f'query{i}']=backend.query(ids[:128])
            np.testing.assert_array_equal(arrays['query0'],arrays['query2'])
            wrapped=CandidateBatchBackend(backend,256)
            arrays['retained']=wrapped.query_retained(ids)
            arrays['accepted']=backend.accept_query(ids[130])
            np.testing.assert_array_equal(arrays['accepted'][0],arrays['retained'][130])
            arrays['next_query']=backend.query(ids[:17])
            arrays['next_ranking']=backend.probabilities()
            if reference is not None:
                for key in arrays:np.testing.assert_array_equal(arrays[key],reference[key],err_msg=key)
            reference=arrays
        np.savez_compressed(output/'outputs.npz',**arrays)
        rows=[json.loads(line) for line in (output/'rpc.jsonl').read_text().splitlines()]
        records[transport]=dict(operations=len(rows),host_seconds=sum(r['host_seconds'] for r in rows),
            worker_seconds=sum(r['worker_seconds'] for r in rows),model_seconds=sum(r['decoder_seconds'] for r in rows),
            query_rows=[r for r in rows if r['op']=='query'])
        print(transport,'passed',flush=True)
    dump(args.output/'validation.json',dict(passed=True,transports=records,
        checks=['full vocabulary ranking vs stable host argsort','repeat query','retained candidate from second physical chunk',
                'all candidate KV bytes across transports','next-prefix KV and ranking across transports']))
    print('PASS:',args.output,flush=True)


if __name__=='__main__':main()
