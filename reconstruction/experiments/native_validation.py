"""Same-APK reference/optimized worker, public inputs, byte-level comparison."""
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
    parser.add_argument('--modes',nargs='+',choices=('reference','optimized','optimized_v2'),default=['reference','optimized'])
    parser.add_argument('--cache-churn',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    profile=load_model_config(DEFAULT_MODEL_CONFIG)
    sections=json.loads(profile['manifest'].read_text())['model_sections']
    prefix=json.loads((ROOT/'experiments/stored_short0001_layer5/input/prefix.json').read_text())
    ids=np.random.default_rng(20261019).choice(262144,145,replace=False).tolist()
    dump(args.output/'protocol.json',dict(prefix=prefix,candidates=ids,
        note='Same APK, socket transport; no target/labels',modes=args.modes,cache_churn=args.cache_churn,
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'native').glob('*')) if p.is_file()}))
    reference=None;records={}
    for mode in args.modes:
        output=args.output/mode;arrays={}
        with NpuBackend(sections,4,output,digest(profile['model']),native_mode=mode) as backend:
            backend.reset(prefix);arrays['ranking']=backend.probabilities()
            for i in range(3):arrays[f'query{i}']=backend.query(ids[:128])
            np.testing.assert_array_equal(arrays['query0'],arrays['query2'])
            wrapped=CandidateBatchBackend(backend,256)
            arrays['retained']=wrapped.query_retained(ids)
            arrays['accepted']=backend.accept_query(ids[130])
            np.testing.assert_array_equal(arrays['accepted'][0],arrays['retained'][130])
            arrays['next_query']=backend.query(ids[:17]);arrays['next_ranking']=backend.probabilities()
            # Reset to a different same-length prefix: position alone is not a cache identity across reset.
            backend.reset(prefix[:-1]+[ids[0]])
            arrays['changed_prefix']=backend.query(ids[:4])
            backend.reset(prefix);arrays['reset_query']=backend.query(ids[:4])
            np.testing.assert_array_equal(arrays['reset_query'],arrays['query0'][:4])
            arrays['single_query']=backend.query(ids[:1])
            np.testing.assert_array_equal(arrays['single_query'],arrays['query0'][:1])
            if args.cache_churn:
                churn=np.random.default_rng(42).choice(262144,1100,replace=False).tolist()
                arrays['churn']=backend.query_bulk(churn)
                arrays['after_churn']=backend.query(ids[:128])
                np.testing.assert_array_equal(arrays['after_churn'],arrays['query0'])
            if reference is not None:
                for key in arrays:np.testing.assert_array_equal(arrays[key],reference[key],err_msg=key)
            # Compare shared checks with the previous APK's saved output as well.
            legacy=ROOT/'results/transport_validation_20261019/socket/outputs.npz'
            with np.load(legacy) as previous:
                for key in previous.files:np.testing.assert_array_equal(arrays[key],previous[key],err_msg='previous APK '+key)
            reference=arrays
        np.savez_compressed(output/'outputs.npz',**arrays)
        rows=[json.loads(x) for x in (output/'rpc.jsonl').read_text().splitlines()]
        records[mode]=dict(host_seconds=sum(x['host_seconds'] for x in rows),
            worker_seconds=sum(x['worker_seconds'] for x in rows),model_seconds=sum(x['decoder_seconds'] for x in rows),
            query_rows=[x for x in rows if x['op']=='query'])
        print(mode,'passed',flush=True)
    dump(args.output/'validation.json',dict(passed=True,results=records,
        checks=['previous APK outputs','full rankings','repeated queries','different candidate counts',
                'retained candidate from second physical batch','commit prefix','reset invalidation']))
    print('PASS',args.output,flush=True)


if __name__=='__main__':main()
