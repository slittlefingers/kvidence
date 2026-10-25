"""Evaluator-only checks of sequential/batched arithmetic and prefix drift."""
import json
from pathlib import Path
from text_recovery import ROOT, TextOracle, load_runtime, np, array, mx, write_json


def main():
    out=ROOT/'results';model,_=load_runtime();rows=[]
    for public in sorted((out/'observed').iterdir()):
        obs=np.load(public/'text_observation.npz')
        truth=np.load(out/'private_eval'/public.name/'truth.npz')
        best=min((out/'recovered'/public.name).glob('seed_*'),
                 key=lambda p:json.loads((p/'run.json').read_text())['best_kv_nmse'])
        recovered=np.load(best/'recovered.npz')['z']
        for mode,z in [('original',truth['z']),('recovered',recovered),('zero',np.zeros_like(recovered))]:
            oracle=TextOracle(model,obs,z);dist=[];batchdiff=[]
            for j,tid in enumerate(truth['text_ids']):
                offsets=[c.offset for c in oracle.cache]
                previous=[array(c.keys).copy() for c in oracle.cache]
                single=array(oracle.query([int(tid)]))[0]
                batch=array(oracle.query([int(tid),123,456]))[0]
                assert [c.offset for c in oracle.cache]==offsets
                assert all(np.array_equal(a,array(c.keys)) for a,c in zip(previous,oracle.cache))
                target=obs['kv'][0,j]
                dist.append(float(np.mean((single-target)**2)/np.mean(target**2)))
                batchdiff.append(float(np.mean((single-batch)**2)/np.mean(single**2)))
                oracle.accept(int(tid))
            row=dict(case=public.name,mode=mode,teacher_forced_true_previous_tokens=True,
                target_nmse=dist,mean_target_nmse=float(np.mean(dist)),max_target_nmse=max(dist),
                batch_vs_single_nmse=batchdiff,queries_leave_prefix_unchanged=True)
            rows.append(row);print('VALIDATION',public.name,mode,row['mean_target_nmse'],row['max_target_nmse'],flush=True)
            del oracle;mx.clear_cache()
    assert max(r['max_target_nmse'] for r in rows if r['mode']=='original')<.002
    write_json(out/'teacher_forced_diagnostic.json',rows)


if __name__=='__main__':main()
