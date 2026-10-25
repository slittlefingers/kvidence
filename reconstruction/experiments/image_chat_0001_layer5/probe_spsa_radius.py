"""Post-run local diagnostic; not another full recovery or Adam trajectory."""
import json
from pathlib import Path
import sys
import numpy as np
from spsa_adam import estimate_gradient,unit
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import dump,EXPERIMENTS,digest
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard


def main():
    private_guard()
    base=HERE/'spsa_adam_device'
    summary=json.loads((base/'result.json').read_text())
    chosen=min((r for r in summary['runs'] if r['kind'].startswith('synthetic')),key=lambda r:r['best_l2'])
    u=np.load(base/chosen['kind']/'best_latent.npy')
    target=np.load(base/'synthetic_target_kv.npy')
    meta=json.loads((HERE/'input/metadata.json').read_text())
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    out=base/'radius_diagnostic';out.mkdir(exist_ok=False)
    radii=[.08,.02,.005];steps=[.005,.02,.05]
    delta=(2*np.random.default_rng(3107).integers(0,2,(64,768))-1).astype(np.float32)
    dump(out/'protocol.json',dict(selected_start=chosen['kind'],radii=radii,update_rms=steps,
        direction_seed=3107,identical_directions_for_all_radii=True,
        purpose='local sensitivity test after baseline; no convergence claim',
        source_sha256=digest(Path(__file__))))
    rows=[]
    with NpuBackend(sections,4,out/'device',meta['model_sha256']) as b:
        b.reset(meta['public_prefix'])
        initial=float(np.linalg.norm(b.values(b.query_vision(u[None]))[0]-target))
        for radius in radii:
            values=b.values(b.query_vision(np.concatenate([u+radius*delta,u-radius*delta])))
            loss=.5*np.mean((values-target)**2,axis=-1)
            g=estimate_gradient(loss[:64],loss[64:],delta,radius)
            g/=np.sqrt(np.mean(g*g)+1e-30)
            candidates=unit(u[None]-np.array(steps)[:,None]*g)
            actual=b.values(b.query_vision(candidates));dist=np.linalg.norm(actual-target,axis=-1)
            np.save(out/f'candidates_radius_{radius}.npy',candidates)
            np.save(out/f'kv_radius_{radius}.npy',actual)
            committed_l2=None
            if dist.min()<4:
                feature=b.project_vision(candidates[int(dist.argmin())][None])
                committed_l2=float(np.linalg.norm(b.values(b.commit_embeddings(feature))[0]-target))
                b.reset(meta['public_prefix'])
            rows.append(dict(radius=radius,initial_l2=initial,update_rms=steps,
                candidate_l2=dist.tolist(),zero_slope_fraction=float(np.mean(loss[:64]==loss[64:])),
                best_l2=float(dist.min()),any_descent=bool(np.any(dist<initial)),
                commit_l2_if_threshold=committed_l2))
        dump(out/'result.json',dict(selected_start=chosen['kind'],rows=rows,
            query_only_local_probe=True,original_media_used=False,does_not_replace_baseline_results=True))
        print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
