"""Image recovery entry point, invoked only after a verified image-start token."""
from .solver import recover_latent


def build_problem(model, input_ids, media_positions, targets, token_types, **options):
    """Use Gemma's vision adapter, not raw pixels or word embeddings."""
    from .gemma4 import Gemma4Problem
    return Gemma4Problem(model, input_ids, media_positions, targets, token_types,
        modality='image', adapter=model.embed_vision,
        adapter_input_dim=model.config.vision_config.hidden_size, **options)


def recover(backend, request):
    if request['modality'] != 'image':
        raise ValueError('Image handler received a different modality')
    if request.get('optimizer') == 'local_ridge_trust_region':
        return recover_trust_region(backend,request,request.get('optimizer_config'))
    return recover_latent(backend, request, problem_factory=build_problem)




def recover_trust_region(backend, request, options=None):
    """Recover image slots causally using the phone adapter and NPU only."""
    from dataclasses import replace
    from pathlib import Path
    import numpy as np
    from .image_trust import TrustConfig, fit_local
    from ..search import measured_values, kv_distance
    from ..common import dump
    from . import MediaUnavailable
    if request['modality']!='image':raise ValueError('Expected image request')
    positions=np.asarray(request['positions'])
    targets=np.asarray(request['targets'])
    if (positions.ndim!=1 or not len(positions) or targets.ndim!=2
            or len(targets)!=len(positions) or not np.isfinite(targets).all()
            or not np.array_equal(positions,np.arange(int(positions[0]),int(positions[0])+len(positions)))):
        raise ValueError('Require nonempty contiguous image positions and aligned targets')
    config=request['config']
    options=options or TrustConfig(epsilon=config.epsilon,metric=config.metric)
    if options.epsilon!=config.epsilon or options.metric!=config.metric:
        raise ValueError('Optimizer and replay gates must agree')
    for name in ('query_vision','project_vision','commit_embeddings'):
        if not callable(getattr(backend,name,None)):raise MediaUnavailable('Phone vision hook missing: '+name)
    output=Path(request['output']);output.mkdir(parents=True,exist_ok=False)
    runs=[];features=[]
    for index,(position,target) in enumerate(zip(request['positions'],request['targets'])):
        position=int(position)
        if backend.position!=position:raise ValueError('Causal image position mismatch')
        def query(rows):
            values=measured_values(backend,backend.query_vision(rows),config)
            if backend.position!=position:raise ValueError('Candidate query advanced prefix')
            return values
        local=replace(options,seed=options.seed+index,batch_size=min(options.batch_size,backend.batch_capacity,128))
        initial=np.random.default_rng(local.seed).standard_normal(768).astype(np.float32)
        initial/=np.sqrt(np.mean(initial**2))
        folder=output/f'position_{position:04d}'
        latent,row=fit_local(query,target,initial,local,folder)
        row['position']=position
        # No failed candidate is committed merely to continue to later slots.
        if row['passes_query']:
            feature=backend.project_vision(latent[None])
            if backend.position!=position:raise ValueError('Adapter advanced prefix')
            committed=measured_values(backend,backend.commit_embeddings(feature),config)[0]
            distance=float(kv_distance(committed,target,config.metric))
            row.update(commit_distance=distance,passes_commit=distance<config.epsilon)
            np.save(folder/'features.npy',feature);np.save(folder/'commit_kv.npy',committed)
            if backend.position!=position+1:raise ValueError('Unexpected commit position')
            if row['passes_commit']:features.append(feature)
        dump(folder/'result.json',row);runs.append(row);dump(output/'runs.json',runs)
        print(f"image position={position} queries={row['queries']} {config.metric}={row['distance']:.6g} commit={row.get('commit_distance')}",flush=True)
        if not row.get('passes_commit',False):break
    if features:np.save(output/'accepted_features.npy',np.concatenate(features))
    complete=len(features)==len(request['positions'])
    report=dict(status='complete' if complete else 'unresolved',accepted_positions=len(features),
        requested_positions=len(request['positions']),runs=runs,optimizer='local_ridge_trust_region',
        distance_space=config.distance_space,distance_metric=config.metric,epsilon=config.epsilon,
        feature_kind='replacement adapter representation, not reconstructed pixels',
        image_content_evaluated=False,subsequent_text_recovered=False)
    dump(output/'result.json',report)
    return report


if __name__ == '__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='trust-region':
        from .image_experiment import main
        main(sys.argv[2:])
    else:
        from .pilot import main
        main('image',build_problem)
