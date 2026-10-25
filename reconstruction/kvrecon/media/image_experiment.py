"""Prepare public image_chat_100 KV observations, then isolated NPU recovery."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import struct
import numpy as np
from ..common import WORKSPACE, EXPERIMENTS, dump, digest, quantization, distance_values
from ..search import SearchConfig
from .image_trust import TrustConfig


def prepare(dataset, sample, output, layer=4):
    from tokenizers import Tokenizer
    dataset=Path(dataset);output=Path(output)
    manifest=json.loads((dataset/'manifest.json').read_text())
    item=next(x for x in manifest['samples'] if x['id']==sample)
    tokenizer=Tokenizer.from_file(str(EXPERIMENTS/'kv_guided_search_gemma4_int8/model/tokenizer.json'))
    prefix=[2]+tokenizer.encode('<|turn>user\n\n\n<|image>',add_special_tokens=False).ids
    # Same publicly known shape/template rule as the original pilot, no pixels.
    factor=np.sqrt(np.float32(2520*16*16)/np.float32(item['width']*item['height']))
    height=int(np.floor(factor*np.float32(item['height'])/48))*48
    width=int(np.floor(factor*np.float32(item['width'])/48))*48
    count=(height//48)*(width//48);start=len(prefix);end=start+count
    snapshot=dataset/'capture'/sample/'all_layers.gkv.gz'
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    with gzip.open(snapshot,'rb') as f:
        header=f.read(16)
        if header[:8]!=b'GHOSTKV1':raise ValueError('Unknown capture format')
        layout=json.loads(f.read(struct.unpack('<Q',header[8:])[0]));payload=f.read()
    if layout.get('ringbuffer') or layout['kv_positions']<end:
        raise ValueError('Image positions unavailable or ring-mapped')
    rows=[]
    for part in ('k','v'):
        b=next(x for x in layout['buffers'] if x['name']==f'kv_cache_{part}_{layer}')
        if b.get('local') or b['dtype']!='int8' or b['capacity']<end:
            raise ValueError('Require retained full-attention INT8 image KV')
        array=np.ndarray(b['shape'],dtype=np.int8,buffer=payload,offset=b['payload_offset'],strides=tuple(b['strides']))
        rows.append(np.moveaxis(array,b['sequence_axis'],0).reshape(b['capacity'],-1)[start:end].copy())
    codes=np.concatenate(rows,axis=1);quant=quantization(sections,layer)
    if codes.shape!=(count,sum(x['width'] for x in quant)):raise ValueError('KV width mismatch')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'observed.npz',codes=codes,positions=np.arange(start,end))
    dump(output/'metadata.json',dict(dataset='image_chat_100',sample=sample,
        model_sha256=manifest['capture_identity']['model_sha256'],snapshot_sha256=digest(snapshot),
        observation_sha256=digest(output/'observed.npz'),layer_index=layer,quantization=quant,
        public_prefix=prefix,media_start=start,media_end=end,image_dimensions=[item['width'],item['height']],
        boundary_source='known Gemma capture template and public image dimensions; no pixel or text reads',
        original_media_accessed=False,original_text_accessed=False))
    print(f'Prepared {sample}: {count} image slots, layer {layer+1}, {output}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--dataset',type=Path,default=WORKSPACE/'kvidence/datasets/shadow_chat/image_chat_100')
    q.add_argument('--sample',default='0001');q.add_argument('--layer',type=int,default=4,help='Zero-based layer')
    q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('run');q.add_argument('--observation',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--max-positions',type=int,default=1,help='Sequential slots; 0 means all')
    q.add_argument('--serial',default='YOUR_DEVICE_SERIAL');q.add_argument('--no-install',action='store_true')
    q.add_argument('--distance-space',choices=['lsb','original'],default='lsb')
    q.add_argument('--distance-metric',choices=['rmse','l2'],default='rmse');q.add_argument('--epsilon',type=float,default=.125)
    for name,kind,default in [('directions',int,16),('max-queries',int,2048),('seed',int,29),('batch-size',int,128),
        ('radius',float,2.),('min-radius',float,.01),('max-radius',float,16.),('ridge',float,.01),('damping',float,.001)]:
        q.add_argument('--'+name,type=kind,default=default)
    args=p.parse_args(argv)
    if args.command=='prepare':return prepare(args.dataset,args.sample,args.output,args.layer)
    from ..cli import private_guard
    from ..npu import NpuBackend
    from .image import recover_trust_region
    private_guard()
    if args.max_positions<0:raise ValueError('max-positions must be nonnegative')
    meta=json.loads((args.observation/'metadata.json').read_text())
    if digest(args.observation/'observed.npz')!=meta['observation_sha256']:raise ValueError('Observation integrity mismatch')
    with np.load(args.observation/'observed.npz',allow_pickle=False) as data:
        codes=data['codes'].copy();positions=data['positions'].copy()
    if not np.array_equal(positions,np.arange(meta['media_start'],meta['media_end'])):raise ValueError('Noncontiguous image slots')
    count=min(args.max_positions,len(positions)) if args.max_positions else len(positions)
    target=distance_values(codes[:count],meta['quantization'],args.distance_space)
    search=SearchConfig(distance_space=args.distance_space,distance_metric=args.distance_metric,epsilon=args.epsilon)
    search.validate()
    options=TrustConfig(**{key:getattr(args,key) for key in ('directions','max_queries','seed','batch_size','radius','min_radius','max_radius','ridge','damping')},epsilon=args.epsilon,metric=args.distance_metric)
    options.validate(768)
    args.output.mkdir(parents=True,exist_ok=False)
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    dump(args.output/'protocol.json',dict(observation=meta,optimizer=asdict(options),distance_space=args.distance_space,
        requested_positions=count,adapter='phone CPU vision_adapter_280',decoder='phone NPU',
        original_media_accessed=False,original_text_accessed=False,target_cache_used_as_prefix=False,
        input_dtype='float32',input_width=768,output_kind='adapter representations, not pixels',
        source_sha256={name:digest(Path(__file__).with_name(name)) for name in ('image.py','image_trust.py','image_experiment.py')}))
    with NpuBackend(sections,meta['layer_index'],args.output/'device',meta['model_sha256'],serial=args.serial,install=not args.no_install) as backend:
        from ..common import validate_quantization_grid
        validate_quantization_grid(meta['quantization'],backend.quant)
        backend.reset(meta['public_prefix'])
        result=recover_trust_region(backend,dict(modality='image',positions=positions[:count],targets=target,
            config=search,output=args.output/'recovery'),options)
        if result['status']=='complete' and count<len(positions):result['status']='position_limit'
        result.update(sample=meta['sample'],total_image_positions=len(positions),
            whole_image_representation_available=result['accepted_positions']==len(positions))
        dump(args.output/'result.json',result)
