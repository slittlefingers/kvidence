"""Validate recovered media and recover unknown text on Pixel; no private input."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump, EXPERIMENTS, digest
from kvrecon.npu import NpuBackend
from kvrecon.engine import reconstruct
from kvrecon.search import SearchConfig
from kvrecon.surrogate import ExistingRidge

class RecoveredPrefix:
    def __init__(self, base, meta, features, eoi):
        self.base,self.meta,self.features,self.eoi=base,meta,features,eoi
    def __getattr__(self,name):return getattr(self.base,name)
    def reset(self,prefix):
        if list(prefix)!=self.meta['public_prefix']:raise ValueError('Unexpected public prefix')
        self.base.reset(prefix)
        self.base.commit_embeddings(self.features)
        self.base.commit_embeddings(self.eoi)
        for t in self.meta['separator_ids']:self.base.commit(t)
        assert self.base.position==self.meta['text_start']

def replay(base,meta,features,eoi):
    base.reset(meta['public_prefix'])
    codes=base.commit_embeddings(features)
    end=base.commit_embeddings(eoi)
    sep=np.concatenate([base.commit(t) for t in meta['separator_ids']])
    assert base.position==meta['text_start']
    return base.values(codes),base.values(end),base.values(sep)

def main():
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    # Initialize trusted dependencies before rejecting reads of private labels.
    surrogate=ExistingRidge(EXPERIMENTS/'embedding_hnsw/assets/embeddings.npy',262144,1536)
    tok=Tokenizer.from_file(str(EXPERIMENTS/'kv_guided_search_gemma4_int8/model/tokenizer.json'))
    from kvrecon.cli import private_guard
    private_guard()
    meta=json.loads((HERE/'input/metadata.json').read_text())
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    with np.load(HERE/'input/observed.npz') as d:targets={k:d[k].copy() for k in d.files}
    eoi=np.load(HERE/'input/end_of_image.npy').reshape(1,1536)
    out=HERE/'pixel';out.mkdir(exist_ok=False)
    dump(out/'protocol.json',dict(target_sha256=digest(HERE/'input/observed.npz'),
        seeds=[17,29],selection='minimum NPU measured media KV squared error, before private evaluation',
        original_image_accessed=False,original_text_accessed=False,target_kv_imported_into_cache=False,
        epsilon=4.0,distance='raw concatenated K+V L2, separately for each position',
        text_method='existing guided 50 probability + 150 random; then 20 proposals; full vocabulary; no query cap',
        media_gate_policy='report media gate separately; downstream text diagnostic also attempted if media gate fails',
        image_evaluation='fixed-question semantic description of recovered latent, not pixel reconstruction'))
    with NpuBackend(sections,4,out/'device',meta['model_sha256']) as b:
        runs=[]
        for seed in (17,29):
            features=np.load(HERE/f'fit/seed_{seed}/features.npy')[0]
            tick=time.perf_counter()
            media,end,sep=replay(b,meta,features,eoi)
            dist=np.linalg.norm(media.astype(np.float64)-targets['media_kv'],axis=-1)
            row=dict(seed=seed,seconds=time.perf_counter()-tick,per_position_l2=dist.tolist(),
                mean_l2=float(dist.mean()),max_l2=float(dist.max()),squared_error=float((dist**2).sum()),
                passing_positions=int((dist<4).sum()),total_positions=len(dist),passes_all=bool((dist<4).all()),
                end_l2=float(np.linalg.norm(end-targets['end_kv'])),
                separator_l2=float(np.linalg.norm(sep-targets['separator_kv'])))
            np.save(out/f'seed_{seed}_kv.npy',media)
            runs.append(row)
            dump(out/'media_replay.json',runs)
            print(f'Pixel seed={seed}: mean L2={dist.mean():.5f}, max={dist.max():.5f}, pass={(dist<4).sum()}/{len(dist)}',flush=True)
        best=min(runs,key=lambda r:r['squared_error'])
        features=np.load(HERE/f"fit/seed_{best['seed']}/features.npy")[0]
        dump(out/'selection.json',best)
        wrapped=RecoveredPrefix(b,meta,features,eoi)
        result=reconstruct(wrapped,targets['text_kv'],targets['text_positions'],meta['public_prefix'],
            SearchConfig(distance_space='original',distance_metric='l2'),out/'text',surrogate=surrogate)
        dump(out/'text_preview.json',dict(text=tok.decode(result['recovered_token_ids']),
            status=result['status'],media_passes_all=best['passes_all'],
            interpretation='Subsequent-text diagnostic; not full pipeline acceptance if media gate fails'))
        # A fresh replay with an invariant question, with no original text/image.
        wrapped.reset(meta['public_prefix'])
        question='Describe the image briefly. Name the visible objects and actions.'
        suffix=question+'<turn|>\n<|turn>model\n'
        for t in tok.encode(suffix,add_special_tokens=False).ids:b.commit(t)
        stop={1,tok.token_to_id('<turn|>'),tok.token_to_id('<|endoftext|>')}
        generated=[]
        for _ in range(160):
            t=int(b.probabilities()[0])
            if t in stop:break
            generated.append(t);b.commit(t)
        dump(out/'image_description.json',dict(question=question,token_ids=generated,
            text=tok.decode(generated),max_tokens=160,truncated=len(generated)==160,
            uses_recovered_features=True,uses_original_image=False))
        print('Pixel recovery and fixed-question image description completed',flush=True)

if __name__=='__main__':main()
