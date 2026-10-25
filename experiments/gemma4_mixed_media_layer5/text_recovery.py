"""Sequential discrete token search with unknown image and unknown text.

Same ridge-guided candidate principle as the existing text experiment; this
isolated MLX implementation supports multimodal prefixes. No language logits,
ground-truth text, or true prior tokens enter the recovered/zero attack modes.
"""
import argparse
import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from attack_runtime import ROOT, load_runtime, array, write_json, mx, np


def project_kv(base,h,offset):
    a=base.layers[4].self_attn;h=base.layers[4].input_layernorm(h)
    n,t,_=h.shape
    k=a.k_norm(a.k_proj(h).reshape(n,t,a.n_kv_heads,a.head_dim))
    k=a.rope(k.transpose(0,2,1,3),offset=offset)
    v=a.v_norm(a.v_proj(h).reshape(n,t,a.n_kv_heads,a.head_dim)).transpose(0,2,1,3)
    return mx.concatenate([k[:,:,-1,:].reshape(n,-1),v[:,:,-1,:].reshape(n,-1)],-1).astype(mx.float32)


class TextOracle:
    def __init__(self,model,obs,z):
        self.base=model.language_model.model
        self.cache=model.make_cache()[:4]
        text_start=int(obs['text_positions'][0])
        ids=obs['skeleton_ids'][:,:text_start].copy()
        assert (ids>=0).all()
        positions=obs['media_positions'];a,b=int(positions[0]),int(positions[-1])+1
        h=self.base.embed_tokens(mx.array(ids))*self.base.embed_scale
        h=mx.concatenate([h[:,:a],mx.array(z).astype(h.dtype),h[:,b:]],1)
        ids[:,a:b]=0
        pli=self.base.project_per_layer_inputs(h,self.base.get_per_layer_inputs(mx.array(ids)))
        masks=self.base._make_masks(h,self.cache,mx.array(obs['token_types'][:,:text_start]))
        for i in range(4):
            h,_,_=self.base.layers[i](h,masks[i],self.cache[i],per_layer_input=pli[:,:,i,:])
        mx.eval(h,*[x for c in self.cache for x in (c.keys,c.values)])
        self.position=text_start

    def step(self,ids,cache):
        ids=mx.array(ids).reshape(-1,1)
        h=self.base.embed_tokens(ids)*self.base.embed_scale
        pli=self.base.project_per_layer_inputs(h,self.base.get_per_layer_inputs(ids))
        offset=cache[0].offset
        masks=self.base._make_masks(h,cache)
        for i in range(4):
            h,_,_=self.base.layers[i](h,masks[i],cache[i],per_layer_input=pli[:,:,i,:])
        kv=project_kv(self.base,h,offset);mx.eval(kv)
        return kv

    def query(self,token_ids,batch=64):
        parts=[]
        for begin in range(0,len(token_ids),batch):
            ids=token_ids[begin:begin+batch];caches=[]
            for c in self.cache:
                fork=copy.copy(c)
                fork.keys=mx.broadcast_to(c.keys,(len(ids),)+c.keys.shape[1:])
                fork.values=mx.broadcast_to(c.values,(len(ids),)+c.values.shape[1:])
                caches.append(fork)
            parts.append(self.step(ids,caches))
        return mx.concatenate(parts,0)

    def accept(self,tid):
        kv=self.step([tid],self.cache)
        self.position+=1
        return kv


def ranking(emb,queried,kv,target,ridge=.01):
    """The old standardized dual-ridge objective, no change to old scripts."""
    x=emb[mx.array(queried)]
    mean=mx.mean(x,axis=0);scale=mx.maximum(mx.std(x,axis=0),1e-4)
    x=(x-mean)/scale
    ym=mx.mean(kv,axis=0);ys=mx.maximum(mx.std(kv,axis=0),1e-4)
    y=(kv-ym)/ys;xm=mx.mean(x,axis=0);ycmean=mx.mean(y,axis=0)
    xc=x-xm;yc=y-ycmean
    gram=xc@xc.T+mx.eye(len(queried))*(ridge*len(queried))
    # CPU solve is the only unsupported GPU primitive; rank on Metal.
    dual=mx.array(np.linalg.solve(array(gram),array(yc)))
    distances=[]
    for start in range(0,len(emb),8192):
        xx=(emb[start:start+8192]-mean)/scale-xm
        pred=((xx@xc.T)@dual+ycmean)*ys+ym
        d=mx.mean((pred-target).square(),axis=-1);mx.eval(d)
        distances.append(np.array(d))
    distances=np.concatenate(distances);distances[np.asarray(queried)]=np.inf
    return np.argsort(distances)


def candidates(model,tokenizer,out):
    specials=set(tokenizer.all_special_ids)
    ids=[i for i in range(len(tokenizer)) if i not in specials and
         re.fullmatch(r'[ -~\n\t\r]{1,32}',tokenizer.decode([i]))]
    emb=model.language_model.model.embed_tokens(mx.array(ids)).astype(mx.float32);mx.eval(emb)
    write_json(out/'candidate_pool.json',dict(rule='All non-special tokens decoding to 1..32 ASCII printable/whitespace characters',
        count=len(ids),token_ids=ids,full_vocabulary_size=len(tokenizer),
        selected_independently_of_target=True))
    return np.asarray(ids),emb


def recover(args):
    out=Path(args.output);model,processor=load_runtime()
    private_allowed=False
    def read_guard(event,details):
        if event!='open' or not isinstance(details[0],(str,bytes,Path)):return
        filename=Path(details[0]).resolve()
        if filename.is_relative_to((ROOT/'assets').resolve()):
            raise PermissionError('Text recovery cannot read media or acquisition fixtures')
        if filename.is_relative_to((out/'private_eval').resolve()) and not private_allowed:
            raise PermissionError('Non-oracle text recovery cannot read private truth')
    sys.addaudithook(read_guard)
    ids,emb=candidates(model,processor.tokenizer,out)
    character_indices=[i for i,tid in enumerate(ids) if len(processor.tokenizer.decode([int(tid)]))==1]
    print('CANDIDATES',len(ids),flush=True)
    for public in sorted((out/'observed').iterdir()):
        if args.cases and public.name not in args.cases:continue
        obs=np.load(public/'text_observation.npz')
        positions=obs['text_positions'];assert (obs['skeleton_ids'][:,positions]==-1).all()
        for mode in args.modes:
            private_allowed=(mode=='original')
            dest=out/args.result_dir/public.name/mode
            dest.mkdir(parents=True,exist_ok=True)
            if mode=='original':
                # Explicit oracle-image CONTROL. Still does not read true text IDs.
                z=np.load(out/'private_eval'/public.name/'truth.npz')['z']
                selected='true_image_control'
            else:
                paths=list((out/'recovered'/public.name).glob('seed_*'))
                selected=min(paths,key=lambda p:json.loads((p/'run.json').read_text())['best_kv_nmse'])
                z=np.load(selected/'recovered.npz')['z'];selected=selected.name
                if mode=='zero':z=np.zeros_like(z)
            oracle=TextOracle(model,obs,z);recovered=[];trace=[]
            begin=time.perf_counter()
            for j,position in enumerate(positions):
                assert oracle.position==int(position)
                target=mx.array(obs['kv'][0,j]);energy=mx.mean(target.square())
                rng=np.random.default_rng(args.seed+j)
                queried=rng.choice(len(ids),size=200,replace=False).tolist()
                if args.char_coverage and args.char_policy=='prior':
                    queried += [i for i in character_indices if i not in set(queried)]
                assert len(queried)<=args.budget
                qstart=time.perf_counter();kv=oracle.query(ids[queried]);query_seconds=time.perf_counter()-qstart
                fitting=0.
                search_budget=args.budget-200 if args.char_coverage and args.char_policy=='post' else args.budget
                while len(queried)<search_budget:
                    tick=time.perf_counter();rank=ranking(emb,queried,kv,target)
                    chosen=rank[:min(args.proposal_batch,search_budget-len(queried))]
                    assert not set(chosen.tolist()) & set(queried)
                    fitting+=time.perf_counter()-tick
                    tick=time.perf_counter();new=oracle.query(ids[chosen]);query_seconds+=time.perf_counter()-tick
                    kv=mx.concatenate([kv,new],0);queried.extend(chosen.tolist())
                if args.char_coverage and args.char_policy=='post':
                    # Preserve the original 600-query surrogate; supplement its
                    # candidate checks without reweighting its ridge training set.
                    additional=[i for i in character_indices if i not in set(queried)]
                    remaining=args.budget-len(queried)-len(additional)
                    pool=np.setdiff1d(np.arange(len(ids)),np.asarray(queried+additional))
                    additional+=rng.choice(pool,size=remaining,replace=False).tolist()
                    tick=time.perf_counter();new=oracle.query(ids[additional]);query_seconds+=time.perf_counter()-tick
                    kv=mx.concatenate([kv,new],0);queried.extend(additional)
                distances=array(mx.mean((kv-target).square(),axis=-1)/energy)
                assert len(set(queried))==args.budget and np.isfinite(distances).all()
                best=int(np.argmin(distances));tid=int(ids[queried[best]])
                mx.eval(oracle.accept(tid));recovered.append(tid)
                order=np.argsort(distances)
                row=dict(position=int(position),index=j,token_id=tid,
                    piece=processor.tokenizer.decode([tid]),kv_nmse=float(distances[best]),
                    runner_up_nmse=float(distances[order[1]]),queries=len(queried),
                    queried_token_ids=ids[queried].tolist(),distances=distances.tolist(),
                    fit_rank_seconds=fitting,query_seconds=query_seconds,
                    recovered_prefix_text=processor.tokenizer.decode(recovered))
                trace.append(row)
                write_json(dest/'trace.json',trace)
                print('TEXT',public.name,mode,j,repr(row['piece']),row['kv_nmse'],
                      round(time.perf_counter()-begin,1),flush=True)
            write_json(dest/'run.json',dict(case=public.name,mode=mode,image_source=selected,
                token_ids=recovered,text=processor.tokenizer.decode(recovered),
                seconds=time.perf_counter()-begin,seed=args.seed,budget_per_token=args.budget,
                initial_queries=200,proposal_batch=args.proposal_batch,candidate_count=len(ids),
                char_coverage=args.char_coverage,character_token_count=len(character_indices),
                character_policy=args.char_policy if args.char_coverage else 'none',
                media_and_private_truth_read_guard=True,
                uses_true_previous_tokens=False,uses_true_text=False,
                uses_true_image=mode=='original',selection='minimum measured layer-5 KV NMSE'))
            del oracle;mx.clear_cache()


def evaluate(args):
    out=Path(args.output);rows=[]
    pool=set(json.loads((out/'candidate_pool.json').read_text())['token_ids'])
    for public in sorted((out/'observed').iterdir()):
        truth=np.load(out/'private_eval'/public.name/'truth.npz')['text_ids'].tolist()
        text=json.loads((out/'private_eval'/public.name/'case.json').read_text())['secret']
        for p in sorted((out/args.result_dir/public.name).glob('*')):
            if not (p/'run.json').exists():continue
            run=json.loads((p/'run.json').read_text());trace=json.loads((p/'trace.json').read_text())
            assert len(truth)==len(run['token_ids'])
            correct=[a==b for a,b in zip(truth,run['token_ids'])]
            hits=[t in r['queried_token_ids'] for t,r in zip(truth,trace)]
            rows.append(dict(**run,true_text=text,true_token_ids=truth,
                token_correct=sum(correct),total_tokens=len(truth),all_exact=all(correct),
                exact_by_position=correct,true_candidate_queried=hits,
                true_tokens_in_pool=sum(t in pool for t in truth),
                true_token_distances=[r['distances'][r['queried_token_ids'].index(t)]
                    if t in r['queried_token_ids'] else None for t,r in zip(truth,trace)]))
    evaluation_name='text_evaluation.json' if args.result_dir=='text_recovered' else args.result_dir+'_evaluation.json'
    write_json(out/evaluation_name,rows)
    for r in rows:print(r['case'],r['mode'],r['token_correct'],r['total_tokens'],repr(r['text']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['recover','evaluate'])
    p.add_argument('--output',default=str(ROOT/'results'));p.add_argument('--cases',nargs='+')
    p.add_argument('--modes',nargs='+',choices=['original','recovered','zero'],default=['original','recovered','zero'])
    p.add_argument('--budget',type=int,default=600);p.add_argument('--proposal-batch',type=int,default=100)
    p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--char-coverage',action='store_true')
    p.add_argument('--char-policy',choices=['prior','post'],default='prior')
    p.add_argument('--result-dir',default='text_recovered')
    args=p.parse_args();{'recover':recover,'evaluate':evaluate}[args.stage](args)
