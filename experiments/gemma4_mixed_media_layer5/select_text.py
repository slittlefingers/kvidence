"""Choose whole recovered sequences by observed KV only, then score separately."""
import argparse
import json
import sys
import time
from pathlib import Path
from text_recovery import TextOracle
from attack_runtime import ROOT, load_runtime, array, write_json, np, mx


def select():
    out=ROOT/'results';model,_=load_runtime();rows=[];private_allowed=False
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,Path)):return
        p=Path(args[0]).resolve()
        if p.is_relative_to((ROOT/'assets').resolve()):raise PermissionError('No acquisition assets')
        if p.is_relative_to((out/'private_eval').resolve()) and not private_allowed:
            raise PermissionError('No private truth for recovered/zero modes')
    sys.addaudithook(guard)
    for case in ['giraffes','lunch']:
        obs=np.load(out/'observed'/case/'text_observation.npz')
        for mode in ['original','recovered','zero']:
            private_allowed=mode=='original'
            if mode=='original':z=np.load(out/'private_eval'/case/'truth.npz')['z']
            else:
                best=min((out/'recovered'/case).glob('seed_*'),
                    key=lambda p:json.loads((p/'run.json').read_text())['best_kv_nmse'])
                z=np.load(best/'recovered.npz')['z']
                if mode=='zero':z=np.zeros_like(z)
            choices=[]
            for variant in ['char800','charpost800']:
                run=json.loads((out/('text_'+variant)/case/mode/'run.json').read_text())
                tick=time.perf_counter();oracle=TextOracle(model,obs,z);kv=[]
                for token in run['token_ids']:kv.append(array(oracle.accept(token))[0])
                kv=np.stack(kv);target=obs['kv'][0]
                score=float(np.sum((kv.astype(float)-target)**2)/np.sum(target.astype(float)**2))
                choices.append(dict(variant=variant,observed_kv_nmse=score,
                    replay_seconds=time.perf_counter()-tick,search_seconds=run['seconds'],
                    token_ids=run['token_ids'],text=run['text']))
                del oracle;mx.clear_cache()
            chosen=min(choices,key=lambda x:x['observed_kv_nmse'])
            row=dict(case=case,mode=mode,choices=choices,selected_variant=chosen['variant'],
                token_ids=chosen['token_ids'],text=chosen['text'],selection='lowest full-sequence observed KV NMSE',
                total_search_queries_per_token=1600,extra_replay_forwards_per_token=2,
                total_seconds=sum(x['search_seconds']+x['replay_seconds'] for x in choices),
                truth_used_for_selection=False,media_and_private_truth_read_guard=True)
            rows.append(row);write_json(out/'selected_text.json',rows)
            print('SELECT',case,mode,[(x['variant'],x['observed_kv_nmse']) for x in choices],chosen['variant'],flush=True)


def evaluate():
    out=ROOT/'results';rows=json.loads((out/'selected_text.json').read_text())
    for r in rows:
        truth=np.load(out/'private_eval'/r['case']/'truth.npz')['text_ids'].tolist()
        r['token_correct']=sum(a==b for a,b in zip(truth,r['token_ids']))
        r['total_tokens']=len(truth);r['all_exact']=truth==r['token_ids']
        print(r['case'],r['mode'],r['token_correct'],r['total_tokens'],repr(r['text']),flush=True)
    write_json(out/'selected_text_evaluation.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','evaluate'])
    args=p.parse_args();{'select':select,'evaluate':evaluate}[args.stage]()
