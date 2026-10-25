"""Independently check dumped INT8 bytes and summarize candidate-batch probes."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import statistics


def analyze(run):
    report=json.loads((run/'data/report.json').read_text())
    job=json.loads((run/'input.json').read_text())
    assert report['completed']
    assert 'OK (1 test)' in (run/'device.log').read_text()
    batch=len(report['candidates'])
    signature=report.get('signature','verify')
    metas=job['model_sections']['tf_lite_prefill_decode']['signatures']
    reference={b['name']:b for b in metas['decode']['outputs']}
    target={b['name']:b for b in metas[signature]['outputs']}
    names=sorted(n for n in target if n.startswith('kv_slice_'))
    assert len(names)==30
    result=dict(batch=batch,signature=signature,independently_verified=True,cases=[])
    for case in report['cases']:
        t=case['prefix_length']
        suffix='batch' if report.get('schema',1)>=2 else 'verify'
        raw={n:array('b',(run/f'data/p{t}-{suffix}-{n}.bin').read_bytes()) for n in names}
        per_candidate=[];rows=[];digests=[]
        for q in range(batch):
            layers={};h=hashlib.sha256()
            for n in names:
                a=array('b',(run/f'data/p{t}-serial{q}-{n}.bin').read_bytes())
                h.update(a.tobytes())
                shape=target[n]['shape']
                assert shape[:2]==[1,1]
                if n.startswith('kv_slice_k_'):
                    assert shape[2]==batch
                    b=raw[n][q*shape[3]:(q+1)*shape[3]]
                else:
                    assert shape[3]==batch
                    b=raw[n][q::batch]
                assert len(a)==len(b)
                delta=[int(x)-int(y) for x,y in zip(a,b)]
                observed=case['comparisons'][q][n]
                mx=max(map(abs,delta));changed=sum(x!=0 for x in delta)
                integer_l2=math.sqrt(sum(x*x for x in delta))
                assert observed['max_abs_lsb']==mx and observed['different']==changed
                assert math.isclose(integer_l2,observed['l2_integer'],abs_tol=1e-10)
                ma,mb=reference[n],target[n]
                assert len(ma['scale'])==len(mb['scale'])==len(ma['zero_point'])==len(mb['zero_point'])==1
                dequant=[(int(x)-ma['zero_point'][0])*ma['scale'][0]-(int(y)-mb['zero_point'][0])*mb['scale'][0] for x,y in zip(a,b)]
                layer=int(n.rsplit('_',1)[1]);z=layers.setdefault(layer,dict(max_abs_lsb=0,squared_integer_l2=0.,squared_dequantized_l2=0.,different=0,elements=0,abs_sum=0))
                z['max_abs_lsb']=max(z['max_abs_lsb'],mx);z['squared_integer_l2']+=integer_l2**2
                z['squared_dequantized_l2']+=sum(x*x for x in dequant)
                z['different']+=changed;z['elements']+=len(delta);z['abs_sum']+=sum(map(abs,delta))
            for z in layers.values():
                z['l2_integer']=math.sqrt(z.pop('squared_integer_l2'))
                z['l2_dequantized']=math.sqrt(z.pop('squared_dequantized_l2'))
                z['mean_abs_lsb']=z.pop('abs_sum')/z['elements']
                rows.append(z)
            per_candidate.append(layers);digests.append(h.hexdigest())
        assert len(set(digests))==batch,'Candidate outputs unexpectedly identical'
        serial=sum(statistics.mean(times) for times in case['serial_seconds'])
        batched=statistics.mean(case.get('batch_seconds',case.get('verify_seconds')))
        fifth=[candidate[4] for candidate in per_candidate]
        isolation=[v for c in case['neighbour_changes'] for v in c.values()]
        negative=list(case['causal_negative_control'].values())
        summary=dict(prefix_length=t,serial_total_ms=serial*1000,batch_ms=batched*1000,speedup=serial/batched,
            all_layers_max_abs_lsb=max(z['max_abs_lsb'] for z in rows),all_layers_changed=sum(z['different'] for z in rows),
            all_layers_elements=sum(z['elements'] for z in rows),
            layer4_max_abs_lsb=max(z['max_abs_lsb'] for z in fifth),
            layer4_max_l2_integer=max(z['l2_integer'] for z in fifth),
            layer4_max_l2_dequantized=max(z['l2_dequantized'] for z in fifth),
            layer4_l2_dequantized_below4=sum(z['l2_dequantized']<4 for z in fifth),
            neighbour_change_max_abs_lsb=max(z['max_abs_lsb'] for z in isolation),
            causal_control_max_abs_lsb=max(z['max_abs_lsb'] for z in negative),
            input_kv_unchanged=case.get('input_kv_unchanged'),
            layer_metrics=per_candidate)
        result['cases'].append(summary)
    (run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args()
    result=analyze(args.run)
    print(json.dumps(dict(batch=result['batch'],signature=result['signature'],cases=[{k:v for k,v in c.items() if k!='layer_metrics'} for c in result['cases']]),indent=2))
