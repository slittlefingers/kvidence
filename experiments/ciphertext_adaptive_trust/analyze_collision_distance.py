"""Read-only post-hoc distance/rank audit of the 999-token Collision+ trace.

No new model inference, no source ciphertext or private evaluation labels.
Distances are real raw K+V L2 (1024 coordinates), not predicted values or LSBs.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'ciphertext_hybrid_l2_long/results'
trace=SOURCE/'collision_plus_l2/trace.jsonl'
integrity=json.loads((SOURCE/'integrity.json').read_text())
sha=hashlib.sha256(trace.read_bytes()).hexdigest()
assert sha==integrity['file_sha256']['collision_plus_l2/trace.jsonl']
ev=json.loads((SOURCE/'evaluation.json').read_text())['methods']['collision_plus_l2']
assert ev['all_exact'] and ev['incorrect_acceptances']==0
rows=[json.loads(line) for line in trace.open(encoding='utf-8')]
assert len(rows)==999
all_values=[];wrong=[];before=[];overshoot=[];ranks=[];cost=[];correct=[]
for j,row in enumerate(rows):
    d=np.asarray(row['raw_l2_distances'],dtype=np.float64)
    ids=row['queried_token_ids'];chosen=row['accepted_token_id']
    assert len(d)==len(ids)==len(set(ids))==row['queries'] and np.isfinite(d).all()
    index=ids.index(chosen)
    assert d[index]<4
    np.testing.assert_allclose(d[index],row['accepted_raw_l2'],rtol=1e-12,atol=1e-12)
    others=np.delete(d,index)
    assert np.all(others>=4)
    all_values.extend(d);wrong.extend(others)
    before.extend(d[:index]);overshoot.extend(d[index+1:])
    ranks.append(index+1);cost.append(len(d));correct.append(float(d[index]))
all_values=np.asarray(all_values);wrong=np.asarray(wrong);before=np.asarray(before)
overshoot=np.asarray(overshoot);ranks=np.asarray(ranks);cost=np.asarray(cost)
assert len(all_values)==ev['total_queries']==431120
assert len(before)+len(overshoot)+len(rows)==len(all_values)

def distribution(a):
    quantiles=(0,.1,.25,.5,.75,.9,.95,.99,1)
    return dict(count=len(a),mean=float(np.mean(a)),
        quantiles={str(q):float(v) for q,v in zip(quantiles,np.quantile(a,quantiles))})

thresholds=[]
for threshold in (4,8,12,16,20,24,32):
    record=dict(raw_l2_at_least=threshold)
    for name,a in (('all_actual_queries',all_values),('incorrect_actual_queries',wrong),
                   ('incorrect_before_first_hit',before),('trailing_batch_overshoot',overshoot)):
        n=int(np.sum(a>=threshold))
        record[name]=dict(count=n,total=len(a),fraction=n/len(a))
    thresholds.append(record)

bands=[]
for lo,hi in ((1,50),(51,200),(201,1000),(1001,2000),(2001,53150)):
    mask=(ranks>=lo)&(ranks<=hi)
    values=[]
    for row,rank in zip(rows,ranks):
        # Wrong candidates before acceptance only; do not include the true hit.
        values.extend(row['raw_l2_distances'][lo-1:min(hi,int(rank)-1)])
    a=np.asarray(values)
    bands.append(dict(rank_from=lo,rank_to=hi,
        positions_with_correct_token_in_band=int(mask.sum()),
        actual_queries_for_those_positions=int(cost[mask].sum()),
        fraction_of_all_actual_queries=float(cost[mask].sum()/cost.sum()),
        wrong_queries_at_candidate_ranks_in_band=distribution(a),
        wrong_fraction_raw_l2_at_least_16=float(np.mean(a>=16))))

rank_tails=[]
for t in (50,200,1000,2000,5000):
    mask=ranks>t
    rank_tails.append(dict(correct_token_rank_greater_than=t,
        positions=int(mask.sum()),fraction_of_positions=float(mask.mean()),
        actual_queries_for_those_positions=int(cost[mask].sum()),
        fraction_of_all_actual_queries=float(cost[mask].sum()/cost.sum())))

result=dict(source=str(trace.relative_to(ROOT.parent)),source_sha256=sha,
    sample_id='0037',target_tokens=999,candidate_pool=53150,
    scope='Existing Collision+ probability-order adaptation with raw L2<4; no new inference',
    metric='Raw real concatenated layer-5 K+V Euclidean L2; 1024 BF16 coordinates read to float32 and compared in float64',
    rank_definition='1-based probability rank within the filtered Base64-piece pool, not full vocabulary',
    all_actual_queries=distribution(all_values),incorrect_actual_queries=distribution(wrong),
    incorrect_before_first_hit=distribution(before),trailing_batch_overshoot=distribution(overshoot),
    correct_hits=distribution(np.asarray(correct)),correct_token_ranks=distribution(ranks),
    far_distance_thresholds=thresholds,rank_bands=bands,correct_rank_tails=rank_tails,
    interpretation_limits=[
        'Very far has no canonical threshold; cumulative raw L2 cutoffs are descriptive, not new acceptance rules.',
        'Batch size 20 means physical query cost includes candidates after the first passing probability rank.',
        'Minimum wrong distance describes only queried candidates, not all vocabulary candidates.',
        'Ranks and later-rank distance groups reflect different surviving positions; do not infer a causal distance trend.',
        'Distances are available after real forwards; far observations alone do not justify zero-cost safe pruning.',
        'One previously used ciphertext, native BF16 KV and Apple MLX GPU; not NPU or KV8 evidence.'])
output=ROOT/'collision_distance_analysis.json'
output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(total_queries=len(all_values),incorrect_queries=len(wrong),
    before_hit=len(before),batch_overshoot=len(overshoot),
    correct_l2_range=[min(correct),max(correct)],wrong_quantiles=distribution(wrong)['quantiles'],
    rank_tails=rank_tails,output=str(output)),ensure_ascii=False,indent=2))
