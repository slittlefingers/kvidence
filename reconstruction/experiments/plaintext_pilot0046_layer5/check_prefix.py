"""Validate only known public prefix rows against the new plaintext APK snapshot."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from kvrecon.common import dump,digest
from kvrecon.config import DEFAULT_MODEL_CONFIG,load_model_config
from kvrecon.observation import load_observation
from kvrecon.npu import NpuBackend
from kvrecon.cpa import component_distances
profile=load_model_config(DEFAULT_MODEL_CONFIG)
sections=json.loads(profile['manifest'].read_text())['model_sections']
meta,_,values,positions,prefix=load_observation(HERE/'input/public_prefix_check')
public=json.loads((HERE/'input/prefix.json').read_text())
output=HERE/'prefix_validation';output.mkdir(exist_ok=False)
rows=[]
with NpuBackend(sections,4,output/'device',meta['model_sha256']) as b:
    b.reset(prefix)
    for i,position in enumerate(positions):
        token=public[int(position)]
        computed=b.values(b.commit(token))[0]
        delta=computed-values[i]
        rows.append(dict(position=int(position),public_token_id=token,l2=float(np.linalg.norm(delta)),
                         kv_l2=component_distances(computed,values[i],512).tolist(),
                         maximum_absolute_difference=float(np.abs(delta).max())))
record=dict(rows=rows,passed=all(r['l2']<4 for r in rows),
            note='Only public prefix inspected; finite alignment check, not a target recovery or accuracy result')
dump(output/'validation.json',record)
print(json.dumps(record,indent=2))
if not record['passed']:raise SystemExit('Public prefix alignment check failed; do not launch recovery')
