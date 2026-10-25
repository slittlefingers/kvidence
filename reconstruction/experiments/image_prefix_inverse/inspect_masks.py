import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import EXPERIMENTS,dump
from kvrecon.npu import NpuBackend
root=HERE/'data/mask_audit'
sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
identity=json.loads((HERE/'data/20261022_pilot_prefill/protocol.json').read_text())['model_sha256']
reports=[]
with NpuBackend(sections,4,root/'device',identity,serial='YOUR_DEVICE_SERIAL') as b:
    for position in [0,5,133,512,1000]:
        b.rpc('inspect_collection_mask',position=position)
        reply=b.calls[-1]
        dump(root/f'mask_{position}.json',reply)
        for name in ['mask_global','mask_local']:
            for row in reply[name]:
                q=row['q']; expected=set(range(position))
                if name=='mask_local':expected={k for k in expected if k>=position+q-511}
                actual=set(row['history']); suffix=set(row['suffix'])
                if expected!=actual or suffix!=set(range(q+1)):
                    reports.append(dict(position=position,mask=name,q=q,
                        missing_history=sorted(expected-actual)[:20],extra_history=sorted(actual-expected)[:20],
                        missing_suffix=sorted(set(range(q+1))-suffix)[:20],
                        extra_suffix=sorted(suffix-set(range(q+1)))[:20]))
dump(root/'summary.json',dict(violations=reports))
print(json.dumps(dict(violations=reports),indent=2))
