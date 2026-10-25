"""Persistent case metadata and fixed recovery templates. No model-generated shell."""
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = HERE / 'cases'
METHODS = {'guided':'Ridge-guided search', 'collision_plus':'Collision+ (shared-RMSE adaptation)',
           'collision_plus_cpa':'Collision+ CPA (independently calibrated)', 'exhaustive':'Exhaustive token-ID search'}
MODELS = {'gemma4-e2b':'gemma4_e2b.json','gemma3-1b':'gemma3_1b.json'}
ALLOWED = {'model_name','observation','methods','calibration','notes','max_positions','max_queries'}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class CaseStore:
    def __init__(self, root=CASES):self.root=Path(root)

    def path(self,name):
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):
            raise ValueError('Case names use 1–64 letters, numbers, underscores or hyphens.')
        return self.root/name

    def load(self,name):return json.loads((self.path(name)/'metadata.json').read_text())

    def create(self,name):
        directory=self.path(name);directory.mkdir(parents=True,exist_ok=False)
        case=dict(schema=1,name=name,created_at=time.time(),updated_at=time.time(),
            fields=dict(model_name=None,observation=None,methods=['guided','collision_plus','exhaustive']),
            defaults=dict(epsilon=.125,distance_metric='rmse',distance_space='auto',batch_size=128),
            evidence={},provenance={},runs=[],conversation_file='conversation.jsonl')
        write(directory/'metadata.json',case);return case

    def save(self,case):
        case['updated_at']=time.time();write(self.path(case['name'])/'metadata.json',case)

    def record(self,name,role,text,**extra):
        directory=self.path(name)
        with (directory/'conversation.jsonl').open('a') as out:
            out.write(json.dumps(dict(time=time.time(),role=role,text=text,**extra),ensure_ascii=False)+'\n')

    def update(self,name,fields,source='user'):
        if not isinstance(fields,dict) or set(fields)-ALLOWED:raise ValueError('Unsupported case metadata field.')
        case=self.load(name)
        for key,value in fields.items():
            if key=='model_name' and value not in (*MODELS,'unknown',None):raise ValueError('Unregistered model; provide a supported model or say unknown.')
            if key=='methods' and (not isinstance(value,list) or not value or len(set(value))!=len(value) or any(x not in METHODS for x in value)):
                raise ValueError('Methods must be guided, collision_plus, collision_plus_cpa, or exhaustive.')
            if key in ('max_positions','max_queries') and value is not None and (type(value) is not int or value<1):raise ValueError('Budgets must be positive integers.')
            if key in ('observation','calibration') and value is not None:
                if not isinstance(value,str) or not value.strip():raise ValueError('A nonempty local path is required.')
                value=str(Path(value).expanduser().resolve())
            if key=='notes' and not isinstance(value,str):raise ValueError('Notes must be text.')
            case['fields'][key]=value;case['provenance'][key]=dict(source=source,time=time.time())
        # Never retain facts from a previously selected observation.
        if 'observation' in fields:case['evidence']={}
        self.save(case);return case

    def inspect(self,name):
        case=self.load(name);fields=case['fields'];errors=[]
        observation=fields.get('observation')
        if observation:
            try:
                path=Path(observation)
                if {'private_eval','datasets'} & set(path.parts):raise ValueError('Use a canonical observation directory, not evaluation labels or a source dataset.')
                metadata=path/'metadata.json';observed=path/'observed.npz'
                info=json.loads(metadata.read_text())
                if not observed.is_file() or sha(observed)!=info['observed_sha256']:raise ValueError('Observation checksum mismatch.')
                if type(info.get('layer_index')) is not int or info['layer_index']<0:raise ValueError('Missing layer_index in observation metadata.')
                from kvrecon.observation import load_observation
                info,codes,_,positions,prefix=load_observation(path,distance_space='auto')
                if codes.shape[1]!=sum(q['width'] for q in info['quantization']):raise ValueError('KV width disagrees with quantization metadata.')
                evidence={k:info[k] for k in ('model_sha256','layer_index','quantization','observed_sha256')}
                evidence.update(metadata_sha256=sha(metadata),source=str(metadata),inspected_at=time.time(),
                                target_positions=len(positions),public_prefix_tokens=len(prefix),kv_width=codes.shape[1],
                                first_position=int(positions[0]),last_position=int(positions[-1]))
                case['evidence']=evidence
                # Match only explicit, existing reference fingerprints. Never infer a model from its filename.
                references={'gemma4-e2b':'af1082986639ecde7db95d91be6fe54f8b6b458104734c5bafc204e69d6852dc'}
                reference=HERE/'fixtures/gemma3_greedy3/observation/metadata.json'
                if reference.is_file():references['gemma3-1b']=json.loads(reference.read_text())['model_sha256']
                matches=[key for key,value in references.items() if value==info['model_sha256']]
                if fields.get('model_name') in (None,'unknown') and len(matches)==1:
                    fields['model_name']=matches[0];case['provenance']['model_name']=dict(source='observation SHA256 matched reference',time=time.time())
                elif fields.get('model_name') in references and references[fields['model_name']]!=info['model_sha256']:
                    errors.append('Selected model conflicts with the observation model SHA256.')
            except (OSError,ValueError,KeyError,TypeError) as e:
                case['evidence']={};errors.append(str(e))
        missing=[]
        if not observation:missing.append('observation')
        if fields.get('model_name') not in MODELS:missing.append('model_name')
        if 'collision_plus_cpa' in fields['methods']:
            if not fields.get('calibration'):missing.append('calibration')
            elif not Path(fields['calibration']).is_file():errors.append('Calibration file does not exist.')
        # Check required host assets without opening any evaluation labels.
        model=fields.get('model_name')
        if model in MODELS:
            config=HERE/'configs'/MODELS[model];settings=json.loads(config.read_text())
            for key in ('model','manifest','tokenizer')+ (('embeddings',) if 'guided' in fields['methods'] else ()):
                if not (config.parent/settings[key]).resolve().is_file():errors.append(f'Missing {key} asset for {model}.')
        case['readiness']=dict(ready=not missing and not errors,missing=missing,errors=errors,checked_at=time.time())
        self.save(case);return case

    def question(self,case):
        ready=case['readiness']
        if ready['errors']:return 'The case needs attention: '+ready['errors'][0]+' Please provide corrected capture information.'
        if 'observation' in ready['missing']:return 'Where is the canonical observation directory containing metadata.json and observed.npz? I will read its model, layer and quantization metadata.'
        if 'model_name' in ready['missing']:return 'Which model produced this capture: Gemma 4 E2B or Gemma 3 1B? If unknown, say so; I will record it and need capture metadata before recovery.'
        if 'calibration' in ready['missing']:return 'Official Collision+ CPA needs an independent known-text calibration JSON. Where is it? The shared-RMSE adaptation does not require this file.'
        labels=', '.join(METHODS[x] for x in case['fields']['methods'])
        return f"{case['name']} is ready. Methods: {labels}. Say ‘start recovery’ to run the fixed templates. Collection must be paused first."

    def plan(self,name):
        case=self.inspect(name)
        if not case['readiness']['ready']:raise ValueError(self.question(case))
        fields=case['fields'];plans=[]
        for method in fields['methods']:
            args=['--observation',fields['observation'],'--model-name',fields['model_name'],'--method',method,
                  '--epsilon',str(case['defaults']['epsilon']),'--distance-metric',case['defaults']['distance_metric'],
                  '--distance-space',case['defaults']['distance_space'],'--batch-size',str(case['defaults']['batch_size'])]
            for key in ('max_positions','max_queries'):
                if fields.get(key):args += ['--'+key.replace('_','-'),str(fields[key])]
            if method=='collision_plus_cpa':args += ['--calibration',fields['calibration']]
            plans.append(dict(method=method,label=METHODS[method],args=args))
        return case,plans
