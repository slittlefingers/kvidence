"""Shadow Chat experiment scopes and fingerprint-matched archived results."""
import json
from pathlib import Path
from casework import HERE,sha,write

DATA=HERE.parent/'datasets/shadow_chat'
DATASETS={'plaintext_chat':'plaintext_chat_100','encrypted_chat':'encrypted_chat_100'}
DEFAULT_METHODS=['guided','collision_plus','exhaustive']


def dataset_name(name):
    aliases={'plaintext':'plaintext_chat','encrypted':'encrypted_chat',**{v:k for k,v in DATASETS.items()}}
    name=aliases.get(name,name)
    if name not in DATASETS:raise ValueError('Choose plaintext_chat or encrypted_chat.')
    return name


class ShadowCatalog:
    def __init__(self,data=DATA,results=HERE/'results'):
        self.data=Path(data);self.datasets={};self.records={}
        for key,folder in DATASETS.items():
            path=self.data/folder/'manifest.json';manifest=json.loads(path.read_text())
            self.datasets[key]=dict(root=path.parent,manifest=manifest,manifest_sha256=sha(path))
        fingerprints={}
        for key,entry in self.datasets.items():
            for sample in entry['manifest']['samples']:
                fingerprints[(entry['manifest']['model_sha256'],sample['snapshot_sha256'])]=(key,sample['id'])
        # Discover outputs by evidence identity, never by a suggestive directory name.
        for path in Path(results).rglob('protocol.json'):
            try:
                if not (path.parent/'result.json').is_file():continue
                if not any((path.parent/file).is_file() for file in ('events.jsonl','trace.jsonl')):continue
                protocol=json.loads(path.read_text());observation=protocol.get('observation_metadata',{})
                key=fingerprints.get((protocol.get('model_sha256'),observation.get('source',{}).get('snapshot_sha256')))
                method=protocol.get('config',{}).get('method')
                if key and method in ('guided','collision_plus','collision_plus_cpa','exhaustive'):
                    self.records.setdefault(key,[]).append(dict(method=method,path=path.parent,protocol=protocol))
            except (ValueError,OSError):continue

    def samples(self,dataset):return self.datasets[dataset_name(dataset)]['manifest']['samples']

    def archived(self,dataset,sample):
        rows=self.records.get((dataset_name(dataset),sample),[])
        chosen={}
        for row in sorted(rows,key=lambda r:(r['path']/'result.json').stat().st_mtime):chosen[row['method']]=row
        return [chosen[k] for k in ('guided','collision_plus','collision_plus_cpa','exhaustive') if k in chosen]

    def specification(self,dataset,sample=None):
        dataset=dataset_name(dataset);entry=self.datasets[dataset];records=entry['manifest']['samples']
        if sample is not None:
            sample=str(sample).zfill(4);records=[r for r in records if r['id']==sample]
            if not records:raise ValueError('Sample ID is not in the selected dataset.')
        return dict(schema=1,dataset=dataset,dataset_directory=str(entry['root']),
            manifest_sha256=entry['manifest_sha256'],model_name='gemma4-e2b',model_sha256=entry['manifest']['model_sha256'],
            scope='sample' if sample else 'dataset',sample_ids=[r['id'] for r in records],
            sample_fingerprints={r['id']:r['snapshot_sha256'] for r in records},methods=DEFAULT_METHODS,
            layer_index=4,target='supplied user prompt; encrypted input recovery is not AES decryption',
            benchmark_length_policy='User-prompt boundaries come from benchmark input tokenization; target token IDs are withheld from recovery.',
            state='planned')


class ExperimentBrowser:
    def __init__(self,dataset=None,sample=None,catalog=None):
        self.catalog=catalog or ShadowCatalog();self.dataset=dataset_name(dataset or 'plaintext_chat')
        self.scope='sample' if sample else 'dataset';self.message='';self.select_sample(sample)

    def select_sample(self,sample=None):
        rows=self.catalog.samples(self.dataset)
        if sample is None:
            sample=self.catalog.datasets[self.dataset]['manifest'].get('validation_sample_id')
            if not sample:sample=next((r['id'] for r in rows if self.catalog.archived(self.dataset,r['id'])),rows[0]['id'])
        sample=str(sample).zfill(4)
        indices=[i for i,row in enumerate(rows) if row['id']==sample]
        if not indices:raise ValueError('Unknown sample in the selected dataset.')
        self.index=indices[0]

    @property
    def sample(self):return self.catalog.samples(self.dataset)[self.index]

    def specification(self):return self.catalog.specification(self.dataset,self.sample['id'] if self.scope=='sample' else None)

    def draw(self,put,rule,height,width,wrap):
        put(1,'KVidence  /  Shadow Chat experiments',1,True)
        put(2,'1 Plaintext chat  ·  2 Encrypted chat',6)
        if height<24:
            put(5,'Resize to at least 24 rows for experiments.',7);put(height-2,'q / Esc back',6);return
        rows=self.catalog.samples(self.dataset);spec=self.specification()
        put(4,self.dataset,0,True)
        put(5,f"Scope: {spec['scope']}  ·  {len(spec['sample_ids'])} sample(s)  ·  Gemma 4 E2B / layer 5",7)
        rule(7)
        put(8,'ID    Tokens    Archived recovery methods',4)
        available=max(1,height-18);start=max(0,min(self.index-available//2,len(rows)-available))
        for y,index in enumerate(range(start,min(len(rows),start+available)),9):
            row=rows[index];methods=', '.join(r['method'] for r in self.catalog.archived(self.dataset,row['id'])) or 'not recorded'
            count=str(row.get('target_tokens','—'))
            put(y,('› ' if index==self.index else '  ')+f"{row['id']}  {count:>5}    {methods}",6 if index==self.index else 7,index==self.index)
        put(height-8,f"Selected: {self.sample['id']}  ·  a all samples / s selected sample",7)
        put(height-7,'Encrypted means recover the supplied ciphertext, not decrypt AES.' if self.dataset=='encrypted_chat' else 'Only dataset-matched results are shown; missing runs stay unrecorded.',4)
        put(height-6,self.message or 'Enter replay  ·  p save plan  ·  r run selected scope',1)
        rule(height-5)
        put(height-3,'1/2 dataset  ·  ↑↓ sample  ·  PgUp/PgDn jump',6)
        put(height-2,'a all  ·  s single  ·  Enter replay  ·  p plan  ·  r run',6)
        put(height-1,'q / Esc back. Selecting or planning does not start the NPU.',4)
