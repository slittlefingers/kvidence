"""Read-only playback of archived, real-NPU text recovery experiments."""
import json
from pathlib import Path
import time
from kvrecon.dashboard import DashboardState

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results/attack_dashboard_validation_20261020'
RUNS=(('guided','Ridge-guided'),('collision_plus_cpa','Collision+ CPA'))


class ExperimentDemo:
    def __init__(self,archive=ARCHIVE,method='guided'):
        from tokenizers import Tokenizer
        self.runs=[];self.speed=1.;self.index=0
        for key,label in RUNS:
            path=Path(archive)/key
            protocol=json.loads((path/'protocol.json').read_text())
            tokenizer_path=protocol['model_config']['tokenizer']
            tokenizer=Tokenizer.from_file(tokenizer_path)
            self.runs.append(dict(key=key,label=label,path=path,protocol=protocol,
                events=[json.loads(row) for row in (path/'events.jsonl').read_text().splitlines()],
                result=json.loads((path/'result.json').read_text()),
                evaluation=json.loads((path/'evaluation.json').read_text()),tokenizer=tokenizer))
        identities={(r['protocol']['model_sha256'],r['protocol']['observation_metadata']['observed_sha256']) for r in self.runs}
        if len(identities)!=1:raise ValueError('Demo records do not share the same model and observation.')
        self.select(method)

    def select(self,method):
        matches=[i for i,r in enumerate(self.runs) if r['key']==method]
        if not matches:raise ValueError('No archived demo for that method. Choose guided or collision_plus_cpa.')
        self.index=matches[0];self.reset()

    def reset(self):
        run=self.runs[self.index]
        self.state=DashboardState('experiment_demo',run['label'],lambda ids:run['tokenizer'].decode(ids,skip_special_tokens=False))
        self.elapsed=0.;self.cursor=0;self.last_clock=None;self.playing=True

    def advance(self,now=None):
        now=time.monotonic() if now is None else now
        if self.last_clock is not None and self.playing:self.elapsed+=max(0,now-self.last_clock)*self.speed
        self.last_clock=now
        events=self.runs[self.index]['events'];start=events[0]['timestamp']
        while self.cursor<len(events) and events[self.cursor]['timestamp']-start<=self.elapsed:
            self.state.apply(events[self.cursor]);self.cursor+=1
        if self.cursor==len(events):self.playing=False

    def finish(self):
        events=self.runs[self.index]['events']
        self.elapsed=max(0,events[-1]['timestamp']-events[0]['timestamp']);self.advance(self.last_clock or 0)

    def key(self,key):
        if key in ('q','\x1b','\x03'):return False
        if key==' ':self.playing=not self.playing
        elif key=='n':self.index=(self.index+1)%len(self.runs);self.reset()
        elif key=='r':self.reset()
        elif key in ('+','='):self.speed=min(8.,self.speed*2)
        elif key=='-':self.speed=max(.25,self.speed/2)
        elif key=='e':self.finish()
        return True

    def draw(self,put,rule,height,width,wrap):
        run=self.runs[self.index];data=self.state.data
        put(1,'KVidence  /  Text recovery demo',1,True)
        if height<24:
            put(4,'Resize to at least 24 rows for the experiment view.',7)
            put(height-2,'q / Esc back',6)
            return
        put(2,'Recorded Pixel 10 / Gemma 4 E2B / Layer 5',4)
        put(4,run['label'],6,True)
        phase='Finished' if self.cursor==len(run['events']) else 'Playing' if self.playing else 'Paused'
        put(5,f'RECORDED REPLAY  ·  {phase}  ·  {self.speed:g}x  ·  no NPU use',7)
        rule(7)
        put(8,'ARCHIVED RESULTS  ·  independent evaluation',1,True)
        put(9,f"{'Method':<15} {'Correct':>7} {'Queries':>8} {'Online':>8}",4)
        for y,row in enumerate(self.runs,10):
            evaluation=row['evaluation'];result=row['result']
            correct=f"{evaluation['correct_ids']}/{evaluation['truth_tokens']}"
            put(y,f"{row['label']:<15} {correct:>7} {evaluation['total_queries']:>8,} {result['online_seconds']:>7.2f}s",7)
        put(12,'Exhaustive       Not recorded for this sample',4)
        put(14,'RECOVERED TEXT',1,True)
        current=data.get('current') or {};batch=data.get('batch') or {}
        put(15,f"Accepted {data['accepted']}/{data.get('planned_positions',21)}  ·  position {current.get('position','—')}  ·  {data['total_queries']:,} candidates",4)
        text=data['recovered_text'] or 'Waiting for the first accepted token…'
        for y,line in enumerate(wrap(text,width-1),16):
            if y>=height-6:break
            put(y,line,2 if data['recovered_text'] else 4)
        if height>=28 and batch:
            put(height-9,f"Batch {batch.get('batch','—')}  ·  {batch.get('phase','—')}  ·  {len(batch.get('token_ids',[]))} candidates",6)
        note='Recorded trial; acceptance rules differ.'
        if run['key']=='collision_plus_cpa':note='Pilot CPA calibration; 1 false acceptance. Accepted != verified.'
        put(height-6,note,4)
        rule(height-5)
        put(height-3,'Space pause  ·  n next method  ·  r replay',6)
        put(height-2,'+/- speed  ·  e end  ·  q / Esc back',4)
        put(height-1,'Playback only. Collection and case metadata are unchanged.',4)
