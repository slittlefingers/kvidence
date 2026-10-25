"""Bounded dashboard state from public reconstruction telemetry (no truth access)."""
import time


class DashboardState:
    def __init__(self, run_id, label, decode=None):
        self.decode=decode or (lambda ids: '')
        self.data=dict(schema=1,run_id=run_id,label=label,status='waiting',positions=[],
            total_queries=0,elapsed_seconds=0,accepted=0,current=None,batch=None,
            recovered_text='',evaluation=None)
        self.first_time=None
        self.position_queries=0

    def apply(self,e):
        d=self.data
        if self.first_time is None:self.first_time=e.get('timestamp',time.time())
        d['event_time']=e.get('timestamp',time.time())
        d['elapsed_seconds']=max(0,d['event_time']-self.first_time)
        kind=e['type']
        if kind=='start':
            d.update(status='running',config=e['config'],planned_positions=e['planned_positions'],
                target_positions=e['target_positions'])
        elif kind=='position_start':
            d.update(current=dict(index=e['index'],position=e['position']),batch=None)
            self.position_queries=0
        elif kind=='batch':
            d['total_queries']+=e['queries']-self.position_queries
            self.position_queries=e['queries']
            batch=dict(e)
            batch['pieces']=[self.decode([t]) for t in e['token_ids']]
            d['batch']=batch
        elif kind=='position_end':
            row={k:v for k,v in e.items() if k not in ('stages','type')}
            accepted=e['accepted_token_id']
            row['piece']=self.decode([accepted]) if accepted is not None else ''
            row['accepted_batch']=next((i+1 for i,s in enumerate(e['stages']) if accepted is not None and accepted in s['passing_ids']),None)
            row['accepted_phase']=next((s['phase'] for s in e['stages'] if accepted is not None and accepted in s['passing_ids']),None)
            d['positions'].append(row)
            ids=[r['accepted_token_id'] for r in d['positions'] if r['committed']]
            d.update(accepted=len(ids),recovered_text=self.decode(ids))
        elif kind=='finish':
            d.update(status=e['result']['status'],result=e['result'])
            d['elapsed_seconds']=e['result']['online_seconds']
        elif kind=='error':
            d.update(status=e['status'],error=e.get('error',''))

    def snapshot(self):
        return dict(self.data,relay_time=time.time())
