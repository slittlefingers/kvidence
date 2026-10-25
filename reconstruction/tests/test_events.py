import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_core import FakeBackend
from kvrecon.engine import reconstruct
from kvrecon.search import SearchConfig
from kvrecon.dashboard import DashboardState
from kvrecon.events import EventLog


class EventTests(unittest.TestCase):
    def test_telemetry_preserves_complete_stage_selection_and_commit(self):
        config=SearchConfig(distance_metric="l2",epsilon=4.,method='guided',probability_seeds=2,random_seeds=1,batch_size=1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=FakeBackend([[2.],[0.],[9.]]);b=FakeBackend([[2.],[0.],[9.]])
            plain=reconstruct(a,np.array([[0.]]),[1],[2],config,root/'plain')
            logged=reconstruct(b,np.array([[0.]]),[1],[2],config,root/'logged',live_events=True)
            events=[json.loads(x) for x in (root/'logged/events.jsonl').read_text().splitlines()]
            self.assertEqual(a.calls,b.calls);self.assertEqual(a.commits,b.commits)
            self.assertEqual(plain['recovered_token_ids'],logged['recovered_token_ids'])
            self.assertEqual([e['type'] for e in events],['start','position_start','batch','batch','position_end','finish'])
            self.assertEqual(events[2]['passing_ids'],[0])
            self.assertEqual(events[4]['accepted_token_id'],1)  # provisional first pass is not final selection
            state=DashboardState('test','test',lambda ids:''.join(map(str,ids)))
            for event in events:state.apply(event)
            self.assertEqual(state.data['total_queries'],2)
            self.assertEqual(state.data['positions'][0]['accepted_batch'],2)
            self.assertEqual(state.data['recovered_text'],'1')
            self.assertIsNone(state.data['evaluation'])
            self.assertFalse((root/'plain/events.jsonl').exists())

    def test_failed_commit_is_not_displayed_as_recovered(self):
        class Divergent(FakeBackend):
            def commit(self,token):
                super().commit(token);return np.array([[20.]])
        with tempfile.TemporaryDirectory() as folder:
            reconstruct(Divergent([[0.],[9.]]),np.array([[0.]]),[1],[2],
                SearchConfig(distance_metric="l2",epsilon=4.,method='collision_plus'),folder,live_events=True)
            state=DashboardState('a','a')
            for line in (Path(folder)/'events.jsonl').read_text().splitlines():state.apply(json.loads(line))
            self.assertEqual(state.data['status'],'commit_mismatch')
            self.assertEqual(state.data['accepted'],0)
            self.assertFalse(state.data['positions'][0]['committed'])

    def test_exception_records_terminal_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'events.jsonl'
            with self.assertRaises(KeyboardInterrupt):
                with EventLog(path) as emit:
                    emit(dict(type='start'));raise KeyboardInterrupt()
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(rows[-1]['status'],'interrupted')
            with self.assertRaises(FileExistsError):
                with EventLog(path):pass

    def test_queries_accumulate_across_positions(self):
        state=DashboardState('a','a')
        for index in range(2):
            state.apply(dict(type='position_start',index=index,position=index+4))
            for count in (2,4):state.apply(dict(type='batch',token_ids=[1,2],queries=count))
        self.assertEqual(state.data['total_queries'],8)

if __name__=='__main__':unittest.main()
