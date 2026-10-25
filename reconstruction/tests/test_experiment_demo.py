import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP))
from experiment_demo import ExperimentDemo


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        events=[dict(type='start',timestamp=0,config={},planned_positions=1,target_positions=1),
            dict(type='position_start',timestamp=.5,index=0,position=4),
            dict(type='batch',timestamp=1,index=0,batch=1,phase='ridge',queries=1,token_ids=[10]),
            dict(type='position_end',timestamp=2,index=0,position=4,accepted_token_id=10,committed=True,
                 stages=[dict(passing_ids=[10],phase='ridge')]),
            dict(type='finish',timestamp=3,result=dict(status='complete',online_seconds=3))]
        for method in ('guided','collision_plus_cpa'):
            path=self.root/method;path.mkdir()
            values={'protocol.json':dict(model_sha256='model',observation_metadata=dict(observed_sha256='obs'),model_config=dict(tokenizer='tokenizer.json')),
                'result.json':dict(status='complete',online_seconds=3),
                'evaluation.json':dict(correct_ids=1 if method=='guided' else 0,truth_tokens=1,total_queries=1)}
            for name,value in values.items():(path/name).write_text(json.dumps(value))
            (path/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
        class Tokenizer:
            def decode(self,ids,**kwargs):return ''.join(str(x) for x in ids)
        self.patch=patch('tokenizers.Tokenizer.from_file',return_value=Tokenizer());self.patch.start()
    def tearDown(self):self.patch.stop();self.tmp.cleanup()

    def test_progressive_replay_and_pause(self):
        demo=ExperimentDemo(self.root);demo.advance(0);demo.advance(1)
        self.assertEqual(demo.state.data['accepted'],0)
        self.assertEqual(demo.state.data['total_queries'],1)
        demo.key(' ');demo.advance(20);self.assertEqual(demo.cursor,3)
        demo.key(' ');demo.advance(21)
        self.assertEqual(demo.state.data['recovered_text'],'10')
        demo.advance(23);self.assertFalse(demo.playing)

    def test_switch_restart_and_skip_to_end(self):
        demo=ExperimentDemo(self.root);demo.key('e')
        self.assertEqual(demo.state.data['status'],'complete')
        demo.key('n');self.assertEqual(demo.runs[demo.index]['key'],'collision_plus_cpa')
        self.assertEqual(demo.cursor,0)
        demo.key('+');self.assertEqual(demo.speed,2)
        demo.key('r');self.assertEqual(demo.cursor,0)
        self.assertFalse(demo.key('q'))

    def test_summary_does_not_fabricate_exhaustive_results(self):
        demo=ExperimentDemo(self.root);drawn=[]
        demo.draw(lambda y,text,*a,**k:drawn.append(text),lambda y:None,30,100,lambda text,width:[text])
        self.assertTrue(any('Not recorded' in text for text in drawn))
        self.assertTrue(any('0/1' in text for text in drawn))
        self.assertTrue(any('RECORDED REPLAY' in text for text in drawn))
