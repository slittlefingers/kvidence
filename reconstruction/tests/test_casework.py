import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP))
import casework
from casework import CaseStore
from case_agent import CaseAgent
from case_commands import CaseCommands
from kvrecon.observation import save_canonical_observation
from kvrecon.search import SearchConfig,recover_position
from kvrecon.engine import reconstruct
from test_core import FakeBackend


class CaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.store=CaseStore(self.root/'cases')
        app=self.root/'app';(app/'configs').mkdir(parents=True)
        config={key:key+'.bin' for key in ('model','manifest','tokenizer','embeddings')}
        for value in config.values():(app/'configs'/value).write_bytes(b'asset')
        (app/'configs/gemma4_e2b.json').write_text(json.dumps(config))
        self.patch=patch.object(casework,'HERE',app);self.patch.start()
        self.observation=self.root/'observation'
        save_canonical_observation(self.observation,np.array([[1,2]],dtype=np.int8),[1],[2],4,
            [dict(width=2,scale=.1,zero_point=0,dtype='int8')],
            'af1082986639ecde7db95d91be6fe54f8b6b458104734c5bafc204e69d6852dc',{})
    def tearDown(self):self.patch.stop();self.temp.cleanup()

    def test_unknown_information_and_path_isolation(self):
        self.store.create('case1');self.store.update('case1',{'model_name':'unknown'})
        case=self.store.inspect('case1')
        self.assertIn('model_name',case['readiness']['missing']);self.assertFalse(case['readiness']['ready'])
        with self.assertRaises(ValueError):self.store.create('../escape')
        with self.assertRaises(FileExistsError):self.store.create('case1')

    def test_fixed_commands_share_validation_and_provenance(self):
        commands=CaseCommands(self.store)
        commands.execute(['new','case1'])
        commands.execute(['set','model_name','unknown'])
        commands.execute(['set','methods','guided,exhaustive'])
        commands.execute(['set','observation',str(self.observation)])
        case=self.store.load('case1')
        self.assertEqual(case['fields']['methods'],['guided','exhaustive'])
        self.assertEqual(case['provenance']['observation']['source'],'fixed command')
        self.assertTrue(commands.execute(['check'])['ready'])
        self.assertIn('guided',commands.execute(['plan'])['reply'])
        self.assertEqual(commands.execute(['run'])['intent'],'run')
        self.assertEqual(commands.execute(['open','case1'])['case'],'case1')

    def test_explicit_case_commands_and_json_import(self):
        commands=CaseCommands(self.store);commands.execute(['new','case1'])
        commands.execute(['set','case1','max_queries','100'])
        file=self.root/'fields.json';file.write_text(json.dumps({'notes':'imported','max_positions':2}))
        commands.execute(['import','case1',str(file)])
        self.assertEqual(self.store.load('case1')['fields']['max_queries'],100)
        self.assertEqual(self.store.load('case1')['fields']['notes'],'imported')
        with self.assertRaises(ValueError):commands.execute(['set','case1','shell','rm -rf'])
        with self.assertRaises(ValueError):commands.execute(['set','case1','max_queries','-1'])
        self.assertIn('case1',commands.execute(['list'])['reply'])

    def test_evidence_infers_model_and_templates(self):
        self.store.create('case1');self.store.update('case1',{'observation':str(self.observation)})
        case,plans=self.store.plan('case1')
        self.assertEqual(case['fields']['model_name'],'gemma4-e2b')
        self.assertEqual(case['evidence']['public_prefix_tokens'],1)
        self.assertEqual([p['method'] for p in plans],['guided','collision_plus','exhaustive'])
        self.assertIn('adaptation',plans[1]['label'])
        self.assertNotIn('shell',case['fields'])

    def test_changed_observation_rejected(self):
        self.store.create('case1');self.store.update('case1',{'observation':str(self.observation)})
        (self.observation/'observed.npz').write_bytes(b'corrupt')
        case=self.store.inspect('case1');self.assertFalse(case['readiness']['ready'])
        self.assertIn('checksum',case['readiness']['errors'][0])

    def test_official_cpa_asks_for_calibration(self):
        self.store.create('case1');self.store.update('case1',{'observation':str(self.observation),'methods':['collision_plus_cpa']})
        case=self.store.inspect('case1');self.assertIn('calibration',case['readiness']['missing'])
        with self.assertRaises(ValueError):self.store.plan('case1')

    def agent(self,plan):
        class Model:
            def infer(self,*args):return dict(plan=plan)
        return CaseAgent(self.store,Model())

    def test_conversation_saved_and_no_guessed_model(self):
        agent=self.agent(dict(intent='create',case_name='case1',fields={'model_name':'gemma4-e2b'}))
        result=agent.handle('create case1')
        self.assertFalse(result['ready']);self.assertIsNone(self.store.load('case1')['fields']['model_name'])
        conversation=(self.store.path('case1')/'conversation.jsonl').read_text().splitlines()
        self.assertEqual(len(conversation),2)

    def test_method_selection_or_negative_request_cannot_start(self):
        self.store.create('case1')
        agent=self.agent(dict(intent='run',case_name='case1',fields={}))
        agent.current='case1'
        for text in ('use ridge and exhaustive','不要开始恢复','do not run recovery'):
            self.assertEqual(agent.handle(text)['intent'],'update')

    def test_arbitrary_shell_and_fabricated_paths_rejected(self):
        self.store.create('case1')
        agent=self.agent(dict(intent='shell',case_name='case1',fields={}))
        with self.assertRaises(ValueError):agent.handle('run shell')
        agent=self.agent(dict(intent='update',case_name='case1',fields={'observation':'/invented/path'}))
        with self.assertRaises(ValueError):agent.handle('I have some evidence')

    def run_supervisor(self,change_after_second=False):
        import run_case
        self.store.create('case1');self.store.update('case1',{'observation':str(self.observation)})
        output=self.root/'case_run';calls=[]
        def launch(argv,**kwargs):
            self.assertIsInstance(argv,list)
            method=argv[argv.index('--method')+1];calls.append(method)
            target=Path(argv[argv.index('--output')+1]);self.assertFalse(target.exists());target.mkdir()
            (target/'result.json').write_text(json.dumps(dict(status='complete',recovered_token_ids=[1],online_seconds=.1)))
            (target/'trace.jsonl').write_text(json.dumps(dict(queries=2))+'\n')
            if change_after_second and len(calls)==2:(self.observation/'metadata.json').write_text('{}')
            class Worker:
                def wait(self,**kwargs):return 0
                def poll(self):return 0
            return Worker()
        with patch.object(run_case,'CaseStore',return_value=self.store),patch('terminal.api',return_value=dict(active=False,mode='paused',cooldown=0)),patch.object(sys,'argv',['run_case.py','case1','--run-dir',str(output)]),patch.object(run_case.subprocess,'Popen',side_effect=launch):
            if change_after_second:
                with self.assertRaisesRegex(ValueError,'changed'):run_case.main()
            else:run_case.main()
        return calls,json.loads((output/'plan.json').read_text())

    def test_case_supervisor_runs_all_registered_templates(self):
        calls,manifest=self.run_supervisor()
        self.assertEqual(calls,['guided','collision_plus','exhaustive'])
        self.assertEqual(manifest['status'],'complete')
        self.assertEqual([x['total_queries'] for x in manifest['results']],[2,2,2])

    def test_case_supervisor_stops_if_evidence_changes(self):
        calls,manifest=self.run_supervisor(change_after_second=True)
        self.assertEqual(calls,['guided','collision_plus'])
        self.assertEqual(manifest['status'],'failed')


class ExhaustiveTests(unittest.TestCase):
    def test_token_id_order_no_surrogate(self):
        backend=FakeBackend([[8.],[8.],[0.],[8.]])
        row=recover_position(backend,[0.],[3,2,1,0],0,SearchConfig(method='exhaustive',epsilon=.1,batch_size=2))
        self.assertEqual(backend.calls,[[0,1],[2,3]])
        self.assertEqual(row['accepted_token_id'],2)
        self.assertFalse(row['uniqueness_proven'])
        self.assertTrue(all(s['phase']=='exhaustive' for s in row['stages']))

    def test_engine_never_requests_probability_ranking(self):
        class Backend(FakeBackend):
            def probabilities(self):raise AssertionError('Probability ranking must be unused')
        with tempfile.TemporaryDirectory() as folder:
            result=reconstruct(Backend([[8.],[0.]]),[[0.]],[1],[2],SearchConfig(method='exhaustive',epsilon=.1,batch_size=1),folder)
            self.assertEqual(result['recovered_token_ids'],[1])

    def test_budget_does_not_accept_nearest(self):
        row=recover_position(FakeBackend([[8.],[0.]]),[0.],[0,1],0,
             SearchConfig(method='exhaustive',epsilon=.1,batch_size=1,max_queries=1))
        self.assertIsNone(row['accepted_token_id']);self.assertEqual(row['stop_reason'],'resource_budget')


if __name__=='__main__':unittest.main()
