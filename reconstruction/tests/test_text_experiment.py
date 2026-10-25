import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from run_text_experiment import options,create_experiment,check_idle,execute

class ExperimentTests(unittest.TestCase):
    def test_method_selection(self):
        args=options(['plaintext_chat','0076','--methods','guided,exhaustive','--max-queries','200'])
        self.assertEqual(args.methods,['guided','exhaustive']);self.assertEqual(args.max_queries,200)
        self.assertIsNone(options([]).methods)
        for methods in ('guided,guided','unknown','collision_plus_cpa'):
            with self.assertRaises(ValueError):options(['--methods',methods])

    def test_fresh_directories_and_scope(self):
        class Catalog:
            def specification(self,dataset,sample):return dict(dataset=dataset,sample_ids=[sample] if sample else ['0001','0002'])
        with tempfile.TemporaryDirectory() as root:
            args=options(['plaintext_chat','0076','--methods','guided'])
            a=create_experiment(args,Catalog(),root);b=create_experiment(args,Catalog(),root)
            self.assertNotEqual(a,b)
            plan=json.loads((a/'experiment.json').read_text())
            self.assertEqual(plan['methods'],['guided']);self.assertEqual(plan['sample_ids'],['0076'])

    def test_sequential_execution_and_evaluation_after_search(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);spec=dict(sample_ids=['0076'],methods=['guided'],model_name='gemma4-e2b',dataset='plaintext_chat')
            (folder/'experiment.json').write_text(json.dumps(spec));calls=[]
            def launch(argv,**kwargs):
                calls.append('search')
                output=Path(argv[argv.index('--run-dir')+1])/'guided';output.mkdir(parents=True)
                (output/'result.json').write_text(json.dumps(dict(status='complete',config={'method':'guided'},recovered_token_ids=[5],online_seconds=1)))
                class Worker:
                    def wait(self):calls.append('search finished');return 0
                return Worker()
            def evaluate(argv,**kwargs):
                calls.append('evaluate' if 'evaluate' in argv else 'decode');return 0
            with patch('run_text_experiment.check_idle'),patch('run_text_experiment.CaseStore'),patch('run_text_experiment.prepare_sample',return_value=(folder/'observation',folder/'private_eval/truth.json')),patch('run_text_experiment.subprocess.Popen',side_effect=launch),patch('run_text_experiment.subprocess.call',side_effect=evaluate):
                execute(folder)
            self.assertEqual(calls,['search','search finished','evaluate','decode'])
            self.assertEqual(json.loads((folder/'experiment.json').read_text())['state'],'complete')
            rows=json.loads((folder/'summary.json').read_text())
            self.assertEqual([r['method'] for r in rows],['guided'])
            self.assertTrue((folder/'summary.csv').exists())

    def test_collection_guard(self):
        for state in ({'active':True},{'mode':'pausing'},{'cooldown':20}):
            with patch('terminal.api',return_value=state),self.assertRaises(ValueError):check_idle()
        with patch('terminal.api',return_value={'active':False,'mode':'paused','cooldown':0}):check_idle()

    def test_blocked_worker_persists_failure_without_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'experiment.json').write_text(json.dumps(dict(sample_ids=['0001'])))
            with patch('run_text_experiment.check_idle',side_effect=ValueError('pause first')),patch('run_text_experiment.prepare_sample') as prepare:
                with self.assertRaises(ValueError):execute(folder)
                prepare.assert_not_called()
            self.assertEqual(json.loads((folder/'experiment.json').read_text())['state'],'failed')
            self.assertEqual(json.loads((folder/'summary.json').read_text()),[])

if __name__=='__main__':unittest.main()
