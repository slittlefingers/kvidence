import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.cli import build_parser,resolve_search_defaults
from kvrecon.comparison import compare,summarize,evaluate_comparison
from kvrecon.common import dump
from kvrecon.observation import save_canonical_observation
from kvrecon.search import SearchConfig,recover_position
from test_core import FakeBackend


class ComparisonTests(unittest.TestCase):
    def test_baseline_ignores_guided_seeds_and_never_fits(self):
        class Forbidden:
            def scores(self,*args):raise AssertionError('Baseline must not fit ridge')
        with patch('numpy.random.default_rng',side_effect=AssertionError('Baseline must not sample')):
            r=recover_position(FakeBackend([[1.],[0.],[9.]]),np.array([0.]),np.arange(3),0,
                SearchConfig(epsilon=4.,method='collision_plus',probability_seeds=0,random_seeds=0,proposals=0),Forbidden())
        self.assertEqual(r['accepted_token_id'],0)  # First passing, not minimum distance.
        self.assertEqual(r['effective_random_seeds'],0)
        self.assertEqual(r['requested_initial_candidates'],0)

    def test_comparison_routing_and_independent_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);q=[dict(width=1,scale=1.,zero_point=0,dtype='float32')]*2
            save_canonical_observation(d/'obs',[[0,0]],[1],[2],4,q,'model',{})
            args=build_parser().parse_args(['compare','--baseline','collision_plus','--observation',str(d/'obs'),'--output',str(d/'pair'),'--repeats','2'])
            resolve_search_defaults(args)
            methods=[]
            def process(command,**kwargs):
                method=command[command.index('--method')+1];methods.append(method)
                self.assertNotIn('--truth',command);self.assertNotIn('auto',command)
                self.assertNotIn('--calibration',command)
                self.assertEqual(command[command.index('--batch-size')+1],'128')
                self.assertEqual(float(command[command.index('--epsilon')+1]),.125)
                folder=Path(command[command.index('--output')+1]);folder.mkdir()
                config=dict(method=method,epsilon=4.,batch_size=128,max_queries=None)
                dump(folder/'protocol.json',dict(config=config,model_sha256='model',manifest_sha256='manifest',
                    model_config={},observation_metadata={},candidate_vocabulary={},source_hashes={}))
                dump(folder/'result.json',dict(config=config,status='complete',recovered_token_ids=[3],
                    target_positions=1,attempted_positions=1,online_seconds=1.,decoder_seconds=.1,public_prefix_seconds=.2))
                row=dict(accepted_token_id=3,accepted_l2=0.,queries=2,queried_token_ids=[3,4],
                    probability_seconds=.1,ranking_seconds=.1,query_transport_seconds=.5)
                (folder/'trace.jsonl').write_text(json.dumps(row)+'\n')
                return SimpleNamespace(returncode=0)
            with patch('kvrecon.cli.private_guard'),patch('kvrecon.comparison.subprocess.run',side_effect=process):
                compare(args,{},'model')
            self.assertEqual(methods,['collision_plus','guided','guided','collision_plus'])
            summary=json.loads((d/'pair/summary.json').read_text())
            self.assertTrue(all(row['evaluation'] is None for row in summary['rows']))
            dump(d/'truth.json',dict(token_ids=[3]))
            evaluate_comparison(SimpleNamespace(comparison=d/'pair',truth=d/'truth.json'))
            summary=json.loads((d/'pair/summary.json').read_text())
            self.assertTrue(all(row['evaluation']['correct_ids']==1 for row in summary['rows']))
            p=d/'pair/001_guided/protocol.json';data=json.loads(p.read_text());data['config']['epsilon']=8;dump(p,data)
            with self.assertRaisesRegex(ValueError,'protocols differ'):summarize(d/'pair')

    def test_prefixless_first_layer_not_silently_compared_to_kv(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);q=[dict(width=1,scale=1.,zero_point=0,dtype='float32')]*2
            save_canonical_observation(d/'obs',[[0,0]],[10],[],0,q,'model',{})
            args=build_parser().parse_args(['compare','--baseline','collision_plus','--batch-size','128','--observation',str(d/'obs'),'--output',str(d/'pair')])
            with patch('kvrecon.cli.private_guard'),patch('kvrecon.comparison.subprocess.run') as process:
                with self.assertRaisesRegex(ValueError,'known prefix'):compare(args,{},'model')
                process.assert_not_called()

    def test_failed_arm_stops_pair_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);q=[dict(width=1,scale=1.,zero_point=0,dtype='float32')]*2
            save_canonical_observation(d/'obs',[[0,0]],[1],[2],4,q,'model',{})
            args=build_parser().parse_args(['compare','--baseline','collision_plus','--batch-size','128','--observation',str(d/'obs'),'--output',str(d/'pair')])
            with patch('kvrecon.cli.private_guard'),patch('kvrecon.comparison.subprocess.run',return_value=SimpleNamespace(returncode=1)) as process:
                with self.assertRaisesRegex(RuntimeError,'failed'):compare(args,{},'model')
                self.assertEqual(process.call_count,1)
            summary=json.loads((d/'pair/summary.json').read_text())
            self.assertEqual(summary['rows'][0]['status'],'failed')


if __name__=='__main__':unittest.main()
