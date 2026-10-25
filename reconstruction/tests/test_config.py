import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.cli import build_parser, resolve_search_defaults
from kvrecon.config import DEFAULT_MODEL_CONFIG, load_model_config, model_dimensions, load_vocabulary
from kvrecon.common import MODEL, DEFAULT_MANIFEST, DEFAULT_EMBEDDINGS, DEFAULT_TOKENIZER
from kvrecon.search import SearchConfig,recover_position
from test_core import FakeBackend


class ConfigurationTests(unittest.TestCase):
    def test_default_assets_and_search_parameters_unchanged(self):
        p=load_model_config(DEFAULT_MODEL_CONFIG)
        self.assertEqual(p['model'],MODEL);self.assertEqual(p['manifest'],DEFAULT_MANIFEST)
        self.assertEqual(p['embeddings'],DEFAULT_EMBEDDINGS);self.assertEqual(p['tokenizer'],DEFAULT_TOKENIZER)
        a=build_parser().parse_args(['run','--observation','obs','--output','out'])
        resolve_search_defaults(a)
        self.assertEqual((a.initial_candidates,a.prediction_candidates,a.proposals,a.batch_size),(200,50,20,128))
        dims=model_dimensions(json.loads(DEFAULT_MANIFEST.read_text())['model_sections'])
        self.assertEqual(dims,dict(vocab_size=262144,embedding_dim=1536,batch_capacity=128,context_capacity=4096))

    def test_dimensions_follow_manifest_and_selected_signature(self):
        sections=json.loads(DEFAULT_MANIFEST.read_text())['model_sections']
        sig=sections['tf_lite_prefill_decode']['signatures']
        next(t for t in sig['decode']['outputs'] if t['name']=='logits')['shape'][-1]=32000
        next(t for t in sig['decode']['inputs'] if t['name']=='embeddings')['shape'][-1]=2048
        dims=model_dimensions(sections,'verify')
        self.assertEqual((dims['vocab_size'],dims['embedding_dim'],dims['batch_capacity']),(32000,2048,4))

    def test_invalid_counts_rejected(self):
        for probability,random in [(51,-1),(-1,201),(0,0)]:
            with self.assertRaises(ValueError):SearchConfig(probability_seeds=probability,random_seeds=random).validate()

    def test_zero_prediction_seeds_and_noncontiguous_pool(self):
        b=FakeBackend(np.full((10,1),9.));pool=np.array([1,5,9])
        class Predictor:
            def scores(self,*args):return np.arange(10,dtype=float)
        r=recover_position(b,np.array([0.]),np.arange(10),0,
            SearchConfig(probability_seeds=0,random_seeds=2,proposals=1),Predictor(),pool)
        self.assertEqual(r['effective_probability_seeds'],0)
        self.assertEqual(r['effective_random_seeds'],2)
        self.assertEqual(sorted(r['queried_token_ids']),[1,5,9])
        self.assertEqual(r['candidate_pool_size'],3)

    def test_pool_filter_is_applied_before_probability_top_k(self):
        b=FakeBackend(np.full((10,1),9.))
        r=recover_position(b,np.array([0.]),np.arange(10),0,
            SearchConfig(probability_seeds=2,random_seeds=1),candidate_ids=[9,5,1])
        self.assertEqual(r['queried_token_ids'][:2],[1,5])
        self.assertEqual(r['queried_token_ids'][2:],[9])

    def test_vocab_is_not_renumbered_and_bad_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'vocab.json';path.write_text('{"token_ids":[9,1,5]}')
            ids,meta=load_vocabulary(path,10)
            self.assertEqual(ids.tolist(),[1,5,9]);self.assertFalse(meta['ids_remapped'])
            for bad in [[1,1],[10],[-1],[1.5],[],[True]]:
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):load_vocabulary(path,10)

    def test_batch_bound_follows_backend_capacity(self):
        b=FakeBackend([[0.],[9.]]);b.batch_capacity=4
        with self.assertRaisesRegex(ValueError,'compiled'):
            recover_position(b,np.array([0.]),np.arange(2),0,SearchConfig(method='collision_plus',batch_size=5))


if __name__=='__main__':unittest.main()
