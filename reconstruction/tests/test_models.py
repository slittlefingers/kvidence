import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.cli import build_parser
from kvrecon.config import MODEL_CONFIGS,load_model_config,model_dimensions
from kvrecon.common import quantization,kv_dtype,dequantize
from kvrecon.observation import save_observation,load_observation
from kvrecon.search import recover_position,SearchConfig
from test_core import FakeBackend


class ModelTests(unittest.TestCase):
    def test_named_selection(self):
        for name in MODEL_CONFIGS:
            args=build_parser().parse_args(['run','--model-name',name,'--observation','obs','--output','out'])
            self.assertEqual(args.model_name,name)
        self.assertIsNone(build_parser().parse_args(['run','--observation','obs','--output','out']).model_name)

    def test_gemma3_dimensions_and_float_kv(self):
        profile=load_model_config(MODEL_CONFIGS['gemma3-1b'])
        sections=json.loads(profile['manifest'].read_text())['model_sections']
        dims=model_dimensions(sections)
        self.assertEqual(dims,dict(vocab_size=262144,embedding_dim=1152,batch_capacity=128,context_capacity=1280))
        self.assertEqual(profile['default_layer'],5)
        self.assertNotIn('tf_lite_per_layer_embedder',sections)
        quant=quantization(sections,5)
        self.assertEqual(kv_dtype(quant),np.dtype('<f4'))
        values=np.full((1,512),1.2345,dtype=np.float32)
        np.testing.assert_array_equal(dequantize(values,quant),values.astype(np.float64))
        with tempfile.TemporaryDirectory() as d:
            save_observation(Path(d)/'obs',values,[1],[2],5,sections,'model','test')
            meta,stored,restored,positions,prefix=load_observation(Path(d)/'obs')
            np.testing.assert_array_equal(stored,values);np.testing.assert_array_equal(restored,values.astype(np.float64))

    def test_accepted_float_values_are_not_truncated(self):
        b=FakeBackend([[1.2345],[8.]])
        result=recover_position(b,np.array([1.2345]),np.arange(2),0,SearchConfig(method='collision_plus'))
        self.assertEqual(result['accepted_codes'],[1.2345])


if __name__=='__main__':unittest.main()
