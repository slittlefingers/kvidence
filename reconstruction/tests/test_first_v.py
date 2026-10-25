import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.first_v import build_table,load_table,table_identity,nearest_v,match_observation,check_independence
from kvrecon.observation import save_canonical_observation,load_observation


QUANT=[dict(width=1,scale=1.,zero_point=0,dtype='float32'),dict(width=2,scale=.5,zero_point=0,dtype='float32')]
SPEC=dict(layer=0,quantization=QUANT,prefix=[2],context_independent_by_architecture=True)


class Backend:
    vocab_size=5
    batch_capacity=2
    table=np.array([[0,0],[10,10],[20,20],[30,30],[20,20]],dtype=np.float32)
    def reset(self,prefix):self.position=len(prefix)
    def query(self,ids):return np.column_stack([np.full(len(ids),self.position,dtype=np.float32),self.table[ids]])
    def commit(self,id):
        result=self.query([id]);self.position+=1;return result


class FirstVTests(unittest.TestCase):
    def test_native_build_identity_and_integrity(self):
        ids=np.arange(5);identity=table_identity('model','manifest','fake','batch',ids)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'table'
            build_table(path,Backend(),SPEC,ids,identity)
            meta,codes=load_table(path,identity,SPEC,ids)
            np.testing.assert_array_equal(codes,Backend.table)
            self.assertTrue(meta['full_vocabulary'])
            with self.assertRaisesRegex(ValueError,'identity'):
                load_table(path,dict(identity,model_sha256='wrong'),SPEC,ids)
            del codes
            with (path/'v.npy').open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'integrity'):load_table(path,identity,SPEC,ids)

    def test_full_scan_ambiguity_and_strict_boundary(self):
        ids=np.arange(5)
        exact=nearest_v(Backend.table,ids,[QUANT[1]],[5,5],1,chunk_size=2)
        self.assertEqual(exact['accepted_token_id'],1)
        ambiguous=nearest_v(Backend.table,ids,[QUANT[1]],[10,10],1,chunk_size=2)
        self.assertEqual(ambiguous['passing_ids'],[2,4]);self.assertIsNone(ambiguous['accepted_token_id'])
        boundary=nearest_v(Backend.table[:1],ids[:1],[QUANT[1]],[1,0],1,distance_metric="l2")
        self.assertEqual(boundary['stop_reason'],'unresolved')

    def test_first_layer_needs_no_prefix_and_positions_are_independent(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);observed=np.array([[999,10,10],[-999,20,20],[777,30,30]],dtype=np.float32)
            save_canonical_observation(d/'obs',observed,[50,60,100],[],0,QUANT,'model',{})
            _,_,targets,positions,prefix=load_observation(d/'obs')
            self.assertEqual(prefix,[])
            meta=dict(spec=SPEC,quantization=[QUANT[1]],full_vocabulary=True)
            result=match_observation(d/'run',Backend.table,np.arange(5),meta,targets,positions,1)
            self.assertEqual(result['recovered_token_ids'],[1,None,3])
            self.assertEqual(result['decoder_seconds'],0)
            np.testing.assert_array_equal(np.load(d/'run/candidates/000001.npy'),[2,4])

    def test_context_dependence_rejected(self):
        class Bad(Backend):
            def query(self,ids):
                row=super().query(ids);row[:,1:]+=self.position;return row
        with self.assertRaises(AssertionError):check_independence(Bad(),SPEC,np.arange(5))


if __name__=='__main__':unittest.main()
