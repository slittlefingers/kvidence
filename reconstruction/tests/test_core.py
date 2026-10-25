import gzip
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.search import SearchConfig, recover_position
from kvrecon.observation import import_snapshot, load_observation
from kvrecon.common import dequantize
from kvrecon.engine import reconstruct


class FakeBackend:
    def __init__(self,values):
        self.table=np.asarray(values,dtype=np.float64);self.vocab_size=len(values)
        self.position=1;self.decoder_seconds=0.;self.calls=[];self.commits=[]
    def query(self,ids):
        self.calls.append(list(ids));return self.table[ids].copy()
    def values(self,codes):return np.asarray(codes,dtype=np.float64)
    def reset(self,prefix):self.position=len(prefix)
    def probabilities(self):return np.arange(self.vocab_size)
    def commit(self,token):
        self.commits.append(token);self.position+=1;return self.table[token:token+1]


class Tests(unittest.TestCase):
    def test_strict_raw_l2_and_full_pool(self):
        b=FakeBackend([[4.],[5.],[3.999]])
        r=recover_position(b,np.array([0.]),np.arange(3),0,SearchConfig(distance_metric="l2",epsilon=4.,method="collision_plus",batch_size=2))
        self.assertEqual(r["accepted_token_id"],2)
        self.assertEqual(r["queries"],3)
        self.assertEqual(b.position,1)

    def test_exhaustion_does_not_accept_nearest(self):
        b=FakeBackend([[9.],[8.]])
        r=recover_position(b,np.array([0.]),np.arange(2),0,SearchConfig(distance_metric="l2",epsilon=4.,method="collision_plus"))
        self.assertIsNone(r["accepted_token_id"]);self.assertEqual(r["stop_reason"],"exhausted")

    def test_budget_is_incomplete_and_no_commit(self):
        b=FakeBackend([[9.],[8.],[0.]])
        with tempfile.TemporaryDirectory() as d:
            r=reconstruct(b,np.array([[0.]]),[1],[2],SearchConfig(distance_metric="l2",epsilon=4.,method="collision_plus",max_queries=2),d)
        self.assertEqual(r["status"],"unresolved");self.assertEqual(b.commits,[])

    def test_ambiguity_is_logged_without_uniqueness_claim(self):
        b=FakeBackend([[1.],[0.],[8.]])
        r=recover_position(b,np.array([0.]),np.arange(3),0,SearchConfig(distance_metric="l2",epsilon=4.,method="collision_plus"))
        self.assertEqual(r["accepted_token_id"],0)
        self.assertEqual(r["passing_ids_in_accepting_batch"],[0,1]);self.assertFalse(r["uniqueness_proven"])

    def test_ridge_proposal_requires_real_verification(self):
        b=FakeBackend(np.full((10,1),9.))
        class WrongPredictor:
            def scores(self,*args):return np.zeros(10)
        c=SearchConfig(distance_metric="l2",epsilon=4.,method="guided",probability_seeds=2,random_seeds=2,proposals=3,batch_size=3)
        r=recover_position(b,np.array([0.]),np.arange(10),0,c,WrongPredictor())
        self.assertIsNone(r["accepted_token_id"]);self.assertEqual(sorted(r["queried_token_ids"]),list(range(10)))
        self.assertIn("ridge",[s["phase"] for s in r["stages"]])

    def test_per_tensor_quantization_not_lsb_or_rms(self):
        got=dequantize(np.array([[2,4,3,5]],dtype=np.int8),[
            dict(width=2,scale=.5,zero_point=0),dict(width=2,scale=2.,zero_point=1)])
        np.testing.assert_array_equal(got,[[1,2,4,8]])

    def test_guided_stage_preserves_best_across_physical_chunks(self):
        # Both pass; querying one slot at a time must not accept the first
        # before evaluating the complete probability seed stage.
        b=FakeBackend([[2.],[0.],[9.]])
        config=SearchConfig(distance_metric="l2",epsilon=4.,method="guided",probability_seeds=2,random_seeds=1,batch_size=1)
        r=recover_position(b,np.array([0.]),np.arange(3),0,config)
        self.assertEqual(r["accepted_token_id"],1)
        self.assertEqual(r["queries"],2)
        self.assertEqual(r["passing_ids_in_accepting_stage"],[0,1])

    def test_strided_snapshot_extracts_only_target_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);k=np.array([[10,11],[20,21],[30,31],[40,41]],dtype=np.int8)
            # V is stored channel-major, K sequence-major; include allocation padding.
            payload=k.tobytes()+b"\x00"*8+(k+2).T.tobytes()
            buffers=[]
            for part,shape,strides,offset,axis in [("k",[1,1,4,2],[8,8,2,1],0,2),("v",[1,1,2,4],[8,8,4,1],16,3)]:
                buffers.append(dict(name=f"kv_cache_{part}_4",shape=shape,strides=strides,payload_offset=offset,dtype="int8",local=False,sequence_axis=axis))
            layout=dict(ringbuffer=False,byte_order="little-endian",kv_positions=4,buffers=buffers)
            encoded=json.dumps(layout).encode()
            with gzip.open(d/"source.gz","wb") as f:f.write(b"GHOSTKV1"+struct.pack("<Q",len(encoded))+encoded+payload)
            outputs=[dict(name="kv_slice_k_4",shape=[1,1,1,2],type=9,scale=[.5],zero_point=[0]),
                     dict(name="kv_slice_v_4",shape=[1,1,2,1],type=9,scale=[2.],zero_point=[0])]
            sections={"tf_lite_prefill_decode":{"signatures":{"decode":{"outputs":outputs}}}}
            import_snapshot(d/"source.gz",d/"out",sections,"test",[2],1,2)
            meta,codes,values,pos,prefix=load_observation(d/"out")
            np.testing.assert_array_equal(codes,[[20,21,22,23],[30,31,32,33]])
            self.assertEqual(prefix,[2]);self.assertEqual(set(np.load(d/"out/observed.npz").files),{"codes","positions","public_prefix_ids"})
            # First local layer can be imported before any wrap/shift, without
            # knowing the historical prefix. Never guess overwritten positions.
            for b in buffers:b['name']=b['name'].replace('_4','_0');b['local']=True
            for tensor in outputs:tensor['name']=tensor['name'].replace('_4','_0')
            def write_local():
                encoded=json.dumps(layout).encode()
                with gzip.open(d/'local.gz','wb') as f:f.write(b'GHOSTKV1'+struct.pack('<Q',len(encoded))+encoded+payload)
            write_local()
            import_snapshot(d/'local.gz',d/'local_out',sections,'test',[],2,2,layer=0)
            _,local_codes,_,positions,prefix=load_observation(d/'local_out')
            np.testing.assert_array_equal(local_codes,[[30,31,32,33],[40,41,42,43]])
            self.assertEqual(prefix,[]);self.assertEqual(positions.tolist(),[2,3])
            layout['kv_positions']=5;write_local()
            with self.assertRaisesRegex(ValueError,'slot map'):
                import_snapshot(d/'local.gz',d/'bad_local',sections,'test',[],2,2,layer=0)
            layout['kv_positions']=4;layout['ringbuffer']=True;write_local()
            with self.assertRaisesRegex(ValueError,'slot map'):
                import_snapshot(d/'local.gz',d/'bad_ring',sections,'test',[],2,2,layer=0)


if __name__=="__main__":unittest.main()
