import unittest
import numpy as np
from kvrecon.npu import NpuBackend


class VisionRpcTests(unittest.TestCase):
    def backend(self):
        b=NpuBackend.__new__(NpuBackend)
        b.sections={'tf_lite_vision_adapter':{}}
        b.batch_capacity=128;b.position=5;b.width=1024;b.dtype=np.dtype('int8')
        b.dimensions={'embedding_dim':1536}
        b.pending_begin=True;b.calls=[]
        def rpc(op,**kwargs):
            b.calls.append((op,kwargs))
            if op=='vision_project':return np.zeros((kwargs['rows'],1536),dtype='<f4').tobytes()
            return np.zeros((kwargs['rows'],1024),dtype='int8').tobytes()
        b.rpc=rpc
        return b

    def test_exact_float_transport_and_non_mutating_query(self):
        b=self.backend()
        u=np.linspace(-1,1,128*768,dtype=np.float32).reshape(128,768)
        self.assertEqual(b.query_vision(u).shape,(128,1024))
        op,args=b.calls[-1]
        self.assertEqual(op,'vision_query');self.assertEqual(args['position'],5)
        self.assertLess(len(args['latent_hex'])+256,1<<20)
        np.testing.assert_array_equal(np.frombuffer(bytes.fromhex(args['latent_hex']),dtype='<f4').reshape(u.shape),u)
        self.assertEqual(b.position,5);self.assertFalse(b.pending_begin)
        self.assertEqual(b.project_vision(u[:3]).shape,(3,1536))

    def test_reject_malformed_before_rpc(self):
        b=self.backend()
        for u in (np.zeros((0,768)),np.zeros((129,768)),np.zeros((1,767)),
                  np.zeros((768,)),np.full((1,768),np.nan)):
            with self.assertRaises(ValueError):b.query_vision(u)
        self.assertEqual(b.calls,[])
        b.sections={}
        with self.assertRaises(ValueError):b.project_vision(np.zeros((1,768)))

    def test_direct_candidates_preserve_values_and_position(self):
        from kvrecon.transport import MAX_JSON
        import json
        b=self.backend();b.sections={}
        z=np.linspace(-3,4,128*1536,dtype=np.float32).reshape(128,1536)
        self.assertEqual(b.query_embeddings(z).shape,(128,1024))
        op,args=b.calls[-1]
        self.assertEqual(op,'query_embeddings')
        np.testing.assert_array_equal(np.frombuffer(bytes.fromhex(args['features_hex']),dtype='<f4').reshape(z.shape),z)
        self.assertLess(len(json.dumps(dict(args,op=op,seq=1,auth='x'*64))),MAX_JSON)
        self.assertEqual(b.position,5)
        self.assertFalse(b.pending_begin)

    def test_direct_rejects_invalid_inputs_without_rpc(self):
        b=self.backend()
        for z in (np.zeros((0,1536)),np.zeros((129,1536)),np.zeros((1,768)),
                  np.zeros(1536),np.full((1,1536),np.inf)):
            with self.assertRaises(ValueError):b.query_embeddings(z)
        b.dimensions['embedding_dim']=1152
        with self.assertRaises(ValueError):b.query_embeddings(np.zeros((1,1536)))
        self.assertEqual(b.calls,[])


if __name__=='__main__':unittest.main()
