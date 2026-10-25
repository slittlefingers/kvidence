"""Synthetic CLI integration adapter. No learned model; not research evidence."""
import json
import numpy as np
from kvrecon.backend import Capabilities
from kvrecon.observation import save_canonical_observation


class Backend:
    vocab_size=8
    batch_capacity=1
    setup_seconds=0.
    decoder_seconds=0.

    def __enter__(self):
        self.reset([])
        return self

    def __exit__(self,*args):
        pass

    def reset(self,prefix):
        self.prefix=list(prefix);self.position=len(prefix)

    def probabilities(self):
        return np.arange(8)

    def query(self,ids):
        assert len(ids)==1
        return np.array([[sum(self.prefix)+ids[0],self.position+2*ids[0]]],dtype=np.float32)

    def commit(self,token):
        row=self.query([token]);self.prefix.append(token);self.position+=1
        return row

    def values(self,codes):
        return np.asarray(codes,dtype=np.float64)


class Adapter:
    def read_manifest(self,path):
        return json.loads(path.read_text())

    def dimensions(self,sections,profile):
        return sections

    def capabilities(self,sections,profile,layer):
        if layer!=0:raise ValueError('Only synthetic layer 0 exists')
        return Capabilities(**sections,layers=(0,),kv_dtype='float32',kv_width=2)

    def open(self,*args,**kwargs):
        return Backend()

    def import_observation(self,*args):
        raise NotImplementedError('Synthetic adapter has no snapshot capture format')

    def save_fixture(self,output,vectors,positions,prefix,layer,sections,model_hash,source):
        quant=[dict(width=1,scale=1.,zero_point=0,dtype='float32') for _ in range(2)]
        return save_canonical_observation(output,vectors,positions,prefix,layer,quant,model_hash,source)


class FirstBackend(Backend):
    def query(self,ids):
        assert len(ids)==1
        return np.array([[self.position,ids[0]*10]],dtype=np.float32)


class FirstAdapter(Adapter):
    def open(self,*args,**kwargs):return FirstBackend()
    def first_v_spec(self,sections,profile):
        return dict(layer=0,prefix=[2],context_independent_by_architecture=True,
            quantization=[dict(width=1,scale=1.,zero_point=0,dtype='float32') for _ in range(2)])


class IntegerBackend(Backend):
    quant=[dict(width=1,scale=.001,zero_point=-3,dtype='int8'),
           dict(width=1,scale=.02,zero_point=7,dtype='int8')]
    def query(self,ids):return super().query(ids).astype(np.int8)
    def values(self,codes):raise AssertionError('Integer LSB recovery must bypass dequantization')


class IntegerAdapter(Adapter):
    def open(self,*args,**kwargs):return IntegerBackend()
    def capabilities(self,sections,profile,layer):
        return Capabilities(**sections,layers=(0,),kv_dtype='int8',kv_width=2)
    def save_fixture(self,output,vectors,positions,prefix,layer,sections,model_hash,source):
        return save_canonical_observation(output,vectors,positions,prefix,layer,IntegerBackend.quant,model_hash,source)
