"""Extract Gemma 3 assets from the installed model, without a second download.

The embedding table is a proposal asset: per-row dequantized INT8 weights times
the model's embedding scale. Its agreement with native embedding execution is
checked separately on the device, not assumed from these tensor names.
"""
import argparse
import json
import mmap
from pathlib import Path
import sys
import numpy as np
from kvrecon.common import ROOT, WORKSPACE, EXPERIMENTS, digest, dump
sys.path.insert(0,str(EXPERIMENTS/"npu_candidate_batch"))
from prepare import inspect, FlatReader


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"assets/gemma3_1b")
    p.add_argument("--model",type=Path,default=WORKSPACE/"annoymous/models/Gemma3-1B-IT_q8_ekv1280_Google_Tensor_G5.litertlm")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sections=inspect(a.model);dump(a.output/"manifest.json",dict(model_sections=sections))
    with a.model.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        r=FlatReader(data);root=32+r.scalar_at(32)
        tokenizers=[]
        for s in r.vector(r.table(root,1),0,tables=True):
            if r.scalar(s,3,"<B")==4:
                begin=r.scalar(s,1,"<Q");end=r.scalar(s,2,"<Q")
                tokenizers.append(bytes(data[begin:end]))
        if len(tokenizers)!=1:raise ValueError("Expected exactly one embedded SentencePiece model")
        (a.output/"tokenizer.model").write_bytes(tokenizers[0])
        begin=sections['tf_lite_embedder']['offset'];model=begin+r.scalar_at(begin)
        graph=r.vector(model,2,tables=True)[0];tensors=r.vector(graph,0,tables=True);buffers=r.vector(model,4,tables=True)
        tables=[t for t in tensors if r.scalar(t,1,"<B")==9 and r.vector(t,0)==[262144,1152] and r.scalar(t,2)>0]
        if len(tables)!=1:raise ValueError("Unexpected Gemma 3 embedding constants")
        t=tables[0];q=r.table(t,4);shape=r.vector(t,0)
        scales=np.asarray(r.vector(q,2,"<f"),dtype=np.float32);zeros=np.asarray(r.vector(q,3,"<q"),dtype=np.float32)
        if r.scalar(q,6)!=0 or scales.shape!=(shape[0],) or zeros.shape!=scales.shape:
            raise ValueError("Expected per-row embedding quantization")
        buf=buffers[r.scalar(t,2)];offset=r.table(buf,0)
        if not offset or r.scalar_at(offset)!=np.prod(shape):raise ValueError("Unexpected embedding buffer length")
        weights=np.ndarray(shape,dtype=np.int8,buffer=data,offset=offset+4)
        constants=[]
        for tensor in tensors:
            if r.scalar(tensor,1,"<B")==0 and r.vector(tensor,0)==[] and r.scalar(tensor,2)>0:
                raw=r.table(buffers[r.scalar(tensor,2)],0)
                if raw and r.scalar_at(raw)==4:constants.append(r.scalar_at(raw+4,"<f"))
        if len(constants)!=1 or not np.isclose(constants[0],np.sqrt(shape[1])):
            raise ValueError("Unexpected embedding output scale")
        multiplier=np.float32(constants[0]);out=np.lib.format.open_memmap(a.output/"embeddings.npy",mode="w+",dtype=np.float32,shape=tuple(shape))
        for i in range(0,shape[0],4096):
            out[i:i+4096]=(weights[i:i+4096].astype(np.float32)-zeros[i:i+4096,None])*scales[i:i+4096,None]*multiplier
        out.flush();del out,weights
    import sentencepiece as spm
    tokenizer=spm.SentencePieceProcessor(model_file=str(a.output/"tokenizer.model"))
    if tokenizer.vocab_size()!=shape[0]:raise ValueError("Tokenizer and embeddings disagree")
    dump(a.output/"provenance.json",dict(model=str(a.model.resolve()),model_sha256=digest(a.model),
        embedding_shape=shape,embedding_scale=float(multiplier),tokenizer_vocab_size=tokenizer.vocab_size(),
        embeddings_sha256=digest(a.output/"embeddings.npy"),tokenizer_sha256=digest(a.output/"tokenizer.model"),
        source="Extracted from the same local Tensor G5 .litertlm",device_embedding_comparison="pending"))
    print("Prepared",a.output,flush=True)


if __name__=="__main__":main()
