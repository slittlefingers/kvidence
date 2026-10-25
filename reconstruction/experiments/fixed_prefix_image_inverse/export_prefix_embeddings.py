"""Export exact public-prefix input embeddings from the capture's phone model."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.npu import NpuBackend
from kvrecon.common import digest,dump,EXPERIMENTS


def main():
 dataset=HERE/'data/overnight_20261025/shard_0001'
 prefix_file=dataset/'prefix.json';prefix=json.loads(prefix_file.read_text())
 ids=np.asarray(prefix['prefix_ids'],dtype=np.int32)
 assert len(ids)==prefix['prefix_tokens']==991
 model_sha=json.loads((dataset/'batch_0001/protocol.json').read_text())['model_sha256']
 out=dataset/'prefix_vectors';out.mkdir(exist_ok=True)
 if (out/'prefix_embeddings.npz').exists():raise FileExistsError('Prefix vectors already exist; refusing overwrite.')
 sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
 vectors={};started=time.time()
 with NpuBackend(sections,4,out/f'device/run_{time.time_ns()}',model_sha,serial='YOUR_DEVICE_SERIAL') as backend:
  for token in np.unique(ids):
   raw=backend.rpc('embedding',id=int(token))
   vector=np.frombuffer(raw,dtype='<f4').copy()
   assert vector.shape==(1536,) and np.isfinite(vector).all()
   vectors[int(token)]=vector
  matrix=np.stack([vectors[int(token)] for token in ids]).astype('<f4')
  # Replay a few token lookups through the model, independent of the local cache.
  check_positions=sorted(set([0,len(ids)//2,len(ids)-1]))
  for pos in check_positions:
   check=np.frombuffer(backend.rpc('embedding',id=int(ids[pos])),dtype='<f4')
   np.testing.assert_array_equal(matrix[pos],check)
 positions=np.arange(len(ids),dtype=np.int32)
 tmp=out/'prefix_embeddings.tmp.npz'
 np.savez_compressed(tmp,token_ids=ids,positions=positions,embeddings=matrix)
 tmp.replace(out/'prefix_embeddings.npz')
 dump(out/'metadata.json',dict(schema=1,representation='native token input embeddings from embedding signature; not contextual hidden states or KV',
  file='prefix_embeddings.npz',shape=list(matrix.shape),dtype='float32',token_count=len(ids),unique_tokens=len(vectors),
  prefix_sha256=prefix['prefix_sha256'],prefix_metadata_sha256=digest(prefix_file),tokenizer_sha256=prefix['tokenizer_sha256'],
  model_sha256=model_sha,file_sha256=digest(out/'prefix_embeddings.npz'),
  row_order='Original prefix_ids order, including BOS, template and image-begin marker IDs',
  positional_encoding_included=False,contextualized=False,per_layer_embeddings_included=False,
  training_role='Public fixed-prefix conditioning only; contains no target image features or labels',
  source='Same phone model as image acquisition; exact native embedding signature output',
  device_metadata=str(next((out/'device').glob('*/device.json')).relative_to(out)),
  repeat_lookup_check_positions=check_positions,elapsed_seconds=time.time()-started))
 print('Saved',out/'prefix_embeddings.npz','shape',matrix.shape,'bytes',(out/'prefix_embeddings.npz').stat().st_size,flush=True)

if __name__=='__main__':main()
