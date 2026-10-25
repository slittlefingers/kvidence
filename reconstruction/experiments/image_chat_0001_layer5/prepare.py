"""Deterministic sample 0001 preparation. Private truth is never an attack input."""
import gzip
import json
import mmap
from pathlib import Path
import struct
import sys
import numpy as np
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
APP=ROOT.parents[1]
sys.path.insert(0,str(APP))
from kvrecon.common import WORKSPACE, EXPERIMENTS, digest, dump, dequantize, quantization
sys.path.insert(0,str(EXPERIMENTS/'npu_candidate_batch'))
from prepare import FlatReader


def main():
    dataset=APP.parent/'datasets/shadow_chat/image_chat_100'
    manifest=json.loads((dataset/'manifest.json').read_text())
    item=next(x for x in manifest['samples'] if x['id']=='0001')
    tokenizer=Tokenizer.from_file(str(EXPERIMENTS/'kv_guided_search_gemma4_int8/model/tokenizer.json'))
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    prefix=[2]+tokenizer.encode('<|turn>user\n\n\n<|image>',add_special_tokens=False).ids
    # Port of LiteRT GetAspectRatioPreservingSize: layout only, no image pixels.
    factor=np.sqrt(np.float32(2520*16*16)/np.float32(item['width']*item['height']))
    height=int(np.floor(factor*np.float32(item['height'])/48))*48
    width=int(np.floor(factor*np.float32(item['width'])/48))*48
    count=(height//48)*(width//48)
    start=len(prefix);end=start+count;separator=tokenizer.encode('\n\n',add_special_tokens=False).ids
    assert len(separator)==1
    text_start=end+1+len(separator)
    # Read source only during preparation; output labels are isolated.
    record=next(json.loads(line) for line in (dataset/'inputs.jsonl').read_text().splitlines()
                if json.loads(line)['id']=='0001')
    suffix=tokenizer.encode('\n\n'+record['user_prompt']+'<turn|>\n<|turn>model\n',add_special_tokens=False).ids
    prompt_ids=tokenizer.encode(record['user_prompt'],add_special_tokens=False).ids
    assert suffix[:1]==separator and suffix[1:1+len(prompt_ids)]==prompt_ids
    snapshot=dataset/'capture/0001/all_layers.gkv.gz'
    with gzip.open(snapshot,'rb') as f:
        h=f.read(16);layout=json.loads(f.read(struct.unpack('<Q',h[8:])[0]));payload=f.read()
    def read(layer,part):
        b=next(b for b in layout['buffers'] if b['name']==f'kv_cache_{part}_{layer}')
        a=np.ndarray(b['shape'],dtype=np.int8,buffer=payload,offset=b['payload_offset'],strides=tuple(b['strides']))
        return np.moveaxis(a,b['sequence_axis'],0).reshape(b['capacity'],-1).copy()
    codes=np.concatenate([read(4,'k'),read(4,'v')],axis=1)
    quant=quantization(sections,4);values=dequantize(codes,quant)
    # Validate ONLY known template positions with first-layer V. No unknown
    # text/media is recovered from layer 1 or given to the recovery process.
    table=np.load(APP/'assets/first_v_gemma4_full/v.npy',mmap_mode='r')
    table_ids=np.load(APP/'assets/first_v_gemma4_full/ids.npy')
    first_v=read(0,'v');checks=[]
    for pos,tid in list(enumerate(prefix))+[(end+1,separator[0])]:
        reference=table[np.flatnonzero(table_ids==tid)[0]]
        np.testing.assert_array_equal(first_v[pos],reference)
        checks.append(dict(position=pos,public_token_id=tid,exact_code_match=True))
    (ROOT/'input').mkdir(exist_ok=False);(ROOT/'private_eval').mkdir(exist_ok=False)
    ids=np.array([prefix+[258880]*count],dtype=np.int32)
    types=np.zeros_like(ids);types[:,start:end]=1
    np.savez(ROOT/'input/observed.npz',input_ids=ids,token_types=types,media_positions=np.arange(start,end),
        media_kv=values[start:end],media_codes=codes[start:end],prefix_kv=values[:start],
        end_kv=values[end],separator_kv=values[end+1:text_start],
        text_kv=values[text_start:text_start+len(prompt_ids)],
        text_positions=np.arange(text_start,text_start+len(prompt_ids)))
    # The LiteRT end-of-image vector is a public model constant, not media truth.
    model=WORKSPACE/'annoymous/models/gemma-4-E2B-it_Google_Tensor_G5.litertlm'
    with model.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        r=FlatReader(data);begin=sections['tf_lite_end_of_vision']['offset'];root=begin+r.scalar_at(begin)
        graph=r.vector(root,2,tables=True)[0];assert not r.vector(graph,3,tables=True)
        tensor=r.vector(graph,0,tables=True)[r.vector(graph,2)[0]]
        buffer=r.vector(root,4,tables=True)[r.scalar(tensor,2)];raw=r.table(buffer,0)
        assert r.scalar_at(raw)==1536*4
        eoi=np.frombuffer(data,dtype='<f4',count=1536,offset=raw+4).copy()
    np.save(ROOT/'input/end_of_image.npy',eoi)
    dump(ROOT/'input/metadata.json',dict(sample='0001',selection='first ID, chosen before recovery',
        dataset='image_chat_100',model_sha256=manifest['capture_identity']['model_sha256'],
        snapshot_sha256=digest(snapshot),observed_sha256=digest(ROOT/'input/observed.npz'),
        layer_index=4,quantization=quant,media_start=start,media_end=end,media_positions=count,
        text_start=text_start,text_count=len(prompt_ids),public_prefix=prefix,separator_ids=separator,
        boundary_source='public template and image dimensions with LiteRT patchify rule; known-template-only layer1 V validation',
        public_template_checks=checks,image_dimensions=[item['width'],item['height']],
        resized_dimensions=[width,height],media_before_text=True,only_layer5_targets_passed_to_attack=True,
        end_of_image_sha256=digest(ROOT/'input/end_of_image.npy'),target_cache_used_as_prefix=False))
    dump(ROOT/'private_eval/truth.json',dict(token_ids=prompt_ids,positions=list(range(text_start,text_start+len(prompt_ids))),
        text=record['user_prompt'],image=str(dataset/item['media']),annotations=str(dataset/item['private_annotations'])))
    print(json.dumps(dict(media_start=start,media_end=end,text_start=text_start,text_count=len(prompt_ids),template_verified=True)))


if __name__=='__main__':main()
