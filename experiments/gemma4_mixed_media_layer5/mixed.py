"""Natural photographs + unrelated, hidden text: layer-5 forensic pilot."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'gemma4_multimodal_layer5'
sys.path.insert(0, str(PARENT))
import mlx.core as mx
import numpy as np
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs
from runtime import Layer5Map, array, load_runtime, nmse, MODEL
from experiment import write_json, invert

def pack(cache, positions):
    # Sequence-major layout, works for any number of KV heads.
    positions=mx.array(positions)
    k=cache.keys[:, :, positions, :].transpose(0,2,1,3)
    v=cache.values[:, :, positions, :].transpose(0,2,1,3)
    return mx.concatenate([k.reshape(1,len(positions),-1),v.reshape(1,len(positions),-1)],-1)


def acquire(args):
    cases=json.loads((ROOT/'assets/acquisition_cases.json').read_text())
    model,processor=load_runtime()
    processor.image_processor.max_soft_tokens=280
    out=Path(args.output)
    manifest=[]
    for case in cases:
        prompt=apply_chat_template(processor,model.config,case['secret'],num_images=1)
        inp=prepare_inputs(processor,prompts=prompt,images=[str(ROOT/'assets'/case['file'])])
        emb=model.get_input_embeddings(**inp)
        ids=np.array(inp['input_ids']); types=np.array(inp['mm_token_type_ids'])
        pos=np.flatnonzero(ids[0]==model.config.image_token_id)
        text_ids=processor.tokenizer.encode(case['secret'],add_special_tokens=False)
        starts=[i for i in range(ids.shape[1]-len(text_ids)+1)
                if ids[0,i:i+len(text_ids)].tolist()==text_ids]
        assert len(starts)==1, (case['id'],ids.tolist(),text_ids)
        textpos=np.arange(starts[0],starts[0]+len(text_ids))
        assert pos[-1]<textpos[0]
        z=emb.inputs_embeds[:,int(pos[0]):int(pos[-1])+1,:]
        cache=model.make_cache()
        logits=model.language_model(inputs_embeds=emb.inputs_embeds,
            per_layer_inputs=emb.per_layer_inputs,mm_token_type_ids=inp['mm_token_type_ids'],
            cache=cache,logits_to_keep=1).logits
        mx.eval(logits)
        original_image=array(pack(cache[4],pos)); original_text=array(pack(cache[4],textpos))
        # Redact ALL secret text IDs. No answer-dependent prompt enters inversion.
        skeleton=ids.copy(); skeleton[:,textpos]=-1
        image_only_ids=ids[:,:int(pos[-1])+1]
        image_only_types=types[:,:int(pos[-1])+1]
        fmap=Layer5Map(model,image_only_ids,pos,image_only_types)
        validation=nmse(array(fmap(z)),original_image)
        assert validation<1e-4, validation
        # Same length, wholly changed suffix: checks actual causal independence.
        replacement=processor.tokenizer.encode(' unrelated',add_special_tokens=False)[0]
        changed=ids.copy(); changed[:,textpos]=replacement
        changed_emb=model.get_input_embeddings(input_ids=mx.array(changed),
                                               cached_image_features=z,pixel_values=inp['pixel_values'])
        changed_cache=model.make_cache()
        logits=model.language_model(inputs_embeds=changed_emb.inputs_embeds,
            per_layer_inputs=changed_emb.per_layer_inputs,mm_token_type_ids=inp['mm_token_type_ids'],
            cache=changed_cache,logits_to_keep=1).logits
        mx.eval(logits)
        suffix_change_nmse=nmse(array(pack(changed_cache[4],pos)),original_image)
        assert suffix_change_nmse==0, suffix_change_nmse
        del changed_cache,changed_emb
        snapshots=[]; done=0
        filler=processor.tokenizer.encode(' filler',add_special_tokens=False)[0]
        for delay in (0,256,512,768):
            while done<delay:
                n=min(64,delay-done)
                logits=model.language_model(mx.full((1,n),filler,dtype=mx.int32),
                                            cache=cache,logits_to_keep=1).logits
                mx.eval(logits);done+=n
            snapshots.append(dict(extra_positions=delay,
                image_kv_nmse=nmse(array(pack(cache[4],pos)),original_image),
                text_kv_nmse=nmse(array(pack(cache[4],textpos)),original_text),
                local=[dict(layer=i+1,offset=cache[i].offset,size=cache[i].size(),
                            stored_positions=cache[i].keys.shape[2],
                            earliest_stored_position=cache[i].offset-cache[i].keys.shape[2])
                       for i in range(4)]))
        assert all(x['earliest_stored_position']>textpos[-1] for x in snapshots[-1]['local'])
        assert all(x['image_kv_nmse']==x['text_kv_nmse']==0 for x in snapshots)
        public=out/'observed'/case['id'];private=out/'private_eval'/case['id']
        public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
        np.savez(public/'observation.npz',kv=array(pack(cache[4],pos)),input_ids=image_only_ids,
                 media_positions=pos,token_types=image_only_types)
        np.savez(public/'text_observation.npz',kv=array(pack(cache[4],textpos)),
                 skeleton_ids=skeleton,text_positions=textpos,media_positions=pos,token_types=types)
        np.savez(private/'truth.npz',z=array(z),input_ids=ids,text_ids=np.array(text_ids))
        write_json(private/'case.json',case)
        write_json(private/'eviction.json',snapshots)
        meta=dict(id=case['id'],modality='image',adapter_input_dim=768,feature_dim=1536,
            media_positions=len(pos),text_positions=len(textpos),full_prompt_positions=ids.shape[1],
            layer_1based=5,kv_dtype=str(cache[4].keys.dtype),weight_bits=8,
            max_soft_tokens=280,unknown_text_ids_redacted=True,
            image_inversion_uses_only_prefix_ending_at_last_media_position=True,
            forward_map_validation_nmse=validation,changed_suffix_image_kv_nmse=suffix_change_nmse,
            use_bidirectional_attention=model.config.text_config.use_bidirectional_attention,
            media_sha256=hashlib.sha256((ROOT/'assets'/case['file']).read_bytes()).hexdigest(),
            backend='Apple MLX GPU',evidence_source='instrumented runtime cache')
        write_json(public/'metadata.json',meta)
        manifest.append(dict(id=case['id'],file=case['file'],coco_id=case['coco_id'],
            source='https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',
            split='COCO train2017',sha256=meta['media_sha256'],
            gemma_training_membership='unknown'))
        print('ACQUIRED',case['id'],json.dumps(meta),flush=True)
        del cache,emb,z,fmap,logits;mx.clear_cache()
    write_json(ROOT/'assets/manifest.json',manifest)
    write_json(out/'provenance.json',dict(
        versions={p:importlib.metadata.version(p) for p in ('mlx','mlx-vlm','transformers','numpy','pillow')},
        device=mx.metal.device_info(),model_path=str(MODEL),
        model_config_sha256=hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest(),
        model_revision='03dcf209f3f549b4075e7191e77cf69b3d48e1b2'))


def evaluate(args):
    model,processor=load_runtime();processor.image_processor.max_soft_tokens=280
    out=Path(args.output);rows=[]
    for public in sorted((out/'observed').iterdir()):
        case=json.loads((out/'private_eval'/public.name/'case.json').read_text())
        truth=np.load(out/'private_eval'/public.name/'truth.npz')['z']
        obs=np.load(public/'observation.npz')
        evidence_map=Layer5Map(model,obs['input_ids'],obs['media_positions'],obs['token_types'])
        runs=sorted((out/'recovered'/public.name).glob('seed_*'))
        variants=[('original',truth),('zero',np.zeros_like(truth))]
        for p in runs:
            variants += [(p.name+'_initial',np.load(p/'initial_z.npy')),
                         (p.name,np.load(p/'recovered.npz')['z'])]
        # New image-related questions were NEVER in the acquired KV / objective.
        for qid,question in [('generic','Describe this image in two sentences.'),('detail',case['question'])]:
            prompt=apply_chat_template(processor,model.config,question,num_images=1)
            inp=prepare_inputs(processor,prompts=prompt,images=[str(ROOT/'assets'/case['file'])])
            ids=np.array(inp['input_ids']);pos=np.flatnonzero(ids[0]==model.config.image_token_id)
            assert len(pos)==truth.shape[1]
            fmap=Layer5Map(model,ids,pos,np.array(inp['mm_token_type_ids']))
            for name,z in variants:
                token_ids=fmap.answer(mx.array(z).astype(mx.bfloat16),128)
                row=dict(case=public.name,question_id=qid,question=question,variant=name,
                    answer=processor.tokenizer.decode(token_ids),answer_token_ids=token_ids,
                    truncated=len(token_ids)==128,feature_nmse=nmse(z,truth),
                    kv_nmse=nmse(array(evidence_map(mx.array(z).astype(mx.bfloat16))),obs['kv']))
                rows.append(row);write_json(out/'image_evaluation.json',rows)
                print('IMAGE_EVAL',json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['acquire','invert','evaluate'])
    p.add_argument('--output',default=str(ROOT/'results'))
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--lr',type=float,default=.08)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29])
    p.add_argument('--cases',nargs='+')
    args=p.parse_args();{'acquire':acquire,'invert':invert,'evaluate':evaluate}[args.stage](args)
