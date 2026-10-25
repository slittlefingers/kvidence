"""Separate acquisition, inversion, and evaluation stages for RQ3/RQ4."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from runtime import ROOT, MODEL, Layer5Map, array, load_runtime, nmse


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n")


def assets():
    out = ROOT/"assets"
    out.mkdir(exist_ok=True)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 90)
    cases = []
    for cid, color, shape, number in [("image01","red","square","42"),
                                      ("image02","blue","circle","57")]:
        im = Image.new("RGB",(384,384),"white")
        d = ImageDraw.Draw(im)
        if shape=="square": d.rectangle((85,25,299,239), fill=color)
        else: d.ellipse((85,25,299,239), fill=color)
        d.text((192,300),number,font=font,fill="black",anchor="mm")
        im.save(out/f"{cid}.png")
        cases.append(dict(id=cid, modality="image",file=f"{cid}.png",
                          truth=[color,shape,number],
                          question="State the shape, its color, and the number in the image. Answer briefly."))
    shutil.copy2(ROOT.parents[2]/"annoymous/app/src/androidTest/assets/test-speech.wav",
                 out/"audio01.wav")
    # Offline macOS speech synthesis, written to a file; no playback.
    if not (out/"audio02.wav").exists():
        subprocess.run(["/usr/bin/say","-v","Samantha","-r","145","-o",str(out/"audio02.aiff"),
                        "The blue door has number fifty seven."], check=True)
        subprocess.run(["/opt/homebrew/bin/ffmpeg","-v","error","-y","-i",str(out/"audio02.aiff"),
                        "-ar","16000","-ac","1",str(out/"audio02.wav")], check=True)
    for cid, truth in [("audio01","The secret number is forty two."),
                       ("audio02","The blue door has number fifty seven.")]:
        cases.append(dict(id=cid,modality="audio",file=f"{cid}.wav",truth=truth,
                          question="Transcribe the audio exactly."))
    write_json(out/"manifest.json",cases)
    return cases


def acquire(args):
    cases = assets()
    model, processor = load_runtime()
    out = Path(args.output)
    for case in cases:
        cid, modality = case["id"], case["modality"]
        prompt = apply_chat_template(processor,model.config,case["question"],
                                    num_images=int(modality=="image"),
                                    num_audios=int(modality=="audio"))
        inp = prepare_inputs(processor,prompts=prompt,
            images=[str(ROOT/"assets"/case["file"])] if modality=="image" else None,
            audio=[str(ROOT/"assets"/case["file"])] if modality=="audio" else None)
        emb = model.get_input_embeddings(**inp)
        mx.eval(emb.inputs_embeds, emb.per_layer_inputs)
        ids = np.array(inp["input_ids"])
        mm_id = model.config.image_token_id if modality=="image" else model.config.audio_token_id
        pos = np.flatnonzero(ids[0]==mm_id)
        z = emb.inputs_embeds[:, int(pos[0]):int(pos[-1])+1, :]
        types = np.array(inp["mm_token_type_ids"]) if "mm_token_type_ids" in inp else None
        fmap = Layer5Map(model,ids,pos,types)
        cache=model.make_cache()
        logits=model.language_model(inputs_embeds=emb.inputs_embeds,per_layer_inputs=emb.per_layer_inputs,
                                    mm_token_type_ids=inp.get("mm_token_type_ids"),
                                    cache=cache,logits_to_keep=1).logits
        mx.eval(logits)
        k=cache[4].keys[:,:,int(pos[0]):int(pos[-1])+1,:]
        v=cache[4].values[:,:,int(pos[0]):int(pos[-1])+1,:]
        target=mx.concatenate([k.reshape(1,len(pos),-1),v.reshape(1,len(pos),-1)],axis=-1)
        check=fmap(z); mx.eval(check)
        validation=nmse(array(check),array(target))
        assert validation < 1e-5, validation
        snapshot = []
        base_end = int(cache[4].offset)
        initial = array(target)
        last_delay=0
        # Real rotating cache updates, not just deleting arrays or masking coordinates.
        for delay in (0,256,512,768):
            while last_delay<delay:
                step=min(64,delay-last_delay)
                # A known filler token. No generated response is required for cache eviction.
                filler=processor.tokenizer.encode(" filler",add_special_tokens=False)[0]
                logits=model.language_model(mx.full((1,step),filler,dtype=mx.int32),
                                             cache=cache,logits_to_keep=1).logits
                mx.eval(logits)
                last_delay += step
            c=cache[4]
            now=mx.concatenate([c.keys[:,:,int(pos[0]):int(pos[-1])+1,:].reshape(1,len(pos),-1),
                                c.values[:,:,int(pos[0]):int(pos[-1])+1,:].reshape(1,len(pos),-1)],axis=-1)
            stable=nmse(array(now),initial)
            assert stable==0
            snapshot.append(dict(extra_positions=delay,full_offset=c.offset,
                layer5_media_nmse_vs_initial=stable,
                local=[dict(layer=i+1,offset=cache[i].offset,size=cache[i].size(),
                            max_size=cache[i].max_size,keep=cache[i].keep,
                            stored_positions=cache[i].keys.shape[2],
                            earliest_stored_position=max(0,cache[i].offset-cache[i].keys.shape[2]),
                            earliest_valid_position=max(0,cache[i].offset-cache[i].size()))
                       for i in range(4)]))
        assert all(item["earliest_stored_position"]>int(pos[-1]) for item in snapshot[-1]["local"])
        # Public inversion inputs are isolated from evaluator-only target features and files.
        public=out/"observed"/cid
        private=out/"private_eval"/cid
        public.mkdir(parents=True,exist_ok=True); private.mkdir(parents=True,exist_ok=True)
        np.savez(public/"observation.npz", kv=array(now), input_ids=ids,
                 media_positions=pos,token_types=types)
        np.savez(private/"truth.npz",z=array(z))
        meta=dict(id=cid,modality=modality,media_positions=len(pos),
                  full_prompt_positions=ids.shape[1],layer_1based=5,kv_dtype=str(v.dtype),
                  feature_dtype=str(z.dtype),weight_bits=8,feature_dim=1536,
                  adapter_input_dim=768 if modality=="image" else 1536,
                  use_bidirectional_attention=model.config.text_config.use_bidirectional_attention,
                  observation_after_extra_positions=768,forward_map_validation_nmse=validation,
                  model_id="mlx-community/gemma-4-e2b-it-8bit",
                  model_revision="03dcf209f3f549b4075e7191e77cf69b3d48e1b2",
                  media_sha256=hashlib.sha256((ROOT/"assets"/case["file"]).read_bytes()).hexdigest(),
                  backend="Apple MLX GPU",evidence_source="instrumented runtime cache")
        write_json(public/"metadata.json",meta)
        write_json(private/"eviction.json",snapshot)
        write_json(private/"case.json",case)
        print("ACQUIRED",cid,json.dumps(meta),flush=True)
        del cache,emb,z,fmap,logits
        mx.clear_cache()
    provenance=dict(versions={p:importlib.metadata.version(p) for p in
                               ("mlx","mlx-vlm","transformers","numpy","pillow")},
                    model_config_sha256=hashlib.sha256((MODEL/"config.json").read_bytes()).hexdigest(),
                    source_model=str(MODEL),device=mx.metal.device_info())
    write_json(out/"provenance.json",provenance)


def invert(args):
    # This function reads observed/ only. No truth.npz, assets, labels or transcripts.
    out=Path(args.output)
    model, _ = load_runtime()
    for path in sorted((out/"observed").iterdir()):
        if args.cases and path.name not in args.cases: continue
        obs=np.load(path/"observation.npz",allow_pickle=False)
        meta=json.loads((path/"metadata.json").read_text())
        fmap=Layer5Map(model,obs["input_ids"],obs["media_positions"],obs["token_types"])
        target=mx.array(obs["kv"]); energy=mx.mean(target.square())
        adapter=model.embed_vision if meta["modality"]=="image" else model.embed_audio
        for seed in args.seeds:
            dest=out/"recovered"/path.name/f"seed_{seed}"
            dest.mkdir(parents=True,exist_ok=True)
            rng=np.random.default_rng(seed)
            u=mx.array(rng.standard_normal((1,fmap.n,meta["adapter_input_dim"])).astype(np.float32))
            initial_z=adapter(u.astype(mx.bfloat16))
            mx.eval(initial_z)
            np.save(dest/"initial_z.npy",array(initial_z))
            def loss_fn(p):
                z=adapter(p.astype(mx.bfloat16))
                return mx.mean((fmap(z)-target).square())/energy
            vg=mx.value_and_grad(loss_fn)
            m=mx.zeros_like(u); v=mx.zeros_like(u)
            start=time.perf_counter()
            best=float("inf"); best_u=u
            trace=[]
            for i in range(args.steps+1):
                loss,grad=vg(u)
                mx.eval(loss,grad)
                value=float(loss.item())
                if not np.isfinite(value): raise RuntimeError("Non-finite objective")
                if value<best:
                    best=value; best_u=u
                if i%25==0 or i==args.steps:
                    row=dict(step=i,kv_nmse=value,best_kv_nmse=best,
                             seconds=time.perf_counter()-start)
                    trace.append(row)
                    print("OPT",path.name,seed,json.dumps(row),flush=True)
                if i==args.steps: break
                # Fixed Adam schedule; no target-feature-based stopping or selection.
                g=grad.astype(mx.float32)
                m=.9*m+.1*g; v=.999*v+.001*g.square()
                lr=args.lr*(.15+.85*.5*(1+np.cos(np.pi*i/args.steps)))
                u=u-lr*(m/(1-.9**(i+1)))/(mx.sqrt(v/(1-.999**(i+1)))+1e-8)
                # Removes radial drift, justified by the adapter's RMS normalisation.
                u=u/mx.sqrt(mx.mean(u.square(),axis=-1,keepdims=True)+1e-8)
                mx.eval(u,m,v)
            recovered=adapter(best_u.astype(mx.bfloat16))
            mx.eval(recovered)
            np.savez(dest/"recovered.npz",z=array(recovered),u=array(best_u))
            write_json(dest/"trace.json",trace)
            write_json(dest/"run.json",dict(seed=seed,steps=args.steps,lr=args.lr,
                best_kv_nmse=best,seconds=time.perf_counter()-start,
                selection="minimum measured KV NMSE only",parameterization="known adapter",
                target_features_accessed=False))
            del u,m,v,grad,best_u,recovered,initial_z
            mx.clear_cache()


def evaluate(args):
    out=Path(args.output)
    model,processor=load_runtime()
    result=[]
    for path in sorted((out/"observed").iterdir()):
        if args.cases and path.name not in args.cases: continue
        obs=np.load(path/"observation.npz")
        truth=np.load(out/"private_eval"/path.name/"truth.npz")["z"]
        case=json.loads((out/"private_eval"/path.name/"case.json").read_text())
        fmap=Layer5Map(model,obs["input_ids"],obs["media_positions"],obs["token_types"])
        variants=[("original",truth),("zero",np.zeros_like(truth))]
        for run in sorted((out/"recovered"/path.name).glob("seed_*")):
            variants.extend([(run.name+"_initial",np.load(run/"initial_z.npy")),
                             (run.name,np.load(run/"recovered.npz")["z"])])
        for name,z in variants:
            zmx=mx.array(z).astype(mx.bfloat16)
            kv=array(fmap(zmx))
            token_ids=fmap.answer(zmx,args.max_tokens)
            text=processor.tokenizer.decode(token_ids)
            row=dict(case=path.name,variant=name,modality=case["modality"],
                     feature_nmse=nmse(z,truth),kv_nmse=nmse(kv,obs["kv"]),
                     feature_rmse=float(np.sqrt(np.mean((z-truth)**2))),
                     feature_cosine=float(np.vdot(z.ravel(),truth.ravel())/
                         max(np.linalg.norm(z)*np.linalg.norm(truth),1e-12)),
                     answer=text,answer_token_ids=token_ids,ground_truth=case["truth"],
                     generation_truncated=len(token_ids)==args.max_tokens)
            result.append(row)
            write_json(out/"evaluation.json",result)
            print("EVAL",json.dumps(row,ensure_ascii=False),flush=True)
    write_json(out/"evaluation.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["acquire","invert","evaluate"])
    p.add_argument("--output",default=str(ROOT/"results"))
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--lr",type=float,default=.08)
    p.add_argument("--seeds",type=int,nargs="+",default=[17,29])
    p.add_argument("--cases",nargs="+")
    p.add_argument("--max-tokens",type=int,default=64)
    args=p.parse_args()
    {"acquire":acquire,"invert":invert,"evaluate":evaluate}[args.stage](args)
