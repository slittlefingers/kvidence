"""Small integration probe; no recovery results are produced by this file."""
import json
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image, ImageDraw
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "kv_guided_search_gemma4_int8/model"

if __name__ == "__main__":
    print("DEVICE", mx.metal.device_info(), flush=True)
    t = time.perf_counter()
    model, processor = load(str(MODEL))
    print("LOADED", time.perf_counter() - t, type(processor), flush=True)
    print("CONFIG", model.config.text_config.layer_types[:5], flush=True)
    print("PROCESSOR", processor.image_processor, flush=True)
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    im = Image.new("RGB", (384, 384), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle((70, 70, 314, 314), fill=(220, 25, 40))
    im.save(assets / "probe_red.png")
    for modality, path, question in [
        ("image", assets / "probe_red.png", "What color is the shape?"),
        ("audio", ROOT.parents[2] / "annoymous/app/src/androidTest/assets/test-speech.wav",
         "Transcribe this audio."),
    ]:
        prompt = apply_chat_template(
            processor, model.config, question,
            num_images=int(modality == "image"), num_audios=int(modality == "audio"),
        )
        inp = prepare_inputs(processor, prompts=prompt,
                             images=[str(path)] if modality == "image" else None,
                             audio=[str(path)] if modality == "audio" else None)
        print("INPUT", modality, {k: (list(v.shape) if hasattr(v, "shape") else str(type(v)))
                                  for k, v in inp.items()}, flush=True)
        t = time.perf_counter()
        emb = model.get_input_embeddings(**inp)
        mx.eval(emb.inputs_embeds, emb.per_layer_inputs)
        print("FEATURES", modality, emb.inputs_embeds.shape, emb.inputs_embeds.dtype,
              time.perf_counter()-t, flush=True)
        cache = model.make_cache()
        out = model.language_model(
            inputs_embeds=emb.inputs_embeds, per_layer_inputs=emb.per_layer_inputs,
            cache=cache, mm_token_type_ids=inp.get("mm_token_type_ids"), logits_to_keep=1)
        mx.eval(out.logits if hasattr(out, "logits") else out)
        print("KV5", cache[4].keys.shape, cache[4].values.dtype, flush=True)
    print("OK", flush=True)
