"""Semantic diagnostic of OLD full-image proposals through the PHONE adapter.

These proposals were optimized with a non-equivalent MLX proxy. They are not
the incomplete phone-only SPSA reconstruction. Keep provenance explicit.
"""
import json
from pathlib import Path
import sys
import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from kvrecon.common import dump, digest, EXPERIMENTS
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard


def main():
    tokenizer = Tokenizer.from_file(str(EXPERIMENTS/'kv_guided_search_gemma4_int8/model/tokenizer.json'))
    private_guard()
    meta = json.loads((HERE/'input/metadata.json').read_text())
    sections = json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    with np.load(HERE/'input/observed.npz') as data:
        target = data['media_kv'].astype(np.float64)
    out = HERE/'full_proposal_phone_content'
    out.mkdir(exist_ok=False)
    question = 'Describe the image briefly. Name the main visible objects and actions. Answer in English.'
    dump(out/'protocol.json', dict(
        purpose='Semantic diagnostic of previous full-image proposals; NOT a completed SPSA recovery',
        proposal_source='fit/seed_17 and seed_29/latent.npy; previously optimized with non-equivalent MLX proxy',
        evaluation_adapter='actual phone CPU vision_adapter_280',
        evaluation_decoder='actual phone NPU decode',
        seeds=[17,29], question=question, generation='greedy', max_new_tokens=256,
        image_acceptance='every position raw concatenated K+V L2 < 5',
        inspect_description_even_if_KV_gate_fails=True,
        original_image_accessed=False, original_text_accessed=False,
        original_encoder_output_accessed=False, target_KV_imported_into_prefix=False,
        model_sha256=meta['model_sha256'], script_sha256=digest(Path(__file__))))
    results=[]
    with NpuBackend(sections,4,out/'device',meta['model_sha256']) as backend:
        for seed in (17,29):
            folder=out/f'seed_{seed}'
            folder.mkdir()
            latent_path=HERE/f'fit/seed_{seed}/latent.npy'
            latent=np.load(latent_path)[0]
            assert latent.shape==(260,768)
            features=np.concatenate([backend.project_vision(latent[i:i+128]) for i in range(0,len(latent),128)])
            np.save(folder/'phone_adapter_features.npy',features)
            backend.reset(meta['public_prefix'])
            kv=backend.values(backend.commit_embeddings(features))
            np.save(folder/'phone_kv.npy',kv)
            distances=np.linalg.norm(kv.astype(np.float64)-target,axis=1)
            metrics=dict(seed=seed,latent_sha256=digest(latent_path),
                raw_l2_per_position=distances.tolist(),mean_l2=float(distances.mean()),
                max_l2=float(distances.max()),passing_positions=int((distances<5).sum()),
                total_positions=len(distances),passes_all=bool((distances<5).all()))
            dump(folder/'kv_metrics.json',metrics)
            backend.commit_embeddings(np.load(HERE/'input/end_of_image.npy').reshape(1,1536))
            for token in meta['separator_ids']:
                backend.commit(token)
            suffix=question+'<turn|>\n<|turn>model\n'
            for token in tokenizer.encode(suffix,add_special_tokens=False).ids:
                backend.commit(token)
            stop={1,tokenizer.token_to_id('<turn|>'),tokenizer.token_to_id('<|endoftext|>')}
            generated=[]
            stop_token=None
            for _ in range(256):
                token=int(backend.probabilities()[0])
                if token in stop:
                    stop_token=token
                    break
                generated.append(token)
                backend.commit(token)
            caption=dict(seed=seed,question=question,token_ids=generated,
                text=tokenizer.decode(generated),raw_text=tokenizer.decode(generated,skip_special_tokens=False),
                stop_token=stop_token,truncated=len(generated)==256,
                uses_recovered_proposal=True,uses_original_image=False,
                is_phone_SPSA_result=False)
            dump(folder/'description.json',caption)
            results.append(dict(metrics=metrics,description=caption))
            dump(out/'results.json',results)
            print(f'seed {seed}: KV mean={distances.mean():.4f}, max={distances.max():.4f}, pass={(distances<5).sum()}/260',flush=True)
            print(caption['text'],flush=True)
    dump(out/'completed.json',dict(completed=True,original_evaluation_labels_accessed=False,runs=len(results)))


if __name__=='__main__':
    main()
