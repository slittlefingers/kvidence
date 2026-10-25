"""Explicit, labeled device integration test; never part of recovery policy."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.common import DEFAULT_MANIFEST, MODEL, digest, dump
from kvrecon.npu import NpuBackend
from kvrecon.observation import load_observation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--no-install",action="store_true")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    meta,codes,targets,positions,prefix=load_observation(a.fixture/"observation")
    truth=json.loads((a.fixture/"private_eval/truth.json").read_text())["token_ids"]
    sections=json.loads(DEFAULT_MANIFEST.read_text())["model_sections"]
    result={"controlled_labeled_test":True,"checks":[]}
    with NpuBackend(sections,4,a.output/"device",digest(MODEL),install=not a.no_install) as b:
        b.reset(prefix)
        # Padding sizes used by initial 50 + random 150 (128 + 22), and generic 200 (128 + 72).
        for size in (1,4,22,50,72,128):
            ids=[truth[0]]+list(range(size-1))
            actual=b.query(ids)
            np.testing.assert_array_equal(actual[0],codes[0]);assert b.position==len(prefix)
            result["checks"].append(dict(kind="padding",valid_candidates=size,max_lsb=0))
        ids=[truth[0]]+list(range(127));expected=b.query(ids)
        order=np.random.default_rng(71).permutation(128);actual=b.query(np.asarray(ids)[order].tolist())
        np.testing.assert_array_equal(actual,expected[order])
        result["checks"].append(dict(kind="all_128_rows_permuted",max_lsb=0))
        np.testing.assert_array_equal(b.commit(truth[0])[0],codes[0])
        np.testing.assert_array_equal(b.query([truth[1]])[0],codes[1])
        b.reset(prefix+[truth[0]])
        np.testing.assert_array_equal(b.query([truth[1]])[0],codes[1])
        result["checks"].append(dict(kind="commit_then_query_equals_fresh_prefix_replay",max_lsb=0))
    result["passed"]=True;dump(a.output/"validation.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
