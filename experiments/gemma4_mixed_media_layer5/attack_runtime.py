"""Attack-side imports and I/O, with no acquisition fixture definitions."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'gemma4_multimodal_layer5'))
from runtime import load_runtime, array, mx, np


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
