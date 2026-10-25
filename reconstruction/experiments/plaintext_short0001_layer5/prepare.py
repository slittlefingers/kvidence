"""Import a real after-response capture of the paired, verbatim plaintext."""
import json
from pathlib import Path
import sys
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from kvrecon.common import dump, digest
from kvrecon.config import DEFAULT_MODEL_CONFIG, load_model_config
from kvrecon.observation import import_snapshot

profile = load_model_config(DEFAULT_MODEL_CONFIG)
source = ROOT.parent / 'datasets/shadow_chat/provenance/pilot_100_short/prompts/plaintext/0001.txt'
capture = HERE / 'capture'
message = source.read_text()
job = json.loads((capture / 'job.json').read_text())
sample = json.loads((capture / '0001/sample.json').read_text())
assert sample['capture_stage'] == 'after_response'
assert digest(source) == sample['prompt_sha256']
assert job['model_sha256'] == digest(profile['model'])
assert digest(capture / '0001/all_layers.gkv.gz') == sample['kv']['gzip_sha256']
before = '<bos><|turn>user\n'
rendered = before + message + '<turn|>\n<|turn>model\n'
start, end = len(before), len(before) + len(message)
tokenizer = Tokenizer.from_file(str(profile['tokenizer']))
encoded = tokenizer.encode(rendered, add_special_tokens=False)
positions = [i for i, (a, b) in enumerate(encoded.offsets) if a < end and b > start]
assert positions == list(range(positions[0], positions[-1] + 1))
prefix = tokenizer.encode(before, add_special_tokens=False).ids
assert prefix == encoded.ids[:positions[0]]
truth = [encoded.ids[i] for i in positions]
assert tokenizer.decode(truth, skip_special_tokens=False) == message
output = HERE / 'input'
output.mkdir(exist_ok=False)
private = output / 'private_eval'
private.mkdir()
dump(output / 'prefix.json', prefix)
dump(private / 'truth.json', dict(token_ids=truth, sample_id='0001',
    source_sha256=digest(source), positions=positions,
    note='Evaluation only; unavailable to the recovery subprocesses'))
sections = json.loads(profile['manifest'].read_text())['model_sections']
snapshot = capture / '0001/all_layers.gkv.gz'
import_snapshot(snapshot, output / 'observation', sections, job['model_sha256'],
                prefix, positions[0], len(positions), 4)
import_snapshot(snapshot, output / 'public_prefix_check', sections, job['model_sha256'],
                prefix[:1], 1, len(prefix)-1, 4)
dump(output / 'provenance.json', dict(sample_id='0001',
    capture_directory=str(capture / '0001'), capture_stage=sample['capture_stage'],
    model_sha256=job['model_sha256'], runtime_sha256=job['runtime_sha256'],
    layer_index=4, layer_1based=5, target='all tokens touching the original plaintext user prompt',
    target_tokens=len(positions), source_utf8_bytes=len(message.encode()),
    start_position=positions[0], end_position_exclusive=positions[-1]+1,
    public_prefix_tokens=len(prefix), rendered_input_tokens=len(encoded.ids),
    candidate_vocabulary='full original 262144 IDs, no language filtering',
    tokenizer_sha256=digest(profile['tokenizer']), source_prompt_sha256=digest(source),
    snapshot_sha256=sample['kv']['gzip_sha256'], no_new_capture=False,
    paired_ciphertext_sample='stored_short0001_layer5',
    input_policy='Original plaintext verbatim, no encryption/storage wrapper, no padding or truncation',
    calibration='Existing independent 8-position / 1024-candidate pilot, not full official calibration'))
print(json.dumps(dict(layer_1based=5, tokens=len(positions), prefix_tokens=len(prefix),
                      positions=[positions[0], positions[-1]])))
