"""Package the verbatim 153-token pilot sample and stage label-free KV input."""
import json
import shutil
import sys
from pathlib import Path
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from kvrecon.common import dump, digest
from kvrecon.config import DEFAULT_MODEL_CONFIG, load_model_config
from kvrecon.observation import import_snapshot

profile = load_model_config(DEFAULT_MODEL_CONFIG)
data = ROOT.parent / 'datasets/shadow_chat/archive/plaintext_pilot_0046'
source_dataset = ROOT.parent / 'datasets/shadow_chat/archive/pilot_100'
source = source_dataset / 'prompts/plaintext/0046.txt'
capture = data / 'capture'
sample = json.loads((capture / '0046/sample.json').read_text())
job = json.loads((capture / 'job.json').read_text())
summary = json.loads((capture / 'summary.json').read_text())
assert summary['passed'] and summary['completed_count'] == 1
assert sample['prompt_sha256'] == digest(source)
assert job['model_sha256'] == digest(profile['model'])
assert sample['capture_stage'] == 'after_response'
snapshot = capture / '0046/all_layers.gkv.gz'
assert digest(snapshot) == sample['kv']['gzip_sha256']
message = source.read_text()
before = '<bos><|turn>user\n'
rendered = before + message + '<turn|>\n<|turn>model\n'
tokenizer = Tokenizer.from_file(str(profile['tokenizer']))
encoded = tokenizer.encode(rendered, add_special_tokens=False)
start, end = len(before), len(before) + len(message)
positions = [i for i, (a, b) in enumerate(encoded.offsets) if a < end and b > start]
prefix = tokenizer.encode(before, add_special_tokens=False).ids
assert positions == list(range(len(prefix), len(prefix) + len(positions)))
assert encoded.ids[:len(prefix)] == prefix
assert len(positions) == 153
truth = [encoded.ids[i] for i in positions]
decoded = tokenizer.decode(truth, skip_special_tokens=False)
assert decoded == message
private = data / 'private_eval/0046'
private.mkdir(parents=True, exist_ok=False)
shutil.copyfile(source, private / 'prompt.txt')
dump(private / 'truth.json', dict(sample_id='0046', token_ids=truth,
    positions=positions, source_sha256=digest(source),
    note='Evaluation only; never read by recovery'))
sections = json.loads(profile['manifest'].read_text())['model_sections']
observation = data / 'observations/0046/layer_04'
import_snapshot(snapshot, observation, sections, job['model_sha256'],
                prefix, positions[0], len(positions), 4)
output = HERE / 'input'
output.mkdir(exist_ok=False)
shutil.copytree(observation, output / 'observation')
dump(output / 'prefix.json', prefix)
import_snapshot(snapshot, output / 'public_prefix_check', sections,
                job['model_sha256'], prefix[:1], 1, len(prefix)-1, 4)
for name in ('metadata.json', 'observed.npz'):
    assert digest(observation / name) == digest(output / 'observation' / name)
provenance = dict(sample_id='0046', source_dataset='pilot_100',
    source_prompt_sha256=digest(source), source_manifest_sha256=digest(source_dataset / 'manifest.json'),
    capture_directory=str(capture / '0046'), capture_stage='after_response',
    model_sha256=job['model_sha256'], runtime_sha256=job['runtime_sha256'],
    tokenizer_sha256=digest(profile['tokenizer']),
    target='all tokens touching the original plaintext user prompt',
    target_tokens=len(positions), source_utf8_bytes=len(message.encode()),
    public_prefix_tokens=len(prefix), rendered_input_tokens=len(encoded.ids),
    start_position=positions[0], end_position_exclusive=positions[-1]+1,
    layer_index=4, layer_1based=5, snapshot_sha256=digest(snapshot),
    selection='Closest complete original prompt to 100 tokens among existing pilot_100; minimum is 153',
    input_policy='Verbatim, no padding/truncation or added storage wrapper',
    recovery_input_policy='Byte-identical observation copy outside datasets; no plaintext or target labels staged',
    candidate_vocabulary='full original 262144 IDs',
    calibration='Independent 8x1024 pilot; not complete official calibration')
dump(output / 'provenance.json', provenance)
dump(data / 'manifest.json', dict(schema=1, dataset_name='plaintext_pilot_0046',
    sample_count=1, backend='npu', device='Pixel 10', model='Gemma 4 E2B',
    capture_stage='after_response', max_output_tokens=job['max_output_tokens'],
    samples=[dict(id='0046', provenance=provenance,
        raw_kv='capture/0046/all_layers.gkv.gz', layout='capture/0046/layout.json',
        response='capture/0046/response.txt', observation='observations/0046/layer_04',
        truth='private_eval/0046/truth.json', original_prompt='private_eval/0046/prompt.txt')],
    storage_note='Raw process-owned TensorBuffer allocations in gzip GHOSTKV1; not arbitrary DRAM or encrypted storage',
    source_license='Original LMSYS terms remain applicable; see ../pilot_100/manifest.json and source card',
    files={str(p.relative_to(data)): digest(p) for p in sorted(data.rglob('*')) if p.is_file()}))
print(json.dumps(dict(dataset=str(data), tokens=len(positions), prefix_tokens=len(prefix),
                      observation=str(output / 'observation'))))
