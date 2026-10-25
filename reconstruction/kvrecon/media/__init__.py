"""Token-triggered media dispatch; ordinary text search remains in engine.py."""
import json
from importlib import import_module
from pathlib import Path
import numpy as np


class MediaUnavailable(RuntimeError):
    """An explicit unresolved condition, never a successful media recovery."""


def resolve_rules(profile, tokenizer):
    rules = profile.get('media', {})
    if not rules:
        return {}
    data = json.loads(Path(tokenizer).read_text())
    tokens = {t['content']: t['id'] for t in data['added_tokens'] if t.get('special')}
    resolved = {}
    used = set()
    for modality, rule in rules.items():
        if modality not in ('image', 'audio') or set(rule) != {'start', 'end', 'soft', 'handler'}:
            raise ValueError('Media rules require image/audio with start, end, soft, handler')
        entry = {'handler': rule['handler']}
        if not isinstance(entry['handler'], str) or ':' not in entry['handler']:
            raise ValueError('Media handler must be a trusted module:function')
        for key in ('start', 'end', 'soft'):
            if rule[key] not in tokens:
                raise ValueError(f'Missing special media token: {rule[key]}')
            token = tokens[rule[key]]
            if token in used:
                raise ValueError('Media token roles must be distinct')
            used.add(token)
            entry[key] = token
        resolved[modality] = entry
    return resolved


def validate_spans(spans, positions):
    """Only structure is admitted: no original media, embeddings or text IDs."""
    if not isinstance(spans, list):
        raise ValueError('Media spans must be an array')
    last = -1
    available = set(map(int, positions))
    for span in spans:
        if not isinstance(span, dict) or set(span) != {'modality', 'marker_position', 'start', 'end', 'source'}:
            raise ValueError('Media spans require modality, marker_position, start, end, source only')
        if span['modality'] not in ('image', 'audio'):
            raise ValueError('Unknown media modality')
        if any(type(span[k]) is not int for k in ('marker_position', 'start', 'end')):
            raise ValueError('Media positions must be integers')
        marker, start, end = (span[k] for k in ('marker_position', 'start', 'end'))
        if marker < 0 or start != marker + 1 or end <= start or marker <= last:
            raise ValueError('Media spans must be ordered and nonoverlapping; start follows marker')
        if marker not in available or not set(range(start, end + 1)).issubset(available):
            raise ValueError('Observation must include the start marker, whole media span and end marker')
        if not isinstance(span['source'], str) or not span['source'].strip():
            raise ValueError('Document how media boundaries are known')
        last = end
    return spans


class MediaRouter:
    def __init__(self, rules, spans, positions):
        self.rules = rules
        self.spans = {s['marker_position']: s for s in validate_spans(spans, positions)}
        if any(s['modality'] not in rules for s in spans):
            raise ValueError('Media span has no selected model rule')

    def kind(self, token, role='start'):
        return next((kind for kind, rule in self.rules.items() if rule[role] == token), None)

    def recover(self, backend, token, marker_position, targets, positions, config, output):
        modality = self.kind(token)
        if modality is None:
            raise ValueError('Only accepted media start tokens may dispatch')
        span = self.spans.get(marker_position)
        if span is None:
            raise MediaUnavailable('media_boundary_unknown: start token does not reveal media length')
        if span['modality'] != modality:
            raise MediaUnavailable('media_boundary_mismatch: recovered marker disagrees with structure')
        indices = np.flatnonzero((positions >= span['start']) & (positions < span['end']))
        module, name = self.rules[modality]['handler'].split(':', 1)
        handler = getattr(import_module(module), name)
        request = dict(modality=modality, span=dict(span), positions=positions[indices].copy(),
                       targets=np.asarray(targets)[indices].copy(), config=config, output=Path(output))
        result = handler(backend, request)
        if not isinstance(result, dict) or result.get('status') != 'complete' or backend.position != span['end']:
            raise MediaUnavailable('media_handler_incomplete: backend has not advanced to the end marker')
        return dict(result, modality=modality, start=span['start'], end=span['end'],
                    handler=self.rules[modality]['handler'], boundary_source=span['source'])
