import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrecon.media import MediaRouter, resolve_rules, validate_spans
from kvrecon.media.solver import fit
from kvrecon.config import load_model_config, DEFAULT_MODEL_CONFIG
from kvrecon.engine import reconstruct
from kvrecon.search import SearchConfig
from kvrecon.cli import evaluate
from test_core import FakeBackend


RULES = {
    'image': dict(start=1, end=2, soft=4, handler='kvrecon.media.image:recover'),
    'audio': dict(start=6, end=7, soft=5, handler='kvrecon.media.audio:recover'),
}
TABLE = [[9.], [10.], [20.], [50.], [30.], [40.], [60.], [70.]]


def span(kind='image', marker=1):
    return dict(modality=kind, marker_position=marker, start=marker+1, end=marker+3,
                source='Synthetic known-boundary fixture, no original media')


class MediaBackend(FakeBackend):
    def __init__(self):
        super().__init__(TABLE)
        self.media_calls = []
        self.bad_query = False
        self.bad_commit = False

    def media_problem(self, kind, positions, targets):
        self.media_calls.append(('fit', kind, self.position))
        # Fixed synthetic features, independent of the supplied target. This
        # fixture validates dispatch/replay, not model inversion accuracy.
        return SimpleNamespace(initial=np.array([[3.], [4.]]), steps=0,
                               loss_and_grad=lambda u: (0., np.zeros_like(u)), project=lambda u: u)

    def query_media(self, features, kind):
        self.media_calls.append(('query', kind, self.position))
        return features*10 + (100 if self.bad_query else 0)

    def commit_media(self, features, kind):
        self.media_calls.append(('commit', kind, self.position))
        self.position += len(features)
        return features*10 + (100 if self.bad_commit else 0)


class MediaTests(unittest.TestCase):
    def run_case(self, output, kind='image', backend=None, spans=None, targets=None, limit=None):
        backend = backend or MediaBackend()
        targets = np.array(targets or ([[10.], [30.], [40.], [20.], [50.]] if kind=='image'
                                     else [[60.], [30.], [40.], [70.], [50.]]))
        positions = np.arange(1, len(targets)+1)
        router = MediaRouter(RULES, [span(kind)] if spans is None else spans, positions)
        result = reconstruct(backend, targets, positions, [0],
                             SearchConfig(method='collision_plus', epsilon=.01), output,
                             max_positions=limit, media_router=router)
        return backend, result

    def test_actual_gemma_rules_resolve_without_fixed_ids(self):
        profile = load_model_config(DEFAULT_MODEL_CONFIG)
        rules = resolve_rules(profile, profile['tokenizer'])
        self.assertEqual(rules['image']['start'], 255999)
        self.assertEqual(rules['audio']['start'], 256000)
        self.assertNotEqual(rules['image']['soft'], rules['image']['start'])
        self.assertEqual(resolve_rules({}, Path('unused')), {})

    def test_each_handler_resumes_text_and_records_real_positions(self):
        for kind, ids in [('image', [1, 2, 3]), ('audio', [6, 7, 3])]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                b, r = self.run_case(d, kind)
                self.assertEqual(r['status'], 'complete')
                self.assertEqual(b.commits, ids)
                self.assertEqual(r['recovered_token_positions'], [1, 4, 5])
                self.assertEqual(r['covered_positions'], 5)
                self.assertEqual(r['media_segments'][0]['modality'], kind)
                self.assertEqual([x[0] for x in b.media_calls], ['fit', 'query', 'commit'])

    def test_plaintext_path_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            b, r = self.run_case(d, spans=[], targets=[[50.]])
            self.assertEqual(r['status'], 'complete')
            self.assertEqual(b.commits, [3]); self.assertEqual(b.media_calls, [])

    def test_missing_boundary_stops_at_marker(self):
        with tempfile.TemporaryDirectory() as d:
            b, r = self.run_case(d, spans=[])
            self.assertEqual(r['status'], 'unresolved_media')
            self.assertEqual(b.commits, [1]); self.assertEqual(b.media_calls, [])

    def test_text_only_backend_reports_unsupported(self):
        with tempfile.TemporaryDirectory() as d:
            b, r = self.run_case(d, backend=FakeBackend(TABLE))
            self.assertEqual(r['status'], 'unresolved_media')
            self.assertEqual(b.commits, [1])
            self.assertIn('media_backend_unsupported', (Path(d)/'trace.jsonl').read_text())

    def test_bad_media_fit_does_not_commit_or_resume(self):
        b = MediaBackend(); b.bad_query = True
        with tempfile.TemporaryDirectory() as d:
            _, r = self.run_case(d, backend=b)
            self.assertEqual(r['status'], 'unresolved_media')
            self.assertEqual(b.position, 2)
            self.assertEqual(b.commits, [1])

    def test_bad_media_commit_does_not_resume(self):
        b = MediaBackend(); b.bad_commit = True
        with tempfile.TemporaryDirectory() as d:
            _, r = self.run_case(d, backend=b)
            self.assertEqual(r['status'], 'unresolved_media')
            self.assertEqual(b.commits, [1])

    def test_wrong_end_marker_not_committed(self):
        with tempfile.TemporaryDirectory() as d:
            b, r = self.run_case(d, targets=[[10.], [30.], [40.], [50.], [50.]])
            self.assertEqual(r['status'], 'unresolved_media'); self.assertEqual(b.commits, [1])

    def test_soft_placeholder_cannot_be_committed_as_text(self):
        with tempfile.TemporaryDirectory() as d:
            b, r = self.run_case(d, spans=[], targets=[[30.]])
            self.assertEqual(r['status'], 'unresolved_media'); self.assertEqual(b.commits, [])

    def test_position_limit_does_not_split_media(self):
        with tempfile.TemporaryDirectory() as d:
            b, r = self.run_case(d, limit=3)
            self.assertEqual(r['status'], 'unresolved_media'); self.assertEqual(b.media_calls, [])

    def test_no_dispatch_on_unaccepted_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            b, r = self.run_case(d, spans=[], targets=[[100.]])
            self.assertEqual(r['status'], 'unresolved'); self.assertEqual(b.media_calls, [])

    def test_repeated_media_spans_and_return_to_text(self):
        with tempfile.TemporaryDirectory() as d:
            b, r = self.run_case(d, spans=[span(), span('audio', 5)],
                                targets=[[10.], [30.], [40.], [20.], [60.], [30.], [40.], [70.], [50.]])
            self.assertEqual(r['status'], 'complete')
            self.assertEqual(b.commits, [1, 2, 6, 7, 3])
            self.assertEqual(len(r['media_segments']), 2)

    def test_reject_overlaps_and_extra_truth_fields(self):
        with self.assertRaises(ValueError):validate_spans([span(), span('audio', 2)], np.arange(1, 8))
        with self.assertRaises(ValueError):validate_spans([dict(span(), features='truth.npy')], np.arange(1, 6))
        with self.assertRaises(ValueError):validate_spans([span()], np.arange(1, 4))

    def test_optimizer_reduces_independent_quadratic(self):
        goal = np.array([[.3, -.4]], dtype=np.float32)
        problem = SimpleNamespace(initial=np.zeros_like(goal), steps=300,
                                  loss_and_grad=lambda u: (np.sum((u-goal)**2), 2*(u-goal)),
                                  project=lambda u: u)
        z, loss, trace = fit(problem)
        self.assertLess(loss, 1e-8); np.testing.assert_allclose(z, goal, atol=1e-4)

    def test_evaluation_requires_positions_after_media(self):
        with tempfile.TemporaryDirectory() as d:
            _, r = self.run_case(Path(d)/'run')
            truth = Path(d)/'truth.json'
            truth.write_text(json.dumps(dict(token_ids=[1, 2, 3])))
            args = SimpleNamespace(run=Path(d)/'run', truth=truth)
            with self.assertRaisesRegex(ValueError, 'explicit unique positions'):evaluate(args)
            truth.write_text(json.dumps(dict(token_ids=[1, 2, 3], positions=[1, 4, 5])))
            evaluate(args)
            self.assertTrue(json.loads((args.run/'evaluation.json').read_text())['exact_match'])


if __name__ == '__main__':
    unittest.main()
