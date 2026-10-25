import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrecon.media import image, audio
from kvrecon.media.gemma4 import validate_structure
from kvrecon.media.solver import fit, recover_latent
from kvrecon.common import EXPERIMENTS


class GemmaMediaContractTests(unittest.TestCase):
    def test_adapters_and_audio_output_projection_dimension(self):
        model = SimpleNamespace(embed_vision=object(), embed_audio=object(),
            config=SimpleNamespace(vision_config=SimpleNamespace(hidden_size=768),
                                   audio_config=SimpleNamespace(hidden_size=1024, output_proj_dims=1536)))
        with patch('kvrecon.media.gemma4.Gemma4Problem') as problem:
            image.build_problem(model, [[2, 10]], [1], [[0]], [[0, 1]], seed=29)
            self.assertIs(problem.call_args.kwargs['adapter'], model.embed_vision)
            self.assertEqual(problem.call_args.kwargs['adapter_input_dim'], 768)
            audio.build_problem(model, [[2, 11]], [1], [[0]], [[0, 3]])
            self.assertIs(problem.call_args.kwargs['adapter'], model.embed_audio)
            self.assertEqual(problem.call_args.kwargs['adapter_input_dim'], 1536)

    def test_audio_unavailable_does_not_fall_back_to_vision(self):
        with self.assertRaisesRegex(ValueError, 'no audio adapter'):
            audio.build_problem(SimpleNamespace(config=SimpleNamespace(audio_config=None)), [], [], [], [])

    def test_structure_excludes_future_text_and_wrong_types(self):
        ids = [[2, 10, 10]]; pos = [1, 2]; types = [[0, 1, 1]]
        validate_structure(ids, pos, types, 10, 1, 20)
        for bad_ids, bad_pos, bad_types in [([[2, 10, 10, 5]], pos, [[0, 1, 1, 0]]),
                                           (ids, pos, [[0, 3, 3]]),
                                           ([[2, 10, 11]], pos, types),
                                           ([[-1, 10, 10]], pos, types),
                                           (ids, [0, 2], types)]:
            with self.assertRaises(ValueError):
                validate_structure(bad_ids, bad_pos, bad_types, 10, 1, 20)

    def test_native_fit_selected_and_nonfinite_rejected(self):
        p = SimpleNamespace(fit_native=lambda: (np.ones((1, 2, 3)), .001, []))
        z, loss, trace = fit(p)
        self.assertEqual(z.shape, (1, 2, 3)); self.assertEqual(loss, .001)
        p.fit_native = lambda: (np.array([np.nan]), .1, [])
        with self.assertRaises(ValueError):fit(p)

    def test_context_connects_real_factory_to_shared_verification(self):
        import tempfile
        from kvrecon.search import SearchConfig
        from test_media import MediaBackend
        backend = MediaBackend(); backend.position = 2
        backend.media_context = lambda kind, positions: {'model': 'reconstructed backend model',
                                                          'input_ids': [[0, 1, 4, 4]], 'token_types': [[0, 0, 1, 1]]}
        with tempfile.TemporaryDirectory() as d:
            request = dict(modality='image', span={'start': 2, 'end': 4}, positions=np.array([2, 3]),
                targets=np.array([[30.], [40.]]), config=SearchConfig(epsilon=.01), output=Path(d)/'result')
            with patch('kvrecon.media.image.build_problem') as factory:
                factory.return_value = SimpleNamespace(initial=np.array([[3.], [4.]]), steps=0,
                    loss_and_grad=lambda u: (0., np.zeros_like(u)), project=lambda u: u)
                report = image.recover(backend, request)
                self.assertEqual(report['status'], 'complete')
                self.assertEqual(factory.call_args.kwargs['model'], 'reconstructed backend model')
                np.testing.assert_array_equal(factory.call_args.kwargs['targets'], request['targets'])
                self.assertEqual([x[0] for x in backend.media_calls], ['query', 'commit'])


@unittest.skipUnless(os.environ.get('KVRECON_TEST_MLX_MEDIA') == '1',
                     'Opt-in real Gemma model verification; requires local MLX assets')
class RealGemmaMediaPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        import importlib.util
        from mlx_vlm import load
        cls.model, _ = load(str(EXPERIMENTS/'kv_guided_search_gemma4_int8/model'))
        cls.model.eval()
        source = EXPERIMENTS/'gemma4_multimodal_layer5/runtime.py'
        spec = importlib.util.spec_from_file_location('legacy_media_reference', source)
        cls.legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.legacy)

    def check_modality(self, kind):
        import mlx.core as mx
        file = EXPERIMENTS/f'gemma4_multimodal_layer5/results_long/observed/{kind}01/observation.npz'
        with np.load(file, allow_pickle=False) as obs:
            pos, target = obs['media_positions'].copy(), obs['kv'].copy()
            end = int(pos[-1])+1
            ids, types = obs['input_ids'][:, :end].copy(), obs['token_types'][:, :end].copy()
        module = image if kind == 'image' else audio
        problem = module.build_problem(self.model, ids, pos, target, types, seed=17, steps=3)
        reference = self.legacy.Layer5Map(self.model, ids, pos, types)
        adapter = self.model.embed_vision if kind=='image' else self.model.embed_audio
        u = mx.array(problem.initial); t = mx.array(target); energy = mx.mean(t.square())
        z = adapter(u.astype(mx.bfloat16))
        np.testing.assert_array_equal(np.array(problem.forward(z)), np.array(reference(z)))
        def old_loss(p):return mx.mean((reference(adapter(p.astype(mx.bfloat16)))-t).square())/energy
        vg = mx.value_and_grad(old_loss)
        old_value, old_grad = vg(u); value, grad = problem.loss_and_grad(problem.initial)
        self.assertEqual(value, float(old_value.item()))
        np.testing.assert_array_equal(grad, np.array(old_grad))
        # Independently repeat the legacy updates; check the migrated optimizer,
        # not just a second call into the new function.
        m = mx.zeros_like(u); v = mx.zeros_like(u); best = float('inf'); best_u = u
        for i in range(4):
            loss, grad = vg(u); mx.eval(loss, grad)
            if float(loss.item()) < best:best, best_u = float(loss.item()), u
            if i == 3:break
            g = grad.astype(mx.float32); m = .9*m+.1*g; v = .999*v+.001*g.square()
            lr = .08*(.15+.85*.5*(1+np.cos(np.pi*i/3)))
            u = u-lr*(m/(1-.9**(i+1)))/(mx.sqrt(v/(1-.999**(i+1)))+1e-8)
            u = u/mx.sqrt(mx.mean(u.square(), axis=-1, keepdims=True)+1e-8)
            mx.eval(u, m, v)
        actual, fitted_loss, _ = fit(problem)
        self.assertEqual(fitted_loss, best)
        expected = np.array(adapter(best_u.astype(mx.bfloat16)).astype(mx.float32))
        np.testing.assert_array_equal(actual, expected)
        replay, cache, logits = problem.replay(actual)
        np.testing.assert_array_equal(replay, np.array(problem.forward(mx.array(actual)))[0])
        self.assertEqual(cache[4].offset, end)

    def test_image_forward_gradient_updates_and_full_replay(self):self.check_modality('image')
    def test_audio_forward_gradient_updates_and_full_replay(self):self.check_modality('audio')


if __name__ == '__main__':unittest.main()
