import sys,json,tempfile
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.media.image_geometry import orthogonal_directions
from kvrecon.media.image_checkpoint import restore_state,save_state

class ImageCheckpointTests(unittest.TestCase):
    def test_replay_measured_probes_restores_adam_and_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);seed=17;target=np.ones(3);rng=np.random.default_rng(seed)
            (p/'protocol.json').write_text(json.dumps(dict(seed=seed,directions=4,geometry=True)))
            trace=[];m=np.zeros(4);v=m.copy()
            for step in (1,2):
                delta=orthogonal_directions(rng,4,4);c=.1
                rows=np.concatenate([c*delta,-c*delta]);values=rows[:,:3].astype(float)
                np.savez(p/f'round_{step:04d}_probes.npz',inputs=rows,kv=values)
                loss=.5*np.mean((values-target)**2,axis=1)
                g=np.mean(((loss[:4]-loss[4:])/(2*c))[:,None]*delta,axis=0)
                m=.9*m+.1*g;v=.999*v+.001*g*g
                trace.append(dict(round=step,directions=4,perturbation=c))
            (p/'optimization.json').write_text(json.dumps(trace))
            restored,rm,rv,steps,source=restore_state(p,target,4)
            np.testing.assert_array_equal(rm,m);np.testing.assert_array_equal(rv,v)
            self.assertEqual(steps,2);self.assertEqual(restored.bit_generator.state,rng.bit_generator.state)
            self.assertEqual(source,'replayed measured probes')
            save_state(p/'optimizer_state.npz',restored,rm,rv,steps)
            second,sm,sv,steps,source=restore_state(p,target,4)
            self.assertEqual(second.bit_generator.state,restored.bit_generator.state)
            np.testing.assert_array_equal(sm,m);np.testing.assert_array_equal(sv,v)
            self.assertEqual(source,'checkpoint')

if __name__=='__main__':unittest.main()
