import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.media.image_expansion import ExpansionPolicy

class ExpansionTests(unittest.TestCase):
    def test_improvement_resets_patience_and_accepts_expansion(self):
        p=ExpansionPolicy(.1,patience=2,trial_rounds=1)
        p.update(False);p.update(True)
        self.assertEqual(p.stale,0)
        p.update(False);e=p.update(False)
        self.assertEqual(e['action'],'expand');self.assertAlmostEqual(p.radius,.2)
        self.assertEqual(p.update(True)['action'],'accept_expansion')
        self.assertFalse(p.trial);self.assertEqual(p.stale,0)
        restored=ExpansionPolicy(**p.state());self.assertEqual(restored.state(),p.state())

    def test_failed_expansions_restore_good_radius_and_stop(self):
        p=ExpansionPolicy(.1,patience=2,trial_rounds=1)
        p.update(False);p.update(False)
        self.assertEqual(p.update(False)['action'],'reject_and_expand')
        self.assertAlmostEqual(p.radius,.4)
        e=p.update(False)
        self.assertEqual(e['action'],'reject_and_stop');self.assertTrue(p.stopped)
        self.assertEqual(p.reason,'expansion_exhausted');self.assertAlmostEqual(p.radius,.1)

    def test_cap_stagnation_stops_without_unbounded_growth(self):
        p=ExpansionPolicy(.1,patience=1,trial_rounds=1)
        p.update(False);p.update(True);p.update(False);p.update(True)
        self.assertAlmostEqual(p.radius,.4)
        self.assertEqual(p.update(False)['action'],'stop_at_cap')
        self.assertEqual(p.reason,'max_scale_plateau')

if __name__=='__main__':unittest.main()
