import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]/'experiments/fixed_prefix_image_inverse'
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('overnight_collect',HERE/'overnight_collect.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ArchiveTests(unittest.TestCase):
 def test_verified_archive_preserves_split_and_removes_only_new_shard(self):
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp)/'local';archive=Path(temp)/'external';shard=base/'shard_0001';batch=shard/'batch_0001';(batch/'samples').mkdir(parents=True)
   (batch/'samples/example.npz').write_bytes(b'example')
   (batch/'index.json').write_text(json.dumps([dict(id='one',split='validation',file='samples/example.npz')]))
   with patch.object(module,'BASE',base),patch.object(module,'ARCHIVE',archive),patch.object(Path,'is_mount',return_value=True):module.archive_shard(shard,partial=True)
   dest=archive/'shard_0001_partial'
   self.assertFalse(shard.exists());self.assertEqual((dest/'batch_0001/samples/example.npz').read_bytes(),b'example')
   rows=json.loads((dest/'index.json').read_text());self.assertEqual(rows[0]['split'],'validation');self.assertEqual(rows[0]['file'],'batch_0001/samples/example.npz')
   self.assertTrue((dest/'archive_checksums.json').exists())
 def test_missing_disk_keeps_source(self):
  with tempfile.TemporaryDirectory() as temp:
   shard=Path(temp)/'shard_0001';shard.mkdir()
   with patch.object(Path,'is_mount',return_value=False),self.assertRaises(RuntimeError):module.archive_shard(shard)
   self.assertTrue(shard.exists())

if __name__=='__main__':unittest.main()
