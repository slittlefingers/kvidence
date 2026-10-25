import json
from pathlib import Path
import tempfile
import unittest
from terminal import overnight_status

class CollectionViewTests(unittest.TestCase):
 def test_live_batch_count_and_cooling_do_not_double_count(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);batch=root/'batch';batch.mkdir()
   (batch/'progress.json').write_text(json.dumps(dict(completed=37)))
   (root/'status.json').write_text(json.dumps(dict(state='collecting',completed=500,batch=str(batch))))
   self.assertEqual(overnight_status(root)['completed'],537)
   (root/'status.json').write_text(json.dumps(dict(state='cooling',completed=537,batch=str(batch))))
   self.assertEqual(overnight_status(root)['completed'],537)

if __name__=='__main__':unittest.main()
