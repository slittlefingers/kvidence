import csv
import json
import tempfile
import unittest
from pathlib import Path
from experiment_results import summarize_run,export_candidates

class ResultsTests(unittest.TestCase):
    def test_measurements_and_candidate_distances(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/'result.json').write_text(json.dumps(dict(status='complete',config={'method':'guided'},recovered_token_ids=[7],online_seconds=2)))
            (p/'protocol.json').write_text(json.dumps(dict(metric='rmse',config={'space':'dequantized'})))
            (p/'trace.jsonl').write_text(json.dumps(dict(position=4,queries=2,queried_token_ids=[3,7],distances=[1.2,.01],accepted_token_id=7,stages=[{'phase':'seed','queries':2}],search_seconds=2))+'\n')
            row=summarize_run(p)
            self.assertEqual(row['total_queries'],2);self.assertEqual(row['accepted_tokens_per_second'],.5)
            self.assertIsNone(row['correct_tokens'])
            self.assertEqual(export_candidates(p,p/'candidates.csv'),2)
            with (p/'candidates.csv').open() as handle:rows=list(csv.DictReader(handle))
            self.assertEqual(rows[0]['distance'],'1.2');self.assertEqual(rows[1]['accepted'],'True')
            self.assertEqual(rows[1]['stage'],'seed')

    def test_missing_trace_is_not_zero_queries(self):
        with tempfile.TemporaryDirectory() as temp:
            row=summarize_run(Path(temp));self.assertIsNone(row['total_queries'])

if __name__=='__main__':unittest.main()
