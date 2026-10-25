import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from capture import load_samples, validate_capture


class CaptureChecks(unittest.TestCase):
    def test_input_keeps_prompt_verbatim_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            prompt='  hello\n"world" 中文\n'
            p.write_text(json.dumps([{'id':'a','user_prompt':prompt}]))
            self.assertEqual(load_samples(p)[0]['user_prompt'],prompt)
            p.write_text(json.dumps([{'id':'a','user_prompt':prompt}]*2))
            with self.assertRaises(ValueError):load_samples(p)

    def test_raw_capture_preserves_layout_and_detects_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.gkv.gz'
            layout={'buffers':[{'name':'kv_cache_k_0','layer':0,'payload_offset':0,
                'allocated_bytes':4,'shape':[1,2],'strides':[2,1],'element_bytes':2}]}
            manifest=json.dumps(layout).encode()
            raw=b'GHOSTKV1'+struct.pack('<Q',len(manifest))+manifest+b'\x01\x00\x02\x00'
            def write(data):
                p.write_bytes(gzip.compress(data))
                return {'kv':{'compressed_bytes':p.stat().st_size,
                    'gzip_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                    'uncompressed_bytes':len(data)}}
            meta=write(raw)
            got,check=validate_capture(p,meta)
            self.assertEqual(got,layout);self.assertTrue(check['passed'])
            meta=write(raw[:-1])
            with self.assertRaises(ValueError):validate_capture(p,meta)

    def test_short100_matches_dataset_exactly(self):
        root=Path(__file__).resolve().parent
        dataset=root.parents[1]/'datasets/shadow_chat/provenance/pilot_100_short/inputs/encrypted.jsonl'
        self.assertEqual(load_samples(root/'short100.json'),load_samples(dataset))
        self.assertEqual(len(load_samples(root/'short100.json')),100)

    def test_media_is_preserved_and_requires_valid_identity_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);media=root/'photo.jpg';media.write_bytes(b'media transport fixture')
            row=dict(id='0001',user_prompt=' original text ',content_order=['media','text'],
                     media=dict(type='image',path='photo.jpg',sha256=hashlib.sha256(media.read_bytes()).hexdigest()))
            p=root/'inputs.jsonl';p.write_text(json.dumps(row))
            self.assertEqual(load_samples(p),[row])
            media.write_bytes(b'changed bytes')
            with self.assertRaisesRegex(ValueError,'SHA-256'):load_samples(p)
            row['media']['sha256']=hashlib.sha256(media.read_bytes()).hexdigest()
            row['content_order']=['text','media'];p.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'content_order'):load_samples(p)
            row['content_order']=['media','text'];row['media']['path']='../outside.jpg'
            p.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'outside dataset'):load_samples(p)


if __name__=='__main__':unittest.main()
