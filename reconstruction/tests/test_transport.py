import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import Mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.transport import exchange, MAX_JSON, MAX_PAYLOAD
from kvrecon.npu import NpuBackend


class FragmentedSocket:
    def __init__(self, response):
        self.response=bytearray(response);self.sent=[]
    def sendall(self, data):self.sent.append(data)
    def recv_into(self, view):
        size=min(3,len(view),len(self.response))
        view[:size]=self.response[:size];del self.response[:size]
        return size


def response(reply, payload=b''):
    data=json.dumps(reply).encode()
    return struct.pack('!I',len(data))+data+payload


class TransportTests(unittest.TestCase):
    def test_fragmented_binary_frame_and_single_send(self):
        payload=bytes(range(256))*7
        socket=FragmentedSocket(response(dict(ok=True,seq=5,payload_bytes=len(payload)),payload))
        reply,actual=exchange(socket,dict(seq=5,op='query'))
        self.assertEqual(actual,payload);self.assertEqual(reply['seq'],5)
        self.assertEqual(len(socket.sent),1)
        command=socket.sent[0]
        self.assertEqual(struct.unpack('!I',command[:4])[0],len(command)-4)

    def test_disconnect_does_not_retry(self):
        socket=FragmentedSocket(response(dict(ok=True,seq=1,payload_bytes=10),b'ab'))
        with self.assertRaises(ConnectionError):exchange(socket,dict(seq=1))
        self.assertEqual(len(socket.sent),1)

    def test_rejects_wrong_sequence_and_unbounded_frames(self):
        for data in [response(dict(ok=True,seq=2,payload_bytes=0)),
                     struct.pack('!I',MAX_JSON+1),
                     response(dict(ok=True,seq=1,payload_bytes=MAX_PAYLOAD+1)),
                     response(dict(ok=True,seq=1,payload_bytes=-1))]:
            with self.subTest(data=data),self.assertRaises(ValueError):
                exchange(FragmentedSocket(data),dict(seq=1))

    def test_worker_error(self):
        with self.assertRaisesRegex(RuntimeError,'prefix mismatch'):
            exchange(FragmentedSocket(response(dict(ok=False,seq=1,error='prefix mismatch'))),dict(seq=1))

    def test_rpc_failure_poisoning_prevents_mutation_replay(self):
        backend=NpuBackend.__new__(NpuBackend)
        backend.rpc_broken=False;backend.seq=0;backend.transport='socket';backend.socket_auth='test'
        backend.connection=FragmentedSocket(b'')
        with self.assertRaises(ConnectionError):backend.rpc('commit',id=42)
        with self.assertRaisesRegex(RuntimeError,'uncertain'):backend.rpc('commit',id=42)
        self.assertEqual(len(backend.connection.sent),1)

    def test_begin_is_fused_only_into_first_physical_query(self):
        backend=NpuBackend.__new__(NpuBackend)
        backend.transport='socket';backend.batch_capacity=128;backend.vocab_size=1000
        backend.position=7;backend.width=2;backend.dtype=np.dtype('i1');backend.pending_begin=False
        backend.rpc=Mock(return_value=b'\x01\x02')
        backend.begin_candidates();backend.rpc.assert_not_called()
        backend.query([11]);backend.query([12])
        self.assertTrue(backend.rpc.call_args_list[0].kwargs['begin_candidates'])
        self.assertFalse(backend.rpc.call_args_list[1].kwargs['begin_candidates'])


if __name__=='__main__':unittest.main()
