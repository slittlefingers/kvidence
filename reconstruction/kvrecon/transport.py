"""Length-framed persistent RPC. Never retry a possibly executed mutation."""
import json
import struct

# 128 continuous Gemma candidates x 1536 float32 values, hex encoded.
MAX_JSON = 2 << 20
MAX_PAYLOAD = 128 << 20


def receive_exact(connection, size):
    chunks = bytearray(size)
    view = memoryview(chunks)
    offset = 0
    while offset < size:
        received = connection.recv_into(view[offset:])
        if not received:
            raise ConnectionError("Worker disconnected during RPC; state is unknown, do not retry")
        offset += received
    return bytes(chunks)


def exchange(connection, command):
    data = json.dumps(command, separators=(",", ":")).encode()
    if len(data) > MAX_JSON:
        raise ValueError("RPC command too large")
    connection.sendall(struct.pack("!I", len(data)) + data)
    size, = struct.unpack("!I", receive_exact(connection, 4))
    if not 0 < size <= MAX_JSON:
        raise ValueError("Invalid RPC response header size")
    reply = json.loads(receive_exact(connection, size))
    if reply.get("seq") != command["seq"]:
        raise ValueError("RPC response sequence mismatch")
    size = reply.get("payload_bytes", 0)
    if type(size) is not int or not 0 <= size <= MAX_PAYLOAD:
        raise ValueError("Invalid RPC payload size")
    payload = receive_exact(connection, size)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error", "Worker failed"))
    return reply, payload
