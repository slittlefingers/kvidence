"""Append-only optional experiment telemetry, never ordinary chat history."""
import json
import time


class EventLog:
    def __init__(self, path):
        self.path=path
        self.sequence=0

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('x',encoding='utf-8')
        return self.emit

    def emit(self, event):
        self.sequence+=1
        self.stream.write(json.dumps(dict(event,sequence=self.sequence,timestamp=time.time()),
            ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
        self.stream.flush()

    def __exit__(self, kind, error, tb):
        try:
            if error is not None:
                self.emit(dict(type='error',status='interrupted' if issubclass(kind,KeyboardInterrupt) else 'failed',
                    error=str(error) or kind.__name__))
        finally:
            self.stream.close()
