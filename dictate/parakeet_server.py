"""Local Parakeet transcription server for dictation (transcribe.cpp bindings).

Usage: .venv/bin/python dictate/parakeet_server.py <model.gguf> [port] [threads]

Loads the model once and keeps it in memory. dictate.lua POSTs
{"path": "/abs/file.wav"} to /inference and gets back {"text": ...}.
Only listens on 127.0.0.1. Needs the transcribe.cpp bindings in .venv:
  uv pip install ./vendor/transcribe.cpp/bindings/python

Unlike whisper-server, Parakeet has no 30 s window, so the whole recording is
transcribed in one pass; there's no vocabulary prompt either.
"""
import json
import os
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRANSCRIBE_LIBRARY", os.path.join(ROOT, "vendor/transcribe.cpp/build/src/libtranscribe.dylib"))
import transcribe_cpp  # noqa: E402

model_path = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8178
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MODEL_NAME = os.path.basename(model_path).removesuffix(".gguf")

model = transcribe_cpp.Model(model_path)
session = model.session(n_threads=THREADS)
session.run(np.zeros(16000, dtype=np.float32))  # warm-up so the first dictation isn't slow


def load_pcm(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send(200, {"model": MODEL_NAME})  # health check

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            t = time.perf_counter()
            text = session.run(load_pcm(req["path"])).text
            self.send(200, {"text": " ".join(text.split()), "chunks": 1, "model": MODEL_NAME,
                            "secs": round(time.perf_counter() - t, 3)})
        except Exception as e:  # report instead of dropping the connection
            self.send(500, {"error": str(e)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    # Single-threaded on purpose: one transcription at a time, since a session
    # can't run two at once. Dictations queue up behind each other.
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"parakeet server: {MODEL_NAME} on 127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
