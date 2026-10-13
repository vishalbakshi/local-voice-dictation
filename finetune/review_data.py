"""Browse the prepared fine-tuning data before uploading it.

Usage: python3 finetune/review_data.py   (then open http://127.0.0.1:8766)

Shows every train/val item from dictations/finetune/ with its audio, its
training label, what the stock model heard, and (for whole clips) your
original transcript. "Drop" adds the item to dictations/train_drop.txt;
rerun finetune/prepare_data.py to apply. Fix wrong text in review/review.py
instead, since that's the source of truth.
"""
import difflib
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dictations", "finetune")
DROP = os.path.join(ROOT, "dictations", "train_drop.txt")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_data.html")
PORT = int(os.environ.get("REVIEW_DATA_PORT", 8766))


def words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()


def mismatch(label, heard):
    """Share of label words that differ from what the stock model heard (0..1+)."""
    a, b = words(label), words(heard)
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    edits = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")
    return edits / max(1, len(a))


def dropped():
    if not os.path.exists(DROP):
        return set()
    return {l.strip() for l in open(DROP) if l.strip() and not l.startswith("#")}


def items():
    out = []
    for split in ("train", "val"):
        path = os.path.join(DATA, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            r = json.loads(line)
            r["split"] = split
            r["mismatch"] = round(mismatch(r["text"], r.get("heard", "")), 3)
            out.append(r)
    return out


def set_drop(item, drop):
    cur = dropped()
    cur = cur | {item} if drop else cur - {item}
    with open(DROP, "w") as f:
        f.write("# Fine-tuning items to leave out (from finetune/review_data.py).\n")
        f.write("".join(f"{i}\n" for i in sorted(cur)))


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json", headers=()):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def send_audio(self, path):
        # Range support so the player can seek.
        data = open(path, "rb").read()
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not m:
            return self.send(200, data, "audio/wav", [("Accept-Ranges", "bytes")])
        start = int(m[1]) if m[1] else max(0, len(data) - int(m[2]))
        end = min(int(m[2]), len(data) - 1) if m[1] and m[2] else len(data) - 1
        self.send(206, data[start:end + 1], "audio/wav",
                  [("Accept-Ranges", "bytes"), ("Content-Range", f"bytes {start}-{end}/{len(data)}")])

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self.send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
        if url.path == "/api/items":
            d = dropped()
            return self.send(200, [{**r, "dropped": r["item"] in d} for r in items()])
        if url.path == "/audio":
            known = {r["item"]: r["audio_filepath"] for r in items()}
            if q.get("item") not in known:
                return self.send(404, {"error": "unknown item"})
            return self.send_audio(os.path.join(DATA, known[q["item"]]))
        self.send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/drop" and body.get("item") in {r["item"] for r in items()}:
            set_drop(body["item"], bool(body.get("drop")))
            return self.send(200, {"ok": True})
        self.send(400, {"error": "bad request"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Training data review: http://127.0.0.1:{PORT}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
