"""Review and correct dictation transcripts.

Usage: python3 review/review.py   (then open http://127.0.0.1:8765)

Each dictation lives in dictations/YYYY-MM-DD/HHMMSS.{wav,txt}. The .txt always
holds the current transcript. Every saved change is also appended to
HHMMSS.versions.json, so the model's original output is never lost:

  {"versions": [
    {"v": 0, "source": "model",    "text": "...", "at": "..."},
    {"v": 1, "source": "edited",   "text": "...", "at": "..."},   # changed by hand
    {"v": 2, "source": "verified", "text": "...", "at": "..."}    # checked, no change
  ]}

A dictation's status is "edited" if any version was edited by hand, else
"verified" if it was marked correct, else "model" (unreviewed).
"""
import json
import os
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DICTATIONS_DIR", os.path.join(ROOT, "dictations"))
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review.html")
ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}/\d{6}$")
PORT = int(os.environ.get("REVIEW_PORT", 8765))


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def paths(item_id):
    if not ID_RE.match(item_id or ""):
        raise ValueError("bad id")
    base = os.path.join(DATA, item_id)
    return base + ".wav", base + ".txt", base + ".versions.json"


def load_versions(item_id):
    _, txt, ver = paths(item_id)
    if os.path.exists(ver):
        return json.load(open(ver))["versions"]
    # First look: the .txt is still the model's output.
    text = open(txt).read().strip() if os.path.exists(txt) else ""
    return [{"v": 0, "source": "model", "text": text, "at": None}]


def status(versions):
    sources = {v["source"] for v in versions}
    return "edited" if "edited" in sources else "verified" if "verified" in sources else "model"


def audio_secs(wav):
    return max(0.0, (os.path.getsize(wav) - 44) / 32000)  # 16 kHz mono s16


def models_by_id():
    """Which model transcribed each dictation, from dictations/log.jsonl.
    Older entries without a model field were Whisper small.en."""
    out = {}
    log = os.path.join(DATA, "log.jsonl")
    if os.path.exists(log):
        for line in open(log):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            item_id = "/".join(r.get("wav", "").removesuffix(".wav").split("/")[-2:])
            out[item_id] = r.get("model") or "small.en"
    return out


def list_items():
    items = []
    models = models_by_id()
    for day in sorted(os.listdir(DATA), reverse=True):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", day):
            continue
        for name in sorted(os.listdir(os.path.join(DATA, day)), reverse=True):
            if not name.endswith(".wav"):
                continue
            item_id = f"{day}/{name[:-4]}"
            versions = load_versions(item_id)
            items.append({
                "id": item_id,
                "secs": round(audio_secs(os.path.join(DATA, day, name)), 1),
                "status": status(versions),
                "preview": versions[-1]["text"][:90],
                "model": models.get(item_id, "unknown"),
            })
    return items


def save(item_id, text, action):
    _, txt, ver = paths(item_id)
    versions = load_versions(item_id)
    text = text.strip()
    if text != versions[-1]["text"]:
        source = "edited"  # changed text always counts as a manual edit
    elif action == "verify":
        source = "verified"
    else:
        return versions  # nothing changed
    versions.append({"v": len(versions), "source": source, "text": text, "at": now()})
    json.dump({"versions": versions}, open(ver, "w"), indent=2)
    with open(txt, "w") as f:
        f.write(text + "\n")
    return versions


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
        # Browsers only let you seek if the server answers Range requests.
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
        try:
            if url.path == "/":
                self.send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
            elif url.path == "/api/list":
                self.send(200, list_items())
            elif url.path == "/api/item":
                versions = load_versions(q["id"])
                self.send(200, {"id": q["id"], "versions": versions, "status": status(versions),
                                "model": models_by_id().get(q["id"], "unknown")})
            elif url.path == "/audio":
                self.send_audio(paths(q["id"])[0])
            else:
                self.send(404, {"error": "not found"})
        except (KeyError, ValueError, FileNotFoundError) as e:
            self.send(400, {"error": str(e)})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            versions = save(body["id"], body["text"], body.get("action", "edit"))
            self.send(200, {"versions": versions, "status": status(versions)})
        except (KeyError, ValueError, FileNotFoundError) as e:
            self.send(400, {"error": str(e)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Review tool: http://127.0.0.1:{PORT}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
