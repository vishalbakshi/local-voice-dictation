"""Compare speech models on accuracy and on the wait after you stop talking.

Runs on transcribe.cpp (Parakeet, Moonshine streaming) and on this repo's
whisper-server path (Whisper + vocab prompt) so the baseline matches dictation.

Setup (once): build vendor/transcribe.cpp with -DTRANSCRIBE_BUILD_SHARED=ON, then
  uv sync && uv pip install ./vendor/transcribe.cpp/bindings/python

Usage:
  uv run bench/compare.py manifest.json --out results.json [--whisper small.en --prompt prompt.txt]

manifest.json: [{"id": "...", "audio": "path.wav", "ref": "reference text"}, ...]
Models (GGUF, from huggingface.co/handy-computer) are found in vendor/models/.

Timing, per model type:
  offline   (Parakeet, Whisper): the whole clip is transcribed after you stop,
            so the wait = transcription time.
  streaming (Moonshine): audio is fed in FEED_MS chunks as if live; partial
            transcripts are off. The wait = finalize time. "keep-up" = compute
            spent during feeding / audio length; must stay well under 1.
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys
import time
import wave

import jiwer
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRANSCRIBE_LIBRARY", os.path.join(ROOT, "vendor/transcribe.cpp/build/src/libtranscribe.dylib"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "dictate"))
from bench import normalize  # noqa: E402  (fixes Whisper normalizer gotchas)

THREADS = int(os.environ.get("THREADS", "8"))
# Feed cadence for streaming. Each feed call has a fixed cost on CPU (~65 ms
# for moonshine-streaming-small on an Intel i9), so 80 ms chunks barely keep
# up with real time; 1 s chunks use ~8% of real time. Leftover audio is
# handled at finalize, which is what the measured wait includes.
FEED_MS = int(os.environ.get("FEED_MS", "1000"))
GGUF = {
    "parakeet-tdt-0.6b-v2": "parakeet-tdt-0.6b-v2-Q8_0.gguf",
    "parakeet-tdt_ctc-110m": "parakeet-tdt_ctc-110m-Q8_0.gguf",
    "moonshine-streaming-small": "moonshine-streaming-small-Q8_0.gguf",
    "moonshine-streaming-medium": "moonshine-streaming-medium-Q8_0.gguf",
}


def load_pcm(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def formatted(s):
    """Case and punctuation count; only quotes and whitespace are unified."""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.split())


def kinds(ref, hyp):
    """Count word-level edits by kind, like the review-progress numbers."""
    c = collections.Counter()
    A, B = hyp.split(), ref.split()
    strip = lambda x: re.sub(r"[^\w\s']", "", x)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B).get_opcodes():
        if op == "equal":
            continue
        a, b = " ".join(A[i1:i2]), " ".join(B[j1:j2])
        if a.lower() == b.lower(): c["caps"] += 1
        elif strip(a) == strip(b): c["punct"] += 1
        elif strip(a).lower() == strip(b).lower(): c["caps+punct"] += 1
        else: c["words"] += 1
    return c


def pieces(path, pipeline):
    """The clip as float32 PCM pieces. With --pipeline, long audio is cut at
    pauses (same rule as dictate/transcribe.py, 10-25 s pieces), the way a
    dictation tool would transcribe finished pieces while you keep talking."""
    import transcribe as T
    samples = T.read_pcm(path)
    bounds = [0] + (T.cut_points(samples) if pipeline and len(samples) / 16000 > T.SINGLE_MAX else []) + [len(samples)]
    pcm = np.frombuffer(samples.tobytes(), dtype=np.int16).astype(np.float32) / 32768
    return [pcm[a:b] for a, b in zip(bounds, bounds[1:])]


def run_transcribe_cpp(name, items, pipeline):
    import transcribe_cpp
    # Known names map to their GGUF; anything else is a file in vendor/models (e.g. ft-full-Q8_0).
    model = transcribe_cpp.Model(os.path.join(ROOT, "vendor", "models", GGUF.get(name, f"{name}.gguf")))
    streaming = name.startswith("moonshine-streaming")
    rows = []
    with model.session(n_threads=THREADS) as session:
        session.run(load_pcm(items[0]["audio"])[:16000])  # warm-up
        for it in items:
            texts, feed, compute = [], 0.0, 0.0
            for pcm in pieces(it["audio"], pipeline):
                if not streaming:
                    t = time.perf_counter()
                    texts.append(session.run(pcm).text)
                    wait = time.perf_counter() - t
                    compute += wait
                    continue
                opts = transcribe_cpp.MoonshineStreamingOptions(min_decode_interval_ms=10**9)  # no partials
                with session.stream(family=opts) as st:
                    step = FEED_MS * 16
                    for i in range(0, len(pcm), step):
                        t = time.perf_counter()
                        st.feed(pcm[i:i + step])
                        feed += time.perf_counter() - t
                    t = time.perf_counter()
                    st.finalize()
                    wait = time.perf_counter() - t
                    compute += wait
                    texts.append(st.snapshot().text)
            # The wait is only the last piece: earlier ones finish while you talk.
            row = {"wait": wait, "compute": compute + feed, "text": " ".join(texts)}
            if streaming:
                row["feed"] = feed
            rows.append(row)
    return rows


def run_whisper(model, items, prompt, pipeline):
    import tempfile
    import bench as B
    import transcribe as T
    server = B.start_server(model)
    rows = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for it in items:
                if not pipeline:  # what dictation does today: everything after you stop
                    t = time.perf_counter()
                    text, _ = T.transcribe(it["audio"], B.PORT, prompt)
                    wait = time.perf_counter() - t
                    rows.append({"wait": wait, "compute": wait, "text": text})
                    continue
                texts, compute = [], 0.0
                for k, pcm in enumerate(pieces(it["audio"], True)):
                    f = os.path.join(tmp, f"{k}.wav")
                    with wave.open(f, "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                        w.writeframes((pcm * 32768).astype(np.int16).tobytes())
                    t = time.perf_counter()
                    texts.append(T.request(f, B.PORT, prompt))
                    wait = time.perf_counter() - t
                    compute += wait
                rows.append({"wait": wait, "compute": compute, "text": " ".join(texts)})
    finally:
        server.kill()
    return rows


def summarize(items, rows):
    durs = [len(load_pcm(i["audio"])) / 16000 for i in items]
    refs = [i["ref"] for i in items]
    hyps = [r["text"] for r in rows]
    nonempty = [k for k, r in enumerate(refs) if r.strip()]
    out = {"n": len(items), "audio_min": sum(durs) / 60}
    if nonempty:
        out["wer"] = jiwer.wer([normalize(refs[k]) for k in nonempty], [normalize(hyps[k]) for k in nonempty])
        out["formatted_wer"] = jiwer.wer([formatted(refs[k]) for k in nonempty], [formatted(hyps[k]) for k in nonempty])
        k = collections.Counter()
        for i in nonempty:
            k.update(kinds(formatted(refs[i]), formatted(hyps[i])))
        out["edits"] = dict(k)
    else:  # silence set: how many words got made up
        out["made_up_words"] = sum(len(h.split()) for h in hyps)
    waits = sorted(r["wait"] for r in rows)
    out["wait_median"] = waits[len(waits) // 2]
    out["wait_p90"] = waits[int(len(waits) * 0.9)]
    out["wait_max"] = waits[-1]
    if "feed" in rows[0]:
        out["keep_up"] = sum(r["feed"] for r in rows) / sum(durs)
    # wait vs clip length, for the "does it grow with length" question
    buckets = collections.defaultdict(list)
    for d, r in zip(durs, rows):
        buckets["<5s" if d < 5 else "5-15s" if d < 15 else "15-30s" if d < 30 else "30-60s" if d < 60 else ">60s"].append(r["wait"])
    out["wait_by_length"] = {b: sum(v) / len(v) for b, v in buckets.items()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(GGUF))
    ap.add_argument("--whisper", help="also run this whisper.cpp model via whisper-server")
    ap.add_argument("--prompt", help="vocab prompt file for Whisper")
    ap.add_argument("--pipeline", action="store_true",
                    help="cut long audio at pauses and only time the last piece")
    a = ap.parse_args()
    items = json.load(open(a.manifest))
    prompt = " ".join(open(a.prompt).read().split()) if a.prompt else ""
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    todo = [m for m in a.models.split(",") if m] + ([f"whisper:{a.whisper}"] if a.whisper else [])
    for m in todo:
        print(f"== {m} on {len(items)} clips", flush=True)
        rows = (run_whisper(m.split(":", 1)[1], items, prompt, a.pipeline) if m.startswith("whisper:")
                else run_transcribe_cpp(m, items, a.pipeline))
        results[m] = {"summary": summarize(items, rows),
                      "rows": [{"id": i["id"], **r} for i, r in zip(items, rows)]}
        print(json.dumps(results[m]["summary"], indent=1), flush=True)
        json.dump(results, open(a.out, "w"), indent=1)
