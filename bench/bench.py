# /// script
# dependencies = ["jiwer", "whisper-normalizer"]
# ///
"""WER + latency for whisper.cpp models, using the same path as dictation
(whisper-server + dictate/transcribe.py).

Usage: uv run bench/bench.py [dir ...] [--models small.en,base.en] [--prompt prompt.txt]
Defaults to every folder in bench/data. Each <name>.wav/.flac needs a <name>.txt.
Writes <dir>/bench_results.json.
"""
import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHISPER = os.path.join(ROOT, "vendor", "whisper.cpp")
sys.path.insert(0, os.path.join(ROOT, "dictate"))
import transcribe  # noqa: E402

PORT = 8179  # not the dictation server's port, so both can run
APPLE_SILICON = platform.machine() == "arm64"
THREADS = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu" if APPLE_SILICON else "hw.physicalcpu"],
                         capture_output=True, text=True).stdout.strip() or "4"

_whisper_norm = EnglishTextNormalizer()


def normalize(s):
    # Whisper's normalizer drops text in ()/[] (it treats it as non-speech) and
    # only knows straight apostrophes. In written references both matter.
    s = s.replace("’", "'").replace("‘", "'")
    return _whisper_norm(re.sub(r"[()\[\]]", ", ", s))


def start_server(model):
    args = [f"{WHISPER}/build/bin/whisper-server", "-m", f"{WHISPER}/models/ggml-{model}.bin",
            "-t", THREADS, "--host", "127.0.0.1", "--port", str(PORT)]
    if not APPLE_SILICON:
        args.append("-ng")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    sys.exit(f"whisper-server didn't start for {model}")


def as_wav(path, tmp):
    """transcribe.py wants 16 kHz mono WAV; convert anything else."""
    if path.endswith(".wav"):
        return path
    out = os.path.join(tmp, os.path.basename(path) + ".wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", "16000", out], check=True)
    return out


def run(d, models, prompt):
    clips = sorted(glob.glob(f"{d}/*.wav") + glob.glob(f"{d}/*.flac"))
    refs = [normalize(open(os.path.splitext(c)[0] + ".txt").read()) for c in clips]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        wavs = [as_wav(c, tmp) for c in clips]
        durs = [len(transcribe.read_pcm(w)) / transcribe.RATE for w in wavs]
        for model in models:
            server = start_server(model)
            rows = []
            try:
                for c, w, dur in zip(clips, wavs, durs):
                    t = time.time()
                    text, chunks = transcribe.transcribe(w, PORT, prompt)
                    rows.append({"clip": os.path.basename(c), "dur": dur, "secs": time.time() - t,
                                 "chunks": chunks, "hyp": text})
                    print(f"{model:20} {rows[-1]['clip']}  {rows[-1]['secs']:5.2f}s  {text[:80]}", flush=True)
            finally:
                server.kill()
            hyps = [normalize(r["hyp"]) for r in rows]
            results[model] = {
                "wer": jiwer.wer(refs, hyps),
                "mean_secs": sum(r["secs"] for r in rows) / len(rows),
                "max_secs": max(r["secs"] for r in rows),
                "rtf": sum(r["secs"] for r in rows) / sum(durs),
                "rows": rows,
            }
    print(f"\n{os.path.relpath(d, ROOT)}: {len(clips)} clips, {sum(durs) / 60:.1f} min")
    print(f"{'model':22}{'WER':>7}{'mean s':>9}{'max s':>8}{'RTF':>7}")
    for m, r in results.items():
        print(f"{m:22}{r['wer'] * 100:6.1f}%{r['mean_secs']:9.2f}{r['max_secs']:8.2f}{r['rtf']:7.2f}")
    json.dump(results, open(f"{d}/bench_results.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--models", default="small.en", help="comma-separated whisper.cpp model names")
    ap.add_argument("--prompt", help="vocabulary prompt file, like prompt.txt")
    a = ap.parse_args()
    dirs = a.dirs or sorted(p for p in glob.glob(f"{ROOT}/bench/data/*") if os.path.isdir(p))
    prompt = " ".join(open(a.prompt).read().split()) if a.prompt else ""
    for d in dirs:
        run(os.path.abspath(d), a.models.split(","), prompt)
