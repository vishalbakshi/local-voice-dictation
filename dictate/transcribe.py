"""Transcribe a WAV with the local whisper-server, splitting long audio at pauses.

Usage: python3 transcribe.py <file.wav> [port] [prompt_file]
Prints JSON: {"text": "...", "chunks": N}

Why split here: whisper-server decodes audio in 30 s windows, and every setting
we tried for multi-window audio failed somewhere. Timestamps off skips text
(a 32 s clip lost 60 of 84 words). Timestamps on loops on a short leftover
tail (a 30.2 s clip took 20 s). The server's own VAD rejoins speech into one
stream, so a 164 s dictation still lost most of its words. Single windows of
under 30 s with timestamps off are stable, so long audio is cut at the
quietest point every 10-25 s and each piece is sent on its own.

Standard library only (Hammerspoon runs this with the system python3).
"""
import array
import json
import math
import os
import subprocess
import sys
import tempfile
import wave

RATE = 16000
SINGLE_MAX = 28.0  # seconds; at or under this, send the file as is
MIN_CHUNK, MAX_CHUNK = 10.0, 25.0
FRAME = 0.02  # seconds per RMS frame
QUIET_SPAN = 0.3  # seconds of audio averaged when looking for a pause


def read_pcm(path):
    with wave.open(path) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
        return array.array("h", w.readframes(w.getnframes()))


def cut_points(samples):
    """Sample indices to cut at: the quietest QUIET_SPAN within each 10-25 s span."""
    step = int(FRAME * RATE)
    # Every 4th sample is plenty for finding pauses and keeps this fast in pure Python.
    rms = [math.sqrt(sum(x * x for x in samples[i:i + step:4]) / (step // 4))
           for i in range(0, len(samples) - step + 1, step)]
    span = max(1, int(QUIET_SPAN / FRAME))
    cuts, start = [], 0  # in frames
    total = len(rms)
    while (total - start) * FRAME > MAX_CHUNK:
        lo, hi = start + int(MIN_CHUNK / FRAME), start + int(MAX_CHUNK / FRAME) - span
        best = min(range(lo, hi), key=lambda i: sum(rms[i:i + span]))
        cut = best + span // 2
        cuts.append(cut * step)
        start = cut
    return cuts


def request(path, port, prompt):
    # audio_ctx=1500 (full window): shrinking it to the clip length made short
    # clips loop for ~7 s or come back truncated. Params are sent every time
    # because the server keeps them between requests.
    out = subprocess.run([
        "curl", "-s", "--max-time", "120", f"http://127.0.0.1:{port}/inference",
        "-F", f"file=@{path}", "-F", "response_format=json", "-F", "language=en",
        "-F", "no_timestamps=true", "-F", "audio_ctx=1500", "-F", "beam_size=-1",
        "-F", f"prompt={prompt}",
    ], capture_output=True, text=True, check=True).stdout
    return " ".join(json.loads(out)["text"].split()).lstrip(".-,:; ")


def transcribe(path, port, prompt):
    samples = read_pcm(path)
    if len(samples) / RATE <= SINGLE_MAX:
        return request(path, port, prompt), 1
    bounds = [0] + cut_points(samples) + [len(samples)]
    texts = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
            piece = os.path.join(tmp, f"{i}.wav")
            with wave.open(piece, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(samples[a:b].tobytes())
            texts.append(request(piece, port, prompt))
    return " ".join(t for t in texts if t), len(texts)


if __name__ == "__main__":
    wav = sys.argv[1]
    port = sys.argv[2] if len(sys.argv) > 2 else "8178"
    prompt = " ".join(open(sys.argv[3]).read().split()) if len(sys.argv) > 3 else ""
    text, n = transcribe(wav, port, prompt)
    print(json.dumps({"text": text, "chunks": n}))
