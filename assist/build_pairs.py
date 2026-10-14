"""Build training pairs for the review assistant: ASR output -> your reviewed text.

Usage: uv run assist/build_pairs.py [--parakeet ft-mix_100-Q8_0] [--no-whisper]

For every reviewed dictation (edited or marked correct, not excluded), re-run
the current dictation model (Parakeet) and, for the "second opinion" test,
Whisper small.en + prompt.txt on the same audio. Target = your final text,
verbatim. Re-running matters: older dictations were transcribed by Whisper,
but the assistant has to fix Parakeet's mistakes.

Split by session, matching the speech-model work: test = dictations/split.json
test ids, val = the Parakeet fine-tune's val clips, train = everything else.
Writes dictations/assist/pairs.jsonl (local, gitignored).
"""
import argparse
import glob
import json
import os
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "dictate"))
os.environ.setdefault("TRANSCRIBE_LIBRARY", os.path.join(ROOT, "vendor/transcribe.cpp/build/src/libtranscribe.dylib"))
OUT = os.path.join(ROOT, "dictations", "assist")


def reviewed():
    excl_path = os.path.join(ROOT, "dictations/exclude.txt")
    excluded = {l.strip() for l in open(excl_path) if l.strip() and not l.startswith("#")} if os.path.exists(excl_path) else set()
    log = {}
    for line in open(os.path.join(ROOT, "dictations/log.jsonl")):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        log["/".join(r["wav"].removesuffix(".wav").split("/")[-2:])] = r
    out = []
    for wav in sorted(glob.glob(os.path.join(ROOT, "dictations/*/*.wav"))):
        iid = "/".join(wav[:-4].split("/")[-2:])
        vpath = wav[:-4] + ".versions.json"
        if iid in excluded or not os.path.exists(vpath):
            continue
        vs = json.load(open(vpath))["versions"]
        if not any(v["source"] in ("edited", "verified") for v in vs):
            continue
        target = vs[-1]["text"].strip()
        if not target:
            continue
        out.append({"id": iid, "wav": wav, "target": target,
                    "edited": any(v["source"] == "edited" for v in vs),
                    "dictated_with": log.get(iid, {}).get("model") or "small.en"})
    return out


def split_of(ids):
    test = set(json.load(open(os.path.join(ROOT, "dictations/split.json")))["test"])
    val_path = os.path.join(ROOT, "dictations/finetune/val.jsonl")
    val = {json.loads(l)["clip"] for l in open(val_path)} if os.path.exists(val_path) else set()
    return {i: "test" if i in test else "val" if i in val else "train" for i in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parakeet", default="ft-mix_100-Q8_0", help="GGUF name in vendor/models (the dictation model)")
    ap.add_argument("--no-whisper", action="store_true", help="skip the Whisper second opinion")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    clips = reviewed()
    splits = split_of([c["id"] for c in clips])
    print(f"{len(clips)} reviewed clips:", {s: sum(1 for c in clips if splits[c['id']] == s) for s in ("train", "val", "test")}, flush=True)

    import transcribe_cpp
    model = transcribe_cpp.Model(os.path.join(ROOT, "vendor/models", f"{a.parakeet}.gguf"))
    with model.session(n_threads=8) as s:
        for c in clips:
            with wave.open(c["wav"]) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            c["secs"] = round(len(pcm) / 16000, 2)
            c["parakeet"] = " ".join(s.run(pcm).text.split())
    print("parakeet done", flush=True)

    if not a.no_whisper:
        import bench as B
        import transcribe as T
        prompt = " ".join(open(os.path.join(ROOT, "prompt.txt")).read().split())
        server = B.start_server("small.en")
        try:
            for k, c in enumerate(clips):
                c["whisper"], _ = T.transcribe(c["wav"], B.PORT, prompt)
                if k % 50 == 0:
                    print(f"whisper {k}/{len(clips)}", flush=True)
        finally:
            server.kill()

    with open(os.path.join(OUT, "pairs.jsonl"), "w") as f:
        for c in clips:
            f.write(json.dumps({"id": c["id"], "split": splits[c["id"]], "secs": c["secs"],
                                "edited": c["edited"], "dictated_with": c["dictated_with"],
                                "parakeet": c["parakeet"], "whisper": c.get("whisper"),
                                "target": c["target"], "parakeet_model": a.parakeet}) + "\n")
    print(f"wrote {OUT}/pairs.jsonl", flush=True)


if __name__ == "__main__":
    main()
