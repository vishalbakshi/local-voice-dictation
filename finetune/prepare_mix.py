"""Extra data for round 3: general speech to mix in, and the blog reads as a
second validation set (to watch for forgetting).

Usage: uv run finetune/prepare_mix.py --librispeech-pc path/to/train-clean-100.json

Writes into dictations/finetune/:
  general.jsonl   ~55 min of LibriSpeech-PC train-clean-100 (fixed random sample,
                  <= 25 s each). Audio paths point at /vol/librispeech/LibriSpeech/...
                  which train_modal.py::fetch_librispeech fills on Modal.
  blog/*.wav, blog.jsonl   bench/data blog reads, labels mapped like training labels.

LibriSpeech-PC's text only uses , . ' ? which is exactly what Parakeet 110M can
write, so its labels are used as-is.
"""
import argparse
import glob
import json
import os
import random
import subprocess

from prepare_data import ROOT, to_label

OUT = os.path.join(ROOT, "dictations", "finetune")
GENERAL_MINUTES = 55
MAX_SECS = 25.0


def general(manifest):
    rows = [json.loads(l) for l in open(manifest)]
    rows = [r for r in rows if r["duration"] <= MAX_SECS]
    random.Random(0).shuffle(rows)
    picked, total = [], 0.0
    for r in rows:
        if total >= GENERAL_MINUTES * 60:
            break
        picked.append({"audio_filepath": f"/vol/librispeech/LibriSpeech/{r['audio_filepath']}",
                       "duration": r["duration"], "text": r["text"]})
        total += r["duration"]
    with open(os.path.join(OUT, "general.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in picked))
    speakers = {r["audio_filepath"].split("/")[-3] for r in picked}
    print(f"general: {len(picked)} utterances, {total / 60:.1f} min, {len(speakers)} speakers")


def blog():
    os.makedirs(os.path.join(OUT, "blog"), exist_ok=True)
    rows = []
    for flac in sorted(glob.glob(os.path.join(ROOT, "bench/data/*/*.flac"))):
        name = f"{flac.split('/')[-2]}_{os.path.basename(flac)[:-5]}"
        wav = os.path.join(OUT, "blog", f"{name}.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", flac, "-ac", "1", "-ar", "16000", wav], check=True)
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav],
                             capture_output=True, text=True, check=True).stdout
        rows.append({"audio_filepath": f"blog/{name}.wav", "duration": round(float(out), 3),
                     "text": to_label(open(flac[:-5] + ".txt").read())})
    with open(os.path.join(OUT, "blog.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print(f"blog: {len(rows)} chunks, {sum(r['duration'] for r in rows) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech-pc", required=True, help="LibriSpeech-PC train-clean-100.json (openslr.org/145)")
    a = ap.parse_args()
    general(a.librispeech_pc)
    blog()
