# Data format

Everything you dictate lands in `dictations/` (gitignored). The idea: use it, fix what it gets wrong, fine-tune on that.

```
dictations/
├── log.jsonl              # one line per dictation
├── ffmpeg.log             # capture health: held time vs audio length, ffmpeg warnings
└── 2026-10-08/
    ├── 080410.wav         # 16 kHz mono
    ├── 080410.txt         # current transcript
    └── 080410.versions.json   # only once you've reviewed it
```

## log.jsonl

```json
{"time": "2026-10-08T15:04:10Z", "wav": ".../080410.wav", "text": "...",
 "audio_secs": 164.1, "held_secs": 174.3, "latency_secs": 19.7,
 "chunks": 9, "model": "small.en", "app": "Notes"}
```

`chunks` is how many pieces long audio got split into. `app` is where the text got pasted.

## versions.json

Written by the review tool. Every save is kept, and v0 is always the model's raw output.

```json
{"versions": [
  {"v": 0, "source": "model",    "text": "...", "at": null},
  {"v": 1, "source": "edited",   "text": "...", "at": "2026-10-08T15:36:51Z"},
  {"v": 2, "source": "verified", "text": "...", "at": "..."}
]}
```

- `model`: what the model said (can show up again if a clip gets re-transcribed; those have a `note`)
- `edited`: you changed the text by hand
- `verified`: you checked it and it was right

A clip's status is `edited` if any version was edited, else `verified`, else unreviewed. Handy for picking training data: hand-fixed only, fixed + verified, or everything.

I'm labeling **verbatim** for now (keep the ums and restarts). Plan is to fine-tune once on verbatim and once on cleaned-up text, then see which feels better.

## Benchmark data

`bench/data/<name>/pNN.{flac,txt}`: one chunk of reading + the exact text. Make your own with `bench/read.py`.
