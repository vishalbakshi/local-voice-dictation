# local-voice-dictation

Pointed Opus-5.5 slop cannon at this problem and fired.

Hold Fn, talk, let go. Your words show up wherever your cursor is. Everything runs on your Mac with [whisper.cpp](https://github.com/ggml-org/whisper.cpp), and nothing leaves it.

I built this on a 2019 Intel MacBook Pro. The bigger goal is fine-tuning a speech model on my own voice, so it also saves every dictation and has a tiny tool for fixing transcripts. That's your training data.

## What it does

- **Hold Fn** to dictate. **Double-tap Fn** for hands-free (Fn again to stop, Esc to cancel). Wait for the "Tink" before talking.
- Pastes at your cursor, then puts your clipboard back.
- A **vocab prompt** (`prompt.txt`) for your names and jargon. Biggest accuracy win per minute spent.
- Saves every dictation (audio + text) to `dictations/`.
- **Review tool:** pick a dictation, listen, fix the text. Every edit is versioned.
- **Benchmark:** WER + latency on ~24 min of me reading my blog posts, or your own recordings.

On my Intel i9 (CPU only, small.en + prompt): under 2 s for short stuff, ~20 s for 3-minute audio. On the bundled test set (of my blog posts) with the generic example prompt:

| test set | audio | word error rate |
|---|---|---|
| mundane | 4 min | 1.8% |
| problem formulation | 6 min | 1.7% |
| anthropomorphism (jargon-heavy) | 14 min | 4.0% |

Your own prompt helps. With mine, anthropomorphism blog post drops to 3.0%.

## Setup

macOS + [Homebrew](https://brew.sh).

```bash
git clone https://github.com/vishalbakshi/local-voice-dictation.git
cd local-voice-dictation
./setup.sh
```

It figures out your chip, builds whisper.cpp, grabs a model, writes `config.lua`, and hooks into Hammerspoon. 
You'll have to do the following:

1. System Settings → Keyboard → "Press fn key to" → **Do Nothing**
2. Quit anything else that uses Fn (Wispr Flow, Superwhisper, ...)
3. Open Hammerspoon → allow **Accessibility** → Reload Config
4. Hold Fn, wait for the Tink, talk, let go. Allow the mic.

Then put your own names and jargon in `prompt.txt`.

**Intel vs Apple Silicon:** I've only tested Intel. There it runs on the CPU, because the AMD GPU is slower (see [findings](docs/findings.md)). On Apple Silicon, setup uses Metal, which should be a lot faster, fast enough that bigger models like `large-v3-turbo` might be worth it. If you try it, I'd love to hear how it goes.

## Fixing transcripts

```bash
python3 review/review.py   # → http://127.0.0.1:8765
```

Dropdown of dictations (• unreviewed, ✓ marked correct, ✎ edited), an audio player, and a plain text box. ⌘S saves. "Mark correct" is for transcripts that were already right. Details in [data format](docs/data-format.md).

## Benchmarking

Needs [uv](https://docs.astral.sh/uv/). `uv sync` sets up a Python 3.12 env in `.venv/`.

```bash
uv sync
uv run bench/bench.py --prompt prompt.txt                  # all of bench/data
uv run bench/bench.py --models small.en,base.en            # compare models
python3 bench/read.py my-post.txt bench/data/my-post       # record your own test set
```

The included audio is me reading three of my blog posts: [mundane](https://vishalbakshi.com/blog/posts/2026-09-07-mundane/), [problem formulation](https://vishalbakshi.com/blog/posts/2026-09-02-problem-formulation/), and [anthropomorphism](https://vishalbakshi.com/blog/posts/2026-09-10-communication/).

## Settings

`config.lua`: model, mic, threads, Fn timing. Edit it, then Reload Config in Hammerspoon.

## Docs

- [findings](docs/findings.md): what I learned (AMD GPU dead end, whisper-server gotchas, scoring gotchas)
- [data format](docs/data-format.md): what gets saved and how versions work

## Next

Fine-tuning on my dictations (Whisper small.en and Parakeet), and trying faster models (Parakeet 110M, Moonshine v2 streaming).

## License

MIT
