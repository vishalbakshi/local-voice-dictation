# Findings

Stuff I learned building this on a 2019 Intel MacBook Pro (i9-9880H, Radeon Pro 5500M). Most of it cost me an afternoon, so hopefully it saves you one.

## The AMD GPU is a dead end (on macOS)

Whisper runs *slower* on the Radeon than on the CPU.

- **Vulkan (MoltenVK):** works, but large-v3-turbo's encoder took 10.5 s vs 7.1 s on CPU. ggml turns off subgroup ops for AMD on macOS because RDNA1 cards produce gibberish with them ([llama.cpp#15846](https://github.com/ggml-org/llama.cpp/issues/15846)). Without them, flash attention falls back to the CPU and matmuls run at ~5–8% of peak.
- **Metal:** no simdgroup matmul on this card, and 3 of 4 configs produced garbage. Same kind of bug, different layer.

Same RDNA1 cards reportedly work fine on Windows, so it's the macOS driver stack, not the hardware. CPU it is.

## Model choice

On ~1,650 words of me reading (Intel CPU, 8 threads):

| model | WER | avg latency |
|---|---|---|
| **small.en + prompt** | **1.8%** | 2.5 s |
| large-v3-turbo (q5_0) | 1.8% | 9.3 s |
| small.en | 2.4% | 2.3 s |
| distil-small.en | 4.0% | 1.1 s |

A vocab prompt (your names + jargon) closed the gap to turbo at ~4x the speed. Most remaining errors are jargon and formatting, which is what fine-tuning is for.

Parakeet TDT 0.6B v3 (also in whisper.cpp) was ~8x faster on short clips (0.2 s) but 2.8% WER, mostly jargon, and it can't take a prompt. Good fine-tuning candidate.

## whisper-server gotchas

The big ones. All of these looked fine in `whisper-cli` and broke in the server.

- **Turn timestamps off** (`no_timestamps=true`). With timestamps + a prompt, short clips looped (13 s for "Conclusion.") or came back empty.
- **Keep the full 30 s window** (`audio_ctx=1500`). Shrinking it to the clip length is a known speed trick, but short clips looped for ~7 s or got truncated, and results depended on the *previous* request.
- **Request params stick.** The server keeps whatever you sent last time. Send everything, every time.
- **Audio over 30 s drops text no matter what.** Timestamps off skips the rest of a window (a 32 s clip lost 60 of 84 words). Beam search didn't fix it. Timestamps on loops on a tiny leftover tail (a 30.2 s clip took 20 s). The built-in VAD stitches speech back together, so a 164 s clip still lost 85%. Fix: split it yourself. `dictate/transcribe.py` cuts at the quietest moment every 10–25 s and sends each piece on its own.

Why it's ~1.5 s even for a 2 s clip: the encoder always processes 30 s of audio (~1.2 s on this CPU), plus ~0.35 s to read the prompt. Decoding is tiny.

## Scoring gotchas

Whisper's `EnglishTextNormalizer` (the standard for WER) will quietly wreck your numbers on written text:

- it **deletes anything in parentheses or brackets** (it treats them as "(applause)"-style noise)
- it doesn't know **curly apostrophes**, so "it’s" becomes "it s"

Both made correct transcripts look 2–3x worse than they were. `bench/bench.py` fixes both.

## Hammerspoon gotchas

- **Store your event taps somewhere global.** Otherwise Lua garbage-collects them and Fn stops working after the first dictation.
- **Fn sends its own key codes** (63, 179). Ignore them or Fn cancels itself.
- **Set the Globe key to "Do Nothing"** or Fn also opens emoji/dictation.
- **ffmpeg takes ~0.8 s to open the mic.** Hence the "Tink": it plays once audio is actually hitting the file.
