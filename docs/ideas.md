# Future ideas

## Jargon spotting, no training

Parakeet no vocab prompt. Jargon wrong.

- Keep CTC head (converter drop it now).
- CTC give per-frame token probs.
- Spell jargon with Parakeet tokenizer. ColBERT -> Col BERT.
- Find frames where pieces score high. In order. Close together. (MaxSim but ordered.)
- Score > threshold -> swap term in.
- Same as NVIDIA CTC-WS. No training.
- Not ColBERT vectors. Sound space != text space.
- Blocker: transcribe.cpp no CTC. Patch it, or ONNX second pass.
