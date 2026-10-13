"""Convert a NeMo Parakeet checkpoint to a transcribe.cpp GGUF, on Modal.

NeMo needs a recent PyTorch, which no longer ships for Intel Macs, so the
conversion runs on Modal (CPU is enough). Quantize locally afterwards.

Usage:
  uv run modal run finetune/convert_modal.py --source nvidia/parakeet-tdt_ctc-110m --out vendor/models/stock-110m-F32.gguf
  # or a fine-tuned checkpoint already in the Modal volume:
  #   --source /vol/runs/<run>/model.nemo
"""
import modal

TRANSCRIBE_CPP_COMMIT = "f1dcd15858c4bbba87ca97862b4e448b2cf15b8f"
NEMO_COMMIT = "6967f48fda2a"  # pinned by transcribe.cpp's scripts/envs/parakeet

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libsndfile1")
    .pip_install("torch>=2.2", "huggingface-hub>=0.20", "soundfile", "numpy>=1.26", "sentencepiece", "gguf>=0.10.0")
    .pip_install(f"nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@{NEMO_COMMIT}")
    .run_commands(
        "git clone https://github.com/handy-computer/transcribe.cpp.git /transcribe.cpp",
        f"cd /transcribe.cpp && git checkout {TRANSCRIBE_CPP_COMMIT}",
    )
)
vol = modal.Volume.from_name("local-voice-dictation", create_if_missing=True)
app = modal.App("parakeet-convert", image=image)


@app.function(volumes={"/vol": vol}, timeout=1800, cpu=4, memory=16384)
def convert(source: str) -> bytes:
    import subprocess
    import os
    # The converter infers the variant from the output folder's name.
    os.makedirs("/tmp/parakeet-tdt_ctc-110m", exist_ok=True)
    out = "/tmp/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-F32.gguf"
    # --repo-id picks the 110M profile when converting a local fine-tuned .nemo.
    subprocess.run(["python", "/transcribe.cpp/scripts/convert-parakeet.py", source, out,
                    "--repo-id", "nvidia/parakeet-tdt_ctc-110m"], check=True)
    return open(out, "rb").read()


@app.local_entrypoint()
def main(source: str = "nvidia/parakeet-tdt_ctc-110m", out: str = "vendor/models/converted-F32.gguf"):
    data = convert.remote(source)
    open(out, "wb").write(data)
    print(f"wrote {out} ({len(data) / 1e6:.0f} MB)")
