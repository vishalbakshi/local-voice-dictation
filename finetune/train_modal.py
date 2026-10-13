"""Fine-tune Parakeet TDT_CTC 110M on your dictations, on Modal.

  uv run modal run finetune/train_modal.py::upload                    # copy dictations/finetune to the Modal volume
  uv run modal run finetune/train_modal.py::main --run smoke          # a few steps, to check the setup
  uv run modal run --detach finetune/train_modal.py::launch          # start the round-2 runs and return;
                                                                      # they keep going if the laptop sleeps
  uv run modal run finetune/train_modal.py::fetch --run full          # download the result

Each run saves to /vol/runs/<run>/: best.nemo (lowest validation WER), the
same model converted to a transcribe.cpp GGUF, log.json with the validation
WER before and after, and metrics.csv (train loss every few steps, val loss
and WER every epoch; synced to the volume after each epoch, so it can be
checked mid-run with `uv run modal volume get local-voice-dictation runs/<run>/metrics.csv`).
"""
import json
import os

import modal

NEMO_COMMIT = "6967f48fda2a"  # same NeMo as the converter, so .nemo files round-trip
TRANSCRIBE_CPP_COMMIT = "f1dcd15858c4bbba87ca97862b4e448b2cf15b8f"  # converter verified in step 0
BASE = "nvidia/parakeet-tdt_ctc-110m"
DATA = "/vol/data"

image = (
    # The TDT loss runs numba CUDA kernels, which need libnvvm from the full
    # CUDA toolkit; the slim image doesn't have it.
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .apt_install("git", "libsndfile1", "ffmpeg")
    # numpy<2.3: numba-cuda still uses numpy.row_stack, which newer numpy removed.
    .pip_install("torch>=2.2", "soundfile", "numpy>=1.26,<2.3", "huggingface-hub>=0.20")
    .pip_install(f"nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@{NEMO_COMMIT}", "numpy<2.3")
    # NeMo's TDT loss kernels need numba-cuda (NVIDIA's split-out numba.cuda) on
    # Linux; plain numba's built-in CUDA target fails with "Signature mismatch".
    .pip_install("numba-cuda[cu12]", "gguf>=0.10.0", "sentencepiece", "numpy<2.3")
    .run_commands(
        "git clone https://github.com/handy-computer/transcribe.cpp.git /transcribe.cpp",
        f"cd /transcribe.cpp && git checkout {TRANSCRIBE_CPP_COMMIT}",
    )
)
vol = modal.Volume.from_name("local-voice-dictation", create_if_missing=True)
app = modal.App("parakeet-finetune", image=image)

# lr: AdamW peak (cosine down to min_lr); train: which weights update ("all",
# "heads" = decoder + joint only, "attn+heads" = encoder self-attention + decoder
# + joint); ctc_weight: share of the aux CTC loss (the CTC head is dropped at
# convert time); monitor: what early stopping and best-checkpoint pick on.
RUNS = {
    # round 1 (2026-10-13): CTC loss (~50) swamped TDT loss (~1) at weight 0.3,
    # so TDT barely trained; early stopping on noisy val_wer cut runs short.
    "full":   {"lr": 5e-5, "train": "all",   "ctc_weight": 0.3, "max_epochs": 40, "patience": 6, "monitor": "val_wer"},
    "frozen": {"lr": 1e-4, "train": "heads", "ctc_weight": 0.3, "max_epochs": 40, "patience": 6, "monitor": "val_wer"},
    # round 2: TDT only, cosine to 1e-6, longer, stop on val loss.
    "tdt_full": {"lr": 1e-4, "train": "all",        "ctc_weight": 0.0, "max_epochs": 60, "patience": 10, "monitor": "val_loss", "min_lr": 1e-6},
    "tdt_attn": {"lr": 3e-4, "train": "attn+heads", "ctc_weight": 0.0, "max_epochs": 60, "patience": 10, "monitor": "val_loss", "min_lr": 1e-6},
    "smoke":    {"lr": 1e-4, "train": "attn+heads", "ctc_weight": 0.0, "max_epochs": 1,  "patience": 1,  "monitor": "val_loss", "limit_batches": 5},
    # round 3: tdt_full's recipe + ~55 min of general speech (LibriSpeech-PC) to
    # stop forgetting, 20 epochs, blog reads as a 2nd validation set, and the
    # "guarded" pick: lowest val WER whose blog WER is <= stock + 0.5 points.
    # mix_100 vs mix_50 (same general data, half the personal data) tests
    # whether more personal data still helps.
    "mix_100":   {"lr": 1e-4, "train": "all", "ctc_weight": 0.0, "max_epochs": 20, "min_lr": 1e-6,
                  "data": "train.jsonl", "mix": True, "select": "guarded"},
    "mix_50":    {"lr": 1e-4, "train": "all", "ctc_weight": 0.0, "max_epochs": 20, "min_lr": 1e-6,
                  "data": "train_half.jsonl", "mix": True, "select": "guarded"},
    "smoke_mix": {"lr": 1e-4, "train": "all", "ctc_weight": 0.0, "max_epochs": 2, "min_lr": 1e-6,
                  "data": "train_half.jsonl", "mix": True, "select": "guarded", "limit_batches": 5},
}
GUARD = 0.005  # blog WER may rise at most 0.5 points over stock


def absolute_manifest(srcs, dst):
    """NeMo wants absolute audio paths; the local manifests are relative.
    (Paths that are already absolute, like general.jsonl's, are kept.)"""
    with open(dst, "w") as fo:
        for src in [srcs] if isinstance(srcs, str) else srcs:
            for line in open(src):
                r = json.loads(line)
                fo.write(json.dumps({"audio_filepath": os.path.join(DATA, r["audio_filepath"]),
                                     "duration": r["duration"], "text": r["text"]}) + "\n")


@app.function(volumes={"/vol": vol}, gpu="A10G", timeout=3 * 3600, memory=32768)
def train(run: str):
    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    import torch
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    class SyncVolume(Callback):
        """Push the metrics log to the volume after each epoch, for mid-run checks."""
        def on_validation_epoch_end(self, trainer, pl_module):
            trainer.logger.save()
            vol.commit()
    from omegaconf import open_dict

    cfg = RUNS[run]
    out = f"/vol/runs/{run}"
    os.makedirs(out, exist_ok=True)
    train_srcs = [f"{DATA}/{cfg.get('data', 'train.jsonl')}"] + ([f"{DATA}/general.jsonl"] if cfg.get("mix") else [])
    absolute_manifest(train_srcs, "/tmp/train.jsonl")
    absolute_manifest(f"{DATA}/val.jsonl", "/tmp/val.jsonl")
    absolute_manifest(f"{DATA}/blog.jsonl", "/tmp/blog.jsonl")
    guarded = cfg.get("select") == "guarded"

    model = nemo_asr.models.ASRModel.from_pretrained(BASE)
    with open_dict(model.cfg):
        common = {"sample_rate": 16000, "num_workers": 4, "pin_memory": True, "use_start_end_token": False}
        model.cfg.train_ds.update({**common, "manifest_filepath": "/tmp/train.jsonl", "batch_size": 16,
                                   "shuffle": True, "max_duration": 25.5, "min_duration": 0.2,
                                   "is_tarred": False, "tarred_audio_filepaths": None, "use_lhotse": False})
        # Guarded runs also validate on the blog reads, to watch for forgetting.
        val_manifests = ["/tmp/val.jsonl", "/tmp/blog.jsonl"] if guarded else "/tmp/val.jsonl"
        model.cfg.validation_ds.update({**common, "manifest_filepath": val_manifests, "batch_size": 16,
                                        "shuffle": False, "use_lhotse": False})
        model.cfg.compute_eval_loss = True  # off by default for TDT; we want val loss in the log
        model.cfg.optim.lr = cfg["lr"]
        model.cfg.optim.weight_decay = 1e-3
        model.cfg.optim.sched = {"name": "CosineAnnealing", "warmup_ratio": 0.05 if cfg["ctc_weight"] == 0 else 0.1,
                                 "min_lr": cfg.get("min_lr", cfg["lr"] / 20)}
        model.cfg.aux_ctc.ctc_loss_weight = cfg["ctc_weight"]
    model.ctc_loss_weight = cfg["ctc_weight"]  # like compute_eval_loss, only read at init
    model.compute_eval_loss = True  # read from cfg only at init, so set it on the model too
    model.setup_training_data(model.cfg.train_ds)
    if guarded:
        model.setup_multiple_validation_data(model.cfg.validation_ds)
    else:
        model.setup_validation_data(model.cfg.validation_ds)
    if cfg["train"] != "all":
        for p in model.parameters():
            p.requires_grad = False
        trainable = [model.decoder, model.joint]
        if cfg["train"] == "attn+heads":
            trainable += [layer.self_attn for layer in model.encoder.layers]
        for m in trainable:
            for p in m.parameters():
                p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"training {n_train / 1e6:.1f}M of {n_all / 1e6:.1f}M parameters")

    class GuardedBest(Callback):
        """Keep the weights with the lowest personal val WER among epochs whose
        blog WER stays within GUARD of stock. Metric, not loss (fastbook ch. 1)."""
        def __init__(self):
            self.active, self.keys, self.blog0 = False, None, None
            self.best, self.best_state, self.best_epoch = float("inf"), None, None
            self.any_best, self.any_state, self.trace = float("inf"), None, []

        def on_train_start(self, trainer, pl_module):
            self.active = True

        def on_validation_end(self, trainer, pl_module):
            if not self.active or trainer.sanity_checking:
                return
            m = trainer.callback_metrics
            p, b = float(m[self.keys[0]]), float(m[self.keys[1]])
            ok = b <= self.blog0 + GUARD
            self.trace.append({"epoch": trainer.current_epoch, "val_wer": p, "blog_wer": b, "passes_guard": ok})
            state = lambda: {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}
            if ok and p < self.best:
                self.best, self.best_state, self.best_epoch = p, state(), trainer.current_epoch
            if p < self.any_best:
                self.any_best, self.any_state = p, state()

    guard = GuardedBest() if guarded else None
    limit = cfg.get("limit_batches")
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision="bf16-mixed", max_epochs=cfg["max_epochs"],
        logger=CSVLogger(out, name="", version=""),  # -> /vol/runs/<run>/metrics.csv
        enable_checkpointing=True, log_every_n_steps=5, gradient_clip_val=1.0,
        limit_train_batches=limit or 1.0, limit_val_batches=limit or 1.0, num_sanity_val_steps=0,
        callbacks=[guard, SyncVolume()] if guarded else [
            EarlyStopping(monitor=cfg["monitor"], mode="min", patience=cfg["patience"]),
            ModelCheckpoint(dirpath=f"/tmp/ckpt-{run}", monitor=cfg["monitor"], mode="min", save_top_k=1),
            SyncVolume(),
        ],
    )
    model.set_trainer(trainer)
    model.setup_optimization(model.cfg.optim)  # after set_trainer, so the schedule knows the step count

    # With several validation sets NeMo logs every set's metrics into every
    # result dict, prefixed by the manifest name: val_val_wer, blog_val_wer.
    keys = ["val_val_wer", "blog_val_wer"] if guarded else ["val_wer"]

    def wers(results):
        return [float(results[0][k]) for k in keys]

    pre = trainer.validate(model, verbose=False)
    before = wers(pre)
    if guarded:
        guard.keys, guard.blog0 = keys, before[1]
    trainer.fit(model)
    if guarded:
        guard.active = False  # the re-validation below isn't a training epoch
    best = ""
    if guarded:
        state = guard.best_state or guard.any_state  # fall back if no epoch passed the guard
        model.load_state_dict(state)
        best = f"epoch {guard.best_epoch}" if guard.best_state else "none passed guard (used best val)"
    else:
        best = trainer.checkpoint_callback.best_model_path
        if best:
            model.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["state_dict"])
    after = wers(trainer.validate(model, verbose=False))

    model.save_to(f"{out}/best.nemo")
    # Convert here too, so a detached run leaves a ready-to-use GGUF behind.
    # The converter infers the variant from the output folder's name.
    import subprocess
    gguf_dir = f"{out}/parakeet-tdt_ctc-110m"
    os.makedirs(gguf_dir, exist_ok=True)
    subprocess.run(["python", "/transcribe.cpp/scripts/convert-parakeet.py", f"{out}/best.nemo",
                    f"{gguf_dir}/parakeet-tdt_ctc-110m-F32.gguf", "--repo-id", BASE], check=True)
    log = {"run": run, "config": cfg, "trainable_params_m": round(n_train / 1e6, 1),
           "val_wer_before": before[0], "val_wer_after": after[0],
           "epochs_run": trainer.fit_loop.epoch_progress.current.completed, "best_checkpoint": os.path.basename(best or "")}
    if guarded:
        log.update({"blog_wer_before": before[1], "blog_wer_after": after[1],
                    "guard_points": GUARD * 100, "selection_trace": guard.trace})
    json.dump(log, open(f"{out}/log.json", "w"), indent=1)
    vol.commit()
    return log


@app.local_entrypoint()
def main(run: str = "smoke"):
    print(json.dumps(train.remote(run), indent=1))


@app.local_entrypoint()
def launch(runs: str = "mix_100,mix_50"):
    """Start runs in the background and return. Use with `modal run --detach`."""
    for r in runs.split(","):
        call = train.spawn(r)
        print(f"started {r}: {call.object_id}")


@app.local_entrypoint()
def fetch(run: str = "full"):
    """Download a run's GGUF and log into vendor/models/."""
    dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "models")
    for src, name in [(f"/runs/{run}/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-F32.gguf", f"ft-{run}-F32.gguf"),
                      (f"/runs/{run}/log.json", f"ft-{run}-log.json")]:
        with open(os.path.join(dest, name), "wb") as f:
            for chunk in vol.read_file(src):
                f.write(chunk)
        print("wrote", os.path.join(dest, name))


@app.function(volumes={"/vol": vol}, timeout=2 * 3600, cpu=2)
def fetch_librispeech():
    """Download LibriSpeech train-clean-100 (~6.3 GB) and keep only the files
    general.jsonl uses. Run once: `modal run finetune/train_modal.py::get_general`."""
    import subprocess
    vol.reload()
    paths = [json.loads(l)["audio_filepath"] for l in open(f"{DATA}/general.jsonl")]
    members = [p.removeprefix("/vol/librispeech/") for p in paths]
    missing = [p for p in paths if not os.path.exists(p)]
    if not missing:
        return f"all {len(paths)} files already present"
    os.makedirs("/vol/librispeech", exist_ok=True)
    open("/tmp/members.txt", "w").write("\n".join(members) + "\n")
    # Download to the volume first (with retries and resume) instead of piping:
    # a dropped connection on a 6.3 GB stream killed the first attempt. (Modal's
    # extra scratch disk starts at 512 GB, so the volume is the practical spot.)
    url = "https://openslr.elda.org/resources/12/train-clean-100.tar.gz"
    tgz = "/vol/librispeech/train-clean-100.tar.gz"
    import time
    import urllib.request
    total = 6387309499
    for attempt in range(20):  # resume with Range requests until complete (no curl in this image)
        have = os.path.getsize(tgz) if os.path.exists(tgz) else 0
        if have >= total:
            break
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tgz, "ab") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as e:
            print(f"download interrupted at {have / 1e9:.2f} GB ({e}); retrying")
            time.sleep(5)
    size = os.path.getsize(tgz)
    assert size == 6387309499, f"unexpected archive size {size}"
    subprocess.run(["tar", "-xzf", tgz, "-C", "/vol/librispeech", "-T", "/tmp/members.txt"], check=True)
    os.remove(tgz)
    vol.commit()
    have = sum(os.path.exists(p) for p in paths)
    return f"extracted {have}/{len(paths)} files"


@app.local_entrypoint()
def get_general():
    print(fetch_librispeech.remote())


@app.local_entrypoint()
def upload():
    """Copy the prepared data (audio + manifests) into the volume at /data."""
    import io
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dictations", "finetune")
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(os.path.join(root, "audio"), "/data/audio")
        batch.put_directory(os.path.join(root, "blog"), "/data/blog")
        for f in ("train.jsonl", "train_half.jsonl", "val.jsonl", "blog.jsonl", "general.jsonl"):
            # Only what training needs; the review-only fields (heard, source) stay local.
            rows = [json.loads(l) for l in open(os.path.join(root, f))]
            keep = "".join(json.dumps({k: r[k] for k in ("audio_filepath", "duration", "text")}) + "\n" for r in rows)
            batch.put_file(io.BytesIO(keep.encode()), f"/data/{f}")
    print("uploaded", root)
