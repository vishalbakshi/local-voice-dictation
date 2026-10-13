"""Turn reviewed dictations into Parakeet fine-tuning data (NeMo manifests).

Usage: uv run finetune/prepare_data.py [--out dictations/finetune]

Reads (all local, all gitignored):
  dictations/*/*.wav + *.versions.json   reviewed clips (edited or marked correct)
  dictations/split.json                  "test" ids are never used here
  dictations/exclude.txt                 ids never used or uploaded
  dictations/train_drop.txt              items dropped in finetune/review_data.py
Writes <out>/audio/*.wav, <out>/train.jsonl, <out>/val.jsonl, <out>/report.json.

Long clips are cut into pieces of at most MAX_SECS: the stock model's word
timestamps say where words fall, its words are lined up against your
corrected text, and cuts go in pauses where both sides line up exactly.
Pieces whose words disagree too much with your text are dropped.
"""
import argparse
import collections
import difflib
import glob
import json
import os
import random
import re
import wave
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "vendor/models/parakeet-tdt_ctc-110m-Q8_0.gguf")
RATE = 16000
MAX_SECS = 25.0
MIN_SECS = 3.0
MIN_GAP_MS = 150      # a pause has to be at least this long to cut in it
MAX_PIECE_DISAGREE = 0.4  # drop a piece if >40% of its words differ from the model's
# Hold out an item when the stock model hears clearly more words than its label
# has: the label is probably missing text (e.g. from the old long-dictation bug)
# and would teach the model to drop words. Extra words on the label side are
# fine; they're usually fillers ("um", "you know") the model leaves out.
MISSING_TEXT_RATIO, MISSING_TEXT_MIN = 1.25, 6
VAL_FRACTION = 0.1


# --- labels -------------------------------------------------------------------
# Parakeet 110M's vocabulary only has letters, digits and . , ? ! '
# Map everything else to something it can say. Formatting it can't produce
# (hyphens, quotes, colons) is restored after transcription instead.

def to_label(text):
    t = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\.{2,}|\u2026", " ", t)                    # trailing-off "..." isn't a sentence end
    t = re.sub(r"(\d):(\d)", r"\1 \2", t)                   # 3:30 -> 3 30 (restored later)
    t = re.sub(r"\$(\d[\d.,]*)", r"\1 dollars", t)
    t = t.replace("%", " percent")
    t = t.replace("+", " plus ")
    t = re.sub(r"\s*[—–]\s*|\s+-\s+", ", ", t)   # dashes between words -> comma
    t = re.sub(r"[:;]", ",", t)
    t = re.sub(r"[-/_]", " ", t)                          # fine-tune -> fine tune
    t = re.sub(r"[\"()\[\]]", "", t)
    t = re.sub(r"[^A-Za-z0-9.,?!' ]", " ", t)             # anything else left over
    t = re.sub(r"\s+([.,?!])", r"\1", t)
    t = re.sub(r"([.,?!])(?:\s*[.,])+", r"\1", t)          # ", ," / ". ," -> first mark
    return " ".join(t.split()).strip(" ,")


def word_key(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


# --- data selection -----------------------------------------------------------

def reviewed_clips():
    split = json.load(open(os.path.join(ROOT, "dictations/split.json")))
    test = set(split["test"])
    excl_path = os.path.join(ROOT, "dictations/exclude.txt")
    excluded = {l.strip() for l in open(excl_path) if l.strip() and not l.startswith("#")} if os.path.exists(excl_path) else set()
    clips, skipped = [], collections.Counter()
    for wav in sorted(glob.glob(os.path.join(ROOT, "dictations/*/*.wav"))):
        base = wav[:-4]
        iid = "/".join(base.split("/")[-2:])
        vpath = base + ".versions.json"
        if not os.path.exists(vpath):
            skipped["unreviewed"] += 1; continue
        vs = json.load(open(vpath))["versions"]
        if not any(v["source"] in ("edited", "verified") for v in vs):
            skipped["unreviewed"] += 1; continue
        if iid in test: skipped["test split"] += 1; continue
        if iid in excluded: skipped["excluded"] += 1; continue
        text = vs[-1]["text"].strip()
        label = to_label(text)
        if not label: skipped["empty transcript"] += 1; continue
        clips.append({"id": iid, "wav": wav, "text": text, "label": label,
                      "at": datetime.strptime(iid.replace("/", ""), "%Y-%m-%d%H%M%S")})
    return clips, skipped


def load_pcm(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


# --- cutting long clips ---------------------------------------------------------

def cut_long(pcm, label, session):
    """Split one long clip into (start_sample, end_sample, label) pieces, or [] if it can't be done safely."""
    res = session.run(pcm.astype(np.float32) / 32768, timestamps="word")
    mw = [(word_key(w.text), w.t0_ms, w.t1_ms) for w in res.words if word_key(w.text)]
    rw = label.split()
    rk = [word_key(w) for w in rw]
    sm = difflib.SequenceMatcher(None, [k for k, _, _ in mw], rk, autojunk=False)
    m2r = {}  # model word index -> ref word index, exact matches only
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            m2r[a + k] = b + k
    # Safe cut after model word i: words i and i+1 both match consecutive ref
    # words, and there's a pause between them.
    cands = []
    for i in range(len(mw) - 1):
        if i in m2r and i + 1 in m2r and m2r[i + 1] == m2r[i] + 1:
            gap = mw[i + 1][1] - mw[i][2]
            if gap >= MIN_GAP_MS:
                cands.append((i, (mw[i][2] + mw[i + 1][1]) / 2, gap))
    total_ms = len(pcm) / RATE * 1000
    # Greedy: from the current start, take the longest-gap cut that keeps the
    # piece within MIN..MAX seconds; prefer later cuts when gaps tie.
    pieces, start_ms, start_ref, start_m = [], 0.0, 0, 0
    while total_ms - start_ms > MAX_SECS * 1000:
        ok = [c for c in cands if MIN_SECS * 1000 <= c[1] - start_ms <= MAX_SECS * 1000 and c[0] >= start_m]
        if not ok:
            return []  # no safe place to cut; drop the clip rather than guess
        i, t, _ = max(ok, key=lambda c: (c[2], c[1]))
        pieces.append((start_ms, t, start_ref, m2r[i] + 1, start_m, i + 1))
        start_ms, start_ref, start_m = t, m2r[i] + 1, i + 1
    pieces.append((start_ms, total_ms, start_ref, len(rw), start_m, len(mw)))
    out = []
    for s_ms, e_ms, r0, r1, m0, m1 in pieces:
        ref_piece = rw[r0:r1]
        # disagreement between the model's words and your words in this piece
        d = difflib.SequenceMatcher(None, [k for k, _, _ in mw[m0:m1]], [word_key(w) for w in ref_piece], autojunk=False)
        agree = sum(n for _, _, n in d.get_matching_blocks())
        disagree = 1 - agree / max(1, len(ref_piece))
        out.append((int(s_ms / 1000 * RATE), int(e_ms / 1000 * RATE), " ".join(ref_piece), disagree))
    return out


def missing_text(label, heard):
    n_label, n_heard = len(label.split()), len(heard.split())
    return n_heard - n_label >= MISSING_TEXT_MIN and n_heard > MISSING_TEXT_RATIO * n_label


def write_wav(path, pcm):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def sessions_of(clips):
    clips = sorted(clips, key=lambda c: c["at"])
    groups = [[clips[0]]]
    for a, b in zip(clips, clips[1:]):
        (groups[-1].append(b) if (b["at"] - a["at"]).total_seconds() < 1800 else groups.append([b]))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dictations/finetune"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True)

    import transcribe_cpp
    clips, skipped = reviewed_clips()
    drop_path = os.path.join(ROOT, "dictations/train_drop.txt")
    dropped = {l.strip() for l in open(drop_path) if l.strip() and not l.startswith("#")} if os.path.exists(drop_path) else set()
    model = transcribe_cpp.Model(MODEL)
    rows, report = [], collections.Counter(skipped)
    dropped_examples, flagged = [], []
    with model.session(n_threads=8) as session:
        for c in clips:
            pcm = load_pcm(c["wav"])
            secs = len(pcm) / RATE
            name = c["id"].replace("/", "_")
            if secs <= MAX_SECS:
                heard = session.run(pcm.astype(np.float32) / 32768).text
                if missing_text(c["label"], heard):
                    report["held out: label likely missing text"] += 1
                    flagged.append({"clip": c["id"], "label_words": len(c["label"].split()), "heard_words": len(heard.split())})
                    continue
                path = os.path.join(a.out, "audio", f"{name}.wav")
                write_wav(path, pcm)
                if name in dropped:
                    report["dropped by you"] += 1
                    continue
                rows.append({"item": name, "clip": c["id"], "audio_filepath": os.path.relpath(path, a.out),
                             "duration": round(secs, 3), "text": c["label"], "heard": heard, "source": c["text"]})
                report["kept whole"] += 1
                continue
            pieces = cut_long(pcm, c["label"], session)
            if not pieces:
                report["long, no safe cut (dropped)"] += 1
                continue
            for k, (s, e, text, disagree) in enumerate(pieces):
                heard = session.run(pcm[s:e].astype(np.float32) / 32768).text
                if missing_text(text, heard):
                    report["held out: label likely missing text"] += 1
                    flagged.append({"clip": c["id"], "piece": k, "label_words": len(text.split()), "heard_words": len(heard.split())})
                    continue
                if disagree > MAX_PIECE_DISAGREE or not text:
                    report["piece dropped (words disagree)"] += 1
                    dropped_examples.append({"clip": c["id"], "piece": k, "disagree": round(disagree, 2), "text": text[:80]})
                    continue
                item = f"{name}_{k:02d}"
                if item in dropped:
                    report["dropped by you"] += 1
                    continue
                path = os.path.join(a.out, "audio", f"{item}.wav")
                write_wav(path, pcm[s:e])
                rows.append({"item": item, "clip": c["id"], "audio_filepath": os.path.relpath(path, a.out),
                             "duration": round((e - s) / RATE, 3), "text": text, "heard": heard,
                             "piece": f"{k + 1} of {len(pieces)}"})
                report["pieces kept"] += 1
            report["long clips cut"] += 1

    # Validation = ~10% of training audio, whole sessions, picked at random (fixed seed).
    ids_by_session = sessions_of(clips)
    random.Random(0).shuffle(ids_by_session)
    total = sum(r["duration"] for r in rows)
    val_ids, acc = set(), 0.0
    for sess in ids_by_session:
        if acc >= VAL_FRACTION * total:
            break
        sess_ids = {c["id"] for c in sess}
        acc += sum(r["duration"] for r in rows if r["clip"] in sess_ids)
        val_ids |= sess_ids
    with open(os.path.join(a.out, "train.jsonl"), "w") as tr, open(os.path.join(a.out, "val.jsonl"), "w") as va:
        for r in rows:
            (va if r["clip"] in val_ids else tr).write(json.dumps(r) + "\n")

    # Half of the training sessions (random, fixed seed) for the data-amount
    # experiment: train_half.jsonl is a subset of train.jsonl.
    train_rows = [r for r in rows if r["clip"] not in val_ids]
    # Sessions vary a lot in size, so try a few orders and keep the closest to half.
    total_train = sum(r["duration"] for r in train_rows)
    dur_of = collections.defaultdict(float)
    for r in train_rows:
        dur_of[r["clip"]] += r["duration"]
    best = None
    for seed in range(200):
        order = sessions_of([c for c in clips if c["id"] not in val_ids])
        random.Random(seed).shuffle(order)
        ids, got = set(), 0.0
        for sess in order:
            d = sum(dur_of[c["id"]] for c in sess)
            if abs(got + d - total_train / 2) < abs(got - total_train / 2):
                ids |= {c["id"] for c in sess}
                got += d
        if best is None or abs(got - total_train / 2) < abs(best[1] - total_train / 2):
            best = (ids, got)
    half_ids, acc = best
    with open(os.path.join(a.out, "train_half.jsonl"), "w") as f:
        for r in train_rows:
            if r["clip"] in half_ids:
                f.write(json.dumps(r) + "\n")

    durs = {k: sum(r["duration"] for r in rows if (r["clip"] in val_ids) == (k == "val")) / 60 for k in ("train", "val")}
    n = {k: sum(1 for r in rows if (r["clip"] in val_ids) == (k == "val")) for k in ("train", "val")}
    summary = {"counts": dict(report), "train": {"items": n["train"], "minutes": round(durs["train"], 1)},
               "val": {"items": n["val"], "minutes": round(durs["val"], 1)},
               "train_half": {"items": sum(1 for r in train_rows if r["clip"] in half_ids), "minutes": round(acc / 60, 1)},
               "longest_item_s": max(r["duration"] for r in rows), "dropped_pieces": dropped_examples,
               "fix_in_review": sorted({f["clip"] for f in flagged}), "flagged": flagged}
    json.dump(summary, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("dropped_pieces", "flagged")}, indent=1))


if __name__ == "__main__":
    main()
