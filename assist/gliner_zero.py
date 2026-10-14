"""Zero-shot GLiNER2 as a problem-span flagger, scored against spans.jsonl.

Usage (GLiNER2 needs torch, which is capped at 2.2.2 on Intel Macs, so it gets
its own env):
  uv run --no-project --python 3.12 --with "gliner2[local]" --with torch==2.2.2 \\
    --with "numpy<2" --with "transformers>=4.43,<4.50" python assist/gliner_zero.py

A gold span counts as found if a predicted span of the same label overlaps it.
Precision: share of predicted spans that overlap a gold span of the same label.
"""
import collections
import json
import os

from gliner2 import GLiNER2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dictations", "assist")
MODEL = os.environ.get("GLINER_MODEL", "fastino/gliner2-base-v1")

# label in spans.jsonl -> (GLiNER label, description it sees)
LABELS = {
    "number": ("spoken number or time", "a number, time, version or amount said out loud as words"),
    "hyphen": ("compound term needing a hyphen", "two or more words that are normally written joined by a hyphen"),
    "caps": ("title or proper noun", "a name, title, product or term that should be capitalized"),
    "punct": ("quoted term or label", "a word or phrase that should be in quotation marks or followed by a colon"),
    "word": ("misheard word", "a word that looks like a speech recognition mistake"),
    "break": ("run-on sentence boundary", "a place where one sentence ends and the next begins but no period was written"),
}


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def main():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "spans.jsonl"))]
    model = GLiNER2.from_pretrained(MODEL)
    schema = {name: desc for name, desc in LABELS.values()}
    back = {name: key for key, (name, _) in LABELS.items()}
    results = {}
    for thr in (0.3, 0.5):
        gold_n, found, pred_n, pred_ok = (collections.Counter() for _ in range(4))
        for subset in ("parakeet_era", "all"):
            pass
        per_row = []
        for r in rows:
            out = model.extract_entities(r["text"], schema, threshold=thr, include_spans=True)["entities"]
            preds = [{"start": e["start"], "end": e["end"], "label": back[name], "text": e["text"]}
                     for name, ents in out.items() for e in ents]
            per_row.append({"id": r["id"], "parakeet_era": r["parakeet_era"], "preds": preds})
            for g in r["spans"]:
                gold_n[g["label"]] += 1
                found[g["label"]] += any(p["label"] == g["label"] and overlaps(p, g) for p in preds)
            for p in preds:
                pred_n[p["label"]] += 1
                pred_ok[p["label"]] += any(g["label"] == p["label"] and overlaps(p, g) for g in r["spans"])
        results[thr] = {k: {"gold": gold_n[k], "recall": found[k] / max(gold_n[k], 1),
                            "predicted": pred_n[k], "precision": pred_ok[k] / max(pred_n[k], 1)} for k in LABELS}
        print(f"\nthreshold {thr}  (all {len(rows)} drafts)")
        print(f"{'category':8} {'gold':>5} {'found':>7} {'flags':>6} {'right':>7}")
        for k, v in results[thr].items():
            print(f"{k:8} {v['gold']:5} {v['recall']*100:6.0f}% {v['predicted']:6} {v['precision']*100:6.0f}%")
        json.dump({"model": MODEL, "threshold": thr, "rows": per_row},
                  open(os.path.join(DATA, f"gliner_zero_{thr}.json"), "w"))
    json.dump(results, open(os.path.join(DATA, "gliner_zero_scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
