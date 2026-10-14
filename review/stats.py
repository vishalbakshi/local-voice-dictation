"""Review progress ("run the numbers"): a table comparing earlier checks with now.

Usage: python3 review/stats.py [--goal-minutes 60] [--no-save]

Reads dictations/ (audio length from WAV size, review status from
*.versions.json, model/engine from log.jsonl). Each run appends a snapshot to
dictations/stats_history.jsonl (gitignored) so the next run can compare.
Prints Markdown: progress table, Parakeet-era goal, edit-type table.
"""
import argparse
import collections
import difflib
import glob
import json
import os
import re
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dictations")
HISTORY = os.path.join(DATA, "stats_history.jsonl")


def load_log():
    out = {}
    path = os.path.join(DATA, "log.jsonl")
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out["/".join(r.get("wav", "").removesuffix(".wav").split("/")[-2:])] = r
    return out


def edit_kind(a, b):
    bare = lambda x: re.sub(r"[^\w\s']", "", x)
    if a.lower() == b.lower():
        return "capitalization"
    if bare(a) == bare(b):
        return "punctuation"
    if bare(a).lower() == bare(b).lower():
        return "caps + punct"
    if not a:
        return "words added"
    if not b:
        return "words removed"
    return "word changes"


def snapshot():
    log = load_log()
    s = collections.Counter()
    kinds, examples = collections.Counter(), collections.defaultdict(collections.Counter)
    for wav in glob.glob(os.path.join(DATA, "*", "*.wav")):
        iid = "/".join(wav[:-4].split("/")[-2:])
        mins = max(0, (os.path.getsize(wav) - 44) / 32000) / 60
        vpath = wav[:-4] + ".versions.json"
        vs = json.load(open(vpath))["versions"] if os.path.exists(vpath) else []
        sources = {v["source"] for v in vs}
        status = "edited" if "edited" in sources else "verified" if "verified" in sources else "unreviewed"
        parakeet = log.get(iid, {}).get("engine") == "parakeet"
        s["n"] += 1; s["min"] += mins
        s[f"{status}_n"] += 1; s[f"{status}_min"] += mins
        if parakeet:
            s["pk_n"] += 1; s["pk_min"] += mins
            if status != "unreviewed":
                s["pk_rev_n"] += 1; s["pk_rev_min"] += mins
        if status == "edited":
            model_text = [v for v in vs if v["source"] == "model"][-1]["text"]
            A, B = model_text.split(), vs[-1]["text"].split()
            s["edited_words"] += len(B)
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
                if op != "equal":
                    a, b = " ".join(A[i1:i2]), " ".join(B[j1:j2])
                    k = edit_kind(a, b)
                    kinds[k] += 1
                    examples[k][f"{a[:30] or '∅'} → {b[:30] or '∅'}"] += 1
    s = {k: round(v, 1) if isinstance(v, float) else v for k, v in s.items()}
    s["reviewed_min"] = round(s.get("edited_min", 0) + s.get("verified_min", 0), 1)
    s["time"] = datetime.now().strftime("%b %-d %H:%M")
    return s, kinds, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-minutes", type=float, default=60, help="Parakeet-era reviewed-audio goal")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    now, kinds, examples = snapshot()
    history = [json.loads(l) for l in open(HISTORY)] if os.path.exists(HISTORY) else []
    cols = history[-2:] + [now]
    head = [c["time"] for c in cols[:-1]] + ["**Now**"]
    cell = lambda c, k: f"{c.get(k + '_n', 0)} ({c.get(k + '_min', 0):.1f} min)"
    rows = [
        ("Dictations", lambda c: f"{c.get('n', 0)} ({c.get('min', 0):.0f} min)"),
        ("✎ edited", lambda c: cell(c, "edited")),
        ("✓ marked correct", lambda c: cell(c, "verified")),
        ("• unreviewed", lambda c: cell(c, "unreviewed")),
        ("**Reviewed audio**", lambda c: f"{c.get('reviewed_min', 0):.1f} min"),
        ("Parakeet-era reviewed", lambda c: f"{c.get('pk_rev_min', 0):.1f} min ({c.get('pk_rev_n', 0)} clips)"),
    ]
    print("| | " + " | ".join(head) + " |")
    print("|---|" + "---|" * len(head))
    for name, f in rows:
        print(f"| {name} | " + " | ".join(f(c) for c in cols[:-1]) + (" | " if len(cols) > 1 else "") + f"**{f(now)}** |")
    done = now.get("pk_rev_min", 0)
    print(f"\n**Parakeet-era goal:** {done:.1f} of {a.goal_minutes:.0f} min ({100 * done / a.goal_minutes:.0f}%), "
          f"{max(0, a.goal_minutes - done):.1f} min to go.")
    total = sum(kinds.values())
    if total:
        print(f"\n**Edits:** {total} changes in {now.get('edited_words', 0)} edited words "
              f"({100 * total / max(now.get('edited_words', 1), 1):.1f} per 100 words)\n")
        print("| Type | Count | Examples |\n|---|---|---|")
        for k, n in kinds.most_common():
            ex = " · ".join(e for e, _ in examples[k].most_common(3))
            print(f"| {k} | {n} | {ex} |")
    if not a.no_save:
        with open(HISTORY, "a") as f:
            f.write(json.dumps(now) + "\n")


if __name__ == "__main__":
    main()
