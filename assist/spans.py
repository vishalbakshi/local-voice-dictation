"""Turn (Parakeet draft -> your reviewed text) pairs into labeled problem spans
in the draft, for training and scoring a span flagger (GLiNER).

Usage: uv run assist/spans.py   -> dictations/assist/spans.jsonl

Each draft/target pair is aligned word by word; every difference becomes a
span over the draft's words, labeled by what kind of fix it needed. Fixes that
insert something between words (a missing colon, a sentence break) are
attached to the word just before the gap.
"""
import difflib
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dictations", "assist")

LABELS = {
    "number": "spoken number or time",
    "hyphen": "compound word that needs a hyphen",
    "punct": "missing quote, colon or punctuation",
    "caps": "title or proper noun needing capitals",
    "break": "missing sentence break",
    "word": "misheard word or term",
}
NUMBER_WORDS = set("""zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen
fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand
million billion point percent""".split())
FILLERS = {"um", "uh", "like", "you", "know", "so", "yeah", "i", "mean", "kind", "of", "sort"}


def bare(s):
    return re.sub(r"[^\w\s']", "", s)


def category(a, b):
    """Kind of fix that turns draft words `a` into target words `b` (None = ignore)."""
    A, B = a.split(), b.split()
    if A and all(w.lower().strip(".,?!") in FILLERS for w in A) and not B:
        return None  # filler you removed: a verbatim-labeling inconsistency, not a draft error
    if B and all(w.lower().strip(".,?!") in FILLERS for w in B) and not A:
        return None
    if any(w.lower().strip(".,?!") in NUMBER_WORDS for w in A) or re.search(r"\d", b):
        return "number"
    if "-" in b and bare(a).replace(" ", "").lower() == bare(b).replace("-", "").replace(" ", "").lower():
        return "hyphen"
    if bare(a).lower() == bare(b).lower():
        # same words: punctuation and/or case changed
        if re.search(r"[.?!]\s+[A-Z]", b) and not re.search(r"[.?!]\s+[A-Z]", a):
            return "break"
        if a.lower() == b.lower():
            return "caps"
        return "punct"
    return "word"


def spans_for(draft, target):
    A, B = draft.split(), target.split()
    # character offsets of each draft word
    offs, pos = [], 0
    for w in A:
        pos = draft.index(w, pos)
        offs.append((pos, pos + len(w)))
        pos += len(w)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        cat = category(" ".join(A[i1:i2]), " ".join(B[j1:j2]))
        if cat is None:
            continue
        if i1 == i2:  # insertion: attach to the word before the gap (or after, at the start)
            if not A:
                continue
            i1, i2 = (i1 - 1, i1) if i1 > 0 else (0, 1)
        start, end = offs[i1][0], offs[i2 - 1][1]
        out.append({"start": start, "end": end, "text": draft[start:end], "label": cat,
                    "fix": " ".join(B[j1:j2])})
    return out


def main():
    pairs = [json.loads(l) for l in open(os.path.join(DATA, "pairs.jsonl"))]
    with open(os.path.join(DATA, "spans.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps({"id": p["id"], "split": p["split"], "parakeet_era": p["dictated_with"].startswith("ft-"),
                                "text": p["parakeet"], "spans": spans_for(p["parakeet"], p["target"])}) + "\n")
    rows = [json.loads(l) for l in open(os.path.join(DATA, "spans.jsonl"))]
    from collections import Counter
    c = Counter(s["label"] for r in rows for s in r["spans"])
    print(f"{len(rows)} drafts, {sum(c.values())} spans:", dict(c.most_common()))


if __name__ == "__main__":
    main()
