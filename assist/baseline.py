"""Untrained baseline for the review assistant: a local LLM + a few of your
corrections as examples, scored on how many edits it would save you.

Usage: uv run assist/baseline.py [--model Qwen3-1.7B-Q8_0] [--split val,test] [--shots 6] [--whisper both|on|off]

Runs llama-server (from LLAMA_SERVER, default ~/llama-vulkan/llama.cpp build)
on 127.0.0.1. Everything stays local. Reads dictations/assist/pairs.jsonl
(from build_pairs.py), writes dictations/assist/baseline_<model>.json.

Score: word edits (case and punctuation count) needed to turn a draft into
your reviewed text. "Saved" = edits from Parakeet's raw text minus edits from
the suggestion. "Harmed" = clips where the suggestion needs MORE edits than raw.
"""
import argparse
import json
import os
import random
import re
import subprocess
import time
import urllib.request

import jiwer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dictations", "assist")
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", os.path.expanduser("~/llama-vulkan/llama.cpp/build/bin/llama-server"))
PORT = 8190

SYSTEM = """You correct speech-recognition transcripts of one person's voice dictations.
Return only the corrected transcript, nothing else.

Rules:
- Keep it verbatim: keep his exact words, including fillers (um, like, you know), repeats and restarts. Never summarize, reword, add, or drop content.
- Fix words the recognizer misheard, especially technical terms and names (see vocabulary).
- Write numbers, times, and versions the way he does: digits ("2:14 PM", "350M", "33x", "v1").
- Fix capitalization and punctuation, including hyphens (fine-tune), quotes, and colons where he'd write them.
- If the transcript already looks right, return it unchanged.

His vocabulary: {vocab}"""


def fmt(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.split())


def edits(target, draft):
    """Word-level edits (substitutions + insertions + deletions), case and punctuation sensitive."""
    if not target:
        return len(draft.split())
    o = jiwer.process_words(fmt(target), fmt(draft) or "∅")
    return o.substitutions + o.deletions + o.insertions


def user_msg(p, use_whisper):
    if use_whisper:
        return (f"Transcript A (main recognizer):\n{p['parakeet']}\n\n"
                f"Transcript B (second recognizer, better at technical terms, sometimes adds or drops words):\n{p['whisper']}\n\n"
                "Write the corrected transcript, based on A.")
    return f"Transcript:\n{p['parakeet']}"


def start_server(model):
    log = open(os.path.join(DATA, "llama-server.log"), "w")
    proc = subprocess.Popen([LLAMA_SERVER, "-m", os.path.join(ROOT, "vendor/models", f"{model}.gguf"),
                             "--host", "127.0.0.1", "--port", str(PORT), "-c", "8192", "--jinja",
                             "-ngl", os.environ.get("NGL", "0"), "-t", "8"], stdout=log, stderr=log)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise SystemExit("llama-server didn't start; see dictations/assist/llama-server.log")


def suggest(messages, max_tokens):
    body = json.dumps({"messages": messages, "temperature": 0, "max_tokens": max_tokens,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    return " ".join(out.strip().strip('"').split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-1.7B-Q8_0")
    ap.add_argument("--split", default="val,test")
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--whisper", default="both", choices=["both", "on", "off"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (quick check)")
    ap.add_argument("--pairs", default="pairs.jsonl", help="pairs file in dictations/assist")
    ap.add_argument("--tag", default="", help="suffix for the results file")
    a = ap.parse_args()

    pairs = [json.loads(l) for l in open(os.path.join(DATA, a.pairs))]
    evalset = [p for p in pairs if p["split"] in a.split.split(",")]
    if a.limit:
        evalset = evalset[:a.limit]
    # Few-shot examples: short training clips you actually edited, varied.
    pool = [p for p in pairs if p["split"] == "train" and p["edited"] and 8 <= len(p["target"].split()) <= 45
            and edits(p["target"], p["parakeet"]) >= 2]
    shots = random.Random(0).sample(pool, min(a.shots, len(pool)))
    vocab = ", ".join(t.strip() for t in re.split(r"[,:;]", open(os.path.join(ROOT, "prompt.txt")).read()) if 0 < len(t.split()) <= 3)

    variants = {"both": [False, True], "on": [True], "off": [False]}[a.whisper]
    server = start_server(a.model)
    results = {}
    try:
        for use_w in variants:
            name = "with_whisper" if use_w else "parakeet_only"
            msgs = [{"role": "system", "content": SYSTEM.format(vocab=vocab)}]
            for s in shots:
                msgs += [{"role": "user", "content": user_msg(s, use_w)}, {"role": "assistant", "content": s["target"]}]
            rows = []
            for k, p in enumerate(evalset):
                t = time.time()
                out = suggest(msgs + [{"role": "user", "content": user_msg(p, use_w)}],
                              max_tokens=int(len(p["parakeet"].split()) * 2.5) + 40)
                rows.append({"id": p["id"], "secs": time.time() - t, "suggestion": out,
                             "raw_edits": edits(p["target"], p["parakeet"]), "sugg_edits": edits(p["target"], out)})
                if k % 20 == 0:
                    print(f"{name} {k}/{len(evalset)}", flush=True)
            raw = sum(r["raw_edits"] for r in rows); sug = sum(r["sugg_edits"] for r in rows)
            results[name] = {
                "clips": len(rows), "edits_from_raw": raw, "edits_from_suggestion": sug,
                "saved_pct": round(100 * (raw - sug) / max(raw, 1), 1),
                "helped": sum(r["sugg_edits"] < r["raw_edits"] for r in rows),
                "harmed": sum(r["sugg_edits"] > r["raw_edits"] for r in rows),
                "median_secs": sorted(r["secs"] for r in rows)[len(rows) // 2], "rows": rows,
            }
            s = results[name]
            print(f"{name}: edits {raw} -> {sug} ({s['saved_pct']}% saved) | helped {s['helped']}, harmed {s['harmed']} of {len(rows)} | {s['median_secs']:.1f}s/clip", flush=True)
    finally:
        server.kill()
    json.dump({"model": a.model, "shots": [s["id"] for s in shots], "results": results},
              open(os.path.join(DATA, f"baseline_{a.model}{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
