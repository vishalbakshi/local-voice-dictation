"""Record yourself reading a text, one chunk at a time, to make a test set.

Usage: python3 bench/read.py my-post.txt bench/data/my-post
Chunks are separated by blank lines; aim for 15-30 s of reading each.
Writes pNN.wav + pNN.txt. Ctrl-C anytime; rerun to pick up where you left off.
Set MIC to your mic's name or index (default 0); list them with:
  ffmpeg -f avfoundation -list_devices true -i ""
"""
import os
import subprocess
import sys

text_file, out = sys.argv[1], sys.argv[2]
MIC = os.environ.get("MIC", "0")
chunks = [p.strip() for p in open(text_file).read().split("\n\n") if p.strip()]
os.makedirs(out, exist_ok=True)

for i, text in enumerate(chunks, 1):
    base = f"{out}/p{i:02d}"
    if os.path.exists(base + ".txt"):  # written only once a take is accepted
        continue
    while True:
        os.system("clear")
        print(f"chunk {i} of {len(chunks)}\n\n{text}\n")
        input("Enter to start recording (q to stop)... ")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
                        "-f", "avfoundation", "-i", f":{MIC}", "-ac", "1", "-ar", "16000", base + ".wav"])
        if input("\nEnter = next, r = redo: ").strip().lower() != "r":
            break
    open(base + ".txt", "w").write(text + "\n")

print(f"\nDone: {len(chunks)} chunks in {out}/")
