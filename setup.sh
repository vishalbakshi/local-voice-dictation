#!/bin/bash
# One-time setup: deps, whisper.cpp build, model, config, Hammerspoon hook.
# Usage: ./setup.sh [--no-hammerspoon]
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

WHISPER_COMMIT=d1be6fde11ac6e0407606b4e42fe72d34add8037  # tested 2026-10
MODEL="${MODEL:-small.en}"
HOOK=1
[[ "${1:-}" == "--no-hammerspoon" ]] && HOOK=0

say() { printf '\n==> %s\n' "$*"; }

[[ "$(uname)" == "Darwin" ]] || { echo "macOS only."; exit 1; }
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }

# --- chip ---------------------------------------------------------------------
if [[ "$(uname -m)" == "arm64" ]]; then
  CHIP="Apple Silicon"; GPU=true
  CMAKE_FLAGS=()  # whisper.cpp defaults to Metal here
  THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
else
  CHIP="Intel"; GPU=false
  # AMD GPUs on Intel Macs are slower than the CPU for Whisper (docs/findings.md).
  CMAKE_FLAGS=(-DGGML_METAL=OFF)
  THREADS=$(sysctl -n hw.physicalcpu)
fi
say "$CHIP Mac, $THREADS threads, gpu=$GPU"

# --- deps ---------------------------------------------------------------------
say "Homebrew deps"
brew install cmake ffmpeg
[[ $HOOK == 1 ]] && { brew list --cask hammerspoon >/dev/null 2>&1 || brew install --cask hammerspoon; }

# --- whisper.cpp ----------------------------------------------------------------
W=vendor/whisper.cpp
if [[ ! -d $W/.git ]]; then
  say "Fetching whisper.cpp @ ${WHISPER_COMMIT:0:7}"
  mkdir -p $W && git -C $W init -q
  git -C $W remote add origin https://github.com/ggml-org/whisper.cpp.git
  git -C $W fetch -q --depth 1 origin $WHISPER_COMMIT
  git -C $W checkout -q FETCH_HEAD
fi
say "Building whisper.cpp"
cmake -S $W -B $W/build -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF ${CMAKE_FLAGS[@]+"${CMAKE_FLAGS[@]}"} >/dev/null
cmake --build $W/build --config Release -j "$THREADS" --target whisper-server whisper-cli >/dev/null

say "Model: $MODEL"
[[ -f $W/models/ggml-$MODEL.bin ]] || $W/models/download-ggml-model.sh "$MODEL" >/dev/null

# --- config -------------------------------------------------------------------
if [[ ! -f config.lua ]]; then
  # Default mic: the first device with "Microphone" in its name, else the first one.
  MICS=$(ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 \
    | sed -n '/audio devices/,$p' | sed -nE 's/.*\] \[[0-9]+\] (.*)/\1/p' || true)
  MIC=$(grep -m1 Microphone <<<"$MICS" || head -1 <<<"$MICS")
  sed -e "s|@MODEL@|$MODEL|" -e "s|@GPU@|$GPU|" -e "s|@THREADS@|$THREADS|" \
      -e "s|@MIC@|$MIC|" -e "s|@FFMPEG@|$(command -v ffmpeg)|" -e "s|@PYTHON@|$(command -v python3)|" \
      config.example.lua > config.lua
  say "Wrote config.lua (mic: $MIC)"
fi
[[ -f prompt.txt ]] || cp prompt.example.txt prompt.txt

# --- Hammerspoon ----------------------------------------------------------------
if [[ $HOOK == 1 ]]; then
  LINE="dofile(\"$ROOT/dictate/dictate.lua\")"
  mkdir -p ~/.hammerspoon
  grep -qF "$LINE" ~/.hammerspoon/init.lua 2>/dev/null || echo "$LINE" >> ~/.hammerspoon/init.lua
  say "Hooked into ~/.hammerspoon/init.lua"
fi

cat <<EOF

Done. Last steps (macOS won't let a script do these):
  1. System Settings > Keyboard > "Press 🌐 key to" > Do Nothing
  2. Quit any other app that uses Fn (Wispr Flow, Superwhisper, ...)
  3. Open Hammerspoon, allow Accessibility, then Reload Config
  4. Hold Fn, wait for the "Tink", talk, let go. Allow the mic prompt.

Edit prompt.txt with your own names and jargon.
EOF
