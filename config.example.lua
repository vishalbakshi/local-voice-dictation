-- Copied to config.lua by setup.sh (which fills in the @...@ values).
-- Edit config.lua, then Hammerspoon > Reload Config.
return {
  -- "whisper" (whisper-server, uses prompt.txt) or "parakeet" (a GGUF in
  -- vendor/models, run by dictate/parakeet_server.py from .venv; no prompt).
  engine = "whisper",
  parakeet_model = "ft-mix_100-Q8_0",  -- vendor/models/<name>.gguf
  parakeet_port = 8180,

  -- whisper.cpp model name (file: vendor/whisper.cpp/models/ggml-<model>.bin).
  -- Download others with: vendor/whisper.cpp/models/download-ggml-model.sh <model>
  model = "@MODEL@",
  gpu = @GPU@,          -- Metal on Apple Silicon; Intel Macs run on CPU (see docs/findings.md)
  threads = @THREADS@,  -- performance cores

  -- Mics by name, in priority order: each dictation uses the first one
  -- that's plugged in. List yours with:
  --   ffmpeg -f avfoundation -list_devices true -i ""
  mics = { "@MIC@" },  -- e.g. { "My USB Headset", "@MIC@" }

  ffmpeg = "@FFMPEG@",
  python = "@PYTHON@",
  port = 8178,

  -- Fn timing (seconds)
  tap_max = 0.3,      -- a press shorter than this counts as a tap
  double_tap = 0.4,   -- max gap between taps for hands-free mode
  max_latched = 300,  -- hands-free auto-stop
}
