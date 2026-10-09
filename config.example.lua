-- Copied to config.lua by setup.sh (which fills in the @...@ values).
-- Edit config.lua, then Hammerspoon > Reload Config.
return {
  -- whisper.cpp model name (file: vendor/whisper.cpp/models/ggml-<model>.bin).
  -- Download others with: vendor/whisper.cpp/models/download-ggml-model.sh <model>
  model = "@MODEL@",
  gpu = @GPU@,          -- Metal on Apple Silicon; Intel Macs run on CPU (see docs/findings.md)
  threads = @THREADS@,  -- performance cores

  -- Mic, by name. List yours with:
  --   ffmpeg -f avfoundation -list_devices true -i ""
  mic = "@MIC@",

  ffmpeg = "@FFMPEG@",
  python = "@PYTHON@",
  port = 8178,

  -- Fn timing (seconds)
  tap_max = 0.3,      -- a press shorter than this counts as a tap
  double_tap = 0.4,   -- max gap between taps for hands-free mode
  max_latched = 300,  -- hands-free auto-stop
}
