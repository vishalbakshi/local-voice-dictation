-- Hold Fn to dictate: record while held, transcribe with a local whisper-server,
-- paste at the cursor. Double-tap Fn for hands-free. Every dictation is saved
-- under dictations/ (audio + transcript) so you can review it and fine-tune later.
--
-- setup.sh adds this to ~/.hammerspoon/init.lua:
--   dofile("<repo>/dictate/dictate.lua")
-- Settings live in <repo>/config.lua (copied from config.example.lua).

local ROOT = debug.getinfo(1, "S").source:match("^@(.*)/dictate/dictate%.lua$")
local cfg = dofile(ROOT .. "/config.lua")

local WHISPER = ROOT .. "/vendor/whisper.cpp"
local SERVER_BIN = WHISPER .. "/build/bin/whisper-server"
local MODEL = WHISPER .. "/models/ggml-" .. cfg.model .. ".bin"
local PROMPT_FILE = ROOT .. "/prompt.txt"
local DATA_DIR = ROOT .. "/dictations"
local TRANSCRIBE = ROOT .. "/dictate/transcribe.py"
local FFMPEG, PYTHON = cfg.ffmpeg, cfg.python
local MIC = cfg.mic -- by name: virtual devices (Zoom, Teams, ...) shift indices
local PORT = cfg.port

-- Engine: "whisper" (whisper-server + vocab prompt) or "parakeet" (a
-- fine-tuned or stock Parakeet GGUF in vendor/models, served by
-- dictate/parakeet_server.py from the project's .venv). Separate ports, so a
-- leftover server from the other engine never answers by mistake.
local ENGINE = cfg.engine or "whisper"
local PK_MODEL = ROOT .. "/vendor/models/" .. (cfg.parakeet_model or "") .. ".gguf"
local PK_SERVER = ROOT .. "/dictate/parakeet_server.py"
local PK_PORT = cfg.parakeet_port or 8180
local VENV_PYTHON = ROOT .. "/.venv/bin/python"
local MODEL_NAME = ENGINE == "parakeet" and cfg.parakeet_model or cfg.model
local MIN_AUDIO = 0.4 -- seconds of captured audio; shorter recordings are discarded
local READY_SOUND = hs.sound.getByName("Tink")

-- Global, so the event taps and menu bar item are not garbage collected
-- (init.lua discards dofile's return value).
dictate = {}
local M = dictate
local server, recorder, wavPath, startedAt, cancelled
local menu = hs.menubar.new()
M.menu = menu

local function setStatus(s)
  menu:setTitle(({ idle = "🎙", wait = "⚪", rec = "🔴", busy = "⏳", err = "⚠️" })[s])
end

local function readFile(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local s = f:read("a")
  f:close()
  return s
end

local function appendFile(path, s)
  local f = io.open(path, "a")
  f:write(s)
  f:close()
end

-- whisper-server ---------------------------------------------------------------

local function startServer()
  if server and server:isRunning() then return end
  if ENGINE == "parakeet" then
    if hs.http.get("http://127.0.0.1:" .. PK_PORT .. "/", nil) > 0 then return end
    server = hs.task.new(VENV_PYTHON, function(code, _, stderr)
      print("dictate: parakeet server exited with code " .. tostring(code) .. " " .. tostring(stderr))
      setStatus("err")
    end, { PK_SERVER, PK_MODEL, tostring(PK_PORT), tostring(cfg.threads) })
    server:start()
    return
  end
  -- A server left over from a previous config load may still hold the port.
  if hs.http.get("http://127.0.0.1:" .. PORT .. "/", nil) > 0 then return end
  local args = { "-m", MODEL, "-t", tostring(cfg.threads), "--host", "127.0.0.1", "--port", tostring(PORT) }
  if not cfg.gpu then table.insert(args, "-ng") end
  server = hs.task.new(SERVER_BIN, function(code)
    print("dictate: whisper-server exited with code " .. tostring(code))
    setStatus("err")
  end, args)
  server:start()
end

-- recording -------------------------------------------------------------------

local function startRecording()
  local now = os.time()
  local dir = DATA_DIR .. "/" .. os.date("%Y-%m-%d", now)
  hs.fs.mkdir(DATA_DIR)
  hs.fs.mkdir(dir)
  wavPath = dir .. "/" .. os.date("%H%M%S", now) .. ".wav"
  startedAt = hs.timer.secondsSinceEpoch()
  cancelled = false
  recorder = hs.task.new(FFMPEG, nil, {
    "-y", "-hide_banner", "-loglevel", "warning",
    "-thread_queue_size", "4096",
    "-f", "avfoundation", "-i", ":" .. MIC,
    "-ac", "1", "-ar", "16000", "-flush_packets", "1", wavPath,
  })
  recorder:start()
  setStatus("wait")
  -- ffmpeg takes ~1 s to open the mic. Signal "recording" only once audio
  -- is actually reaching the file (WAV header is 44 bytes; 3200 = 0.1 s).
  local task = recorder
  M.readyTimer = hs.timer.doWhile(function() return recorder == task end, function()
    local attrs = hs.fs.attributes(wavPath)
    if attrs and attrs.size > 44 + 3200 then
      M.readyTimer:stop()
      setStatus("rec")
      READY_SOUND:play()
    end
  end, 0.05)
end

local function audioSecs(wav)
  -- 16 kHz mono s16 => 32000 bytes/s after the 44-byte WAV header.
  local attrs = hs.fs.attributes(wav)
  return attrs and (attrs.size - 44) / 32000 or 0
end

-- paste without clobbering the clipboard
local function pasteText(text)
  local saved = hs.pasteboard.readAllData()
  hs.pasteboard.setContents(text)
  hs.eventtap.keyStroke({ "cmd" }, "v", 0)
  hs.timer.doAfter(0.5, function() hs.pasteboard.writeAllData(saved) end)
end

local function transcribe(wav, heldFor)
  local secs = audioSecs(wav)
  local app = hs.application.frontmostApplication()
  local appName = app and app:name() or ""
  local t0 = hs.timer.secondsSinceEpoch()
  setStatus("busy")
  local function done(ok, resp, errInfo)
    setStatus("idle")
    if not ok or not resp or not resp.text then
      print("dictate: transcription failed: " .. tostring(errInfo))
      setStatus("err")
      return
    end
    local text = resp.text:gsub("%s+", " "):gsub("^[%s%.%-,:;]+", ""):gsub(" $", "")
    local latency = hs.timer.secondsSinceEpoch() - t0
    if text ~= "" and not text:match("^%[.*%]$") and not text:match("^%(.*%)$") then
      pasteText(text)
    end
    appendFile(wav:gsub("%.wav$", ".txt"), text .. "\n")
    appendFile(DATA_DIR .. "/log.jsonl", hs.json.encode({
      time = os.date("!%Y-%m-%dT%H:%M:%SZ"), wav = wav, text = text,
      audio_secs = secs, held_secs = heldFor, latency_secs = latency,
      chunks = resp.chunks, model = MODEL_NAME, engine = ENGINE, app = appName,
    }) .. "\n")
  end
  if ENGINE == "parakeet" then
    hs.http.asyncPost("http://127.0.0.1:" .. PK_PORT .. "/inference", hs.json.encode({ path = wav }),
      { ["Content-Type"] = "application/json" }, function(status, body)
        local ok, resp = pcall(hs.json.decode, body or "")
        done(status == 200 and ok, resp, body)
      end)
    return
  end
  -- transcribe.py sends the request(s) to whisper-server; it splits audio over
  -- ~28 s at pauses, since multi-window decoding on the server drops text.
  hs.task.new(PYTHON, function(code, stdout, stderr)
    local ok, resp = pcall(hs.json.decode, stdout or "")
    done(code == 0 and ok, resp, tostring(stdout) .. tostring(stderr))
  end, { TRANSCRIBE, wav, tostring(PORT), PROMPT_FILE }):start()
end

local function stopRecording()
  if not recorder then return end
  local wav, heldFor = wavPath, hs.timer.secondsSinceEpoch() - startedAt
  local task, wasCancelled = recorder, cancelled
  recorder = nil
  if M.readyTimer then M.readyTimer:stop() end
  task:setCallback(function(_, _, stderr)
    local secs = audioSecs(wav)
    -- Record capture health: audio length vs wall time, plus any ffmpeg warnings.
    appendFile(DATA_DIR .. "/ffmpeg.log", string.format("%s held=%.1fs audio=%.1fs\n%s",
      wav, heldFor, secs, stderr or ""))
    if wasCancelled or secs < MIN_AUDIO then
      os.remove(wav)
      setStatus("idle")
    else
      transcribe(wav, heldFor)
    end
  end)
  task:interrupt() -- SIGINT: ffmpeg finalizes the WAV header and exits
end

-- Fn key ----------------------------------------------------------------------
--
-- Hold Fn: record while held.
-- Double-tap Fn: hands-free recording; press Fn once to stop, Esc to cancel.
--
-- mode: nil (idle) | "hold" (Fn is down, recording) | "pending" (a quick tap
-- just ended; waiting to see if a second tap follows) | "latched" (hands-free)
-- | "release" (Fn pressed to stop a latched recording; ignore its release)

local TAP_MAX = cfg.tap_max         -- a press shorter than this is a tap
local DOUBLE_TAP = cfg.double_tap   -- max gap between the two taps
local MAX_LATCHED = cfg.max_latched -- auto-stop hands-free recordings
local ESC = hs.keycodes.map["escape"]

local mode, fnPressedAt
local fnIsDown = false

local function finish(cancel)
  if M.pendingTimer then M.pendingTimer:stop() end
  if M.latchTimer then M.latchTimer:stop() end
  if cancel then cancelled = true end
  stopRecording()
end

local DEBUG = false
local function dbg(msg)
  if DEBUG then print(string.format("dictate[%.3f] %s (mode=%s)", hs.timer.secondsSinceEpoch() % 100, msg, tostring(mode))) end
end

local function onFnDown()
  dbg("fn down")
  fnPressedAt = hs.timer.secondsSinceEpoch()
  if mode == nil then
    mode = "hold"
    startRecording()
  elseif mode == "pending" then -- second tap: keep the same recording going
    M.pendingTimer:stop()
    mode = "latched"
    dbg("latched")
    M.latchTimer = hs.timer.doAfter(MAX_LATCHED, function()
      mode = nil
      finish(false)
    end)
  elseif mode == "latched" then
    mode = "release"
    finish(false)
  end
end

local function onFnUp()
  dbg(string.format("fn up after %.2fs", hs.timer.secondsSinceEpoch() - fnPressedAt))
  if mode == "hold" then
    if hs.timer.secondsSinceEpoch() - fnPressedAt < TAP_MAX then
      mode = "pending"
      M.pendingTimer = hs.timer.doAfter(DOUBLE_TAP, function()
        dbg("no second tap, discarding")
        mode = nil
        finish(true) -- a lone tap: discard
      end)
    else
      mode = nil
      finish(false)
    end
  elseif mode == "release" then
    mode = nil
  end
end

-- Fn alone drives recording; another modifier with Fn cancels it.
M.flagsTap = hs.eventtap.new({ hs.eventtap.event.types.flagsChanged }, function(e)
  local f = e:getFlags()
  local others = f.cmd or f.alt or f.ctrl or f.shift
  dbg("flagsChanged fn=" .. tostring(f.fn) .. " others=" .. tostring(others))
  if f.fn and not fnIsDown and not others then
    fnIsDown = true
    onFnDown()
  elseif fnIsDown and (not f.fn or others) then
    fnIsDown = false
    if others and mode == "hold" then
      mode = nil
      finish(true)
    else
      onFnUp()
    end
  end
  return false
end)

-- Key codes the Fn/Globe key itself can produce (kVK_Function, Globe key).
local FN_KEYCODES = { [63] = true, [179] = true }

M.keyTap = hs.eventtap.new({ hs.eventtap.event.types.keyDown }, function(e)
  local code = e:getKeyCode()
  if mode then dbg("keyDown code=" .. code) end
  if FN_KEYCODES[code] then return false end
  if mode == "latched" and code == ESC then
    mode = nil
    finish(true)
    return true -- swallow the Esc
  end
  -- Fn used as a modifier (Fn+arrow, Fn+F-key, Fn+Delete) cancels a hold.
  if mode == "hold" and fnIsDown then
    mode = nil
    finish(true)
  end
  return false
end)

startServer()
-- The Fn listener needs Accessibility; passing true shows the system prompt.
if not hs.accessibilityState(true) then
  setStatus("err")
  hs.alert.show("Dictation needs Accessibility: enable Hammerspoon in Privacy & Security, then Reload config", 8)
  print("dictate: NOT ready, Accessibility permission is missing")
  return M
end
M.flagsTap:start()
M.keyTap:start()
setStatus("idle")
print("dictate: ready (hold Fn, or double-tap Fn for hands-free) · " .. ENGINE .. ": " .. tostring(MODEL_NAME))

return M
