# Athena — Always-On Voice Shell: Design

**Date:** 2026-09-15
**Status:** Design approved, awaiting spec review
**Scope:** The always-on voice shell only. Wake word training, computer use, and long-term memory are separate sub-projects.

---

## 1. Goal

A voice assistant that is present whenever the laptop is on. You say "Athena", it hears you, answers out loud in a female voice, and holds a conversation for a short window without needing the wake word again. You can talk over it and it stops.

It does not use tools, does not touch the browser or the desktop, and does not remember anything across sessions. Those are later slices. This slice exists to get the audio plumbing, the state machine, and the host lifecycle right — and to prove that a small local model can hold a spoken conversation with acceptable latency on this hardware.

## 2. Success criteria

All five are manual, listened-to checks. No test suite — consistent with the rest of this project.

1. Say "Athena" from across the room → wake event in the log, tray flips to listening.
2. Talk over a response → playback stops within ~300 ms, and the follow-up demonstrates Gemma knows what it had already said.
3. Stay silent for 7 s → returns to IDLE, and the log shows nothing was transcribed while idle.
4. Change the default audio output mid-session (unplug headphones, replug) → no crash, AEC mode switches, change is logged.
5. Stop `llama-server` → it speaks one short line, tray shows error, returns to IDLE, and recovers when the server returns.

## 3. Hardware and existing services

| Thing | Detail |
|---|---|
| Machine | ASUS Vivobook S 16, Ryzen AI 9 HX 370, Radeon 890M (gfx1150), 32 GB unified |
| Execution tiers | Vulkan iGPU (llama.cpp), ROCm iGPU (ComfyUI), XDNA 2 NPU (ONNX, idle) |
| Brain | Gemma 4 E2B uncensored GGUF, `llama-server` on `127.0.0.1:8080`, ~30 tok/s decode, 131k ctx |
| Neighbours | Open WebUI on `:8000`, ComfyUI on `:8188` — neither is touched by this slice |
| Python | 3.12; uv profiles under `C:\Users\tnmh\.virtualenvs` |

This slice adds no GPU load: Gemma keeps the Vulkan path, ComfyUI keeps ROCm. All audio work is CPU, with the NPU as an optional accelerator.

## 4. Decisions

| Decision | Choice | Why |
|---|---|---|
| Wake word | **"Athena"**, `/əˈθiːnə/`, standalone | 3 syllables standalone is a stronger signature than 2 + a trigger prefix (cf. "Alexa"). The voiceless `/θ/` is rare in English and spectrally distinctive. |
| Pre-roll | 2 s ring buffer | Covers the gap between the user starting to speak and the detector firing, so "Athena, what's the weather" doesn't lose the question. |
| Session shape | Conversation window, ~7 s | Handles both one-shot commands and "no, the other one" follow-ups without re-waking. |
| Barge-in | VAD-based, adaptive AEC | User interrupts by speaking. Requires echo cancellation on speakers. |
| Topology | **One process, threads + queues** | Smallest thing that guarantees the audio path never blocks. Service-per-stage is ceremony for one user on one machine. |
| Supervision | **Task Scheduler**, restart-on-failure | Task Scheduler already does this. No custom supervisor, no second process, no watcher-of-the-watcher. |
| Startup | At logon | Cannot be a Windows *service* — services run in session 0 and cannot open the user's audio endpoints. |
| Dependencies | Isolated `voice` uv profile, ONNX-first | Contains the departure from the project's stdlib-only convention to one deletable venv. |
| Acceleration | `EP = "auto" \| "npu" \| "cpu"` | NPU is a probe, never a hard dependency. The Ryzen AI stack is unproven on this box. |

### Rejected

| Option | Why not |
|---|---|
| LFM2-VL-3B-heretic | Too weak for the task; its GGUF chat template has no tool support. Deleted. |
| Porcupine (Picovoice) | Free tier ended; access keys reported dead as of 2026-06-30. |
| "Jane" | `/dʒeɪn/` rhymes with train, plane, chain, again, explain — unusable false-accept rate. |
| "Nova" / "Hey Nova" | Viable, but 3 syllables standalone beats 2 + prefix. No cost either way since both need training. |
| Service-per-stage (Wyoming) | 4–5 supervised processes and IPC latency, for isolation that doesn't matter on one machine. |
| Home Assistant Assist / Rhasspy | Conversation model is built for home-automation intents; wrong shape for computer use. |
| Custom supervisor process | Task Scheduler covers it. |
| Ryzen AI as a hard dependency | Unverified. Would risk debugging AMD's stack instead of building an assistant. |

## 5. Architecture

One Python process, started at logon, restarted by Task Scheduler on failure.

```
Task Scheduler (logon, restart-on-failure)
        │
        ▼
   athena.ps1 ──► python   [single-instance mutex]
                    │
   ┌────────────────┼─────────────────┐
   │                │                 │
[audio]        [detector]        [tray/main]
   │                │                 │
   ├─► ring buffer  │                 │  status · mute
   │   (pre-roll)   │                 │  restart · quit
   └─► frames ──────┘                 │
                    │                 │
              wake event              │
                    ▼                 │
              [session state machine]─┘
                    │
                    ├──► [STT worker]
                    ├──► Gemma @ 127.0.0.1:8080
                    └──► [TTS worker] ──► speakers
```

**The load-bearing property:** the audio and detector threads share nothing with STT, the brain, or TTS except queues. A multi-second wait on Gemma cannot stop it hearing the wake word.

The tray icon runs on its own thread. Windows tray icons need a message pump or they freeze whenever anything else is busy.

## 6. Audio path and state machine

```
                    ┌────────────────────────────────────┐
                    │              IDLE                  │
                    │  frames → ring buffer + detector   │
                    │  nothing transcribed, nothing kept │
                    └──────────────┬─────────────────────┘
                                   │ wake event
                                   ▼
                            ┌─────────────┐
                            │  LISTENING  │  pre-roll + live audio
                            └──────┬──────┘  VAD endpointer
                                   │         cap 20 s
                          800 ms silence
                                   ▼
                            ┌─────────────┐
                            │TRANSCRIBING │
                            └──────┬──────┘
                                   ▼
                            ┌─────────────┐
                            │  THINKING   │  → Gemma, streamed
                            └──────┬──────┘
                        first sentence│
                                   ▼
                            ┌─────────────┐
                            │  SPEAKING   │  sentence-by-sentence TTS
                            └──────┬──────┘
                     playback done │
                                   ▼
                       ┌───────────────────────┐
                       │ CONVERSATION WINDOW   │
                       └───────┬───────────────┘
              speech           │            7 s silence
                               ▼                   │
                        [ LISTENING ]              ▼
                                              [ IDLE ]
```

Two transitions that don't fit the vertical flow:

- **LISTENING → IDLE** if no speech is detected within 3 s of the wake event (a false trigger).
- **SPEAKING → LISTENING** on barge-in: 300 ms of speech above threshold stops playback and returns to listening.

| Parameter | Value | Rationale |
|---|---|---|
| Ring buffer | 2 s | Covers start-of-speech to detection gap |
| Frame size | 80 ms @ 16 kHz | Native openWakeWord frame |
| End-of-utterance | 800 ms silence | Shorter clips mid-thought; longer feels laggy |
| Max utterance | 20 s | Stops runaway capture if VAD never trips |
| Conversation window | 7 s silence | The follow-up gap |
| Barge-in trigger | 300 ms of speech | Fast enough to feel responsive, slow enough not to trip on noise |

**Barge-in.** During SPEAKING the wake detector is suppressed, or it would trigger on the assistant's own voice. Interruption is VAD-based instead. The partial response stays in conversation history, so Gemma knows what it already said and does not restart mid-sentence.

**Adaptive AEC.** On wake and on every output-device change, query the default endpoint. Headphones → raw VAD barge-in, no DSP. Speakers → AEC using the TTS stream as reference, with a raised VAD threshold. If AEC is unavailable or unreliable, degrade to half-duplex: mute the mic during playback, lose barge-in, keep working.

**Invariants:**

1. Nothing leaves IDLE without a wake event. No speech is transcribed, stored, or transmitted until the detector fires. The ring buffer is RAM-only and continuously overwritten.
2. The tray icon never misrepresents state.

## 7. Components

Each is a one-method seam. This is what makes the Athena swap and the NPU question non-events.

| Component | Interface | Backend now | Swappable to |
|---|---|---|---|
| `WakeDetector` | `score(frame) -> float` | openWakeWord pretrained | trained `athena.onnx` |
| `Transcriber` | `transcribe(pcm) -> str` | sherpa-onnx Parakeet | faster-whisper, NPU path |
| `Speaker` | `say(text) -> chunks` | kokoro-onnx | piper, SAPI failsafe |
| `Brain` | `stream(messages) -> text` | llama-server `:8080` | any OpenAI-compatible endpoint |

`AudioIO`, `SessionStateMachine`, `TrayApp`, and `config` carry no model knowledge.

**Execution provider.** One config value, `EP = "auto" | "npu" | "cpu"`. On `auto`, probe the Ryzen AI / DirectML provider once at startup, log the result, fall back to CPU silently if absent.

### Layout

```
local-llm/
├── athena.ps1              launcher, matches start-*.ps1 convention
├── athena.log / .log.err
├── athena/
│   ├── main.py             wiring, single-instance mutex, tray
│   ├── config.py           every tunable from §6 and §8 in one place
│   ├── audio.py            capture, playback, device watch, AEC mode
│   ├── detector.py         WakeDetector + backends
│   ├── transcribe.py       Transcriber + backends
│   ├── speak.py            Speaker + backends
│   ├── brain.py            llama-server client, streaming
│   ├── session.py          the state machine
│   └── tray.py             status, mute, restart, quit
└── models/wake/            athena.onnx lands here after training
```

A package rather than flat `run_*.py` scripts is a deliberate departure — this has more moving parts than a single script.

### Dependencies

`uv venv voice --python 3.12`, then: `sounddevice`, `numpy`, `onnxruntime`, `openwakeword`, `sherpa-onnx`, `kokoro-onnx`, `pystray`, `pillow`.

Also install `ffmpeg` at the system level — currently absent, and Open WebUI already warns about it.

## 8. Failure handling

Kept short deliberately. Four principles: fail loud in the log and quiet in the room; IDLE is home; the audio thread never blocks and never throws; the tray always reflects reality.

| Failure | Behaviour |
|---|---|
| Wake fires, no speech | Silent return to IDLE after ~3 s. No spoken apology. |
| STT empty or garbage | Silent return to IDLE, counted in the log. |
| `llama-server` down | One short spoken line, tray → error, return to IDLE. Connect timeout 3 s. |
| Gemma slow to first token | One brief non-verbal filler at 2.5 s, once per turn, configurable. |
| Gemma stalls mid-response | 20 s watchdog; speak what exists, keep partial in history. |
| TTS backend fails | Fall back to SAPI; then log + tray error. Never crash the session. |
| Mic unplugged / endpoint change | Re-open capture on new default device, log, stay IDLE. No device → tray shows "no microphone", retry every 5 s. |
| Output device changes mid-speech | Rebind playback, finish current sentence on the new device. |
| Sleep / resume | Detect clock jump, tear down and rebuild audio streams, reset to IDLE. Streams do not survive suspend. |
| AEC unreliable | Degrade to half-duplex. |
| Second instance | Mutex rejects, logs, exits. |
| Crash | Task Scheduler restarts. Nothing replayed or resumed. |
| Wake model missing | Fail fast and loud at startup. |
| Log growth | Rotate at 10 MB, one generation kept. |

Plus a global mute hotkey (`Ctrl+Alt+M`) that gates the mic at capture level, independent of state machine health.

## 9. Out of scope

Deferred to their own specs:

- **Wake word training.** "Athena" needs a trained model. The shell is built against the `WakeDetector` seam and ships on a pretrained openWakeWord model as a stand-in. Training is its own sub-project: data generation, negative mining, and a tuning loop against the real room and voice. Expect ~10–17 h CPU or 2–4 h GPU; the trainers are PyTorch and the ROCm iGPU may accelerate them.
- **Computer use.** Browser via Playwright (`aria_snapshot()` for a text view of the page — note Chrome ≥136 ignores `--remote-debugging-port` against a default profile), desktop via UI Automation (`uiautomation` 2.0.29), shell.
- **Long-term memory.** Not a fine-tune. Vector store plus an append-only journal.
- **Fine-tuning Gemma.** Nothing in this slice needs it.

## 10. Open risks

| Risk | Impact | Mitigation |
|---|---|---|
| Ryzen AI NPU stack unproven on this box | NPU path dead on arrival | `EP = "auto"` probes and falls back to CPU. Nothing depends on it. |
| `sherpa-onnx` bundles its own onnxruntime | NPU provider may not reach STT | Accept CPU STT. Parakeet on Zen 5 with AVX-512 is well above real-time. |
| AEC quality unknown | Barge-in thrash loops | Half-duplex fallback. Headphones sidestep entirely. |
| Gemma at ~30 tok/s decode | Feels slow | Streaming TTS sentence-by-sentence; filler at 2.5 s. Measure before optimising. |
| Gemma tool calling unverified | Blocks the *later* hands slice, not this one | Verify separately before designing the brain slice. One log line (`Failed to parse tools: Missing tool type`) is ambiguous and may just have been a malformed request. |
| Untrained stand-in wake word | Higher false-accept rate during development | Accept for now; swap when training lands. |

## 11. Coexistence

The project already contains a half-finished image-generation integration (plan stops at task 3 of 7; `openwebui/dreamshaper_action.py` is missing). This slice neither depends on nor repairs it. It binds only `127.0.0.1`, adds one Task Scheduler entry, and adds one uv profile.
