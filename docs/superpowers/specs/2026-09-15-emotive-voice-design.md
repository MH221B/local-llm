# Emotive Voice — Emotion-Tagged Speech: Design

**Date:** 2026-09-15
**Status:** Design approved, awaiting spec review
**Scope:** A standalone Python package that turns text + an emotion tag into audible speech that sounds different per emotion. Voice first; no assistant, no brain, no wake word.

---

## 1. Goal

A single Python call takes text, one base voice, an emotion tag (or a weighted mix of several), and an intensity, and produces speech whose delivery reflects that emotion.

```python
say("I told you it would work.", voice="bf_emma", emotion="sarcastic", intensity=70)
```

**The base voice is a control, not a variable.** Every emotion renders with the same voice, so a comparison across emotions isolates the emotion and nothing else. Emotion is carried by prosody and voice *quality* — rate, pitch, energy, pauses, contour, spectral tilt, drive, breath — never by swapping in a different speaker. A listener should hear one person in eight moods, not eight people.

The result should be audibly different from `neutral` — demonstrably, by ear and by measurement. This is the voice slice of the Athena assistant, built and verified independently so that Athena can import it later without inheriting any of its problems.

## 2. Success criteria

Manual, listened-to checks. No test suite, consistent with the rest of this project.

1. One fixed sentence, one fixed voice, rendered in all 8 emotions plays back 8 audibly distinct ways.
2. Through every render the speaker identity stays constant — no emotion sounds like a different person.
3. A weighted mix (e.g. 60% happy / 40% sarcastic) sounds like a blend of both, not one or the other.
4. `intensity=10` vs `intensity=90` on the same emotion is clearly more and less emotional, with `neutral` unaffected by intensity.
5. Measured pitch, rate, and energy differ per emotion and move in the expected direction (happy higher/faster than sad).
6. Passing an unknown emotion name speaks as `neutral` and logs a warning instead of crashing.

## 3. Why emotion is post-hoc, not model-side

Verified against `thewh1teagle/kokoro-onnx` source: the ONNX graph takes exactly three inputs — phoneme `tokens` (≤510), one 256-d `style` vector, and a scalar `speed` (clamped 0.5–2.0). There is **no tag vocabulary and no instruction input**; `[sad]` in the text is phonemized as literal words. Kokoro's style embedding is length-conditioned per voice and was trained only against those per-voice rows, so no external emotion embedding can be fed to it.

The levers that genuinely exist:

| Lever | API | Effect |
|---|---|---|
| Voice choice | `voice="bf_emma"` | Identity/timbre. Set once per install; never per emotion. |
| Speed | `speed=0.5..2.0` | Global rate |
| Pauses | `sentence_pause`, `clause_pause` | Rhythm, perceived weight |
| Direct prosody | `is_phonemes=True` + punctuation | Word-level emphasis |
| Voice blend | `voice=<(1,256) ndarray>` | Timbre mix — advanced, off by default |

Everything else is our own signal processing: pitch shift, energy and contour shaping, spectral tilt, drive, and breath.

### Rejected

| Option | Why not |
|---|---|
| Emotion tags inside Kokoro | Impossible without retraining. The model has no tag input. |
| Native-emotion TTS: Parler-TTS mini (caption control) | 10–100× Kokoro's compute; batch-ish latency, not sub-second. Fails the real-time story. |
| XTTS-v2 (reference-clip emotion) | ~2 GB RAM, RTF well above 1 on CPU, non-commercial license. |
| ChatTTS (`[laugh]`, prosody tokens) | Closest to real tags, but needs custom token manipulation, quality is inconsistent, dormant since 2024. |
| StyleTTS2 raw | Architecture Kokoro derives from, but unmaintained repo and old deps. High friction for no gain over Kokoro. |
| Piper / KittenTTS | Faster, but zero emotion affordance — would still need the same DSP. No reason to switch. |
| Cross-voice blending per emotion (happy=af_bella, etc.) | Changes *who* is speaking, so an emotion comparison no longer isolates emotion, and the assistant's identity drifts between sentences. Rejected as the emotion mechanism. |
| DSP only, no voice-quality shaping | Rate/pitch/energy alone reads as one person doing an impression. Kept, but extended with tilt/drive/breath so timbre shifts without swapping speaker. |

Native-emotion models stay available as a later swap behind the same `say()` seam.

## 4. Public interface

```python
say(
    text: str,
    voice: str = "bf_emma",
    emotion: str | list[tuple[str, float]] = "neutral",
    intensity: int = 50,
    play: bool = True,
    out_dir: str | None = None,
) -> dict
```

`bf_emma` is a British female voice. The other British female options are `bf_isabella`, `bf_alice`, `bf_lily`; American female voices (`af_*`) are also available. Whatever is chosen is set once and used for every emotion.

Returns:

```python
{
    "wav_path": "out/voice-20260915-221500-happy.wav",
    "emotion": "happy",
    "voice": "bf_emma",
    "intensity": 50,
    "params": {...},        # resolved Speed/Pitch/Energy/PauseScale/Tilt/Drive/Breath/Contour
    "metrics": {...},       # measured pitch_hz, rate_wps, energy_rms
    "spoken": "I told you it would work.",
}
```

- `emotion` accepts a single name or a weighted list: `[("happy", 60), ("sarcastic", 40)]`. A bare string is sugar for `[(name, 100)]`.
- Weights are normalized; they need not sum to 100.
- `intensity` 0–100 scales the total offset from neutral. At `0` every emotion collapses to neutral delivery. It is global, not per-emotion — per-emotion intensity is YAGNI until a real mixer asks for it.
- `play=False` renders and measures without touching the audio device.
- Blocking. One-shot: the whole utterance is rendered, then played. Streaming is explicitly out of scope for this slice.

## 5. Emotion presets

Pure data. Adding an emotion means appending one row here — no other file changes, no new branch anywhere.

```python
Params = Params(
    speed=1.0,           # Kokoro speed, clamped 0.5..2.0
    pitch_semis=0.0,     # post-DSP pitch shift in semitones
    energy=0.0,          # post-DSP gain in dB
    pause_scale=1.0,     # multiplies sentence/clause pauses
    contour="flat",      # "rise" | "fall" | "flat" | "vary" — pitch contour
    tilt=0.0,            # spectral tilt in dB/octave — bright (+) vs dark (−)
    drive=0.0,           # 0..1 soft harmonic saturation — push/roughness
    breath=0.0,          # 0..1 breathiness via high-band noise
)

EMOTIONS = {
    "neutral":   ...,
    "happy":     ...,
    "sad":       ...,
    "angry":     ...,
    "calm":      ...,
    "curious":   ...,
    "serious":   ...,
    "sarcastic": ...,
}
```

Starting values, all on one voice — to be tuned by ear during verification:

| Emotion | Speed | Pitch | Energy | Pause | Tilt | Drive | Breath | Contour |
|---|---|---|---|---|---|---|---|---|
| neutral | 1.00 | 0 | 0 | 1.0 | 0 | 0 | 0 | flat |
| happy | 1.08 | +1.5 | +1.5 | 0.8 | +1.5 | 0 | +0.1 | rise |
| sad | 0.85 | −2.0 | −2.0 | 1.4 | −1.5 | 0 | +0.2 | fall |
| angry | 1.10 | +0.5 | +3.0 | 0.7 | +1.0 | 0.5 | 0 | vary |
| calm | 0.92 | −0.5 | −1.0 | 1.2 | −1.0 | 0 | +0.3 | flat |
| curious | 1.05 | +1.0 | +0.5 | 0.9 | +1.0 | 0 | +0.05 | rise |
| serious | 0.95 | −1.0 | +0.5 | 1.1 | 0 | 0 | 0 | flat |
| sarcastic | 0.92 | −0.5 | −0.5 | 1.3 | 0 | 0 | 0.05 | flat |

Adding an emotion means appending one row. No new code, no new branch, no new voice.

**Sarcastic is deadpan, not acted irony.** Slightly low pitch, flat contour, slower rate, long pauses. Real sarcasm lives in word choice, which is the brain's problem, not the synthesiser's.

**Combos.** Weighted numeric interpolation of every field, then `intensity` scales the offset from neutral:

```
final = neutral + Σ(weight_i × (preset_i − neutral)) × (intensity / 100)
```

Contour blends by dominant weight (the only non-numeric field); flag it in the log when the two contributors disagree so a muddy result is explainable.

**Advanced: same-voice blend.** For extra timbre instability, `voice_blend` mixes the original rendering with a re-render at a slightly different speed (default `1.03`) at `blend_amount` ≤ 0.3. Same speaker, thicker sound. Off by default; not an emotion input.

**Unknown name.** Falls back to `neutral`, logs a warning, never raises.

## 6. Components

```
voice/
  __init__.py     say() — the only public call; orchestrates the rest
  emotions.py     EMOTIONS table, weighting, intensity scaling, validation
  synth.py        Kokoro wrapper: text + resolved Params -> 24 kHz pcm
  prosody.py      pitch/energy/contour DSP + metric extraction
  timbre.py       spectral tilt, drive, breath — voice quality without identity change
  play.py         sounddevice playback
  config.py       model paths, sample rate, defaults, VOICE
test_voice.py     renders the whole emotion set + combos, plays, prints metrics
models/           kokoro-v1.0.onnx + voices-v1.0.bin
```

Each module is one job. Only `synth.py` knows Kokoro exists:

- `emotions.resolve(emotion, intensity) -> Params` — pure, no side effects, trivially unit-inspectable.
- `synth.render(text, voice, params) -> np.ndarray` — the only Kokoro-aware file.
- `prosody.apply(pcm, params) -> pcm` and `prosody.measure(pcm) -> metrics` — DSP in, numbers out.
- `timbre.apply(pcm, params) -> pcm` — tilt/drive/breath shaping.
- `play.play(pcm)` — device I/O, nothing else.

`voice` is a one-line change in `config.py`; emotion never touches it. The whole package is deletable, and Athena depends on exactly one function.

## 7. Dependencies and layout

Own uv profile, separate from Athena's:

```
uv venv voice --python 3.12
uv pip install kokoro-onnx sounddevice numpy onnxruntime soundfile
```

Kokoro models (~330 MB fp32, or ~80 MB q8) download into `models/`. CPU by default; `EP = "cpu" | "auto"` accepted by `synth.py` so the DirectML path can be probed later without an API change.

Conventions followed: loopback nothing (no network at all), no git, no Docker, no test suite, stdlib preference relaxed only inside this one deletable venv.

## 8. Failure handling

| Failure | Behaviour |
|---|---|
| Model file missing | Fail fast and loud at import/init. No silent fallback. |
| Unknown emotion name | `neutral` + warning in the log. |
| Unknown `voice` name | Kokoro's own error propagates; message names the voice. |
| No audio output device | `say(play=True)` raises with a clear message; wav is still written if `out_dir` was given. |
| Kokoro raises on text | Propagates to the caller; no partial wav is left behind. |
| Text too long for one render | Split on sentence boundaries, render each, concatenate. Cap at 510 phonemes per chunk. |
| Empty text | Returns a metrics-only result with no wav; logs. |
| No SAPI fallback | Deliberate. Speech fallback is Athena's concern, not the voice library's. |

## 9. Out of scope

- Streaming / sentence-by-sentence incremental synthesis.
- Athena integration, `Speaker` seam adaptation, SAPI fallback.
- Wake word, STT, brain, tray, scheduling.
- Fine-tuning anything. Emotion control is data + DSP here.
- Changing the voice per emotion, or any per-emotion speaker selection. One voice, always.
- True acted sarcasm or irony; multi-speaker dialogue; singing.
- Per-emotion intensity weighting (global intensity only).

## 10. Verification

`test_voice.py`: one fixed sentence ("I told you it would work.") and one fixed voice (`bf_emma`), rendered for all 8 emotions, then 2 combos, written to `out/`, played in sequence, with pitch/rate/energy printed after each.

Pass conditions, all by ear plus the printed numbers:

1. All 8 sound distinct.
2. The speaker is recognisably the same woman throughout — emotion, not identity.
3. The 2 combos sound like blends.
4. `intensity=90` vs `intensity=10` on `sad` and `angry` is obvious; `neutral` is unchanged at both.
5. Metrics move in the expected direction: happy above sad on pitch and rate; angry above calm on energy.
6. `say(..., emotion="smug")` speaks neutral and logs the warning.

## 11. Open risks

| Risk | Impact | Mitigation |
|---|---|---|
| DSP pitch shift artifacts | Chipmunk/robotic timbre | Keep shifts within ±2 semitones; prefer rate, tilt, and breath for the rest. Fall back to rate+energy only if artifacts persist. |
| Voice-quality shaping overshoots | Robotic, lisping, or noisy output | `drive` ≤ 0.5, `breath` ≤ 0.3, `tilt` ≤ ±2 dB; disable a shaper that underperforms per emotion without touching the others. |
| One fixed timbre limits expressiveness | Emotions feel like one person's impressions | Accepted deliberately: identity stability beats peak expressiveness here. `drive`/`breath`/`tilt` extend the palette without changing speaker. |
| Emotion is perceived as an effect, not emotion | Doesn't meet the user's bar | Measure, then tune the table by ear before wiring into Athena; the table is the only thing that changes. |
| Kokoro CPU latency on long text | Slow for real use | One-shot for now; chunking exists; streaming is the Athena slice's job. |
| Metrics don't correlate with perceived emotion | Numbers mislead the tuning loop | Treat metrics as a guide, not a gate; the ear decides. |
