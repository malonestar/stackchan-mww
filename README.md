# microWakeWord for the M5Stack StackChan

> **Note on credit:** the wake-word engine isn't mine — `micro_wake_word.cc/.h` is
> mostly recycled from ESPHome's `micro_wake_word` component (GPLv3). My part is
> the multi-model picker and wiring it into the StackChan firmware. Full credit to
> ESPHome and OHF-Voice/micro-wake-word; that's why it's GPL-3.0.

This is the microWakeWord setup I got working on the M5Stack StackChan factory
firmware. It lets the device listen for a wake phrase you train yourself instead
of the fixed one baked into the stock firmware. It comes with three example
models and an on-screen menu to switch between them.

Treat it as a starting point. If you run the stock factory firmware, you can drop
these files onto a fresh clone and build. If you've already got your own firmware
built off the factory tree, the files still show you what to change and where —
the layout matches the factory firmware exactly.

## Read this first: what you give up by flashing your own firmware

This isn't specific to this project. It's true of any firmware you build and
flash onto a StackChan yourself.

The official "StackChan World" phone app won't pair with self-built firmware.
When the app connects over Bluetooth it asks the device to prove it's running
genuine M5Stack firmware, and that proof needs a signing step only M5Stack's own
build can do. The open-source firmware only ships a placeholder for it, and
M5Stack keeps the real one private. So anything you compile yourself fails to
pair — you'll usually see "failed to connect" or "failed to get device data".
Binding to an M5Stack account fails for the same reason.

So you lose the phone app, account binding, and the cloud features that ride on
them.

And it's worse than just the app — you can't set the stock factory firmware up on
the device at all. There's no on-screen WiFi entry, no setup hotspot gets
generated, and every path funnels you back to "bind in the app", which is the
thing that doesn't work. A self-built stock factory firmware is basically a dead
end for an actual deployment.

The real fix is to build a custom firmware that severs the M5Stack cloud
connections. That's what I run: a custom firmware on top of the M5Stack factory
firmware and xiaozhi-esp32-server, pointed at all-local models with gpt-oss:20b in
my network stack. You give up the app, but I control everything from my phone and
the web UI the StackChan serves itself — added to my iPhone home screen and
launched as a web app, it works great. This kit is meant to drop into that kind of
build.

So if you want the stock app-and-cloud experience, stay on M5Stack's shipped
firmware. If you're building (or already running) a custom firmware that cuts the
cloud, this is what adds custom wake words to it.

## What's in here

A few groups of changes.

The wake word engine

- `micro_wake_word.cc`, `micro_wake_word.h` — the provider, and the core of all
  this. It takes mic audio, runs the streaming TensorFlow Lite Micro model, and
  says when it hears the wake phrase. It plugs into the same wake-word interface
  the firmware already has.

The models

- three `.tflite` files in `audio/wake_words/models/` — example wake words ("Hey
  Stack Chan", "Hey Frank", "Hey M5"), embedded into the firmware image. Swap in
  your own for a different phrase.

The on-screen picker

- `app_setup/app_setup.cpp` — adds a "Wake Word" item to the Setup > Device menu.
- `app_setup/workers/system.cpp`, `workers/workers.h` — the picker page: a
  dropdown of the embedded models. Saving writes your pick and reboots so it
  loads on the next start.
- `hal/hal.h`, `hal/hal_wake_model.cpp` — read and store the chosen model in the
  device's settings.

Build wiring

- `main/CMakeLists.txt` — compiles the provider and embeds the models.
- `main/idf_component.yml` — pulls in the two runtime deps the provider needs
  (esp-tflite-micro and esp-micro-speech-features). esp-sr stays — the firmware
  still uses it for audio processing.

The one change to the upstream engine

- a patch to `audio_service.cc` — points the wake-word provider at microWakeWord
  instead of the built-in ESP-SR one. It's the only edit to the xiaozhi-esp32
  source; everything else is a new file or a change to M5Stack's own files.

The CoreS3 audio fix

- `hal/board/cores3_audio_codec.cc`, `.h` — nothing to do with wake words, but
  you need it. On this board the mic sometimes comes up dead after a reboot.
  These reset the audio input and run a throwaway open/close on first use so it
  starts clean every time. Want it on a CoreS3; ignore it on anything else.

## Building it

The files match the factory firmware's layout. On a fresh clone of
m5stack/StackChan, run its `fetch_repos.py` to pull in xiaozhi-esp32, copy these
files over the matching paths, and apply the one patch to `audio_service.cc` with
`git apply`.

Then build and flash with ESP-IDF (5.5.2 or newer):

```
idf.py set-target esp32s3
idf.py build
idf.py -p <port> flash
```

Two things that'll trip you up:

- The first build needs network — the component manager fetches
  `esp-tflite-micro` and `esp-micro-speech-features`.
- If a new source file doesn't get compiled, run `idf.py reconfigure`. The
  factory CMake globs `hal/` and `apps/`, so new files there need a reconfigure
  to get picked up.

Flash over USB the first few times, not OTA. The ESP32-S3 recovers over USB even
from a bad flash so you won't brick it, but save OTA until you've seen it boot
clean.

Windows: set `git config --global core.autocrlf false` before you clone.
Otherwise git rewrites the line endings and the patch won't apply.

## Adding your own model

The whole point of this is to run your own wake word. Training a model with
microWakeWord produces two files — a `.tflite` and a matching `.json` — and the
`kModels[]` entry you add is built from that `.json`. Here's how each value maps:

| `.json` value | Where it goes in the code | Notes |
|---|---|---|
| `model` (the `.tflite` filename) | the file in `main/audio/wake_words/models/`, the `EMBED_FILES` line, and the `kModels[]` basename | the filename sets the symbol name — `hey_robot_v1.tflite` becomes `_binary_hey_robot_v1_tflite_start` |
| `micro.probability_cutoff` (0.0–1.0) | `kModels[]` cutoff, as `round(probability_cutoff × 255)` | the model reports its probability as a byte (0–255), so 0.81 becomes 207. this is the one number you tune per model |
| `wake_word` | the `kModels[]` phrase and the `_wake_model_list` label | display text — write it to read nicely, it doesn't have to match the `.json` verbatim |
| (you pick it) | the `kModels[]` id and the `_wake_model_list` key | a short NVS-safe slug like `robot`; it has to match in both spots |
| `micro.sliding_window_size` | `SLIDING_WINDOW_SIZE` in `micro_wake_word.h` | shared by every model; the bundled ones use 5, only change it if yours differs |
| `micro.feature_step_size` | `features_step_size_` in `micro_wake_word.h` | shared; the bundled value is 10 |
| `micro.tensor_arena_size` | `TENSOR_ARENA_SIZE` in `micro_wake_word.h` | the provider over-allocates 65536, so your model's value just needs to fit under that |

The edits, in order (paths are relative to the factory firmware's `firmware/`):

1. Drop the `.tflite` in `main/audio/wake_words/models/`.
2. Add it to the `EMBED_FILES` list in `main/CMakeLists.txt`.
3. In `micro_wake_word.cc`, declare the `_binary_..._start` symbol and add a row
   to `kModels[]`: the id, the basename, the data pointer, the cutoff, and the
   phrase. Bump `kModelCount` if it's a separate constant.
4. In `app_setup/workers/system.cpp`, add a matching row to `_wake_model_list` —
   the on-screen label and the same id you used in `kModels[]`.

Rebuild, flash, and it shows up under Setup > Device > Wake Word. Removing one is
the same edits in reverse.

For reference, here's how the three bundled models fill that in. The shipped
cutoffs are tuned up from the `.json` starting points — the `.json` values turned
out conservative once I tested on the device and watched how high each model's
detection probability actually peaked:

| Model (`.tflite`) | Phrase | NVS id | `.json` cutoff | × 255 | shipped cutoff |
|---|---|---|---|---|---|
| hey_stackchan_v1 | Hey Stack Chan | `stackchan` | 0.81 | 207 | 217 |
| hey_frank_v5 | Hey Frank | `frank` | 0.85 | 217 | 204 |
| hey_m5_v3 | Hey M5 | `m5` | 0.50 | 128 | 217 |

So treat the `.json` number as a starting point: derive the cutoff with `× 255`,
then tune it on the device for your own setup.

## How I trained the bundled models

In case it helps with training your own.

I train in WSL on Windows on an NVIDIA 5070 Ti laptop GPU (CUDA). It's about 8
hours per model version on the GPU, and I've put a lot of time into learning how
to do it.

The positive set is mostly Piper samples — the phrase generated with a bunch of
different voices, speeds, and pitches — plus a small proportion of real voice
samples. For the real ones I ran a script that auto-fired a recording every couple
seconds off my laptop mic, and I kept varying my pitch, speed, and tone while
moving around the room to change the distance — a few hundred samples in about 5
minutes. Small part of the set, but they add strength.

Two things matter the most. First is phoneme input. Piper from plain text gets
sloppy at the ends of words — for "hey frank" it spat out a ton of samples that
were really "hey fran", and the early model false-triggered constantly off ambient
noise. Feeding phonemes instead let me force a hard "k" at the end of every
sample, which fixed it.

Second is similar-phrase negatives. I explicitly train out phrases that sound like
the wake word so the model learns what isn't it. How many depends on the word —
"hey frank" is super common and a lot of stuff trips it from ambient noise, so I
trained against "hey fran", "hey frin", "hey finn", "hey flinn", "hey friend", etc
(Piper + phoneme input here too). That's why the models stay strong even at a
lower cutoff — they know the near-misses aren't the wake word — and it lets you use
a wider variety of wake words and still get a solid model.

A unique phrase just trains into a really strong model with basically nothing that
trips it. "Hey stackchan" is like that — it pins right at the top, and the only
near-phrase that fires it is "hi stackchan", which is harmless. A common one like
"hey frank" needs all that negative work to get there.

One thing to be upfront about: I'm a male English speaker from the US, and the
real voice samples (the small slice that's an actual person) are all me. The vast
majority is Piper across different voices, speeds, and pitches, so I'm hoping it
generalizes fine, but I can't promise the same accuracy for very different voices
or accents. If a bundled model doesn't work great for you, just train your own
(above).

## Caveats

This replaces the wake engine, it doesn't sit next to it. There's no automatic
fall back to the original. If a model fails to load you get a device that still
boots, joins WiFi, and is reflashable — just no working voice. Recoverable, but
don't count on a safety net.

I've only run it on the CoreS3 StackChan. The audio fix especially is specific to
that board.

Test it on real hardware before you trust it. The audio issue above only showed
up across a lot of reboots, not one or two.

## License

GPLv3 or later — see `LICENSE`. It has to be GPLv3 because the provider is adapted
from ESPHome's `micro_wake_word` component, and ESPHome's C++ is GPLv3.

Everything else keeps its own license, all GPLv3-compatible: the modified M5Stack
files stay MIT (their headers are untouched), the xiaozhi-esp32 patch is against
MIT code, and the runtime deps (esp-tflite-micro, esp-micro-speech-features) plus
the microWakeWord model framework (OHF-Voice/micro-wake-word) are Apache-2.0.
Full attribution is in `NOTICE`.

Not affiliated with or endorsed by M5Stack, Espressif, the Open Home Foundation,
or ESPHome. Provided as-is, no warranty.
