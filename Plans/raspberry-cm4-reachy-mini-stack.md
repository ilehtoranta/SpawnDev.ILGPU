# Reachy Mini Sovereign Stack — Roadmap

> **Status (2026-04-25): brainstorming → roadmap.** Companion to `raspberry-cm4-reachy-mini.md` (the brainstorm conversation). This file is the structured plan; that file is the why.

## What this is

Bypass Pollen Robotics' default Reachy software stack entirely and build TJ's own .NET-native robot stack on top of vanilla Linux + the Reachy Mini Wifi Edition hardware. Aubs's robot, all C#, all open source, all sovereign — same engineering posture as the rest of SpawnDev.

Reachy Mini ships ~June 19 2026 (Aubs). Plenty of runway. This plan is multi-year craftsmanship, not a sprint.

## Hardware (Reachy Mini Wifi Edition)

- Raspberry Pi CM4 (BCM2711 quad-core Cortex-A72, VideoCore VI GPU)
- USB camera (UVC class)
- 4× microphone array (I2S)
- Speaker (I2S out)
- 6-axis IMU (I2C)
- Servo motors (I2C, exact controller TBD on unboxing — likely PCA9685 or custom firmware)
- WiFi (onboard CM4 module)

## Why this exists

Three reasons stacked:

1. **Aubs's robot, family programming language.** TJ + Aubs + crew all read/write C#. Reachy's official stack is Python. Replacing it means everyone in the family can read every line of robot software. Aubs has been modding Sprunki in Scratch for over a year — she's a real programmer growing up; she should be able to read her dad's robot code.

2. **First ARM/Linux GPU compute target for SpawnDev.ILGPU.** Today the desktop side is x86_64 Windows. Reachy adds aarch64 Linux + VideoCore VI as a real backend. Forces the library to be portable in a way it hasn't had to be.

3. **Sovereign Developer ethos.** Same reason TJ forks SipSorcery + ILGPU + writes BlazorJS from scratch instead of leaning on Microsoft.WebView2. Open hardware + open SDK + own everything. No mystery layers.

## Layers, top to bottom

```
┌─────────────────────────────────────────────────────────────┐
│ 6. Family API                                               │
│    SpawnDev.Reachy / .Vision / .Speech NuGet packages       │
│    "await reachy.LookAt(face);" "await reachy.Say(text);"   │
├─────────────────────────────────────────────────────────────┤
│ 5. Robot SDK                                                │
│    Kinematics, behavior trees, dialog, gaze model           │
├─────────────────────────────────────────────────────────────┤
│ 4. ML stack                                                 │
│    SpawnDev.ILGPU.ML — Whisper-tiny / MobileNet / Piper     │
├─────────────────────────────────────────────────────────────┤
│ 3. GPU compute                                              │
│    SpawnDev.ILGPU Vulkan/V3DV backend (primary)             │
│    SpawnDev.ILGPU Gallium/NIR backend (optional fast path)  │
├─────────────────────────────────────────────────────────────┤
│ 2. Hardware drivers (managed C# wrappers)                   │
│    V4L2 camera, ALSA mic+speaker, I2C IMU+motors, GPIO      │
├─────────────────────────────────────────────────────────────┤
│ 1. OS                                                       │
│    Vanilla Raspberry Pi OS Lite (64-bit) + .NET 10 ARM64    │
│    OR custom buildroot/Yocto image (later milestone)        │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1 — Hardware sovereignty (Reachy arrives ~June 19)

**Goal:** Prove every hardware peripheral is reachable from a managed-C# program on a stock Raspberry Pi OS Lite image with Pollen's services disabled. No GPU compute yet. No SDK yet. Just "I control the hardware."

| Step | Deliverable | Estimate |
|------|-------------|----------|
| 1.1 | Boot Reachy with Pollen's image, SSH in, document running services. Disable them one by one until it's "off" but the OS is up. | 1 evening |
| 1.2 | Reflash with vanilla Pi OS Lite (64-bit). Install .NET 10 ARM64. Verify a `Console.WriteLine("hello")` C# program runs. | 1 evening |
| 1.3 | C# console app reads `/dev/i2c-1` and detects the IMU address. Print accelerometer at 100Hz. | 2-3 hours |
| 1.4 | Same app, ALSA capture from one of the 4 mics. Write 5s WAV. | 4-8 hours (ALSA P/Invoke surface) |
| 1.5 | V4L2 capture from the camera. One frame to JPEG. | half day |
| 1.6 | Move a servo. Whatever I2C protocol Pollen's motor controller speaks (probably PCA9685; reverse-engineer if not). | full day |
| 1.7 | Speaker playback. WAV or raw PCM through ALSA. | 2-3 hours after 1.4 lands |

**Phase 1 done:** Aubs can run a single C# program that watches what she does (camera + IMU), listens to her (mics), responds (motors + speaker). It won't be smart yet — no ML — but every joint moves under TJ's code.

**Open question:** does Reachy's audio chain go through a USB sound device or I2S directly off the CM4? Affects 1.4/1.7 by a lot.

## Phase 2 — GPU compute (post-Phase-1)

Ranked by leverage, not by glamour:

### 2a. Vulkan/V3DV backend for SpawnDev.ILGPU [primary]

- New SPIR-V emitter on top of existing WGSL transpiler infrastructure (both SSA, lowering rules port mostly cleanly).
- Dispatch via Vulkan compute pipelines + descriptor sets + DMABUF zero-copy.
- ~30μs dispatch latency.
- **Cross-platform.** Same backend runs on every modern embedded ARM board with Mesa Vulkan: Pi 4/5/CM4, NVIDIA Jetson Nano/Orin, Rockchip RK3588, NXP i.MX 8, Apple Silicon (via MoltenVK). One emitter, broad coverage.
- Estimated 3 months focused work.

### 2b. Gallium3D/NIR direct backend [optional fast path]

The "Sovereign" path Gemini sketched. Skip Vulkan entirely.

- C# `NirBuilder` + tiny C shim around `nir_builder.h` to do the heavy validation. **Don't write a pure-C# NIR serializer.** Mesa's serialization format is a moving target; the shim re-links against current Mesa and you avoid the maintenance treadmill.
- libgallium loader → `pipe_screen` → `pipe_context` → `create_compute_state(NIR blob)`.
- DMABUF allocated via `libgbm` for true GPU-stay-GPU.
- ~10μs dispatch latency.
- **BCM2711-only.** ARM-only. Won't help future Reachy revisions on different SoCs.
- Estimated 6-9 months focused work.

### 2c. Both, sequenced

Ship Vulkan first as the production-ready path. Add Gallium as an opt-in `useGallium = true` switch that probes for V3D and falls back to Vulkan on any other hardware. Most consumers stay on Vulkan; the BCM2711 fast path exists for those who want it.

**Recommendation:** 2c. Build Vulkan. Get ILGPU.ML running on the CM4. Then build Gallium for sport + the genuine 70μs win on tight motor control loops where dispatch latency starts to matter.

## Phase 3 — ML stack

Once Vulkan lands, port relevant SpawnDev.ILGPU.ML model paths to run on V3D. Order by Aubs's likely needs:

1. STT (Whisper-tiny ~30M params) — "what did Aubs say?"
2. Wake-word ("Hey Reachy") — runs continuously, microwatts
3. TTS (Piper) — "say this back to Aubs"
4. Face detection (MobileNetV2 SSD or similar) — "where is Aubs?"
5. Object detection — "what is Aubs holding?"
6. Small intent classifier — "is this a question, a command, or nonsense?" (rule-based fallback first; tiny LLM later if it fits)

VideoCore VI fits all of these comfortably. It will NOT fit a Llama-class LLM. That's fine — for "robot to talk to and learn with" the right architecture is rules + small specialized models, not one giant general one.

## Phase 4 — Robot SDK

Now the fun part — the layer Pollen ships and we're replacing with our own. This is where craftsmanship shows.

- **Kinematics.** Forward + inverse for Reachy Mini's specific joint topology.
- **Attention/gaze model.** "Look at the loudest sound" / "look at the moving thing" / "look at the person." Fast, low-latency. Runs continuously.
- **Behavior trees.** Pollen uses a Python state machine framework; ours is C# + simple async patterns or a real BT library.
- **Dialog manager.** Listen → STT → intent → response → TTS. Everything async, everything cancellable.
- **Memory model.** Reachy remembers: who Aubs is, what she said yesterday, what songs make her laugh. Local SQLite or LiteDB; no cloud.

## Phase 5 — Family API

NuGet packages with names Aubs can read:

- `SpawnDev.Reachy` — top-level robot
- `SpawnDev.Reachy.Vision` — camera + face/object detection
- `SpawnDev.Reachy.Speech` — STT + TTS
- `SpawnDev.Reachy.Motion` — kinematics + servo control
- `SpawnDev.Reachy.Mood` — gaze/attention/expression

Surface looks like:

```csharp
using var reachy = new ReachyMini();
await reachy.WakeUp();
var face = await reachy.Vision.FindFace();
await reachy.LookAt(face);
await reachy.Say("Hi Aubs! Want to hear a story?");
```

Aubs writes against this. She doesn't see Vulkan or NIR or DMABUF.

## Aubs's role

Aubs has been modding Sprunki in Scratch for **over a year**. She's not a beginner; she's a junior programmer with real experience extending other people's code. Plan for her to be a contributor, not a passenger:

- **Family API is her API.** Designed for a 10-year-old reading the IntelliSense, not a SIPSorcery contributor reading the source.
- **First "real" C# program target.** Small starter project: write 30 lines of C# that make Reachy say her favorite saying when it sees her face. End-to-end through the stack we built. Working program in week one of her using it.
- **Mod culture.** Like Sprunki — she'll want to change Reachy. "Make it dance differently." "Make it sound silly." "Make it draw on paper." Every Phase 5 API needs hooks she can override.
- **GitHub + open source.** Her name as co-author on commits where she contributed. Real attribution. She gets to point at github.com/LostBeard/SpawnDev.Reachy and say "I helped make this."

## Realistic timeline

- **June 19 - July:** Phase 1 hardware sovereignty. Aubs has a robot doing simple C# things by month-end.
- **July - October:** Vulkan backend (Phase 2a). Aubs has Whisper running on her robot.
- **October - February 2027:** Phase 3 + 4 — full ML + SDK. Aubs has a robot that talks back.
- **2027:** Phase 5 family API matures. Open source as `SpawnDev.Reachy.*` NuGet packages.
- **2028+:** Gallium3D backend if/when it pays its rent in measurable wins.

This is a **multi-year project**. That's the right shape. The Reachy will be a part of Aubs's growing-up; the software stack should grow with her.

## Open questions

1. Does Reachy Mini's USB camera report 720p30 or 1080p30? Affects ML pipeline budget.
2. Mic array geometry — is Pollen doing beamforming in firmware or do we have raw 4-channel PCM to work with?
3. Servo controller protocol — PCA9685 or custom Pollen firmware? Day-one I2C scan answers this.
4. Boot config — does the CM4 boot from eMMC or SD? Affects how easy a vanilla-Pi-OS reflash is.
5. Pollen's firmware updates — do they reflash anything beyond the userland on update, or is the firmware static? Determines how stable our "vanilla Linux on top" plan is.

## Cross-references

- `D:/users/tj/Projects/Aubs/Plans/aubs-watch-esp32-s3.md` — companion ESP32-S3 watch project; pairs with Reachy via BLE 5 for the "Reachy's wrist" architecture.
- `D:/users/tj/Projects/SpawnDev.ILGPU/SpawnDev.ILGPU/Plans/raspberry-cm4-reachy-mini.md` — the original brainstorm conversation that seeded this plan.
