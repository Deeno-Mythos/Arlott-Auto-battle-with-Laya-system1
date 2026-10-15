# Arlott + Laya + scrcpy

Experimental Windows controller for Arlott in Mobile Legends, using Laya and the
official scrcpy video and touch protocol. Tested with Python 3.12, Laya 0.4.7
(CPU) and scrcpy 5.0.1. Setup verifies the scrcpy archive's SHA-256.

This repository includes controller/training code, reviewed annotation intervals,
small gameplay regression fixtures and result reports. Model weights, source
recordings, extracted datasets, runtime logs and personal device settings stay
local. The published example uses stock multilingual Laya; the recorded baseline
results require the separately trained v1 checkpoint. The ONNX candidate did not
pass the accuracy gates and has not replaced that baseline. True game-event-to-
phone-touch latency remains unverified.

## First setup

Install Python 3.12 for Windows, then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

Setup creates a virtual environment, installs pinned dependencies, downloads
scrcpy and copies `config.example.json` to `config.json` if needed. Connect one
Android phone with USB debugging authorized. A null `serial` selects the single
connected device automatically; set a local serial for multiple-device setups.
Stock Laya downloads its checkpoint on first model-enabled start. Use
`start-controller.cmd --no-model` for manual control without loading a model.
Calibrate the controls, health bar and HUD before enabling automatic play.

## Start

Double-click `start-controller.cmd` and open <http://127.0.0.1:8765>.
Keep the phone connected over USB with USB debugging authorized.
The controller starts in Manual mode; the Laya model loads in the background.
Close the terminal or press Ctrl+C to shut it down.

PowerShell alternative:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-controller.ps1
```

For ordinary scrcpy mirroring:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\mirror.ps1
```

Use one control window at a time to avoid conflicting fingers from two sessions.

## General gameplay controls

| Input | Action |
| --- | --- |
| WASD / arrows | Hold and move the joystick; diagonals supported |
| Hold Space | Repeat basic attack |
| 1 / 2 / 3 | Arlott skill 1 / skill 2 / ultimate with game auto-aim |
| C / Arlott combo button | Skill 1, Skill 2, basic attack, ultimate, Skill 2, basic attack |
| B / E / R | Battle spell / regen / recall |
| Mouse drag on the live screen | Direct touch, including aimed skill casts |
| Escape / Stop | Release every finger and return to Manual |

Keyboard movement and mouse skill aiming use separate pointer IDs, so you can
move and cast at the same time. The keyboard only controls the phone when this
browser page is focused. Switching focus to the chat releases manual keys but
keeps auto active while the controller remains visible. Hiding the page, closing
it, losing the heartbeat or receiving stale video releases all inputs and pauses auto.

## Calibration

`config.json` stores normalized coordinates for the 1600 x 720 example layout.
It is ignored by Git; the example starts without a health/HUD calibration.
If the phone, orientation or control layout changes, recalibrate in the dashboard:

1. Enter a battle with full health. Choose a control from the dropdown, click
   **Set button position**, then its center on the live screen. Repeat for all
   eight controls.
2. Click **Set full health bar** and select the top-left and bottom-right corners
   of the complete green fill above your hero.
3. Click **Save battle HUD** with the basic attack button visible and inactive.

Calibration clicks only save positions. They do not tap your phone.

## What the AI can currently do

The locally trained baseline is **Laya plus a video-trained Arlott neural adapter**
in `models/arlott-video-v1`. Its weights are not included in Git. The supplied
recording was decoded locally;
no phone or live game was used during training. Actual gradient updates trained
Laya's native scorer and a 5,224-parameter residual adapter. The text encoder and
transformer body stay frozen. The adapter reads health, target range/type, visible
cooldown state, a weak mark hint, and the most recent skill input. It predicts hold,
approach, retreat, basic attack, skill 1, skill 2, ultimate, or reposition.

The first scorer-only experiment failed. The baseline adapter was selected using
validation combat macro F1, with class-balanced, confidence-weighted training.
See [the training report](training/reports/arlott-training-report.md) for the actual
results, failed experiments, per-action counts, and limitations. This is one edited
match with noisy CV labels; these figures do not establish win rate or generalization
to your phone. Basic attacks and retreat are still poorly learned.

**Auto combat: trained Arlott + fallback** evaluates the small neural adapter on
each new decoded screen frame, reusing Laya context for at most 1.5 seconds.
The background Laya transformer supplies fresh context. On this CPU a full Laya
pass takes about 0.42 seconds; the adapter alone takes about 0.07 ms using native
NumPy matrix operations. No slower Python pixel loops or language rewrite is needed.

Fresh, confident neural choices take priority. The controller checks target range,
health and visible cooldowns before sending a skill. A learned basic attack does
not start the fixed combo. Uncertain, stale or unusable choices use the explicitly
labelled combat fallback: approach a target, attack in melee range, or retreat at
low health. Reposition is not a learned map route; it uses combat fallback when a
target is present and otherwise waits. `combat_fallback: false` disables fallback.
The dashboard separates the proposed action from the actual input source, and logs
are saved to `logs/decisions.jsonl`.

The manual C combo and fallback auto combo remain available. They adapt to visible
readiness and skip unavailable skills. They use the game's auto-aim. The mark hint
is a screen-color heuristic, not reliable mark confirmation. Shop, leveling,
objectives, walls and map navigation have not been trained. Enemy health bars are
CV candidates; they cannot reliably identify Nolan or distinguish every hero/minion.
Stale video, death, a missing HUD and a lost browser heartbeat release inputs.

## Offline training and replay

These commands do not connect to Android or start the controller. Extraction and
normalization are calibrated for this specific 854 x 480 recording. Generated
frames, observations and labels are under `training/data`; weights stay local in
`models/arlott-video-v1`.

```powershell
.\.venv\Scripts\python.exe training\inspect_video.py "C:\Videos\arlott.mp4"
.\.venv\Scripts\python.exe training\extract_demonstrations.py "C:\Videos\arlott.mp4"
.\.venv\Scripts\python.exe training\build_dataset.py
.\.venv\Scripts\python.exe training\finetune_arlott.py
.\.venv\Scripts\python.exe training\train_adapter.py
.\.venv\Scripts\python.exe training\verify_checkpoint.py
.\.venv\Scripts\python.exe training\profile_policy.py
.\.venv\Scripts\python.exe training\write_report.py
```

Training inputs are pre-action states. Cooldown changes in subsequent frames are
label evidence only. Joystick evidence labels movement but never enters the model
features. Train: before 500 seconds; validation: 510–620 seconds; test: 630–787 seconds,
with ten-second gaps. Identical state strings can recur in different stretches;
the report includes their count and a separate novel-state score. The temporal
test was used for development diagnostics; it is not a pristine independent-match
benchmark. More independent recordings and live practice evaluation are necessary.

Load the complete policy through `gameplay.LayaPolicy`, which verifies that the
adapter matches the native scorer. `laya.load` alone loads only the native model,
without the trained sidecar adapter. To use your completed v1 training output, set
`model` to `models/arlott-video-v1` and `policy_schema` to `arlott-v1` in your local
config. Extraction geometry and `annotations_v2.json` intervals are specific to the
original source recording (identified by SHA-256 in the reports). Review and
replace them for a different video. A fresh clone cannot regenerate the published
metrics without that source data and the generated checkpoints.

## Real-time optimization and limits

See [the real-time report](training/reports/v2/real-time-report.md) for confusion
matrices, failed quality gates, ONNX/INT8 comparison and latency breakdown.
The candidate has not met the requested accuracy gates, so the original local
deployment retains baseline v1. The public example uses stock Laya. The sole
original test retreat is actually a dead/spectating frame;
audited labels are versioned separately, and the original replay is hash-locked.

The runtime now wakes on frames, uses a latest-frame buffer and caches the HUD
template. Taps send DOWN immediately and schedule UP separately. Stop cancels
pending pulses. All timing uses the high-resolution performance counter. New
optional ONNX policies use the per-frame path and avoid the large transformer.
Native OpenCV masks preserve the original feature thresholds.

The ONNX candidate's host-only replay measured 21.25 ms p95 at 30 Hz and 15.95 ms
at 60 Hz for actual DOWN/MOVE socket sends. Whole-frame dispatch p95 was 20.67 /
17.02 ms, respectively. These are different sample sets and are not additive.
They exclude phone capture, USB and touch registration. Native 60 FPS phone
capture, browser load and game-event-to-phone-touch latency remain unverified.
The default tap hold remains 55 ms; cast-on-release latency includes that hold.

To collect local manual demonstrations when the game is available:

```powershell
.\.venv\Scripts\python.exe controller.py --fps 60 --record-demos
```

This starts in Manual as usual. It saves pre-input frames, manual button/movement/
mouse intent and high-resolution timestamps under `logs/demonstrations`. Frames
are sampled at up to 8 FPS; automated combo/policy taps are excluded from input
labels. Intent does not confirm a successful cast, and movement direction still
needs retreat/approach annotation. Recording is off unless requested by this flag.
`--tap-ms` can be calibrated later against actual accepted inputs; it defaults to 55.

Offline optimization workflow:

```powershell
.\.venv\Scripts\python.exe training\diagnose_v2.py
.\.venv\Scripts\python.exe training\extract_v2_features.py
.\.venv\Scripts\python.exe training\build_v2_dataset.py
.\.venv\Scripts\python.exe training\train_rt.py
.\.venv\Scripts\python.exe training\benchmark_onnx.py
.\.venv\Scripts\python.exe training\benchmark_pipeline.py
.\.venv\Scripts\python.exe training\write_rt_report.py
.\.venv\Scripts\python.exe -m unittest test_gameplay test_realtime -v
```

The pipeline benchmark uses an offline session and loopback socket; it never calls
ADB or starts a phone session. Model weights are in `models/arlott-rt-v2`; reports
are in `training/reports/v2`. FP32 was faster and preserved more validation
decisions than INT8 on this CPU. TensorRT was not used on the Intel GPU.

## Repair / verify

Run `setup.ps1` to recreate dependencies. `requirements-lock.txt` pins the tested
environment. Run:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\tools\scrcpy-win64-v5.0.1\adb.exe devices -l
.\.venv\Scripts\python.exe controller.py --no-model
.\.venv\Scripts\python.exe -m unittest -v test_gameplay
```

Run all 14 offline controller/transport checks with
`python -m unittest test_gameplay test_realtime -v` after activating `.venv`.
They use the checked-in `tests/fixtures` calibration and practice screenshots;
no phone, model download, source recording or personal `config.json` is required.

If the device is unauthorized, unlock it and approve the USB debugging dialog.
If screen mirroring works but touches fail on a Xiaomi phone, scrcpy's official
docs describe the additional USB debugging (Security Settings) option. It has
not been changed by this setup.

## Sources

- [Laya source and documentation](https://github.com/NandhaKishorM/laya)
- [Laya model card and limitations](https://huggingface.co/convaiinnovations/laya)
- [Official scrcpy](https://github.com/Genymobile/scrcpy)
- [scrcpy protocol documentation](https://github.com/Genymobile/scrcpy/blob/v5.0.1/doc/develop.md)

`scrcpy_transport.py` implements the pinned 5.0.1 protocol. Upgrading the server
requires checking video/session headers and control message serialization.
