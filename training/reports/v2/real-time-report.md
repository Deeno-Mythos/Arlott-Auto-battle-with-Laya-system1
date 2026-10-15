# Arlott real-time optimization — targets and evidence

**Targets are not met.** The runtime is faster, but the new compact model does not
pass the accuracy gates. The original local deployment retains baseline v1 weights;
the public example uses stock Laya because trained weights are not redistributed. The game was not
opened and no phone input was sent during this task. All socket tests used loopback.

| Requested gate | Current result | Decision |
| --- | --- | --- |
| ≥18/24 skill casts, frozen replay | 11/24, same total as baseline | Fail |
| Attack F1 ≥0.80 | 0.00 frozen; 0.00 audited | Fail |
| Retreat F1 ≥0.80 | 0.00 frozen, with an invalid sole label; 0.35 audited | Not established |
| Game event → phone touch p95 ≤33 ms at 30 FPS | No phone registration timestamps available | Unverified |
| Game event → phone touch p95 ≤16 ms at 60 FPS | No native 60 FPS phone capture or registration timestamps available | Unverified |
| Keep all 10 controller checks green | 10/10 original + 4/4 new tests pass | Pass |

## Diagnosis first

1. Original training had 14 attack and 4 retreat examples; validation had 1 attack
   and **zero retreat** examples. Those were not enough to select a reliable policy.
2. Twelve of 14 attack training rows and all three attack test rows had no detected
   target. The perception failure removes the information needed to choose attacks.
3. Retreat weak labels required opposite joystick direction AND health below 0.60.
   High-health disengagements were assigned to reposition. Several zero-health rows
   were nevertheless labeled as movement or casting.
4. The original test retreat at 766.293 seconds is **Arlott dead/spectating**. See
   `training/data/v2/review/0766.00.jpg`: View Death Replay and a respawn countdown
   are visible. Training a policy to retreat there would reward incorrect behavior.
   Skill2 rows at 762.553 and 764.156 seconds are also after death. The original
   94-row / 24-cast benchmark is preserved byte-for-byte; corrections are separate.
5. Movement states and cast events have different durations and may overlap. A
   single exclusive eight-way label hides simultaneous movement and attacks.
6. Raw visual patches overfit scene appearance. Adding neighboring frames from one
   recording did not provide independent combat situations. These failures support
   collecting logged inputs, rather than claiming synthetic labels solve the gap.

## Data and sequence experiment

Explicit assistant-reviewed intervals expanded training attacks from **14 to
55** and retreats from **4 to 37**,
across 7 attack and
5 retreat episodes. Validation now has 3 attack
and 12 retreat rows, but each class still comes from only one episode.

The compact model consumes eight causal observations over 0.875 seconds, including
health uncertainty/damage trend, target/skill state, past cooldown transitions and
a small world-region visual descriptor. It excludes joystick input, control HUD,
future frames and timestamps as features. Normalization uses training rows only.
Three seeded runs were selected using validation skill accuracy and attack/retreat
F1. The model has 81,176 parameters.

All original data hashes were checked against `baseline_lock.json`. The v2 audited
test adds reviewed intervals and corrects death labels; it does not replace the
frozen test. Its 169 correlated rows include only **one attack and two retreat
episodes**. Neither set establishes independent-match performance.

## Confusion matrices

Rows are true labels; columns are predictions. These are ungated neural choices,
not simulated win rate or claims of inputs the phone would accept.

### Baseline v1, original 94 rows

| Truth → prediction | hold | approach | retreat | attack | skill1 | skill2 | ultimate | reposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hold | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| approach | 0 | 0 | 0 | 0 | 1 | 4 | 0 | 0 |
| retreat | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| attack | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 |
| skill1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 2 |
| skill2 | 0 | 2 | 0 | 0 | 2 | 9 | 0 | 4 |
| ultimate | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| reposition | 0 | 6 | 0 | 0 | 1 | 5 | 6 | 33 |

### Candidate v2, same frozen 94 rows

| Truth → prediction | hold | approach | retreat | attack | skill1 | skill2 | ultimate | reposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hold | 9 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| approach | 0 | 1 | 1 | 0 | 2 | 0 | 0 | 1 |
| retreat | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| attack | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| skill1 | 0 | 0 | 0 | 0 | 4 | 0 | 1 | 0 |
| skill2 | 0 | 1 | 0 | 0 | 3 | 7 | 2 | 4 |
| ultimate | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| reposition | 1 | 4 | 2 | 0 | 7 | 8 | 1 | 28 |

### Candidate v2, separately audited 169 rows

| Truth → prediction | hold | approach | retreat | attack | skill1 | skill2 | ultimate | reposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hold | 20 | 26 | 0 | 0 | 4 | 2 | 2 | 16 |
| approach | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 |
| retreat | 0 | 0 | 4 | 0 | 2 | 2 | 0 | 8 |
| attack | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| skill1 | 0 | 0 | 0 | 0 | 4 | 0 | 1 | 0 |
| skill2 | 0 | 0 | 0 | 0 | 3 | 7 | 2 | 3 |
| ultimate | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| reposition | 1 | 1 | 2 | 0 | 6 | 7 | 1 | 27 |

CSV files beside this report contain the same matrices. The candidate did not
improve skill-match total or attack F1, so it was **not promoted**. Its audited
retreat F1 is 0.348 on two episodes.

## Runtime changes implemented

- Frame notifications replace the 50 ms watchdog poll and 120 ms combat interval.
  The controller processes the latest frame once; it does not build an old-frame queue.
- DOWN is sent immediately; a separate scheduler sends UP after the tap hold.
  Stop cancels pending releases and releases immediately. Pointer reuse cannot be
  released by an old timer. The default hold is still 55 ms; if the game casts on
  UP, action-effect latency includes that hold. `--tap-ms` allows later calibration.
- HUD templates are cached; frame readers avoid copying the full published image.
- Equivalent color-mask calculations run in native OpenCV instead of allocating
  multiple NumPy boolean images. OpenCV uses two threads, ONNX one.
- Runtime timestamps now use QueryPerformanceCounter (`perf_counter`). This Python
  build's `monotonic` uses GetTickCount64 with 15.625 ms resolution, so the initial
  coarse timing results were discarded and measurements repeated.
- The control worker uses Windows above-normal thread priority only. No global
  power plan or realtime process priority was changed.
- ONNX candidate decisions run per frame, with no Laya transformer on that path.
  Baseline Laya remains available and is the default configured model.
- `--record-demos` opt-in records manual pre-input frames at up to 8 FPS and exact
  button/movement/touch intent timestamps. JPEG writing is asynchronous; automated
  combo/policy taps are excluded from manual input labels. Game acceptance is
  explicitly unconfirmed. Movement still needs retreat/approach intent annotation.

## ONNX FP32 versus dynamic INT8

| Backend | Model p95 | Size | Validation argmax parity with PyTorch |
| --- | ---: | ---: | ---: |
| FP32 | 0.0177 ms | 331,508 B | 100.00% |
| INT8 | 0.0190 ms | 90,938 B | 98.91% |

Each backend ran 2,000 warm measurements on validation inputs. FP32 was selected;
INT8 was slightly slower and changed one validation decision. Quantization overhead
can outweigh savings for small networks; ONNX documents the extra dynamic-scale
work ([official quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)).
This PC has Intel UHD graphics; TensorRT's NVIDIA hardware requirement makes it
unsuitable here ([NVIDIA support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)).

## Latency breakdown: offline host replay only

The following is decoded/published frame → host dispatch. It excludes phone
display/capture/encoding, USB transport and Android input registration. The 60 Hz
run reuses a 29.952 FPS source at synthetic pacing. It is not native 60 FPS capture.
Each run published 300 frames and used a real local control socket. Fixed fallback
combos and browser JPEG display were excluded; this is the unpromoted ONNX candidate.

| Stage, p95 | 30 Hz | 60 Hz |
| --- | ---: | ---: |
| Frame wake/queue | 0.187 ms | 0.432 ms |
| Perception + sequence features | 20.305 ms | 16.159 ms |
| ONNX decision + guards | 0.331 ms | 0.327 ms |
| Control dispatch | 0.211 ms | 0.242 ms |
| Whole frame → dispatch | 20.668 ms | 17.020 ms |
| Actual DOWN/MOVE → socket sent | 21.249 ms | 15.953 ms |

Stage percentiles are not additive. Whole-frame p95 is measured directly. There
were 300/300 and 300/300 processed frames,
and 156 / 154
actual DOWN/MOVE sends. Send timing includes selected input frames only; all-frame
p95 at 60 Hz still exceeds 16 ms. These host figures do **not** pass either requested
game-event-to-phone-touch gate. Earlier host processing p95 at 60 Hz was 34.307 ms;
the current result is 17.020 ms.

Recorded-file decode p95: 1.556 ms;
BGR conversion + normalization p95: 2.618 ms.
Those describe this 854x480 file, not a 1600x720 live phone encoder/USB stream.

## Step-by-step path to the requested gates

1. **Establish valid replay truth.** Review all 24 cast events at native frame rate;
   keep invalid death/cut labels visible in the original benchmark and publish a
   versioned corrected set. Label attack start/hold/stop, retreat intent, death,
   target boxes and actual successful skill casts. Keep evaluation labels separate
   from pseudo-labels. One death row cannot validate retreat F1.
2. **Collect independent manual episodes when the game is available.** Use
   `python controller.py --fps 60 --record-demos`. Include sustained attacks against
   minions, towers and heroes; low/high-health retreats; and approach/reposition
   negatives. Separate training/validation/test by match. Recordings are local under
   `logs/demonstrations`; no collection started during this task.
3. **Fix perception and action semantics before more tuning.** Recover targets in
   attack frames, distinguish missing HP from death, verify cooldown transitions,
   and add target motion/damage history. Use separate movement and button heads so
   attacking while moving is representable. Retain a fixed projection when comparing
   against the original eight-way replay. Tune on validation only; report per-class
   precision/recall/F1 and skill counts on a locked, independent-match test.
4. **Keep FP32 ONNX and reduce the remaining perception cost.** Model inference is
   already below 0.02 ms; perception dominates. Optimize tracked ROIs/scale only
   with detection and replay parity checks. Compare INT8 again if a larger detector
   is introduced. Re-run at least 1,000 paced frames per rate with the browser and
   intended background workload, reporting dropped frames and p95/p99.
5. **Measure the actual phone boundary.** With a live controlled scenario, collect
   synchronized game-display and Android touch-registration timestamps, or use
   sufficiently fast external capture. Measure DOWN, UP and cast effect separately;
   host send timestamps and unsynchronized scrcpy PTS do not establish phone latency.
   Calibrate `--tap-ms` against successful casts rather than shortening it blindly.
6. **Promote only after all gates pass.** Require ≥18/24 on valid documented casts,
   attack and retreat F1 ≥0.80 on adequate independent episodes, true event-to-touch
   p95 ≤33/16 ms at 30/60 FPS, and all 10 original controller checks green. Until
   then, retain baseline weights and mark the ONNX model as a candidate.

## Reproduce / artifacts

`training/diagnose_v2.py` → audit + hash lock; `annotations_v2.json` → explicit labels;
`extract_v2_features.py`, `build_v2_dataset.py`, `train_rt.py` → candidate training;
`benchmark_onnx.py` → FP32/INT8 validation; `benchmark_pipeline.py` → offline timings;
`python -m unittest test_gameplay test_realtime -v` → 14 passing checks.

Weights and ONNX models: `models/arlott-rt-v2`. Full metrics: `evaluation.json`.
Latencies: `onnx_benchmark.json`, `pipeline_benchmark.json`. Original data hashes:
`baseline_lock.json`. Weights, source video, extracted datasets and temporary source
snapshots are excluded from Git. These are recorded experiment results; reproducing
them requires the original source and regenerated local artifacts.
