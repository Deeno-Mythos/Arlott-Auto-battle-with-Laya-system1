"""Publish complete failures and measured scopes, without promoting a bad model."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'training/reports/v2'
ACTIONS=('hold','approach','retreat','attack','skill1','skill2','ultimate','reposition')


def table(matrix):
    out='| Truth → prediction | '+' | '.join(ACTIONS)+' |\n'
    out+='| --- |'+' ---: |'*8+'\n'
    for label,row in zip(ACTIONS,matrix):out+='| '+label+' | '+' | '.join(map(str,row))+' |\n'
    return out


def main():
    evaluation=json.loads((REPORT/'evaluation.json').read_text())
    onnx=json.loads((REPORT/'onnx_benchmark.json').read_text())
    pipeline=json.loads((REPORT/'pipeline_benchmark.json').read_text())
    original=json.loads((ROOT/'training/reports/evaluation.json').read_text())
    manifest=evaluation['v2_manifest'];locked=json.loads((REPORT/'baseline_lock.json').read_text())
    for filename,digest in locked.items():
        assert hashlib.sha256((ROOT/'training/data'/filename).read_bytes()).hexdigest()==digest
    for name,matrix in [('baseline_v1',original['trained']['test']['confusion']),
                        ('candidate_v2_frozen',evaluation['frozen_original']['confusion']),
                        ('candidate_v2_audited',evaluation['audited']['test']['confusion'])]:
        with (REPORT/f'{name}_confusion.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['truth/prediction',*ACTIONS])
            writer.writerows([[a,*row] for a,row in zip(ACTIONS,matrix)])
    train=manifest['splits']['train'];test=evaluation['audited']['test'];frozen=evaluation['frozen_original']
    rates=pipeline['rates'];p30=rates['30'];p60=rates['60']
    casts=round(evaluation['frozen_24_casts']['accuracy']*24)
    summary={'status':'TARGETS NOT MET; optimized runtime implemented; candidate not promoted',
        'skill_match_frozen':f'{casts}/24','attack_f1_frozen':frozen['per_action']['attack']['f1'],
        'retreat_f1_frozen':frozen['per_action']['retreat']['f1'],'retreat_original_label_valid':False,
        'audited_attack_f1':test['per_action']['attack']['f1'],'audited_retreat_f1':test['per_action']['retreat']['f1'],
        'host_control_send_p95_30':p30['actual_loopback_down_or_move_sent']['p95_ms'],
        'host_control_send_p95_60':p60['actual_loopback_down_or_move_sent']['p95_ms'],
        'true_game_event_phone_touch_p95':'UNVERIFIED at both rates',
        'original_checks':'10/10 green','additional_checks':'4/4 green','default_model':'baseline v1 retained',
        'frozen_data_hashes_verified':True}
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2))
    text=f'''# Arlott real-time optimization — targets and evidence

**Targets are not met.** The runtime is faster, but the new compact model does not
pass the accuracy gates. Baseline v1 weights remain configured. The game was not
opened and no phone input was sent during this task. All socket tests used loopback.

| Requested gate | Current result | Decision |
| --- | --- | --- |
| ≥18/24 skill casts, frozen replay | {casts}/24, same total as baseline | Fail |
| Attack F1 ≥0.80 | {frozen['per_action']['attack']['f1']:.2f} frozen; {test['per_action']['attack']['f1']:.2f} audited | Fail |
| Retreat F1 ≥0.80 | {frozen['per_action']['retreat']['f1']:.2f} frozen, with an invalid sole label; {test['per_action']['retreat']['f1']:.2f} audited | Not established |
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
{train['labels']['attack']}** and retreats from **4 to {train['labels']['retreat']}**,
across {train['explicit_episodes']['attack']} attack and
{train['explicit_episodes']['retreat']} retreat episodes. Validation now has 3 attack
and 12 retreat rows, but each class still comes from only one episode.

The compact model consumes eight causal observations over 0.875 seconds, including
health uncertainty/damage trend, target/skill state, past cooldown transitions and
a small world-region visual descriptor. It excludes joystick input, control HUD,
future frames and timestamps as features. Normalization uses training rows only.
Three seeded runs were selected using validation skill accuracy and attack/retreat
F1. The model has {evaluation['config']['model_parameters']:,} parameters.

All original data hashes were checked against `baseline_lock.json`. The v2 audited
test adds reviewed intervals and corrects death labels; it does not replace the
frozen test. Its 169 correlated rows include only **one attack and two retreat
episodes**. Neither set establishes independent-match performance.

## Confusion matrices

Rows are true labels; columns are predictions. These are ungated neural choices,
not simulated win rate or claims of inputs the phone would accept.

### Baseline v1, original 94 rows

{table(original['trained']['test']['confusion'])}
### Candidate v2, same frozen 94 rows

{table(frozen['confusion'])}
### Candidate v2, separately audited 169 rows

{table(test['confusion'])}
CSV files beside this report contain the same matrices. The candidate did not
improve skill-match total or attack F1, so it was **not promoted**. Its audited
retreat F1 is {test['per_action']['retreat']['f1']:.3f} on two episodes.

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
| FP32 | {onnx['models']['fp32']['p95_ms']:.4f} ms | {onnx['models']['fp32']['bytes']:,} B | {100*onnx['models']['fp32']['argmax_agreement_vs_pytorch']:.2f}% |
| INT8 | {onnx['models']['int8']['p95_ms']:.4f} ms | {onnx['models']['int8']['bytes']:,} B | {100*onnx['models']['int8']['argmax_agreement_vs_pytorch']:.2f}% |

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
'''
    for key,label in [('queue_ms','Frame wake/queue'),('perception_ms','Perception + sequence features'),('policy_guard_ms','ONNX decision + guards'),('dispatch_ms','Control dispatch'),('decoded_to_dispatch_ms','Whole frame → dispatch')]:
        text+=f"| {label} | {p30['stages'][key]['p95_ms']:.3f} ms | {p60['stages'][key]['p95_ms']:.3f} ms |\n"
    text+=f"| Actual DOWN/MOVE → socket sent | {p30['actual_loopback_down_or_move_sent']['p95_ms']:.3f} ms | {p60['actual_loopback_down_or_move_sent']['p95_ms']:.3f} ms |\n"
    text+=f'''
Stage percentiles are not additive. Whole-frame p95 is measured directly. There
were {p30['processed_frames']}/300 and {p60['processed_frames']}/300 processed frames,
and {p30['actual_loopback_down_or_move_sent']['n']} / {p60['actual_loopback_down_or_move_sent']['n']}
actual DOWN/MOVE sends. Send timing includes selected input frames only; all-frame
p95 at 60 Hz still exceeds 16 ms. These host figures do **not** pass either requested
game-event-to-phone-touch gate. Earlier host processing p95 at 60 Hz was 34.307 ms;
the current result is {p60['stages']['decoded_to_dispatch_ms']['p95_ms']:.3f} ms.

Recorded-file decode p95: {pipeline['recorded_file_decode']['p95_ms']:.3f} ms;
BGR conversion + normalization p95: {pipeline['bgr_conversion_and_normalization']['p95_ms']:.3f} ms.
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
`baseline_lock.json`. Baseline source snapshots are preserved beside this report.
'''
    (REPORT/'real-time-report.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
