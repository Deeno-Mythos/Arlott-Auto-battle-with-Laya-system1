# Arlott video training — offline results

The original local deployment loaded Laya with a trained neural state
adapter (weights are not included in Git). Gradient updates changed 6 native scorer tensors
and trained 5,224 adapter parameters. The encoder
and transformer body remain frozen. This is a small imitation-learning experiment
from one edited recording, not evidence of autonomous match performance.

## Results from the final stretch of the same recording

| Policy | Action agreement | Macro F1 |
| --- | ---: | ---: |
| Stock Laya, same eight-action schema | 7.4% | 0.082 |
| Always choose most common training action | 54.3% | 0.088 |
| Native scorer fine-tuning alone (failed) | 11.7% | 0.085 |
| First adapter, selected for general agreement | 64.9% | 0.373 |
| Final adapter, selected for combat decisions | 57.5% | 0.325 |

The final pass favors combat coverage over movement agreement. It matched
**11/24 detected skill-cast labels**;
skill-cast macro F1 is **0.527**, versus
0.453 for the first adapter. Among detected
casts above 0.60 model confidence, agreement was 72.7%
on only 11 cases. This small subset
does not establish reliable calibration or game performance.

| Action | Test labels | Precision | Recall |
| --- | ---: | ---: | ---: |
| hold | 10 | 100.0% | 100.0% |
| approach | 5 | 0.0% | 0.0% |
| retreat | 1 | 0.0% | 0.0% |
| attack | 3 | 0.0% | 0.0% |
| skill1 | 5 | 16.7% | 20.0% |
| skill2 | 17 | 45.0% | 52.9% |
| ultimate | 2 | 12.5% | 50.0% |
| reposition | 51 | 84.6% | 64.7% |

Basic attack and retreat were not learned reliably. Ultimate has only two test
examples. The controller therefore retains the explicit combat fallback and checks
health, target range, cooldown readings and context age before applying learned
choices. Predicted reposition does not supply a navigation route.

## What was trained

Source: a user-supplied local Arlott recording; source footage is not redistributed.

Source SHA-256: `1d9b7db7a719af56571101fa2fe1d362dd0f05898d9fdbd7193716f1746e791d`

PyAV decoded the 787-second recording locally. The game viewport was normalized
to the existing HUD detector. Before-action health, target range/type, skill
readiness, weak mark hint and previous cast are serialized in the same schema used
at runtime. Future frames supply cooldown-transition labels only. Joystick evidence,
timestamps, future state and review images are excluded from model inputs.

Cooldown and movement labels are imperfect CV observations. A small set of attacks
and rotations received assistant visual review, not human expert annotation. No
claims are based on an audio transcript. The supplied video was treated as data.

| Split | Time range | Examples |
| --- | --- | ---: |
| Train | before 500 s | 327 |
| Validation | 510–620 s | 77 |
| Temporal test | 630–787 s | 94 |

Ten-second gaps separate stretches. 23
test examples share identical serialized states with training despite using
different timestamps. On the remaining 71 novel
state strings, agreement is 47.9%.
Excluding weak movement labels, agreement is
60.5% on
43 examples. The temporal test was inspected
during development; it is not an untouched independent-match benchmark.

The native scorer-only pass failed and was retained in
`native_scorer_evaluation.json`. The final residual adapter is a 64/32-unit ReLU
MLP over explicit state features and frozen Laya context. Training uses
confidence-weighted, class-balanced cross entropy. Epoch 46
was selected from 126 epochs using validation combat macro F1
(70%) and overall validation macro F1 (30%). Temperature scaling used validation
labels only. NumPy and PyTorch adapter probabilities agreed within
2.4e-07.

## Saved model and runtime verification

Weights: `models/arlott-video-v1/model.safetensors`

Adapter: `models/arlott-video-v1/state_adapter.npz` and `state_adapter.json`

Native weights SHA-256: `1cbaf32da9b7f26e35249cc770b03d3e552334d7c0a5305e94965d242285e216`

Load through `gameplay.LayaPolicy`; plain `laya.load` does not apply the sidecar
adapter. A scorer fingerprint prevents combining an adapter with different native
weights. `config.json` selects the trained policy. The pre-training configuration
is preserved in `training/reports/config-before-video-training.json`.

Saved-model replay: 94 examples, 100.0%
agreement with exported training predictions. Maximum confidence discrepancy:
4.9e-05. Same-state fast refresh agreement:
100.0%.

Full Laya CPU inference: median 421 ms,
95th percentile 453 ms. Four CPU threads were fastest among
1/2/4 in the validation benchmark. Native NumPy adapter-only mean:
0.068 ms; current replay median refresh:
0.112 ms. These adapter numbers exclude the transformer
and screen processing. The 120 ms control loop refreshes the adapter on current
state while cached Laya context remains under 1.5 seconds old. Stop invalidates it.
Reusing context across a changed scene is an approximation; only same-state refresh
parity was verified, and no live-game performance is claimed.

Offline replay and controller regressions use no Android connection. No controller
or live game was launched during this training task. Live practice testing on the
phone remains outstanding; perception may differ from this recording.

## Reproduce

See the offline commands in the project README. Detailed metrics and confusion
matrices: `evaluation.json`; raw predictions: `test_predictions.jsonl` and
`replay.jsonl`; dataset provenance and hashes: `training/data/dataset_manifest.json`.
The data builder documents each annotation and its confidence.
