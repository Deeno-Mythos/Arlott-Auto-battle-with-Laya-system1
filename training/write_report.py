"""Render the actual saved evaluation results into a readable local report."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'training/reports'


def main():
    e=json.loads((REPORT/'evaluation.json').read_text())
    v=json.loads((REPORT/'reload_verification.json').read_text())
    p=json.loads((REPORT/'performance.json').read_text())
    assert abs(v['accuracy']-e['trained']['test']['accuracy'])<.0001, 'Verify current checkpoint before writing report'
    splits=e['manifest']['splits'];test=e['trained']['test'];casts=e['test_cast_events']
    first=json.loads((REPORT/'overall-adapter-first-pass.json').read_text())
    def pc(value):return f'{100*value:.1f}%'
    text=f'''# Arlott video training — offline results

The configured controller now loads Laya with a genuinely trained neural state
adapter. Gradient updates changed {e['scorer_tensors_changed']} native scorer tensors
and trained {e['adapter']['trainable_parameters']:,} adapter parameters. The encoder
and transformer body remain frozen. This is a small imitation-learning experiment
from one edited recording, not evidence of autonomous match performance.

## Results from the final stretch of the same recording

| Policy | Action agreement | Macro F1 |
| --- | ---: | ---: |
| Stock Laya, same eight-action schema | {pc(e['original']['test']['accuracy'])} | {e['original']['test']['macro_f1']:.3f} |
| Always choose most common training action | {pc(e['test_majority']['accuracy'])} | {e['test_majority']['macro_f1']:.3f} |
| Native scorer fine-tuning alone (failed) | {pc(e['native_scorer_test']['accuracy'])} | {e['native_scorer_test']['macro_f1']:.3f} |
| First adapter, selected for general agreement | {pc(first['trained']['test']['accuracy'])} | {first['trained']['test']['macro_f1']:.3f} |
| Final adapter, selected for combat decisions | {pc(test['accuracy'])} | {test['macro_f1']:.3f} |

The final pass favors combat coverage over movement agreement. It matched
**{round(casts['n']*casts['accuracy'])}/{casts['n']} detected skill-cast labels**;
skill-cast macro F1 is **{casts['macro_f1']:.3f}**, versus
{first['test_cast_events']['macro_f1']:.3f} for the first adapter. Among detected
casts above 0.60 model confidence, agreement was {pc(casts['confidence_0.6_accuracy'])}
on only {round(casts['n']*casts['confidence_0.6_coverage'])} cases. This small subset
does not establish reliable calibration or game performance.

| Action | Test labels | Precision | Recall |
| --- | ---: | ---: | ---: |
'''
    for name,m in test['per_action'].items():
        text+=f"| {name} | {m['n']} | {pc(m['precision'])} | {pc(m['recall'])} |\n"
    text+=f'''
Basic attack and retreat were not learned reliably. Ultimate has only two test
examples. The controller therefore retains the explicit combat fallback and checks
health, target range, cooldown readings and context age before applying learned
choices. Predicted reposition does not supply a navigation route.

## What was trained

Source: `{e['manifest']['source']['path']}`

Source SHA-256: `{e['manifest']['source']['sha256']}`

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
| Train | before 500 s | {splits['train']['rows']} |
| Validation | 510–620 s | {splits['validation']['rows']} |
| Temporal test | 630–787 s | {splits['test']['rows']} |

Ten-second gaps separate stretches. {splits['test']['identical_state_strings_in_train']}
test examples share identical serialized states with training despite using
different timestamps. On the remaining {e['test_novel_state_strings']['n']} novel
state strings, agreement is {pc(e['test_novel_state_strings']['accuracy'])}.
Excluding weak movement labels, agreement is
{pc(e['test_without_weak_movement']['accuracy'])} on
{e['test_without_weak_movement']['n']} examples. The temporal test was inspected
during development; it is not an untouched independent-match benchmark.

The native scorer-only pass failed and was retained in
`native_scorer_evaluation.json`. The final residual adapter is a 64/32-unit ReLU
MLP over explicit state features and frozen Laya context. Training uses
confidence-weighted, class-balanced cross entropy. Epoch {e['adapter']['best_epoch']}
was selected from {e['epochs_run']} epochs using validation combat macro F1
(70%) and overall validation macro F1 (30%). Temperature scaling used validation
labels only. NumPy and PyTorch adapter probabilities agreed within
{e['adapter_numpy_pytorch_max_error']:.2g}.

## Saved model and runtime verification

Weights: `models/arlott-video-v1/model.safetensors`

Adapter: `models/arlott-video-v1/state_adapter.npz` and `state_adapter.json`

Native weights SHA-256: `{v['weights_sha256']}`

Load through `gameplay.LayaPolicy`; plain `laya.load` does not apply the sidecar
adapter. A scorer fingerprint prevents combining an adapter with different native
weights. `config.json` selects the trained policy. The pre-training configuration
is preserved in `training/reports/config-before-video-training.json`.

Saved-model replay: {v['rows']} examples, {pc(v['reload_prediction_agreement'])}
agreement with exported training predictions. Maximum confidence discrepancy:
{v['confidence_max_error']:.2g}. Same-state fast refresh agreement:
{pc(v['fast_refresh_same_state_agreement'])}.

Full Laya CPU inference: median {v['latency_median_ms']} ms,
95th percentile {v['latency_p95_ms']} ms. Four CPU threads were fastest among
1/2/4 in the validation benchmark. Native NumPy adapter-only mean:
{p['adapter_only_mean_ms']:.3f} ms; current replay median refresh:
{v['fast_refresh_median_ms']:.3f} ms. These adapter numbers exclude the transformer
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
'''
    (REPORT/'arlott-training-report.md').write_text(text,encoding='utf-8')
    print(str(REPORT/'arlott-training-report.md'))


if __name__=='__main__':main()
