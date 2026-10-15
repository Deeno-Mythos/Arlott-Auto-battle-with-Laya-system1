"""Train structured-state residual neural adapter on the same temporal split."""
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TF','0')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from laya.train import items_from_rows, encode_item
from arlott_schema import ACTION_NAMES
from arlott_adapter import vectorize, ArlottAdapter, scorer_fingerprint
from training.finetune_arlott import metrics, rows

DATA=ROOT/'training/data';REPORT=ROOT/'training/reports';OUT=ROOT/'models/arlott-video-v1'


def main():
    torch.set_num_threads(4);torch.manual_seed(43);random.seed(43);np.random.seed(43)
    start=time.monotonic();sets={p:rows(p) for p in ('train','validation','test')}
    previous=REPORT/'native_scorer_evaluation.json'
    native=json.loads((previous if previous.exists() else REPORT/'evaluation.json').read_text())
    (REPORT/'native_scorer_evaluation.json').write_text(json.dumps(native,indent=2))
    cache=torch.load(DATA/'laya-marker-cache.pt',weights_only=True)
    d=cache['x'].shape[-1]
    scorer=torch.nn.Sequential(torch.nn.LayerNorm(d),torch.nn.Linear(d,d),torch.nn.GELU(),torch.nn.Linear(d,1))
    with safe_open(OUT/'model.safetensors',framework='pt',device='cpu') as weights:
        scorer.load_state_dict({key[len('scorer.'):]:weights.get_tensor(key).float() for key in weights.keys() if key.startswith('scorer.')})
    scorer.eval()
    with torch.no_grad(): native_z=scorer(cache['x']).squeeze(-1)
    tok=AutoTokenizer.from_pretrained(str(OUT/'tokenizer'))
    unique={};encoded=[];mapping={}
    for part,data in sets.items():
        items,skipped=items_from_rows(tok,data,512,192);assert not skipped
        mapping[part]=[]
        for item in items:
            e=encode_item(tok,item,512,192);key=tuple(e['ids'])
            if key not in unique: unique[key]=len(encoded);encoded.append(e)
            mapping[part].append(unique[key])
    assert hashlib.sha256(json.dumps([e['ids'] for e in encoded]).encode()).hexdigest()==cache['signature']
    temperature_base=native['temperature']
    base={p:native_z[mapping[p]]/temperature_base for p in sets}
    # Match rounded probabilities returned by Laya's public runtime exactly.
    for p in base:
        probs=base[p].softmax(-1).numpy().round(4).clip(1e-8)
        base[p]=torch.tensor(np.log(probs));base[p]-=base[p].mean(-1,keepdim=True)
    x={p:torch.cat([torch.tensor(np.stack([vectorize(r['features']) for r in sets[p]])),base[p]],-1) for p in sets}
    y={p:torch.tensor([ACTION_NAMES.index(r['expected']['action']) for r in sets[p]]) for p in sets}
    adapter=torch.nn.Sequential(torch.nn.Linear(x['train'].shape[-1],64),torch.nn.ReLU(),
                                 torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,len(ACTION_NAMES)))
    opt=torch.optim.AdamW(adapter.parameters(),lr=.003,weight_decay=.02)
    counts=torch.bincount(y['train'],minlength=len(ACTION_NAMES)).float().clamp_min(1)
    class_weights=(counts.max()/counts).sqrt().clamp_max(4)
    sample_weights=torch.tensor([r['weight'] for r in sets['train']])*class_weights[y['train']]
    best=-1;best_epoch=0;curve=[];best_weights=None
    combat_val=[i for i,r in enumerate(sets['validation'])
                if r['expected']['action'] in ('attack','skill1','skill2','ultimate')
                and r['provenance']['evidence']!='joystick_direction_weak']
    def output(part):
        adapter.eval()
        with torch.no_grad():return base[part]+adapter(x[part])
    for epoch in range(1,601):
        adapter.train();order=torch.randperm(len(x['train']));total=0
        for offset in range(0,len(order),64):
            ix=order[offset:offset+64];opt.zero_grad(set_to_none=True)
            z=base['train'][ix]+adapter(x['train'][ix])
            losses=torch.nn.functional.cross_entropy(z,y['train'][ix],label_smoothing=.04,reduction='none')
            loss=(losses*sample_weights[ix]).sum()/sample_weights[ix].sum()
            loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(ix)
        val=metrics(output('validation'),sets['validation'])
        combat=metrics(output('validation'),sets['validation'],indices=combat_val)
        score=.7*combat['macro_f1']+.3*val['macro_f1']
        if score>best+1e-6:
            best=score;best_epoch=epoch;best_weights=copy.deepcopy(adapter.state_dict())
        entry={'epoch':epoch,'train_loss':round(total/len(order),4),'validation_accuracy':val['accuracy'],'validation_macro_f1':val['macro_f1'],
               'validation_combat_macro_f1':combat['macro_f1'],'best_epoch':best_epoch}
        curve.append(entry)
        if epoch%10==0:print(json.dumps(entry),flush=True)
        if epoch-best_epoch>=80:break
    adapter.load_state_dict(best_weights);adapter.eval()
    val_z=output('validation')
    temp=float(min(torch.linspace(.5,5,91),key=lambda t:float(torch.nn.functional.cross_entropy(val_z/t,y['validation']))))
    np.savez_compressed(OUT/'state_adapter.npz',**{k:v.detach().numpy() for k,v in best_weights.items()})
    config={'schema':'arlott-v1','kind':'structured-state residual neural adapter','hidden':[64,32],'activation':'relu',
            'temperature':temp,'base_temperature':temperature_base,'actions':ACTION_NAMES,'seed':43,'best_epoch':best_epoch,
            'trainable_parameters':sum(p.numel() for p in adapter.parameters()),'features_exclude':['joystick','timestamp','future observations'],
            'base_scorer_sha256':scorer_fingerprint(scorer.state_dict())}
    (OUT/'state_adapter.json').write_text(json.dumps(config,indent=2))
    trained={p:output(p) for p in sets}
    strong=[i for i,r in enumerate(sets['test']) if r['provenance']['evidence']!='joystick_direction_weak']
    casts=[i for i,r in enumerate(sets['test']) if r['provenance']['evidence']=='observed_ready_to_cooldown']
    train_states={r['state'] for r in sets['train']};unseen=[i for i,r in enumerate(sets['test']) if r['state'] not in train_states]
    report={k:v for k,v in native.items() if k not in ('curve','trained','test_without_weak_movement','test_cast_events','test_novel_state_strings')}
    report.update(method='Fine-tuned Laya scorer plus trained structured-state residual neural adapter. Encoder/transformer frozen.',
                  native_scorer_test=native['trained']['test'],adapter=config,adapter_curve=curve,
                  best_epoch=best_epoch,epochs_run=len(curve),temperature=temp,adapter_seconds=round(time.monotonic()-start,2),
                  selection='Adapter epoch maximizes 0.7 validation combat macro F1 + 0.3 overall validation macro F1. Class-balanced confidence-weighted loss. Validation only temperature. Previous failed/imbalanced runs retained. Same temporal test used for diagnostic iterations; no independent-match test.',
                  trained={p:metrics(trained[p],sets[p],temp) for p in sets},
                  test_without_weak_movement=metrics(trained['test'],sets['test'],temp,strong),
                  test_cast_events=metrics(trained['test'],sets['test'],temp,casts),
                  test_novel_state_strings=metrics(trained['test'],sets['test'],temp,unseen))
    runtime=ArlottAdapter(OUT);parity=[]
    predictions=[]
    for i,r in enumerate(sets['test']):
        p=(trained['test'][i]/temp).softmax(-1).numpy()
        original_probs={a:float(v) for a,v in zip(ACTION_NAMES,base['test'][i].softmax(-1).numpy())}
        np_probs=runtime.predict(r['features'],original_probs)
        parity.append(max(abs(np_probs[a]-p[k]) for k,a in enumerate(ACTION_NAMES)))
        predictions.append({'id':r['id'],'time':r['time'],'expected':r['expected']['action'],
            'trained':ACTION_NAMES[int(p.argmax())],'confidence':round(float(p.max()),4),'evidence':r['provenance']['evidence']})
    report['adapter_numpy_pytorch_max_error']=float(max(parity))
    assert max(parity)<1e-5
    (REPORT/'evaluation.json').write_text(json.dumps(report,indent=2))
    (REPORT/'test_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    print(json.dumps({'stage':'adapter_saved','test':report['trained']['test'],'casts':report['test_cast_events'],'validation':report['trained']['validation'],'best_epoch':best_epoch,'seconds':report['adapter_seconds']}),flush=True)


if __name__=='__main__':main()
