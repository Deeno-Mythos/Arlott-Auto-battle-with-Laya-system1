"""CPU Laya fine-tuning, using exact frozen-body activations for speed.

Train the native Laya LayerNorm/MLP scorer; encoder, transformer decision head,
and act head remain frozen. Checkpoint is directly loadable with laya.load.
Only validation selects epoch and temperature. Test labels are scored once.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault('USE_TF','0')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arlott_schema import ACTION_NAMES
import numpy as np
import torch
from laya.common import collate_items
from laya.train import encode_item, items_from_rows, load_checkpoint, save_checkpoint

DATA=ROOT/'training/data'
OUT=ROOT/'models/arlott-video-v1'
REPORT=ROOT/'training/reports'


def rows(part):
    return [json.loads(line) for line in (DATA/f'{part}.jsonl').read_text().splitlines()]


def metrics(logits,data,temperature=1.0,indices=None):
    ix=list(range(len(data))) if indices is None else indices
    if not ix: return {'n':0}
    z=logits[ix]/temperature
    p=z.softmax(-1); predictions=p.argmax(-1).numpy()
    gold=np.array([ACTION_NAMES.index(data[i]['expected']['action']) for i in ix])
    confusion=np.zeros((len(ACTION_NAMES),len(ACTION_NAMES)),dtype=int)
    for y,h in zip(gold,predictions): confusion[y,h]+=1
    per={}; f1=[]; recalls=[]
    for k,name in enumerate(ACTION_NAMES):
        n=int(confusion[k].sum()); tp=int(confusion[k,k]); pred=int(confusion[:,k].sum())
        precision=tp/max(1,pred); recall=tp/max(1,n)
        score=2*precision*recall/max(1e-12,precision+recall)
        per[name]={'n':n,'precision':round(precision,4),'recall':round(recall,4),'f1':round(score,4)}
        if n: f1.append(score); recalls.append(recall)
    conf=p.max(-1).values.numpy(); correct=(gold==predictions)
    ece=sum(float(((conf>lo)&(conf<=lo+.1)).mean())*abs(float(conf[(conf>lo)&(conf<=lo+.1)].mean())-float(correct[(conf>lo)&(conf<=lo+.1)].mean()))
            for lo in np.arange(0,1,.1) if ((conf>lo)&(conf<=lo+.1)).any())
    selected=conf>=.6
    return {'n':len(ix),'accuracy':round(float(correct.mean()),4),'macro_f1':round(float(np.mean(f1)),4),
            'balanced_accuracy':round(float(np.mean(recalls)),4),
            'nll':round(float(torch.nn.functional.cross_entropy(z,torch.tensor(gold))),4),
            'ece':round(ece,4),'confidence_0.6_coverage':round(float(selected.mean()),4),
            'confidence_0.6_accuracy':round(float(correct[selected].mean()),4) if selected.any() else None,
            'per_action':per,'confusion':confusion.tolist()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--threads',type=int,default=4);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--batch',type=int,default=32)
    args=parser.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    REPORT.mkdir(parents=True,exist_ok=True)
    sets={p:rows(p) for p in ('train','validation','test')}
    model,tok,cfg=load_checkpoint('multilingual');model.eval()
    cfg.update(max_len=512,head_max_len=192,option_layout='sequential')
    frozen_hash=hashlib.sha256(model.scorer[1].weight.detach().numpy().tobytes()).hexdigest()
    initial=copy.deepcopy(model.scorer.state_dict())
    encoded=[];mapping={}; unique={}; lengths=[]
    for part,data in sets.items():
        items,skipped=items_from_rows(tok,data,512,192)
        if skipped or len(items)!=len(data): raise RuntimeError(f'Tokenization dropped data: {skipped}')
        mapping[part]=[]
        for item in items:
            e=encode_item(tok,item,512,192)
            key=tuple(e['ids'])
            if key not in unique:
                unique[key]=len(encoded);encoded.append(e);lengths.append(len(key))
            mapping[part].append(unique[key])
    print(json.dumps({'stage':'cache_frozen_body','unique_inputs':len(encoded),'rows':{p:len(x) for p,x in sets.items()},'token_lengths':[min(lengths),max(lengths)]}),flush=True)
    # Hook native model's scorer input, keeping exact attention/marker semantics.
    captured=[]
    handle=model.scorer.register_forward_pre_hook(lambda _module,inputs: captured.append(inputs[0].detach().clone()))
    activations=[];baseline=[];start=time.monotonic()
    cachefile=DATA/'laya-marker-cache.pt'
    signature=hashlib.sha256(json.dumps([e['ids'] for e in encoded]).encode()).hexdigest()
    cached=torch.load(cachefile,weights_only=True) if cachefile.exists() else None
    if cached is not None and cached['signature']==signature and cached['base_scorer_hash']==frozen_hash:
        x=cached['x']; base=cached['logits']; parity=cached['parity']
        print(json.dumps({'stage':'cache_reused'}),flush=True)
    else:
        with torch.inference_mode():
            for offset in range(0,len(encoded),4):
                batch=collate_items([encoded[offset:offset+4]],tok.pad_token_id)
                logits,_=model(**{k:batch[k] for k in ('input_ids','attention_mask','marker_pos','marker_mask','qtype')})
                activations.append(captured.pop());baseline.append(logits.detach().clone())
                if offset%40==0: print(json.dumps({'stage':'cache','done':min(offset+4,len(encoded)),'total':len(encoded),'seconds':round(time.monotonic()-start,1)}),flush=True)
        x=torch.cat(activations);base=torch.cat(baseline)
        with torch.inference_mode(): parity=float((model.scorer(x).squeeze(-1)-base).abs().max())
        if parity>1e-4: raise RuntimeError(f'Cached/native scorer mismatch {parity}')
        torch.save({'signature':signature,'base_scorer_hash':frozen_hash,'x':x,'logits':base,'parity':parity},cachefile)
    handle.remove()
    # inference_mode tensors cannot participate in autograd; clone into normal tensors.
    x=x.clone(); xs={p:x[mapping[p]].clone() for p in sets}
    for param in model.parameters(): param.requires_grad_(False)
    for param in model.scorer.parameters(): param.requires_grad_(True)
    opt=torch.optim.AdamW(model.scorer.parameters(),lr=3e-4,weight_decay=.01)
    labels=torch.tensor([ACTION_NAMES.index(r['expected']['action']) for r in sets['train']])
    counts=torch.bincount(labels,minlength=len(ACTION_NAMES)).float().clamp_min(1)
    class_weights=(counts.max()/counts).sqrt().clamp_max(3)
    sample_weights=torch.tensor([r['weight'] for r in sets['train']])*class_weights[labels]
    best=-1;best_epoch=0;best_weights=None;curve=[]
    def logits_for(part):
        model.scorer.eval()
        with torch.no_grad(): return model.scorer(xs[part]).squeeze(-1)
    baseline_metrics={p:metrics(base[mapping[p]],sets[p]) for p in sets}
    for epoch in range(1,args.epochs+1):
        order=torch.randperm(len(labels));loss_sum=0
        model.scorer.train()
        for offset in range(0,len(order),args.batch):
            ix=order[offset:offset+args.batch];opt.zero_grad(set_to_none=True)
            z=model.scorer(xs['train'][ix]).squeeze(-1)
            losses=torch.nn.functional.cross_entropy(z,labels[ix],label_smoothing=.05,reduction='none')
            loss=(losses*sample_weights[ix]).sum()/sample_weights[ix].sum()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.scorer.parameters(),1);opt.step()
            loss_sum+=float(loss.detach())*len(ix)
        v=metrics(logits_for('validation'),sets['validation'])
        score=v['macro_f1']
        if score>best+1e-5:
            best=score;best_epoch=epoch;best_weights=copy.deepcopy(model.scorer.state_dict())
        entry={'epoch':epoch,'loss':round(loss_sum/len(labels),4),'validation_accuracy':v['accuracy'],'validation_macro_f1':v['macro_f1'],'best_epoch':best_epoch}
        curve.append(entry);print(json.dumps(entry),flush=True)
        (REPORT/'training_progress.json').write_text(json.dumps(curve,indent=2))
        if epoch-best_epoch>=15: break
    model.scorer.load_state_dict(best_weights);model.eval()
    val_logits=logits_for('validation')
    val_labels=torch.tensor([ACTION_NAMES.index(r['expected']['action']) for r in sets['validation']])
    grid=torch.linspace(.5,5,91)
    temp=float(min(grid,key=lambda t:float(torch.nn.functional.cross_entropy(val_logits/t,val_labels))))
    cfg['temperature']=[1.,1.,1.];cfg['temperature_by_options']={'choice:6-10':temp}
    cfg.pop('lang_temperatures',None)
    cfg['arlott_training']={'schema':'arlott-v1','method':'frozen-body scorer fine-tuning','seed':args.seed,'best_epoch':best_epoch,
                            'source_sha256':json.loads((DATA/'dataset_manifest.json').read_text())['source']['sha256']}
    save_checkpoint(model,tok,cfg,str(OUT))
    trained_logits={p:logits_for(p) for p in sets}
    majority=Counter(r['expected']['action'] for r in sets['train']).most_common(1)[0][0]
    constant=torch.zeros(len(sets['test']),len(ACTION_NAMES));constant[:,ACTION_NAMES.index(majority)]=1
    strong=[i for i,r in enumerate(sets['test']) if r['provenance']['evidence']!='joystick_direction_weak']
    cast=[i for i,r in enumerate(sets['test']) if r['provenance']['evidence']=='observed_ready_to_cooldown']
    train_strings={r['state'] for r in sets['train']};unseen=[i for i,r in enumerate(sets['test']) if r['state'] not in train_strings]
    predictions=[{'id':r['id'],'time':r['time'],'expected':r['expected']['action'],
                  'trained':ACTION_NAMES[int(trained_logits['test'][i].argmax())],
                  'original':ACTION_NAMES[int(base[mapping['test']][i].argmax())],
                  'confidence':round(float((trained_logits['test'][i]/temp).softmax(-1).max()),4),
                  'evidence':r['provenance']['evidence']} for i,r in enumerate(sets['test'])]
    changed=sum(not torch.equal(initial[k],best_weights[k]) for k in initial)
    report={'method':'Actual gradient updates to Laya native scorer; encoder and transformer body frozen.',
            'model':str(OUT),'trainable_parameters':sum(p.numel() for p in model.scorer.parameters()),
            'scorer_tensors_changed':changed,'seed':args.seed,'best_epoch':best_epoch,'epochs_run':len(curve),
            'selection':'validation macro F1 only; temperature fitted on validation NLL; no test selection',
            'cached_native_max_logit_error':parity,'seconds':round(time.monotonic()-start,2),'temperature':temp,
            'manifest':json.loads((DATA/'dataset_manifest.json').read_text()),'curve':curve,
            'original':baseline_metrics,'trained':{p:metrics(trained_logits[p],sets[p],temp) for p in sets},
            'test_majority':{'label':majority,**metrics(constant,sets['test'])},
            'test_without_weak_movement':{'trained':metrics(trained_logits['test'],sets['test'],temp,strong),'original':metrics(base[mapping['test']],sets['test'],1.,strong)},
            'test_cast_events':{'trained':metrics(trained_logits['test'],sets['test'],temp,cast),'original':metrics(base[mapping['test']],sets['test'],1.,cast)},
            'test_novel_state_strings':{'trained':metrics(trained_logits['test'],sets['test'],temp,unseen),'original':metrics(base[mapping['test']],sets['test'],1.,unseen)},
            'limitations':['One edited match; no win-rate or live-game validation.','Automated cooldown labels can be wrong.','Movement labels are weak. Attack and retreat labels are sparse.','Reposition class learns when to rotate, not a navigation route.','CV is not a trained object detector and cannot reliably identify Nolan or marks.']}
    (REPORT/'evaluation.json').write_text(json.dumps(report,indent=2))
    (REPORT/'native_scorer_evaluation.json').write_text(json.dumps(report,indent=2))
    (REPORT/'test_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    print(json.dumps({'stage':'saved','model':str(OUT),'test_accuracy':report['trained']['test']['accuracy'],'original_accuracy':report['original']['test']['accuracy'],'majority_accuracy':report['test_majority']['accuracy'],'macro_f1':report['trained']['test']['macro_f1'],'best_epoch':best_epoch,'temperature':temp}),flush=True)


if __name__=='__main__': main()
