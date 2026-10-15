"""Train/export a standalone sequence student. Selection uses validation only."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from arlott_schema import ACTION_NAMES
from training.finetune_arlott import metrics

DATA=ROOT/'training/data/v2';OUT=ROOT/'models/arlott-rt-v2';REPORT=ROOT/'training/reports/v2'


def export_graph(net,mean,std,path):
    import onnx
    from onnx import helper as h,numpy_helper as nh,TensorProto as T
    initial=[nh.from_array(mean.astype(np.float32),'mean'),nh.from_array(std.astype(np.float32),'std')]
    nodes=[h.make_node('Sub',['context','mean'],['centered']),h.make_node('Div',['centered','std'],['normalized'])]
    previous='normalized'
    for j,k in enumerate((0,2,4)):
        w=net[k].weight.detach().numpy().T.copy();b=net[k].bias.detach().numpy()
        initial += [nh.from_array(w,f'w{j}'),nh.from_array(b,f'b{j}')]
        nodes += [h.make_node('MatMul',[previous,f'w{j}'],[f'm{j}']),h.make_node('Add',[f'm{j}',f'b{j}'],[f'z{j}'])]
        if j<2:nodes.append(h.make_node('Relu',[f'z{j}'],[f'a{j}']));previous=f'a{j}'
    nodes.append(h.make_node('Identity',['z2'],['logits']))
    graph=h.make_graph(nodes,'arlott_causal_student',[h.make_tensor_value_info('context',T.FLOAT,[1,len(mean)])],
                       [h.make_tensor_value_info('logits',T.FLOAT,[1,8])],initial)
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=10
    onnx.checker.check_model(model);onnx.save(model,str(path))


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    archive=np.load(DATA/'contexts.npz',allow_pickle=False);contexts=archive['contexts'];times=archive['times']
    sets={p:[json.loads(x) for x in (DATA/f'{p}.jsonl').read_text().splitlines()] for p in ('train','validation','test')}
    x={p:contexts[[r['index'] for r in data]].reshape(len(data),-1) for p,data in sets.items()}
    # One normalization fit, training rows only. The ONNX graph includes it.
    mean=x['train'].mean(0);std=np.maximum(x['train'].std(0),.08)
    xs={p:torch.tensor((values-mean)/std) for p,values in x.items()}
    labels={p:torch.tensor([ACTION_NAMES.index(r['label']) for r in data]) for p,data in sets.items()}
    scored={p:[{'expected':{'action':r['label']}} for r in data] for p,data in sets.items()}
    count=torch.bincount(labels['train'],minlength=8).float().clamp_min(1)
    sample=torch.tensor([r['weight'] for r in sets['train']])*(count.max()/count[labels['train']]).sqrt().clamp_max(4)
    combat=[i for i,r in enumerate(sets['validation']) if r['label'] in ('skill1','skill2','ultimate')]
    best=-1;best_net=None;selection={};curves=[];start=time.monotonic()
    for seed in (43,44,45):
        torch.manual_seed(seed)
        net=torch.nn.Sequential(torch.nn.Linear(xs['train'].shape[1],96),torch.nn.ReLU(),
                                torch.nn.Linear(96,48),torch.nn.ReLU(),torch.nn.Linear(48,8))
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.02)
        local_best=-1;best_epoch=0
        for epoch in range(1,241):
            net.train();order=torch.randperm(len(sample));total=0
            for offset in range(0,len(order),64):
                ix=order[offset:offset+64];opt.zero_grad(set_to_none=True)
                z=net(torch.nn.functional.dropout(xs['train'][ix],p=.05,training=True))
                loss=(torch.nn.functional.cross_entropy(z,labels['train'][ix],reduction='none',label_smoothing=.03)*sample[ix]).sum()/sample[ix].sum()
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();total+=float(loss.detach())*len(ix)
            net.eval()
            with torch.no_grad():val_z=net(xs['validation'])
            val=metrics(val_z,scored['validation']);skill=metrics(val_z,scored['validation'],indices=combat)
            score=.4*skill['accuracy']+.2*val['per_action']['attack']['f1']+.2*val['per_action']['retreat']['f1']+.2*val['macro_f1']
            if score>local_best+1e-6:local_best=score;best_epoch=epoch
            if score>best+1e-6:
                best=score;best_net=copy.deepcopy(net.state_dict());selection={'seed':seed,'epoch':epoch,'score':score,'validation':val,'validation_skills':skill}
            if epoch%20==0:
                row={'seed':seed,'epoch':epoch,'loss':round(total/len(sample),4),'validation_score':round(score,4),'best':round(best,4)}
                curves.append(row);print(json.dumps(row),flush=True)
            if epoch-best_epoch>=45:break
    net.load_state_dict(best_net);net.eval()
    with torch.no_grad():val_z=net(xs['validation'])
    temperature=float(min(torch.linspace(.5,5,91),key=lambda t:float(torch.nn.functional.cross_entropy(val_z/t,labels['validation']))))
    config={'schema':'arlott-rt-v2','actions':ACTION_NAMES,'context_seconds':.875,'context_offsets':[.875,.75,.625,.5,.375,.25,.125,0],
            'feature_dimension':contexts.shape[-1],'input_dimension':xs['train'].shape[-1],'hidden':[96,48],
            'temperature':temperature,'selection':selection,'model_parameters':sum(p.numel() for p in net.parameters()),
            'source':'same supplied video; no additional recordings','promotion':'candidate only; true event-to-phone-touch latency unmeasured'}
    (OUT/'policy.json').write_text(json.dumps(config,indent=2))
    torch.save({'state_dict':best_net,'mean':torch.tensor(mean),'std':torch.tensor(std)},OUT/'student.pt')
    export_graph(net,mean,std,OUT/'policy-fp32.onnx')
    from onnxruntime.quantization import quantize_dynamic,QuantType
    quantize_dynamic(str(OUT/'policy-fp32.onnx'),str(OUT/'policy-int8.onnx'),weight_type=QuantType.QInt8,per_channel=True,reduce_range=True)
    with torch.no_grad():pred={p:net(xs[p]) for p in sets}
    frozen=[json.loads(r) for r in (ROOT/'training/data/test.jsonl').read_text().splitlines()]
    indices=[max(0,int(np.searchsorted(times,r['time'],side='right'))-1) for r in frozen]
    with torch.no_grad():frozen_z=net(torch.tensor((contexts[indices].reshape(len(indices),-1)-mean)/std))
    cast_ix=[i for i,r in enumerate(frozen) if r['provenance']['evidence']=='observed_ready_to_cooldown']
    predictions=[]
    for i,r in enumerate(frozen):
        p=(frozen_z[i]/temperature).softmax(-1).numpy()
        predictions.append({'id':r['id'],'time':r['time'],'expected':r['expected']['action'],'predicted':ACTION_NAMES[int(p.argmax())],
                            'confidence':float(p.max()),'label_invalid_death':r['time']>=759.5})
    report={'method':'Standalone causal supervised student; full Laya removed from per-frame inference. No future labels or joystick in features.',
            'selection':selection,'config':config,'curve':curves,'seconds':time.monotonic()-start,
            'v2_manifest':json.loads((DATA/'manifest.json').read_text()),
            'audited':{p:metrics(pred[p],scored[p],temperature) for p in sets},
            'frozen_original':metrics(frozen_z,frozen,temperature),'frozen_24_casts':metrics(frozen_z,frozen,temperature,cast_ix),
            'criteria':{'skill_18_of_24':sum(predictions[i]['predicted']==predictions[i]['expected'] for i in cast_ix)>=18,
                        'attack_f1_0.80':metrics(frozen_z,frozen,temperature)['per_action']['attack']['f1']>=.8,
                        'retreat_f1_0.80_valid':False,
                        'retreat_reason':'Sole original retreat is dead/spectating; audited set has only two retreat episodes.',
                        'true_event_touch_latency':'unverified; no phone game running'}}
    (REPORT/'evaluation.json').write_text(json.dumps(report,indent=2))
    (REPORT/'frozen_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    np.savez_compressed(DATA/'export_validation.npz',inputs=x['validation'],expected_logits=pred['validation'].numpy())
    print(json.dumps({'saved':str(OUT),'frozen':report['frozen_original'],'casts':report['frozen_24_casts'],'audited':report['audited']['test'],'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
