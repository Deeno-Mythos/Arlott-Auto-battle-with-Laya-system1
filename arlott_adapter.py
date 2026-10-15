"""Video-trained neural residual adapter. NumPy inference uses native matrix math."""
import json
import hashlib
from pathlib import Path

import numpy as np

NUMERIC = ('hp_fraction','enemy_distance','enemy_count')
CATEGORIES = {'hp':('unknown','critical','low','healthy'),
              'enemy_range':('none','melee','near','far'),
              'enemy_hp':('unknown','finisher','healthy'),
              'target_kind':('unknown','hero','unit'),
              's1':('ready','cooldown'), 's2':('ready','cooldown'), 'ult':('ready','cooldown'),
              'previous_action':('unknown','hold','approach','retreat','attack','skill1','skill2','ultimate','reposition'),
              'previous_action_age':('recent','older')}


def scorer_fingerprint(state_dict):
    digest=hashlib.sha256()
    for name,value in sorted(state_dict.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()


def vectorize(features):
    values=[float(features.get('battle',False)),float(features.get('mark_hint',False))]
    for name in NUMERIC:
        val=features.get(name)
        val=-1. if val is None else float(val)
        values.append(val/4 if name=='enemy_count' else val)
    for name,choices in CATEGORIES.items():
        values.extend(float(features.get(name)==x) for x in choices)
    return np.asarray(values,dtype=np.float32)


class ArlottAdapter:
    def __init__(self,directory):
        directory=Path(directory)
        self.config=json.loads((directory/'state_adapter.json').read_text())
        if self.config['schema']!='arlott-v1':
            raise ValueError('Unsupported Arlott adapter schema')
        with np.load(directory/'state_adapter.npz',allow_pickle=False) as archive:
            self.weights={k:archive[k].copy() for k in archive.files}

    def predict(self,features,base_probabilities):
        from arlott_schema import ACTION_NAMES
        base=np.log(np.maximum([base_probabilities[name] for name in ACTION_NAMES],1e-8)).astype(np.float32)
        base-=base.mean()
        h=np.concatenate([vectorize(features),base])
        for index in (0,2):
            h=np.maximum(0,h@self.weights[f'{index}.weight'].T+self.weights[f'{index}.bias'])
        z=base+h@self.weights['4.weight'].T+self.weights['4.bias']
        z=z/float(self.config['temperature']);p=np.exp(z-z.max());p/=p.sum()
        return {name:float(value) for name,value in zip(ACTION_NAMES,p)}
