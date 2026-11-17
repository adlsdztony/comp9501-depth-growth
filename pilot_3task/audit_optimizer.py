"""Read-only named-Adam audit via native module/group reconstruction.
Saved IDs are never paired with state_dict tensor order. Unsupported layouts fail.
"""
import copy
from pathlib import Path
import re
from unittest.mock import patch
import torch
from torch import nn
from backend import NativeFactory
from growth import append_layer
from prune_ops import equal,named_steps,model_digest


def load_native_checkpoint(path,author_root,device='cpu',task_names=None,factory=None):
    path=Path(path);saved=torch.load(path,map_location=device,weights_only=False)
    state=saved['algorithm'];expected=saved['optimizer']
    if task_names is None:task_names=saved.get('state',{}).get('task_names')
    if factory is None:factory=NativeFactory(author_root,device=device,task_names=task_names)
    names=[]
    for key,value in state.items():
        match=re.fullmatch(r'policy\.actor\.actor_body\.((?:adaptive|growth)_\d+)\.0\.weight',key)
        if match:
            assert tuple(value.shape)==(400,400)
            names.append(match.group(1))
    assert len(names)<=2 and len(names)==len(set(names))
    groups=expected['param_groups'];assert groups
    if len(groups)>1:
        assert names==[f'growth_{i}' for i in range(1,len(names)+1)]
        assert len(groups)==len(names)+1 and len(groups[0]['params'])==19
        assert all(len(g['params'])==1 and g['growth_index']==i for i,g in enumerate(groups[1:],1))
        algorithm=factory()
        for i in range(len(names)):append_layer(algorithm,'identity',9491+i)
        layout='native base19 plus appended one-parameter growth groups'
    else:
        count=len(groups[0]['params']);private=count==20+len(names)
        assert count==(20 if private else 19)+len(names)
        assert names in ([],['adaptive_1'])
        original_actor=factory.train.MultiTaskActor;original_critic=factory.train.MultiTaskCritic
        def actor_constructor(**kwargs):
            actor=original_actor(**kwargs)
            for name in names:
                layer=nn.Linear(400,400,bias=False,device=device,dtype=actor.actor_output.weight.dtype)
                actor.actor_body.add_module(name,nn.Sequential(layer,nn.ReLU()))
            return actor
        def critic_constructor(**kwargs):
            if private:kwargs['preprocess_net']=copy.deepcopy(kwargs['preprocess_net'])
            return original_critic(**kwargs)
        with patch.object(factory.train,'MultiTaskActor',actor_constructor),patch.object(factory.train,'MultiTaskCritic',critic_constructor):algorithm=factory()
        layout='native single group; '+('private' if private else 'shared')+' dummy preprocess'
    raw=algorithm._get_raw_optimizer();by_id={id(p):name for name,p in algorithm.named_parameters()}
    group_names=[[by_id[id(p)] for p in group['params']] for group in raw.param_groups]
    assert [len(g) for g in group_names]==[len(g['params']) for g in groups]
    assert {n for g in group_names for n in g}==set(by_id.values())
    algorithm.load_state_dict(state);raw.load_state_dict(expected)
    assert equal(raw.state_dict(),expected)
    for name,p in algorithm.named_parameters():assert equal(p,state[name]),name
    for prefix,module in [('policy.actor.',algorithm.policy.actor),('critic.',algorithm.critic)]:
        assert equal(module.state_dict(),{n[len(prefix):]:v for n,v in state.items() if n.startswith(prefix)})
    report=dict(layout=layout,optimizer_parameter_names_by_group=group_names,named_adam_steps=named_steps(algorithm),
        parameters_without_initialized_Adam=[n for n,p in algorithm.named_parameters() if p not in raw.state],
        actor_parameters=sum(p.numel() for p in algorithm.policy.actor.parameters()),critic_parameters=sum(p.numel() for p in algorithm.critic.parameters()),
        loaded_model_fingerprints=model_digest(algorithm),all_raw_Adam_tensors_and_options_exact=True,
        ownership_basis='Native construction and named_parameters, never state_dict tensor order',scope='Read-only engineering audit, no simulator resume')
    return algorithm,report
