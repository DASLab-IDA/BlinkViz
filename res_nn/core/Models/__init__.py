import torch
def build_network(cfg):
    name = cfg.model 
    if name == 'base':
        from .BaseModel.network import network
    elif name == 'norm':
        from .NormModel.network import network
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid optimizer!")

    return network(cfg[name])
