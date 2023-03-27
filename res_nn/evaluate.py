import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from configs.default import get_cfg
from core.utils.misc import process_cfg
import datasets
import statistics
import platform
from core.Models import build_network

@torch.no_grad()
def validate(model, cfg):
    model.eval()
    results = {}

    #val_dataset = datasets.SPNDataset(is_test=True, cfg=cfg)
    test_loader = datasets.fetch_test_dataloader(cfg)
    relative_error_list = []
    total_time = 0.0 


    #for val_id in range(len(val_dataset)):

    #count = 0
    for i_batch, data_blob in enumerate(test_loader):
        spn_values, spn_preds, gt = [x.to(cfg.device) for x in data_blob]  

        # spn_values = spn_values[None].to(cfg.device)
        # spn_preds = spn_preds[None].to(cfg.device)

        time_start = time.perf_counter()
        preds = model(spn_values, spn_preds, {})
        time_end = time.perf_counter()
        total_time = total_time + (time_end - time_start)

        
        gt = gt.cpu()
        relative_error = ((preds.cpu()-gt) / gt).abs()
        print("preds:", preds)
        print("gt:", gt)
        
        relative_error_list.append(relative_error.view(-1).numpy())
        

    print("total_time:", total_time)   

    relative_error_all = np.concatenate(relative_error_list)
    
    error = np.mean(relative_error_all)   
    
    print("Validation: {}".format(error))
    return 

  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    #parser.add_argument('--dataset_batch', default="/home/qym/datasets/single_queries/")
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_network(cfg))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    with torch.no_grad():
        validate(model.module, cfg)