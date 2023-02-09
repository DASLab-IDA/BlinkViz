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
#from torchsummary import summary

# from FlowFormer import FlowFormer
from core.Models import build_network

@torch.no_grad()
def validate(model, cfg):
    model.eval()
    results = {}

    #val_dataset = datasets.SPNDataset(is_test=True, cfg=cfg)
    test_loader = datasets.fetch_test_dataloader(cfg)
    relative_error_list = []
    q_error_list = []

    _max = -9999
    _index = -99
    total_time = 0.0

    
    count_large = 0
    
    # 10%-20%
    sel10_20 = 0.0 # prediction error
    count10_20 = 0 # count query
    spn10_20 = 0.0 # mean of spn error
    spn_mean_10_20 = 0. # spn mean error

    # 1%-10%
    sel1_10 = 0.0
    count1_10 = 0
    spn1_10 = 0.0
    spn_mean_1_10 = 0.0

    # 0.1%-1%
    sel01_1 = 0.0
    count01_1 = 0
    spn01_1 = 0.0
    spn_mean_01_1 = 0.0

    # 0.01%-0.1%
    sel001_01 = 0.0
    count001_01 = 0.0
    spn001_01 = 0.0
    spn_mean_001_01 = 0.0

    # 0.001%-0.01%
    sel0001_001 = 0.0
    count0001_001 = 0
    spn0001_001 = 0.0
    spn_mean_0001_001 = 0.0

    # 0.0001%-0.001%
    sel00001_0001 = 0.0
    count00001_0001 = 0
    spn00001_0001 = 0.0
    spn_mean_00001_0001 = 0.0
    
    
    # 0.00001%-0.0001%
    sel000001_00001 = 0.0
    count000001_00001 = 0
    spn000001_00001 = 0.0
    spn_mean_000001_00001 = 0.0
    
    # minor
    sel_minor = 0.0
    count_minor = 0
    spn_minor = 0.0
    spn_mean_minor = 0.0



    #for val_id in range(len(val_dataset)):

    #count = 0
    for i_batch, data_blob in enumerate(test_loader):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",i_batch)
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
        
        '''
        if count<26:
            query2.append((preds, gt))
            q2 = q2+((preds.cpu()-gt) / gt).abs()
        elif count<(26+26):
            query3.append((preds, gt))
            q3 = q3+((preds.cpu()-gt) / gt).abs()
        elif count<(26+26+22):
            query4.append((preds, gt))
            q4 = q4+((preds.cpu()-gt) / gt).abs()
        elif count<(26+26+22+22):
            query5.append((preds, gt))
            q5 = q5+((preds.cpu()-gt) / gt).abs()
        elif count<(26+26+22+22+53):
            query6.append((preds, gt))
            q6 = q6+((preds.cpu()-gt) / gt).abs()
        elif count<(26+26+22+22+53+53):
            query8.append((preds, gt))
            q8 = q8+((preds.cpu()-gt) / gt).abs()
        elif count<(26+26+22+22+53+53+26):
            query9.append((preds, gt))
            q9 = q9+((preds.cpu()-gt) / gt).abs()
        elif count<(26+26+22+22+53+53+26+22):
            query10.append((preds, gt))
            q10 = q10+((preds.cpu()-gt) / gt).abs()
        elif count<(26+26+22+22+53+53+26+22+53):
            query11.append((preds, gt))
            q11 = q11+((preds.cpu()-gt) / gt).abs()
        
        count = count + 1
        '''
        """
        # 22 53 26 22
        if count<22:
            query5.append((preds, gt))
            q5 = q5+((preds.cpu()-gt) / gt).abs()
        elif count<(22+53):
            query6.append((preds, gt))
            q6 = q6+((preds.cpu()-gt) / gt).abs()
        elif count<(22+53+26):
            query7.append((preds, gt))
            q7 = q7+((preds.cpu()-gt) / gt).abs()
        elif count<(22+53+26+22):
            query10.append((preds, gt))
            q10 = q10+((preds.cpu()-gt) / gt).abs()
        """
        
        # res_gt = gt.abs()
        # res_pred = preds.cpu().abs()
        # if res_gt.item()!=0 and res_pred.item()!=0:
        #     q_error = max(res_gt.item(), res_pred.item()) / min(res_gt.item(), res_pred.item())
        #     q_error_list.append(q_error)        
        
        # count norm 1000-mean
        #print("spn_preds:", (spn_preds/1000.0+torch.tensor([0.00393749319269767,0.00393749319269767,0.00393749319269767]).to(cfg.device))*499921200)
        #print("preds:", (preds/1000.0+0.00393749319269767)*499921200)
        #print("gt:", (gt/1000.0+0.00393749319269767)*499921200)
    
        #gt = (gt/1000.0+0.00393749319269767)*499921200

        #relative_error = (((preds.cpu()/1000.0+0.00393749319269767)*499921200-(gt/1000.0+0.00393749319269767)*499921200) / ((gt/1000.0+0.00393749319269767)*499921200)).abs()
        #spn_error = (((((spn_preds[0][0].cpu()/1000.0+0.00393749319269767)*499921200-(gt/1000.0+0.00393749319269767)*499921200) / ((gt/1000.0+0.00393749319269767)*499921200)).abs()) + ((((spn_preds[0][1].cpu()/1000.0+0.00393749319269767)*499921200-(gt/1000.0+0.00393749319269767)*499921200) / ((gt/1000.0+0.00393749319269767)*499921200)).abs()) + ((((spn_preds[0][2].cpu()/1000.0+0.00393749319269767)*499921200-(gt/1000.0+0.00393749319269767)*499921200) / ((gt/1000.0+0.00393749319269767)*499921200)).abs())) / 3
        #spn_mean_error = ((((spn_preds[0][0].cpu()+spn_preds[0][1].cpu()+spn_preds[0][2].cpu())/3000.0+0.00393749319269767)*499921200-(gt/1000.0+0.00393749319269767)*499921200) / ((gt/1000.0+0.00393749319269767)*499921200)).abs()
        
        #res_gt = ((gt/1000.0+0.00393749319269767)*499921200).abs()
        #res_pred = ((preds.cpu()/1000.0+0.00393749319269767)*499921200).abs()
        #if res_gt.item()!=0 and res_pred.item()!=0:
        #    q_error = max(res_gt.item(), res_pred.item()) / min(res_gt.item(), res_pred.item())
        #    q_error_list.append(q_error)
        #print(val_id, relative_error)

        # count norm 1000 
        #print("spn_preds:", (spn_preds/1000.0*499921200))
        #print("preds:", (preds/1000.0)*499921200)
        #print("gt:", (gt/1000.0)*499921200)
        #gt = gt.cpu()
        #relative_error = (((preds.cpu()/1000.0)*499921200-(gt/1000.0)*499921200) / ((gt/1000.0)*499921200)).abs()
        #spn_mean_error = ((((spn_preds[0][0].cpu()+spn_preds[0][1].cpu()+spn_preds[0][2].cpu())/3000.0)*499921200-(gt/1000.0)*499921200) / ((gt/1000.0)*499921200)).abs()
        #spn_error = (((((spn_preds[0][0].cpu()/1000.0)*499921200-(gt/1000.0)*499921200) / ((gt/1000.0)*499921200)).abs()) + ((((spn_preds[0][1].cpu()/1000.0)*499921200-(gt/1000.0)*499921200) / ((gt/1000.0)*499921200)).abs()) + ((((spn_preds[0][2].cpu()/1000.0)*499921200-(gt/1000.0)*499921200) / ((gt/1000.0)*499921200)).abs())) / 3
        
        #gt = (gt/1000.0)*499921200

        #if relative_error.numpy()[0][0] > _max:
        #    _max = relative_error.numpy()[0][0]
        #    _index = val_id
        
        
        """
        if gt/499921200<0.0000001:
            sel_minor = sel_minor + relative_error
            count_minor = count_minor + 1
            spn_minor = spn_minor + spn_error
            spn_mean_minor = spn_mean_minor + spn_mean_error
        elif gt/499921200<0.000001:
            sel000001_00001 = sel000001_00001 + relative_error
            count000001_00001 = count000001_00001 + 1
            spn000001_00001 = spn000001_00001 + spn_error
            spn_mean_000001_00001 = spn_mean_000001_00001 + spn_mean_error
        elif gt/499921200<0.00001:
            sel00001_0001 = sel00001_0001 + relative_error
            count00001_0001 = count00001_0001 + 1
            spn00001_0001 = spn00001_0001 + spn_error
            spn_mean_00001_0001 = spn_mean_00001_0001 + spn_mean_error
        elif gt/499921200<0.0001:
            sel0001_001 = sel0001_001 + relative_error
            count0001_001 = count0001_001 + 1
            spn0001_001 = spn0001_001 + spn_error
            spn_mean_0001_001 = spn_mean_0001_001 + spn_mean_error
        elif gt/499921200<0.001:
            sel001_01 = sel001_01 + relative_error
            count001_01 = count001_01 + 1
            spn001_01 = spn001_01 + spn_error
            spn_mean_001_01 = spn_mean_001_01 + spn_mean_error
        elif gt/499921200<0.01:
            sel01_1 = sel01_1 + relative_error
            count01_1 = count01_1 + 1
            spn01_1 = spn01_1 + spn_error
            spn_mean_01_1 = spn_mean_01_1 + spn_mean_error
        elif gt/499921200<0.1:
            sel1_10 = sel1_10 + relative_error
            count1_10 = count1_10 + 1
            spn1_10 = spn1_10 + spn_error
            spn_mean_1_10 = spn_mean_1_10 + spn_mean_error
        elif gt/499921200<0.2:
            sel10_20 = sel10_20 + relative_error
            count10_20 = count10_20 + 1
            spn10_20 = spn10_20 + spn_error
            spn_mean_10_20 = spn_mean_10_20 + spn_mean_error
        else:
            count_large = count_large + 1
        """
        """
        if (gt/1000.0+0.00393749319269767)<0.0000001:
            sel_minor = sel_minor + relative_error
            count_minor = count_minor + 1
            spn_minor = spn_minor + spn_error
            spn_mean_minor = spn_mean_minor + spn_mean_error
        elif (gt/1000.0+0.00393749319269767)<0.000001:
            sel000001_00001 = sel000001_00001 + relative_error
            count000001_00001 = count000001_00001 + 1
            spn000001_00001 = spn000001_00001 + spn_error
            spn_mean_000001_00001 = spn_mean_000001_00001 + spn_mean_error
        elif (gt/1000.0+0.00393749319269767)<0.00001:
            sel00001_0001 = sel00001_0001 + relative_error
            count00001_0001 = count00001_0001 + 1
            spn00001_0001 = spn00001_0001 + spn_error
            spn_mean_00001_0001 = spn_mean_00001_0001 + spn_mean_error
        elif (gt/1000.0+0.00393749319269767)<0.0001:
            sel0001_001 = sel0001_001 + relative_error
            count0001_001 = count0001_001 + 1
            spn0001_001 = spn0001_001 + spn_error
            spn_mean_0001_001 = spn_mean_0001_001 + spn_mean_error
        elif (gt/1000.0+0.00393749319269767)<0.001:
            sel001_01 = sel001_01 + relative_error
            count001_01 = count001_01 + 1
            spn001_01 = spn001_01 + spn_error
            spn_mean_001_01 = spn_mean_001_01 + spn_mean_error
        elif (gt/1000.0+0.00393749319269767)<0.01:
            sel01_1 = sel01_1 + relative_error
            count01_1 = count01_1 + 1
            spn01_1 = spn01_1 + spn_error
            spn_mean_01_1 = spn_mean_01_1 + spn_mean_error
        elif (gt/1000.0+0.00393749319269767)<0.1:
            sel1_10 = sel1_10 + relative_error
            count1_10 = count1_10 + 1
            spn1_10 = spn1_10 + spn_error
            spn_mean_1_10 = spn_mean_1_10 + spn_mean_error
        elif (gt/1000.0+0.00393749319269767)<0.2:
            sel10_20 = sel10_20 + relative_error
            count10_20 = count10_20 + 1
            spn10_20 = spn10_20 + spn_error
            spn_mean_10_20 = spn_mean_10_20 + spn_mean_error
        else:
            count_large = count_large + 1
        """
        relative_error_list.append(relative_error.view(-1).numpy())
        

    print("total_time:", total_time)   




    relative_error_all = np.concatenate(relative_error_list)
    # print(relative_error_all.shape, _max, _index)
    # print("q_error:", statistics.mean(q_error_list))
    
    error = np.mean(relative_error_all)
    """
    print("sel minor:", sel_minor)
    print("spn mean:", spn_mean_minor)
    print("spn:", spn_minor)
    print(count_minor)
    print("sel 0.00001\%-0.0001\%:", sel000001_00001/count000001_00001)
    print("spn mean:", spn_mean_000001_00001/count000001_00001)
    print("spn:", spn000001_00001/count000001_00001)
    print(count000001_00001)
    print("sel 0.0001\%-0.001\%:", sel00001_0001/count00001_0001)
    print("spn mean:", spn_mean_00001_0001/count00001_0001)
    print("spn:", spn00001_0001/count00001_0001)
    print(count00001_0001)
    print("sel 0.001\%-0.01\%:", sel0001_001/count0001_001)
    print("spn mean:", spn_mean_0001_001/count0001_001)
    print("spn:", spn0001_001/count0001_001)
    print(count0001_001)
    print("sel 0.01\%-0.1\%:", sel001_01/count001_01)
    print("spn mean:", spn_mean_001_01/count001_01)
    print("spn:", spn001_01/count001_01)
    print(count001_01)
    print("sel 0.1\%-1\%:", sel01_1/count01_1)
    print("spn mean:", spn_mean_01_1/count01_1)
    print("spn:", spn01_1/count01_1)
    print(count01_1)
    print("sel 1\%-10\%:", sel1_10/count1_10)
    print("spn mean:", spn_mean_1_10/count1_10)
    print("spn:", spn1_10/count1_10)
    print(count1_10)
    print("sel 10\%-20\%:", sel10_20/count10_20)
    print("spn mean:", spn_mean_10_20/count10_20)
    print("spn:", spn10_20/count10_20)
    print(count10_20)
    print(count_large)
    """
    
    '''
    print("query2:", query2)
    print("q2:", q2/26)
    print("query3:", query3)
    print("q3:", q3/26)
    print("query4:", query4)
    print("q4:", q4/22)
    
    print("query5:", query5)
    print("q5:", q5/22)
    print("query6:", query6)
    print("q6:", q6/53)
    print("query7:", query7)
    print("q7:", q7/26)
    
    print("query8:", query8)
    print("q8:", q8/53)
    print("query9:", query9)
    print("q9:", q9/26)
    
    print("query10:", query10)
    print("q10:", q10/22)
    print("query11:", query11)
    print("q11:", q11/53)
    '''
    
    
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

    #print(model)
    #exit()
    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    with torch.no_grad():
        validate(model.module, cfg)
