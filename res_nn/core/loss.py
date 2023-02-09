import torch
import torch.nn as nn
import torch.nn.functional as F

def L1Loss(preds, gt):
    loss = (preds-gt).abs().mean()
    metrics = {
        'loss': loss.item(),
    }

    return loss, metrics

'''
def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)
'''

def qerrorLoss(preds, gt):
    qerror = []
    if (preds>gt).cpu().data.numpy()[0]:
        qerror.append(preds.abs() / (gt.abs()+0.000001))
    else:
        qerror.append(gt.abs() / (preds.abs()+0.000001))

    metrics = {
        'qerror': torch.mean(torch.cat(qerror)).item(),
    }
    return torch.mean(torch.cat(qerror)), metrics

'''
def qerrorLoss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
        
    metrics = {
        'q-error loss': qerror
    }
    return torch.mean(torch.cat(qerror))
'''