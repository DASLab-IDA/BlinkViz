import torch
import torch.nn as nn
import torch.nn.functional as F

def L1Loss(preds, gt):
    loss = (preds-gt).abs().mean()
    metrics = {
        'loss': loss.item(),
    }

    return loss, metrics

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
