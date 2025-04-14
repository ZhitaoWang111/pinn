import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from utils.util import AverageMeter,get_logger,eval_metrix
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Test(model,testloader):
    model.eval()
    label = []
    pred = []
    with torch.no_grad():
        for iter,(x1,_,y1,_) in enumerate(testloader):
            x1 = x1.to(device)
            u1 = model.predict(x1)
            label.append(y1)
            pred.append(u1.cpu().detach().numpy())
    pred = np.concatenate(pred,axis=0)
    label = np.concatenate(label,axis=0)
    label[:,0] = label[:,0]*1.19
    label[:,1] = label[:,1]*3000
    pred[:,0] = pred[:,0]*1.19
    pred[:,1] = pred[:,1]*3000
    

    return label,pred