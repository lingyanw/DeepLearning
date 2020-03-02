from scipy import misc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib as mp
import torch.optim as optim
import torch
from scipy.stats import pearsonr

class LeNet(nn.Module):
    def __init__(self, _n_neuron):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 9, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 9, padding=1)
        self.pool = nn.MaxPool2d(4, padding=1)
        self.fc1 = nn.Linear(7 * 14 * 64, 1024)
        self.fc2 = nn.Linear(1024, _n_neuron)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.conv1(x))
        self.conv1_act = x.detach().clone().cpu().numpy()
        x = self.pool(x)
        x = self.act(self.conv2(x))
        self.conv2_act = x.detach().clone().cpu().numpy()
        x = self.pool(x)
        x = x.view(-1, 7 * 14 * 64)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(directory='./', device='cuda'):
    
    n = 5000
    ntrain = 4900
    ntest = 100
    data = np.load(directory+'dataset.npy',allow_pickle = True).item()
    images = data['images']
    response = data['responses']
    perm = np.arange(n)
    np.random.shuffle(perm)
    
    Train = images[perm[0:ntrain],:,:,:]
    LTrain = response[perm[0:ntrain]][:,None]
            
    Test = images[perm[ntrain:5000],:,:,:]
    LTest = response[perm[ntrain:5000]][:,None]
    
    Train = torch.tensor(Train, device=device, dtype=torch.float32)
    LTrain = torch.tensor(LTrain, device=device, dtype=torch.float32)
    Test = torch.tensor(Test, device=device, dtype=torch.float32)
    LTest = torch.tensor(LTest, device=device, dtype=torch.float32)
    
    return Train, LTrain, Test, LTest

def corr(y1, y2, axis=-1, eps=1e-8, **kwargs):
    """
    Compute the correlation between two matrices along certain dimensions.

    Args:
        y1:      first matrix
        y2:      second matrix
        axis:    dimension along which the correlation is computed.
        eps:     offset to the standard deviation to make sure the correlation is well defined (default 1e-8)
        **kwargs passed to final `mean` of standardized y1 * y2

    Returns: correlation vector

    """
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=1) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=1) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)

def accuracy(net, imgs, labels, batchsize = 100):  # need change 
    with torch.no_grad():
        net.eval()
        ys = net(imgs).cpu().numpy()
        net.train()
    targets = labels.cpu().numpy()
    return corr(ys, targets, axis=0).mean()

def add_all(writer, var, var_name, iter_n):  # no need to change 
    writer.add_scalar(var_name + '_mean', var.mean(), iter_n)
    writer.add_scalar(var_name + '_max', var.max(), iter_n)
    writer.add_scalar(var_name + '_min', var.mean(), iter_n)
    writer.add_scalar(var_name + '_std', var.std(), iter_n)
    writer.add_histogram(var_name + '_hist', var, iter_n)
