import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fasttext
import torch.nn.functional as F
from torch.utils import data

class REEModel(nn.Module):
    def __init__(self):
        super(REEModel, self).__init__()

        self.rnn = nn.LSTM(     
            input_size=100,      # 图片每行的数据像素点
            hidden_size=64,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.hidden = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2)    # 输出层

    def forward(self, x_l, x_r):
        r_out_l, (h_n_l, h_c_l) = self.rnn(x_l, None)  
        r_out_r, (h_n_r, h_c_r) = self.rnn(x_r, None)   

        out_l = (r_out_l[:, -1, :])
        out_r = (r_out_r[:, -1, :])

        dis = torch.sub(out_l, out_r)
        sim_rep = dis.pow(2)
        sim_rep = self.hidden(sim_rep)
        out = self.out(sim_rep)
        return out
    @staticmethod
    def euclidean_distance(l, r):
        dis = torch.sub(l, r)
        dis = dis.pow(2)
        return dis
        
    @staticmethod
    def cos_distance(l,r):
        l = F.normalize(l, dim=-1)
        r = F.normalize(r, dim=-1)
        cose = torch.mm(l,r)
        return 1 - cose
    @staticmethod
    def element_wise(l,r):            
        pass
