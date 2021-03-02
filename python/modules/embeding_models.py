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

class FastTextEmbeding():
    def __init__(self, bin_pth):
        self.model = fasttext.load_model(bin_pth)

    def seq_embeding(self, seq_words):
        if(type(seq_words) == float or seq_words is None):
            seq_words = "?"
        words = seq_words.split()
        vec_seq = []
        for w_ in words:
            vec_ = self.model.get_word_vector(w_)   
            vec_seq.append(vec_)
        return np.array(vec_seq)

    def avg_embeding(self, seq_words):
        vec_seq = self.seq_embeding(seq_words)
        vec_seq = np.array(vec_seq)
    
        len_ = len(vec_seq)
        sum_vec = np.zeros(np.shape(vec_seq)[1])
        for vec_ in vec_seq:
            sum_vec += vec_
        avg_vec =  sum_vec / len_
        return np.array(avg_vec)

    def sum_embeding(self, seq_words):
        vec_seq = self.avg_embeding(seq_words)
        vec_seq = np.array(vec_seq)
        len_ = len(vec_seq)
        sum_vec = np.zeros(np.shape(vec_seq)[1])
        for vec_ in vec_seq:
            sum_vec += vec_
        return np.array(sum_vec)

    def dataset_embeding(self, data_, embeding_style):
        data_embeding = []
        switch = {
            'avg':self.avg_embeding, 
            'max':self.sum_embeding,
            'seq':self.seq_embeding,
        }
        eb_model = switch.get(embeding_style)
        for i in data_:
           eb_l = eb_model(i[0])
           eb_r = eb_model(i[1])
           data_embeding.append([eb_l, eb_r])
        return np.array(data_embeding)
