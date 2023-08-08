import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import time

from filters import generate_filter, preprocess_v1

class FastThumbnail(nn.Module):
    def __init__(self, dims, filter_sd=0.1, n_layers=3, pad_param=0.1, n_channels=10):
        super().__init__()
        self.low = dims[0]
        self.high = dims[1]
        self.filter_sd = filter_sd
        self.n_layers = n_layers
        self.pad_param = pad_param
        self.n_channels = n_channels

        filters = self.create_filterbank()
        self.filterbank = nn.ParameterList(filters)

    def create_filterbank(self):
        filter_list = []
        for ix in range(self.low, self.high + 1, self.n_layers):
            filter_list.append(nn.Parameter(generate_filter(ix,self.filter_sd)))

        return filter_list

    def compute_pooled_features(self, S):
        pooled_features = []
        for ix in range(self.n_layers):
            if ix==0:
                sub_S = S
            else:
                sub_S = F.avg_pool2d(S, kernel_size=ix+1, stride=1)
                
            print(sub_S.shape)
            pooled_features.append(nn.Parameter(sub_S))
        
        return nn.ParameterList(pooled_features)

    def compute_coverage(self, conv_out, idxs):
        coverage = torch.stack([conv_out[idxs[i],:,i] for i in range(len(idxs))])
        cov_clean = coverage
        cov_clean[cov_clean <= torch.median(coverage)] = 0.0
        cov_clean /= (2**0.5)
        cov_scores = torch.sum(cov_clean,dim=1)/((1+self.pad_param)*cov_clean.shape[-1])
        return cov_scores


    def forward(self, S):
        t1 = time.time()
        S = preprocess_v1(S, pad_param=self.pad_param, n_channels=self.n_channels)
        t2 = time.time()
        print(f'Preprocess time: {t2 - t1}')
        # assert self.high < S.shape[-1]
        input_features = self.compute_pooled_features(S)
        cov_scores = []
        rep_scores = []

        with torch.no_grad():
            for filter in tqdm(self.filterbank):
                kernel = filter.view(1, 1, filter.shape[0], -1).repeat(self.n_channels, 1, 1, 1)
                for feature in input_features:
                    output = F.conv2d(feature, kernel, stride=(kernel.shape[-1],1), groups=10).squeeze(0)
                    channel_score = output.sum(dim=-2)
                    rep = torch.max(channel_score,dim=-2).values
                    max_idxs = torch.max(channel_score,dim=-2).indices
                    cov = self.compute_coverage(conv_out=output, idxs=max_idxs)
                    cov_scores.append(cov)
                    rep_scores.append(rep)        

        return cov_scores, rep_scores  