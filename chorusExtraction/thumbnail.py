import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import time

from .filters import generate_filter, preprocess

class FastThumbnail(nn.Module):
    def __init__(self, cfg, dims=None, device='cuda'):
        super().__init__()
        if dims is None:
            dims = [1, cfg['max_dim']]
        self.low = dims[0]
        self.high = dims[1]
        self.pad_param = cfg['pad_param']
        self.n_channels = cfg['n_channels']
        self.device = device

        self.filterbank = self.create_filterbank(c=cfg['filter_width'])
        # self.filterbank = nn.ParameterList(filters)


    def create_filterbank(self, c):
        F = generate_filter(self.high, c)
        filter_list = []
        for ix in range(self.low, self.high):
            f = F.clone()
            k = (self.high - ix)//2
            f = f[k:k+ix, k:k+ix]
            filter_list.append(f)

        return filter_list

    def compute_coverage(self, conv_out, idxs):
        coverage = torch.stack([conv_out[idxs[i],:,i] for i in range(len(idxs))])
        cov_clean = coverage
        cov_clean[cov_clean <= torch.median(coverage)] = 0.0
        cov_clean /= (2**0.5)
        cov_scores = torch.sum(cov_clean,dim=1)/((1+self.pad_param)*cov_clean.shape[-1])
        return cov_scores


    def forward(self, S):
        S = preprocess(S, pad_param=self.pad_param, n_channels=self.n_channels)
        cov_scores = []
        rep_scores = []

        with torch.no_grad():
            for filter in tqdm(self.filterbank):
                print(f"Filter shape: {filter.shape}")
                if filter.shape[0] > S.shape[-1]:
                    break
                kernel = filter.view(1, 1, filter.shape[0], -1).repeat(self.n_channels, 1, 1, 1)
                kernel = kernel.to(self.device)
                S = S.to(self.device)
                output = F.conv2d(S, kernel, stride=(kernel.shape[-1],1), groups=self.n_channels).squeeze(0)
                channel_score = output.sum(dim=-2)
                rep = torch.max(channel_score,dim=-2).values
                max_idxs = torch.max(channel_score,dim=-2).indices
                cov = self.compute_coverage(conv_out=output, idxs=max_idxs)
                cov_scores.append(cov)
                rep_scores.append(rep)

        return cov_scores, rep_scores
