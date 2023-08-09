import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import pandas as pd

from util import *
from preprocess import extract_stems, compute_cqt_spectrogram
from chorusExtraction.utils import compute_sm_ti

def pad_ssm(tensor_list):
    max_size = max(tensor.size(0) for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_size = max_size - tensor.size(0)
        padded_tensor = F.pad(tensor, (0, pad_size, 0, pad_size))
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors, dim=0)

def collate_fn(batch):
    """
    batch = [dataset[i] for i in N]
    """
    size = len(batch[0])
    if size == 3:
        S, I1, I2 = zip(*batch)
        I1 = pad_ssm(I2)
        I2 = pad_ssm(I2)
    else:
        S = zip(*batch)
  
    S = pad_sequence(S, batch_first=True).permute(0,2,3,1)

    if size == 3:
        return S, I1, I2
    else:
        return S

class LIMEDataset(Dataset):
    def __init__(self, cfg, transform=None, train=True):
        self.metadata = pd.read_csv(cfg['metadata_path'])
        self.transform = transform
        self.ignore_index = []
        self.crema_threshold = cfg['crema_threshold']
        self.lyr_enc_threshold = cfg['lyr_enc_threshold']
        self.norm = cfg['norm']
        self.train = train

    def __getitem__(self, idx):
        if idx in self.ignore_index:
            return self[idx + 1]

        row = self.metadata.iloc[idx] 
        try:
            if self.train:
                cqt = np.abs(np.load(row['cqt_path']))
                cqt = qtile_normalize(cqt, self.norm)
                lyr_enc = np.load(row['lyr_enc_path'])
                crema_pcp = np.load(row['crema_path'])
            else:
                raise NotImplementedError
            
        except Exception:
            self.ignore_index.append(idx)
            return self[idx + 1]
        
        
        if self.transform is not None:
            cqt = self.transform(cqt)

        # Checking number of frames in crema_pcp and lyr_enc
        if crema_pcp.shape[1] != lyr_enc.shape[1]:
            print(f"crema_pcp.shape[1] - lyr_enc.shape[1] = {crema_pcp.shape[1] - lyr_enc.shape[1]}")
            if crema_pcp.shape[1] < lyr_enc.shape[1]:
                crop = (lyr_enc.shape[1] - crema_pcp.shape[1]) // 2
                lyr_enc = lyr_enc[:, crop:crop+crema_pcp.shape[1]]

            else:
                pad = (crema_pcp.shape[1] - lyr_enc.shape[1]) // 2
                lyr_enc = np.pad(lyr_enc, ((0,0), (pad, pad)), mode='constant', constant_values=0)

        crema_SSM = compute_sm_ti(crema_pcp)
        if self.crema_threshold is not None:
            crema_SSM[crema_SSM < self.crema_threshold] = 0
        else:
            crema_SSM[crema_SSM < np.median(crema_SSM)] = 0

        lyr_SSM = compute_sm_ti(lyr_enc)
        lyr_SSM[lyr_SSM < self.lyr_enc_threshold] = 0
        
        
        cqt = torch.from_numpy(cqt).permute(2,1,0).to(torch.float32) # Shape compatible with the collate_fn
        crema_SSM = torch.from_numpy(crema_SSM).to(torch.float32)
        lyr_SSM = torch.from_numpy(lyr_SSM).to(torch.float32)

        return cqt, lyr_SSM, crema_SSM


        