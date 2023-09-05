import torch
import numpy as np
import math

def generate_filter(N, c=5):

    filt = torch.zeros((N,N))
    for x in range(N):
        for y in range(N):
            if abs(x - y) <= c:
                filt[x,y] = 1.0
  
    return filt

def preprocess(S, pad_param=0.1, n_channels=10):

    # Mask leading diagonal
    mask = np.round(1 - generate_filter(S.shape[0]).to('cpu'))
    S_clean = np.multiply(S,mask)
    
    # Insert random empty rows and stack resultant SSM into channels
    pad = np.zeros(S_clean.shape[0])
    ssm_list = []
    for _ in range(n_channels):
        pad_idxs = np.random.randint(0, S_clean.shape[0]-1, size=int(pad_param*S_clean.shape[0]))
        S_pad = np.insert(S_clean, pad_idxs, pad, axis=0)
        ssm_list.append(S_pad)

    feature_tensor = torch.from_numpy(np.stack(ssm_list,axis=0)).unsqueeze(0).to(torch.float32)

    return feature_tensor


