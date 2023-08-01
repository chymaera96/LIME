import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import TimeMasking, FrequencyMasking

class Augment(nn.Module):
    def __init__(self, cfg, gpu=True):
        super().__init__()
        self.cfg = cfg
        self.time_mask = TimeMasking(cfg['time_mask'])
        self.freq_mask = FrequencyMasking(cfg['freq_mask'])
        self.gpu = gpu
        if not self.cfg['noise_std']:
            self.gaussian_noise = lambda x: x

    def gaussian_noise(self, x):
        return x + torch.randn(x.size()) * self.cfg['noise_std']

    def forward(self, x):
        if self.gpu:
            x = self.time_mask(x)
            x = self.freq_mask(x)
        else:
            raise NotImplementedError

        return x