import numpy as np
import os
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import yaml

def load_audio(path, sr=16000):
    try:
        audio, sr = librosa.load(path, sr=sr, mono=False)
    except Exception as e:
        print('Librosa error')
        audio, sr = sf.read(path)
        if len(audio.shape) == 1:
            audio = np.expand_dims(audio, axis=0)
        
    return audio, sr

def load_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def qtile_normalize(y, q, eps=1e-8):
    return y / (eps + np.quantile(y,q=q))

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['loss']

def save_ckp(state,model_name,model_folder,text):
    if not os.path.exists(model_folder): 
        print("Creating checkpoint directory...")
        os.mkdir(model_folder)
    torch.save(state, "{}/model_{}_{}.pth".format(model_folder, model_name, text))

def diagonal_smoothing(matrix_batch, window_size):

    batch_size, seq_len, _ = matrix_batch.shape
    device = matrix_batch.device

    # Define the custom kernel for forward diagonal median filtering
    kernel_forward = torch.eye(window_size, window_size, device=device)
    kernel_forward /= window_size

    # Perform 2D convolution with the custom kernel
    smoothed_batch = F.conv2d(matrix_batch.view(batch_size, 1, seq_len, seq_len), kernel_forward.view(1, 1, window_size, window_size), padding=(window_size // 2, window_size // 2))

    return smoothed_batch.view(batch_size, seq_len, seq_len)

def flip(matrix_batch):
    flipped_batch = torch.flip(matrix_batch, [1, 2])
    return flipped_batch

def compute_smooth_ssm(emb_batch, thresh=None, L=5):
    ssm = torch.bmm(emb_batch.transpose(1,2), emb_batch)

    if thresh is not None:
        if thresh == 'median':
            thresh = torch.median(ssm)
        ssm[ssm < thresh] = 0.0

    # Forward-backward diagonal smoothing
    ssm_f = diagonal_smoothing(ssm, L)
    ssm_b = flip(diagonal_smoothing(flip(ssm_f), L))
    ssm = flip(ssm_b)

    return ssm