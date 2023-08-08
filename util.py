import numpy as np
import os
import torch
import librosa
import soundfile as sf
import yaml

def load_audio(path, sr=16000):
    try:
        audio, sr = librosa.load(path, sr=sr, mono=False)
    except Exception as e:
        print(e)
        audio, sr = sf.read(path)  
        print(audio.shape)
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