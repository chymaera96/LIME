import numpy as np
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