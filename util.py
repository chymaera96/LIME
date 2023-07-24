import librosa
import soundfile as sf
import yaml

def load_audio(path, sr=16000):
    try:
        audio, sr = librosa.load(path, sr=sr, mono=False)
    except Exception as e:
        audio, sr = sf.read(path)    
    return audio, sr

def load_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config
