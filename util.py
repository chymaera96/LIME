import librosa
import soundfile as sf

def load_audio(path, sr=16000):
    try:
        audio, sr = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        audio, sr = sf.read(path)    
    return audio, sr

        
