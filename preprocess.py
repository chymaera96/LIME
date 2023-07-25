import numpy as np
import os
import torch
import argparse
import glob
import pandas as pd
import librosa
import crema
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import tensorflow as tf
import warnings

from util import load_audio, load_config
from LyricsAlignment.wrapper import extract_phonemegram, align

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/config0.yaml', help='configuration file')

def extract_stems(audio, separator=None):  
    # separator = Separator('spleeter:4stems')
    stems = separator.separate(audio.T)
    # print(stems['vocals'].shape)
    stems = {k: v.T.mean(axis=0) for k, v in stems.items()}
    return stems


def extract_lyrics_vectors(ala_file):
    df = pd.read_csv(ala_file, header=None)
    s_t = np.round(list(df[0]*2))
    e_t = np.round(list(df[1]*2))
    words = df[2]
    vocab = list(set(words))
    lvecs = np.zeros((len(vocab),int(e_t[-1])))
    inv_w_ix = {k:v for v, k in enumerate(vocab)}

    for i in range(len(df)):
        for j in range(int(s_t[i]),int(e_t[i]) + 1):
            lvecs[inv_w_ix[df[2][i]], j - 1] = 1
    return lvecs

def compute_cqt_spectrogram(stems, cfg):
    audio = np.concatenate([stems['vocals'], stems['drums'], stems['bass'], stems['other']])
    cqt = librosa.cqt(audio, sr=cfg['sr_h'], hop_length=cfg['hop_length'])
    return cqt

def main():
    args = parser.parse_args()
    cfg = load_config(args.config)


    # Preprocessing code
    columns = ['audio_path', 'audio_length', 'lyrics_path', 'cqt_path', 'crema_path', 'pgram_path']
    df = pd.read_csv(cfg['metadata_path'])
    fpaths = list(df['audio_path'])
    metadata = df.to_dict('records')
    separator = Separator('spleeter:4stems')

    for ix, fpath in enumerate(fpaths):

        if ix % 10 == 0:
            print(f'Processing {ix} of {len(fpaths)}...')
            if ix != 0:
                df = pd.DataFrame(metadata)
                df.to_csv(cfg['metadata_path'], index=False)

        if fpath.split('.')[-1] not in cfg['audio_exts']:
            continue
        # if any([fpath == m['audio_path'] for m in metadata]):
        #     continue

        try:
            audio, sr_h = load_audio(fpath, sr=cfg['sr_h'])
            audio_length = audio.mean(axis=0).shape[0]/sr_h
            if audio_length > 360:
                continue
        except Exception as e:
            print(e)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stems = extract_stems(audio, separator=separator)
        except Exception as e:
            print(e)
            continue
        
        pgram_path = os.path.join(cfg['pgram_dir'], fpath.split('/')[-1].split('.')[0] + '.pt')
        if os.path.exists(pgram_path):
            continue
        
        pgram = extract_phonemegram(stems['vocals'], method='MTL', cuda=False)
        torch.save(pgram, pgram_path)

        # Downsample stems to 16 kHz
        for k, v in stems.items():
            stems[k] = librosa.resample(v, orig_sr=sr_h, target_sr=cfg['sr_l'])
        cqt = compute_cqt_spectrogram(stems, cfg)
        cqt_path = os.path.join(cfg['cqt_dir'], fpath.split('/')[-1].split('.')[0] + '.pt')
        torch.save(cqt, cqt_path)

        # Add pgram and cqt paths to metadata based on audio path
        for m in metadata:
            if m['audio_path'] == fpath:
                m['cqt_path'] = cqt_path
                m['pgram_path'] = pgram_path



if __name__ == '__main__':
    main()
