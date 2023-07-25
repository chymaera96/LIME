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

from util import load_audio, load_config
from LyricsAlignment.wrapper import extract_phonemegram, align

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/config0.yaml', help='configuration file')

def compute_crema_pcp(audio, sr, model=None, feature_rate=2):
    print('Computing crema_pcp...')
    print(f'audio shape: {audio.shape}')
    out = model.outputs(y=audio.mean(axis=0),sr=sr)
    pcp = out['chord_pitch'].T + out['chord_root'].T[:-1] + out['chord_bass'].T[:-1]
    # tf.keras.backend.clear_session()

    crema_pcp = 1/(1 + np.exp(-pcp))
    fr = crema_pcp.shape[1]/len(audio)*sr
    crema_rs = librosa.resample(crema_pcp, orig_sr=fr, target_sr=feature_rate)

    return crema_rs

def extract_stems(audio):  
    separator = Separator('spleeter:4stems')
    stems = separator.separate(audio.T)
    print(stems['vocals'].shape)
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
    tf.compat.v1.enable_eager_execution()

    # Loading paths from artist text file
    df = pd.read_csv(cfg['artists'], sep='\t', header=None)
    artists = list(df[0])
    fpaths = []
    print('Loading audio paths...')
    for name in artists:
        artists_dir = os.path.join(cfg['audio_dir'], name)
        if not os.path.exists(artists_dir):
            print(f'{name} does not exist in {cfg["audio_dir"]}')
            continue
        path_list = glob.glob(os.path.join(artists_dir, '**/*.*'))
        fpaths.extend(path_list)

    print(f'{len(fpaths)} paths loaded!')

    # Preprocessing code
    columns = ['audio_path', 'audio_length', 'lyrics_path', 'cqt_path', 'crema_path', 'pgram_path']
    if not os.path.exists(cfg['metadata_path']):
        metadata = []
    else:
        df = pd.read_csv(cfg['metadata_path'])
        metadata = df.to_dict('records')


    model = crema.models.chord.ChordModel()

    for ix, fpath in enumerate(fpaths):

        if ix % 10 == 0:
            print(f'Processing {ix} of {len(fpaths)}...')

        if fpath.split('.')[-1] not in cfg['audio_exts']:
            continue
        if any([fpath == m['audio_path'] for m in metadata]):
            continue

        try:
            audio, sr_h = load_audio(fpath, sr=cfg['sr_h'])
            audio_length = len(audio)/sr_h
        except Exception as e:
            print(e)
            continue

        crema_pcp = compute_crema_pcp(audio, sr_h, model=model)
        break
        crema_path = os.path.join(cfg['crema_dir'], fpath.split('/')[-1].split('.')[0] + '.pt')
        torch.save(crema_pcp, crema_path)

        try:
            stems = extract_stems(audio)

        except Exception as e:
            print(e)
            continue
        
        # print(stems['vocals'].shape)
        pgram = extract_phonemegram(stems['vocals'], method='MTL', cuda=False)
        pgram_path = os.path.join(cfg['pgram_dir'], fpath.split('/')[-1].split('.')[0] + '.pt')
        torch.save(pgram, pgram_path)

        # Downsample stems to 16 kHz
        for k, v in stems.items():
            stems[k] = librosa.resample(v, orig_sr=sr_h, target_sr=cfg['sr_l'])
        cqt = compute_cqt_spectrogram(stems, cfg)
        cqt_path = os.path.join(cfg['cqt_dir'], fpath.split('/')[-1].split('.')[0] + '.pt')
        torch.save(cqt, cqt_path)

        # Not storing lyrics as of now
        lyrics_path = np.nan

        metadata.append(dict(zip(columns, [fpath, audio_length, lyrics_path, cqt_path, crema_path, pgram_path])))

        # Checking if the metadata is being saved correctly
        df = pd.DataFrame(metadata)
        df.to_csv(cfg['metadata_path'], index=False)

if __name__ == '__main__':
    main()
