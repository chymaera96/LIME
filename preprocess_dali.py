import os
import crema
import torch
import DALI as dali_code
import pandas as pd
import librosa
import json
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/dali.yaml', help='configuration file')

from preprocess import extract_lyrics_vectors
from preprocess_crema import compute_crema_pcp
from util import load_audio, load_config

def ala_extractor(dali_data, annot_path, audio_length):
    id = annot_path.split('/')[-1].split('.')[0]
    entry = dali_data[id]
    df = pd.DataFrame.from_dict(entry.annotations['annot']['words'])
    df = df[['text','time']]
    time = pd.DataFrame(df['time'].to_list(), columns=['start','end'])
    ala = pd.concat([time, df['text']], axis = 1)
    try:
        lvecs = extract_lyrics_vectors(ala=ala, size=audio_length)
    except IndexError:
        print(f'audio_length: {audio_length}; id: {id}')
        return None
    return lvecs

    
def main():
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Preprocessing code
    if not os.path.exists(cfg['metadata_path']):
        metadata = []
    else:
        df = pd.read_csv(cfg['metadata_path'])
        metadata = df.to_dict('records')

    # Load DALI subset filenames
    with open(cfg['dali_subset_flist'], 'r') as fp:
        fnames = json.load(fp)

    print('Loading DALI annotations...')
    dali_data = dali_code.get_the_DALI_dataset(cfg['dali_annot_dir'], skip=[], keep=[])
    print('Creating CREMA model...')
    model = crema.models.chord.ChordModel()

    for ix, fname in enumerate(fnames):

        if ix % 10 == 0:
            print(f'Processing {ix} of {len(fnames)}...')
            if ix != 0:
                df = pd.DataFrame(metadata)
                df.to_csv(cfg['metadata_path'], index=False)

        audio_path = os.path.join(cfg['dali_audio_dir'], fname.split('.gz')[0] + '.mp3')
        annot_path = os.path.join(cfg['dali_annot_dir'], fname)

        if any([audio_path == m['audio_path'] for m in metadata]):
            continue

        try:
            audio, sr_h = load_audio(audio_path, sr=cfg['sr_h'])
            audio_length = audio.mean(axis=0).shape[0]/sr_h
            if audio_length > 300 or audio_length < 30:
                continue

        except Exception as e:
            print(e)
            continue

        lvecs = ala_extractor(dali_data, annot_path, audio_length)
        if lvecs is None:
            continue
        lvec_path = os.path.join(cfg['lvec_dir'], fname.split('.')[0] + '.npy')
        np.save(lvec_path, lvecs)

        crema_pcp = compute_crema_pcp(audio, sr_h, model=model)
        crema_path = os.path.join(cfg['crema_dir'], fname.split('.')[0] + '.npy')
        np.save(crema_path, crema_pcp)

        cqt_path = ''
        lyrics_path = ''
        pgram_path = ''

        metadata.append({
            'audio_path': audio_path,
            'audio_length': audio_length,
            'cqt_path': cqt_path,
            'crema_path': crema_path,
            'lvec_path': lvec_path,
            'lyrics_path': lyrics_path,
            'pgram_path': pgram_path
        })

if __name__ == '__main__':
    main()
