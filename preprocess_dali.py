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

def ala_extractor(dali_data, annot_path, audio_path):
    entry = dali_data[annot_path.split('.gz')[0]]
    df = pd.DataFrame.from_dict(entry.annotations['annot']['words'])
    df = df[['text','time']]
    time = pd.DataFrame(df['time'].to_list(), columns=['start','end'])
    ala = pd.concat([time, df['text']], axis = 1)
    # Extract length of audio
    audio_length = librosa.get_duration(filename=audio_path)
    if audio_length > 300 or audio_length < 30:
        return None
    lvecs = extract_lyrics_vectors(ala=ala, size=audio_length)
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
        fpaths = json.load(fp)

    print('Loading DALI annotations...')
    dali_data = dali_code.get_the_DALI_dataset(cfg['dali_annot_dir'], skip=[], keep=[])
    print('Creating CREMA model...')
    model = crema.models.chord.ChordModel()

    for ix, fpath in enumerate(os.listdir(fpaths)):

        if ix % 10 == 0:
            print(f'Processing {ix} of {len(fpaths)}...')
            if ix != 0:
                df = pd.DataFrame(metadata)
                df.to_csv(cfg['metadata_path'], index=False)

        if any([fpath == m['audio_path'] for m in metadata]):
            continue

        audio_path = os.path.join(cfg['dali_audio_dir'], fpath.split('.gz')[0] + '.mp3')
        annot_path = os.path.join(cfg['dali_subset_flist'], fpath)

        lvecs = ala_extractor(dali_data, annot_path, audio_path)
        if lvecs is None:
            continue
        lvec_path = os.path.join(cfg['lvec_dir'], fpath.split('/')[-1].split('.')[0] + '.pt')
        np.save(lvecs, lvec_path)

        try:
            audio, sr_h = load_audio(fpath, sr=cfg['sr_h'])
            audio_length = audio.mean(axis=0).shape[0]/sr_h

        except Exception as e:
            print(e)
            continue

        crema_pcp = compute_crema_pcp(audio, sr_h, model=model)
        crema_path = os.path.join(cfg['crema_dir'], fpath.split('/')[-1].split('.')[0] + '.pt')
        np.save(crema_pcp, crema_path)

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
