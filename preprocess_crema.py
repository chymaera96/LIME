import numpy as np
import os
import torch
import argparse
import glob
import pandas as pd
import librosa
import crema


from util import load_audio, load_config
from LyricsAlignment.wrapper import extract_phonemegram, align

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/config0.yaml', help='configuration file')

def compute_crema_pcp(audio, sr, model=None, feature_rate=2):
    out = model.outputs(y=audio.mean(axis=0),sr=sr)
    pcp = out['chord_pitch'].T + out['chord_root'].T[:-1] + out['chord_bass'].T[:-1]
    crema_pcp = 1/(1 + np.exp(-pcp))
    fr = crema_pcp.shape[1]/len(audio)*sr
    crema_rs = librosa.resample(crema_pcp, orig_sr=fr, target_sr=feature_rate)

    return crema_rs


def main():
    args = parser.parse_args()
    cfg = load_config(args.config)

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
    columns = ['audio_path', 'audio_length', 'cqt_path', 'crema_path', 'lvec_path', 'lyrics_path', 'pgram_path']
    if not os.path.exists(cfg['metadata_path']):
        metadata = []
    else:
        df = pd.read_csv(cfg['metadata_path'])
        metadata = df.to_dict('records')


    model = crema.models.chord.ChordModel()

    for ix, fpath in enumerate(fpaths):

        if ix % 10 == 0:
            print(f'Processing {ix} of {len(fpaths)}...')
            if ix != 0:
                df = pd.DataFrame(metadata)
                df.to_csv(cfg['metadata_path'], index=False)

        if fpath.split('.')[-1] not in cfg['audio_exts']:
            continue
        if any([fpath == m['audio_path'] for m in metadata]):
            continue

        try:
            audio, sr_h = load_audio(fpath, sr=cfg['sr_h'])
            audio = audio.mean(axis=0)
            audio_length = audio.shape[0]/sr_h
            if audio_length > 360:
                continue
            # print(f"Audio length: {audio_length}")
        except Exception as e:
            print(e)
            continue
        crema_pcp = compute_crema_pcp(audio, sr_h, model=model)
        crema_path = os.path.join(cfg['crema_dir'], fpath.split('/')[-1].split('.')[0] + '.pt')
        np.save(crema_pcp, crema_path)

        pgram_path = ''
        cqt_path = ''
        lyrics_path = ''
        lvec_path = ''

        metadata.append({
            'fpath': fpath,
            'audio_length': audio_length,
            'cqt_path': cqt_path,
            'crema_path': crema_path,
            'lvec_path': lvec_path,
            'lyrics_path': lyrics_path,
            'pgram_path': pgram_path
        })


if __name__ == '__main__':
    main()
