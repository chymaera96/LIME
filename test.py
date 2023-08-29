import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import json
import glob
from spleeter.separator import Separator

from model.embedding import EmbeddingNetwork, SEBasicBlock
from util import load_audio, load_config
from preprocess import extract_stems, compute_cqt_spectrogram

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser(description='LIME embedding extractor')
parser.add_argument('--config', type=str, default='dali.yaml', 
                    help='path to config file')
parser.add_argument('--emb_dir', type=str, default='data/embeddings', 
                    help='path to store embeddings')
parser.add_argument('--test_dir', type=str, default=None, 
                    help='test audio directory')

def embedding_extractor(cfg, audio, separator):
    stems = extract_stems(audio, separator=separator)
    cqt = compute_cqt_spectrogram(stems, cfg)
    cqt = torch.Tensor(cqt).unsqueeze(0)
    model = EmbeddingNetwork(cfg)
    model.eval()
    with torch.no_grad():
        output = model(cqt)
        
    return output.squeeze(0).cpu().numpy()

def main():
    args = parser.parse_args()
    cfg = load_config(args.config)

    separator = Separator('spleeter:4stems')

    for fpath in glob.glob('checkpoints/ *.pt'):
        print(f"Loading checkpoint {fpath} ...")
        model = EmbeddingNetwork(cfg, SEBasicBlock).to(device) 
        model.load_state_dict(torch.load(cfg['model_path'], map_location=device))

        for fpath in glob.glob(os.path.join(args.test_dir, '*.*')):
            if fpath.split('.')[-1] not in cfg['audio_exts']:
                continue
            print(f"Processing {fpath} ...")
            audio = load_audio(fpath, cfg)
            separator = Separator('spleeter:4stems')
            emb = embedding_extractor(cfg, audio, separator)
            fname = os.path.basename(fpath).split('.')[0]
            np.save(os.path.join(args.emb_dir, f"{fname}.npy"), emb)



