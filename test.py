import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import argparse
import json
import glob
from spleeter.separator import Separator
import matplotlib.pyplot as plt
from mir_eval.segment import detection

from model.embedding import EmbeddingNetwork, SEBasicBlock
from util import load_audio, load_config, compute_smooth_ssm
from preprocess import extract_stems, compute_cqt_spectrogram
from chorusExtraction.thumbnail import FastThumbnail
from chorusExtraction.utils import compute_scape_plot, extract_chorus_segments

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser(description='LIME embedding extractor')
parser.add_argument('--config', type=str, default='config/dali.yaml', 
                    help='path to config file')
parser.add_argument('--emb_dir', type=str, default='data/embeddings', 
                    help='path to store embeddings')
parser.add_argument('--test_dir', type=str, default=None, 
                    help='test audio directory')
parser.add_argument('--scape_plot', type=bool, default=False, 
                    help='save example scape plot')
parser.add_argument('--preprocess', type=bool, default=False, 
                    help='preprocess test audio to cqt')

def create_test_metadata(cfg, test_dir, ground_truth_path):
    metadata = []
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    separator = Separator('spleeter:4stems')
    for fpath in glob.glob(os.path.join(test_dir, '*/*.*')):
        if fpath.split('.')[-1] not in cfg['audio_exts']:
            continue
        audio_id = fpath.split('/')[-2]
        if audio_id not in ground_truth.keys():
            continue
        audio, sr_l = load_audio(fpath, sr=cfg['sr_l'])
        print(audio.shape)
        stems = extract_stems(audio, separator=separator)
        cqt = compute_cqt_spectrogram(stems, cfg)
        cqt_fname = f"{audio_id}.npy"
        cqt_fpath = os.path.join('data/test', cqt_fname)
        np.save(cqt_fpath, cqt)
        metadata.append({'audio_id': audio_id, 'cqt_path': cqt_fpath, 'cqt_shape': cqt.shape})

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join('data/test/salami_test.csv'), index=False)
    return df


def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    ground_truth_path = 'data/ground_truth.json'
    print("Loading ground truth ...")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    if args.preprocess and len(os.listdir('data/test')) == 0:
        print("Creating test metadata ...")
        df = create_test_metadata(cfg, args.test_dir, ground_truth_path)
    else:
        print("Loading test metadata ...")
        df = pd.read_csv('data/test/salami_test.csv')
    score_ckp = {}
    for fpath in glob.glob('checkpoint/*.pth'):
        print(f"Loading checkpoint {fpath} ...")
        ckp_name = fpath.split('/')[-1].split('.')[0]
        model = EmbeddingNetwork(cfg, SEBasicBlock).to(device) 
        ckp = torch.load(fpath, map_location=device)
        model.load_state_dict(ckp['state_dict'])
        model.eval()
        scores_annot1 = []
        scores_annot2 = []
        for ix, row in df.iterrows():
            audio_id = str(row['audio_id'])
            if audio_id != '1620':
                continue
            cqt = np.load(row['cqt_path'])
            cqt = torch.Tensor(cqt).unsqueeze(0).to(device)
            emb = model(cqt)
            print(f"Processing audio {audio_id} ...")
            # torch.save(os.path.join(args.emb_dir, f"{audio_id}.pt"), emb)
            S = compute_smooth_ssm(emb).squeeze(0)
            S[S < torch.median(S)] = 0
            if ix == 0:
                ssm= S.detach().cpu().numpy()
                plt.imshow(ssm, cmap='gray_r', origin='lower')
                plt.savefig(f"plots/{audio_id}_{ckp_name}_ssm.png")
                plt.close()
            audThumb = FastThumbnail(cfg=cfg)
            cov, rep = audThumb(S)
            print(f"Ssm shape: {S.shape}")
            print(rep.shape)
            scape = compute_scape_plot(ssm, fitness=rep)
            if ix == 0 and args.scape_plot:
                plt.imshow(scape, cmap='hot', origin='lower')
                plt.savefig(f"plots/{audio_id}_{ckp_name}_scape.png")
                plt.close()

            est = extract_chorus_segments(ssm, scape)
            # Sorting estimates according to start time
            est = sorted(est, key=lambda x: x[0])
            ref1 = ground_truth[audio_id]["annot1"]
            ref2 = ground_truth[audio_id]["annot2"]

            # Computing F-measure for annot1
            est = np.array(est)
            ref1 = np.array(ref1)
            if not len(ref1) == 0:
                f1, p1, r1 = detection(ref1, est, window=3.0)
                scores_annot1.append([f1, p1, r1])

            # Computing F-measure for annot2
            ref2 = np.array(ref2)
            if not len(ref2) == 0:
                f2, p2, r2 = detection(ref2, est, window=3.0)
                scores_annot2.append([f2, p2, r2])

        scores_annot1 = np.array(scores_annot1)
        scores_annot2 = np.array(scores_annot2)
        print(f"Average F-measure for annot1: {np.mean(scores_annot1[:,0])}")
        print(f"Average F-measure for annot2: {np.mean(scores_annot2[:,0])}")
        print(f"Average Precision for annot1: {np.mean(scores_annot1[:,1])}")
        print(f"Average Precision for annot2: {np.mean(scores_annot2[:,1])}")
        print(f"Average Recall for annot1: {np.mean(scores_annot1[:,2])}")
        print(f"Average Recall for annot2: {np.mean(scores_annot2[:,2])}")
        score_ckp[ckp_name] = [np.mean(scores_annot1[:,1]), np.mean(scores_annot2[:,1])]


if __name__ == '__main__':
    main()

 



