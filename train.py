import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


from util import *
from model.augment import Augment
from model.data import LIMEDataset, collate_fn
from model.embedding import EmbeddingNetwork, SEBasicBlock

root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

device = torch.device("cuda")

parser = argparse.ArgumentParser(description='LIME training')
parser.add_argument('--config', type=str, default='config0.yaml', 
                    help='path to config file')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--ckp', default='lime_config0_0', type=str,
                    help='checkpoint_name')

def train(cfg, train_loader, model, optimizer, augment=None):
    model.train()
    loss_epoch = 0
    for idx, (S, I1, I2) in enumerate(train_loader):
        S, I1, I2 = S.to(device), I1.to(device), I2.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            S = augment(S) if augment is not None else S
        output = model(S)
        emb_ssm = compute_smooth_ssm(output)
        assert emb_ssm.shape == I1.shape == I2.shape
        loss1 = F.mse_loss(emb_ssm, I1)
        loss2 = F.mse_loss(emb_ssm, I2)
        loss = loss1 + cfg['gamma'] * loss2

        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(train_loader)}]\t Loss1: {loss1.item()}\t Loss2: {loss2.item()}")

        loss_epoch += loss.item()


    return loss_epoch / len(train_loader)


def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg['seed'])
    writer = SummaryWriter(f'runs/{args.ckp}')
    np.random.seed(cfg['seed'])

    # Hyperparameters
    batch_size = cfg['batch_size']
    lr = cfg['lr']
    num_epochs = cfg['n_epochs']
    model_name = args.ckp

    gpu_augment = Augment(cfg, gpu=True)

    print("Loading dataset...")
    if not os.path.exists(cfg['metadata_path']):
        raise FileNotFoundError(f"Metadata file not found at {cfg['metadata_path']}")
    
    train_dataset = LIMEDataset(cfg, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=cfg['num_workers'],
                              shuffle=False, collate_fn=collate_fn)
    
    print("Loading model...")
    model = EmbeddingNetwork(SEBasicBlock, [1,2,1,2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg['T_max'], eta_min = 1e-7)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model, optimizer, scheduler, start_epoch, loss_log = load_ckp(args.resume, model, optimizer, scheduler)
        else:
            start_epoch = 0
            loss_log = []
    
    print("Training...")
    best_loss = float('inf')
    for epoch in range(start_epoch+1, num_epochs+1):
        print("#######Epoch {}#######".format(epoch))
        loss_epoch = train(cfg, train_loader, model, optimizer, augment=gpu_augment)
        writer.add_scalar("Loss/train", loss_epoch, epoch)
        loss_log.append(loss_epoch)

        if loss_epoch < best_loss:
            best_loss = loss_epoch
            checkpoint = {
                'epoch': epoch,
                'loss': loss_log,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            save_ckp(checkpoint,epoch, model_name, model_folder)
            scheduler.step()
    

