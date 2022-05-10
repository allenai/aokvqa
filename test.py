import os
import argparse
import pathlib

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ClipCap.model import load_model
from ClipCap.data import load_data
from ClipCap.train import load_config, loop


parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type=pathlib.Path, required=True, dest='log_dir')
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test_w_ans'], required=True)
parser.add_argument('--features', type=pathlib.Path, required=True, dest='features')
parser.add_argument('--bs', type=int, default=16)
cfg = parser.parse_args()

cfg = argparse.Namespace(
    **vars(cfg),
    **vars(load_config(os.path.join(cfg.log_dir, 'model_config.json'))),
    **{
        'finetune_gpt': False,
        f"{cfg.split}_features": cfg.features
    }
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(
    cfg,
    os.path.join(cfg.log_dir, 'checkpoints', f"ckpt-{cfg.epoch:03d}.pt")
)
model = model.to(device)

dataset = load_data(cfg, cfg.split)
dataloader = DataLoader(dataset, batch_size=cfg.bs, shuffle=False)

with torch.no_grad(), SummaryWriter(log_dir=cfg.log_dir) as writer:
    loop(cfg.split, cfg.epoch, model, dataloader, device, cfg.generation_target, cfg.prefix_length, tokenizer=dataset.tokenizer, train=False, tb_writer=writer)
