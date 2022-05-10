import os
import random
import json
from tqdm import tqdm
import numpy as np
import argparse
import pathlib

import torch
import torch.nn.functional as F

from ClipCap.data import load_data
from ClipCap.train import load_model
from ClipCap.predict import generate, generate_beam


random.seed(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=pathlib.Path, required=True, dest='checkpoint_path')

    parser.add_argument('--mapping', type=str, choices=['mlp', 'transformer'], required=True, dest='mapping_type')
    parser.add_argument('--prefix-length', type=int, dest='prefix_length')
    parser.add_argument('--clip-model-type', type=str, choices=('RN50x4', 'ViT-B/32'), dest='clip_model_type')
    parser.add_argument('--normalize-prefix', type=bool, dest='normalize_prefix')
    parser.add_argument('--num-layers', type=int, dest='num_layers')

    parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--eval-features', type=pathlib.Path, required=True, dest='eval_features')
    parser.add_argument('--beam-search', action='store_true', dest='beam_search')
    parser.add_argument('--out', type=argparse.FileType('w'), required=True, dest='output_file')
    cfg = parser.parse_args()

    if cfg.mapping_type == 'mlp':
        cfg.prefix_length = 10
        cfg.clip_model_type = 'ViT-B/32'
        cfg.normalize_prefix = False
    elif cfg.mapping_type == 'transformer':
        cfg.prefix_length = 40
        cfg.clip_model_type = 'RN50x4'
        cfg.normalize_prefix = True
    cfg.num_layers = 8

    cfg = argparse.Namespace(
        **vars(cfg),
        **{
            'prompt_with_choices': False,
            'generation_target': 'answer',
            'finetune_gpt': False,
            f"{cfg.split}_features": cfg.eval_features
        }
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(cfg, cfg.checkpoint_path)
    model = model.to(device)

    dataset = load_data(cfg, cfg.split, eval=True)
    tokenizer = dataset.tokenizer

    ## Run inference
    predictions = {}

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            q = dataset.question_ids[i]
            prefix = dataset[i][0].unsqueeze(0).to(device)
            embed = model.clip_project(prefix).view(-1, model.prefix_length, model.gpt_embedding_size)

            if cfg.beam_search:
                generated_text = generate_beam(model, tokenizer, embed, device, beam_size=5, return_top_pred=True, stop_token_index=tokenizer.eos_token_id)
            else:
                generated_text = generate(model, tokenizer, embed=embed, stop_token_index=tokenizer.eos_token_id)

            predictions[q] = generated_text

    json.dump(predictions, cfg.output_file)


if __name__ == '__main__':
    main()
