import os
import sys
import argparse
import pathlib
import json
from tqdm import tqdm
import collections

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from datasets import load_metric

from ClipCap.train_clipcap import ClipCocoDataset
from ClipCap.data import load_data
from ClipCap.model import load_model, ClipCapVQAModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log-dir', type=pathlib.Path, required=True, dest='log_dir')
    parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
    parser.add_argument('--train-features', type=pathlib.Path, required=True, dest='train_features')
    parser.add_argument('--val-features', type=pathlib.Path, required=True, dest='val_features')
    parser.add_argument('--pretrained-model', type=pathlib.Path, dest='pretrained_model')

    parser.add_argument('--prompt-with-choices', action='store_true', dest='prompt_with_choices')
    parser.add_argument('--generation-target', type=str, choices=['answer', 'rationale'], required=True, dest='generation_target')

    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)

    # Model hyperparams
    parser.add_argument('--mapping', type=str, choices=['mlp', 'transformer'], required=True, dest='mapping_type')
    parser.add_argument('--prefix-length', type=int, dest='prefix_length')
    parser.add_argument('--clip-model-type', type=str, choices=('RN50x4', 'ViT-B/32'), dest='clip_model_type')
    parser.add_argument('--normalize-prefix', type=bool, dest='normalize_prefix')
    parser.add_argument('--num-layers', type=int, dest='num_layers')

    parser.add_argument('--finetune-gpt', action='store_true', dest='finetune_gpt')

    cfg = parser.parse_args()

    if cfg.mapping_type == 'mlp':
        cfg.prefix_length = cfg.prefix_length or 10
        cfg.clip_model_type = cfg.clip_model_type or 'ViT-B/32'
        cfg.normalize_prefix = cfg.normalize_prefix or False
    elif cfg.mapping_type == 'transformer':
        cfg.prefix_length = cfg.prefix_length or 40
        cfg.clip_model_type = cfg.clip_model_type or 'RN50x4'
        cfg.normalize_prefix = cfg.normalize_prefix or True
    cfg.num_layers = 8

    # Load data and model

    dataset = load_data(cfg, 'train')
    val_dataset = load_data(cfg, 'val')
    model = load_model(cfg, cfg.pretrained_model)

    # Logging & run training/val loops

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.log_dir, 'checkpoints'), exist_ok=True)

    with SummaryWriter(log_dir=cfg.log_dir) as writer:
        run(dataset, val_dataset, model, cfg, tb_writer=writer)

## Saving and loading configs

def save_config(cfg: argparse.Namespace):
    with open(os.path.join(cfg.log_dir, "model_config.json"), 'w') as outfile:
        json.dump({
            'mapping_type' : cfg.mapping_type,
            'prefix_length' : cfg.prefix_length,
            'clip_model_type' : cfg.clip_model_type,
            'normalize_prefix' : cfg.normalize_prefix,
            'num_layers' : cfg.num_layers,
            'prompt_with_choices' : cfg.prompt_with_choices,
            'generation_target' : cfg.generation_target
        }, outfile)

def load_config(config_path: str):
    return argparse.Namespace(
        **json.load(open(config_path))
    )

## Training functions

def run(
    dataset: ClipCocoDataset, val_dataset: ClipCocoDataset, model: ClipCapVQAModel,
    cfg, tb_writer: SummaryWriter = None
):
    save_config(cfg)

    device = torch.device('cuda:0')
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    train_dataloader = DataLoader(dataset, batch_size=cfg.bs, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.bs, shuffle=False)

    # Compute train/val metrics before training (e.g. for pretrained weights)
    with torch.no_grad():
        model = loop('train', 0, model, train_dataloader, device, cfg.generation_target, cfg.prefix_length, tokenizer=dataset.tokenizer, train=False, tb_writer=tb_writer)
        model = loop('val', 0, model, val_dataloader, device, cfg.generation_target, cfg.prefix_length, tokenizer=val_dataset.tokenizer, train=False, tb_writer=tb_writer)

    # Training and validation loops
    for epoch in range(1, cfg.epochs + 1):

        model = loop(
            'train', epoch, model, train_dataloader, device,
            cfg.generation_target, cfg.prefix_length, tokenizer=dataset.tokenizer,
            train=True, optimizer=optimizer,
            tb_writer=tb_writer
        )

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(cfg.log_dir, 'checkpoints', f"ckpt-{epoch:03d}.pt"),
            )

        with torch.no_grad():
            model = loop(
                'val', epoch, model, val_dataloader, device,
                cfg.generation_target, cfg.prefix_length, tokenizer=val_dataset.tokenizer,
                train=False, tb_writer=tb_writer
            )

    return model


def loop(
    split, epoch,
    model, dataloader, device,
    generation_target, prefix_len, tokenizer=None,
    train=True, optimizer=None,
    tb_writer=None
):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    print(f">>> epoch {epoch}: {split} set " + ('(train)' if train else '(eval)'))
    sys.stdout.flush()

    if train:
        model.train()

    total_loss = 0.0

    if generation_target == 'rationale':
        metrics = { 'bleu' : load_metric('sacrebleu'),
                    'meteor' : load_metric('meteor') }
    elif generation_target == 'answer':
        metrics = { 'f1' : [],
                    'exact' : [] }

    for prefix, input_tokens, prompt_len, target_len in tqdm(dataloader):
        if train:
            model.zero_grad()

        prefix = prefix.to(device)
        input_tokens = input_tokens.to(device)
        prompt_len = prompt_len.to(device)
        target_len = target_len.to(device)

        loss = compute_step(model, prefix, input_tokens, prefix_len, prompt_len, target_len, metrics, tokenizer)

        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    if generation_target == 'rationale':
        scores = { 'bleu' : metrics['bleu'].compute()['score'],
                   'meteor' : metrics['meteor'].compute()['meteor'] }
    if generation_target == 'answer':
        scores = { 'f1' : sum(metrics['f1']) / len(metrics['f1']),
                   'exact' : sum(metrics['exact']) / len(metrics['exact']) }

    if tb_writer is not None:
        tb_writer.add_scalar(f'{split}/loss', total_loss / len(dataloader), epoch)
        for metric, score in scores.items():
            tb_writer.add_scalar(f'{split}/{metric}', score, epoch)

    return model


def compute_step( model, prefix, input_tokens,
                  prefix_len, prompt_len, target_len,
                  metrics=None, tokenizer=None ):

    outputs = model(prefix, input_tokens)

    ## Compute loss (comparing [target, eos] indices)

    target_logits = [
        l[s:e] for l, s, e in zip(
            outputs.logits,
            prefix_len + prompt_len - 1,
            prefix_len + prompt_len + target_len
        )
    ]

    target_tokens = [
        t[s:e] for t, s, e in zip(
            input_tokens,
            prompt_len,
            prompt_len + target_len + 1
        )
    ]

    loss = F.cross_entropy(
        torch.cat(target_logits),
        torch.cat(target_tokens)
    )

    ## Compute metrics (generated text vs target text)
    if metrics is not None:
        assert tokenizer is not None

        # All tokens after prompt
        generated_tokens = [
            list(l[s:-1].argmax(dim=1)) for l, s in zip(
                outputs.logits,
                prefix_len + prompt_len - 1
            )
        ]

        # Remove tokens at or after eos_token
        generated_tokens = [
            gen_t[:gen_t.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in gen_t
            else gen_t
            for gen_t in generated_tokens
        ]
        target_tokens = [tt[:-1] for tt in target_tokens]

        generated_text = [tokenizer.decode(gen_t) for gen_t in generated_tokens]
        target_text = [[tokenizer.decode(target_t)] for target_t in target_tokens]

        if 'bleu' in metrics:
            metrics['bleu'].add_batch(predictions=generated_text, references=target_text)
        if 'meteor' in metrics:
            metrics['meteor'].add_batch(predictions=generated_text, references=target_text)
        if 'f1' in metrics:
            for gen, target in zip(generated_text, target_text):
                metrics['f1'].append(compute_f1(target[0], gen))
        if 'exact' in metrics:
            for gen, target in zip(generated_text, target_text):
                metrics['exact'].append(compute_exact(target[0], gen))

    return loss

## From HuggingFaces Datasets (SquadV2 metric)

def compute_exact(a_gold, a_pred):
    return int(a_gold == a_pred)

def compute_f1(a_gold, a_pred):
    gold_toks = a_gold.split()
    pred_toks = a_pred.split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


if __name__ == '__main__':
    main()
