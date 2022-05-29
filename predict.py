import os
import random
import json
from tqdm import tqdm
import numpy as np
import argparse
import pathlib

import torch
import torch.nn.functional as F

from load_aokvqa import load_aokvqa
from evaluation.remap_predictions import map_to_choices
from ClipCap.data import prompt_text, load_data
from ClipCap.train import load_config, load_model


random.seed(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=pathlib.Path, required=True, dest='log_dir')
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--eval-features', type=pathlib.Path, required=True, dest='eval_features')
    parser.add_argument('--map-to-choices', action='store_true', dest='map_to_choices')
    parser.add_argument('--beam-search', action='store_true', dest='beam_search')
    parser.add_argument('--out', type=argparse.FileType('w'), required=True, dest='output_file')
    cfg = parser.parse_args()

    cfg = argparse.Namespace(
        **vars(cfg),
        **vars(load_config(os.path.join(cfg.log_dir, 'model_config.json'))),
        **{
            'finetune_gpt': False,
            f"{cfg.split}_features": cfg.eval_features
        }
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        cfg,
        os.path.join(cfg.log_dir, 'checkpoints', f"ckpt-{cfg.epoch:03d}.pt")
    )
    model = model.to(device)

    dataset = load_data(cfg, cfg.split, eval=True)
    tokenizer = dataset.tokenizer
    train_seq_lengths = np.array(load_data(cfg, 'train').seq_lengths)
    entry_length = int(train_seq_lengths.max() + 3 * train_seq_lengths.std())

    ## Run inference
    predictions = {}

    with torch.no_grad():
        for i in range(len(dataset)):
            q = dataset.question_ids[i]
            prefix, prompt_tokens, prompt_len = dataset[i]

            prefix = prefix.unsqueeze(0).to(device)
            prompt_tokens = prompt_tokens.unsqueeze(0).to(device)
            embedding_text = model.gpt.transformer.wte(prompt_tokens)
            prefix_projections = model.clip_project(prefix).view(-1, model.prefix_length, model.gpt_embedding_size)
            embed = torch.cat(( prefix_projections, embedding_text ), dim=1)

            if cfg.beam_search:
                generated_text = generate_beam(model, tokenizer, embed, device, beam_size=5, return_top_pred=True, entry_length=entry_length, stop_token_index=tokenizer.eos_token_id)
            else:
                generated_text = generate(model, tokenizer, embed=embed, entry_length=entry_length, stop_token_index=tokenizer.eos_token_id)

            predictions[q] = generated_text

    if cfg.map_to_choices:
        aokvqa_set = load_aokvqa(cfg.aokvqa_dir, cfg.split)
        aokvqa_set = { aokvqa_set[i]['question_id'] : aokvqa_set[i] for i in range(len(aokvqa_set)) }
        predictions = map_to_choices(aokvqa_set, predictions)

    json.dump(predictions, cfg.output_file)


def generate(model, tokenizer, embed, entry_length=67, top_p=0.8, temperature=1., stop_token_index: int = 50256):

    generated = embed
    tokens = None

    for _ in range(entry_length):

        outputs = model.gpt(inputs_embeds=generated)

        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = -float("Inf")

        next_token = torch.argmax(logits, -1).unsqueeze(0)

        if stop_token_index == next_token.item():
            break

        if tokens is None:
            tokens = next_token
        else:
            tokens = torch.cat((tokens, next_token), dim=1)

        next_token_embed = model.gpt.transformer.wte(next_token)
        generated = torch.cat((generated, next_token_embed), dim=1)

    if tokens is None:
        return ''

    output_list = list(tokens[0].cpu().numpy())
    output_text = tokenizer.decode(output_list).strip()
    return output_text


def generate_beam(model, tokenizer, embed, device, beam_size: int = 5, return_top_pred: bool = False, entry_length=67, temperature=1.0, stop_token_index: int = 50256):
    generated = embed
    tokens = None

    scores = None
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    for _ in range(entry_length):
        outputs = model.gpt(inputs_embeds=generated)

        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits = logits.softmax(-1).log()

        if scores is None:
            scores, next_tokens = logits.topk(beam_size, -1)
            generated = generated.expand(beam_size, *generated.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            if tokens is None:
                tokens = next_tokens
            else:
                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_tokens), dim=1)
        else:
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                beam_size, -1
            )
            next_tokens_source = next_tokens // scores_sum.shape[1]
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = next_tokens % scores_sum.shape[1]
            next_tokens = next_tokens.unsqueeze(1)
            tokens = tokens[next_tokens_source]
            tokens = torch.cat((tokens, next_tokens), dim=1)
            generated = generated[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]

        next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        generated = torch.cat((generated, next_token_embed), dim=1)
        is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
        if is_stopped.all():
            break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length) - 1]).strip()
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    if return_top_pred:
        return output_texts[0]
    return output_texts


if __name__ == '__main__':
    main()
