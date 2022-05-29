from typing import Optional
import argparse
import pathlib
import pickle
import random

import torch
from transformers import GPT2Tokenizer

from load_aokvqa import load_aokvqa
from ClipCap.train_clipcap import ClipCocoDataset


def load_data(cfg: argparse.Namespace, split: str, eval: bool = False):
    features = vars(cfg).get(f"{split}_features", None)
    return AokvqaDataset(
        cfg.aokvqa_dir, split, features,
        cfg.prompt_with_choices, cfg.generation_target,
        cfg.prefix_length, normalize_prefix=cfg.normalize_prefix,
        gpt2_type='gpt2',
        eval=eval
    )


class AokvqaDataset(ClipCocoDataset):
    def __init__(self,
        aokvqa_dir: pathlib.Path, split: str, features: Optional[pathlib.Path],
        prompt_with_choices: bool, generation_target: str,
        prefix_length: int, normalize_prefix: bool, gpt2_type: str,
        eval: bool = False
    ):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.eval = eval

        aokvqa_set = load_aokvqa(aokvqa_dir, split)
        if features is not None:
            embeddings = torch.load(features)
            self.prefixes = []

        self.question_ids = []
        self.prompt_tokens = []

        if self.eval is False:
            self.target_tokens = []
            self.seq_lengths = []

        for d in aokvqa_set:
            q = d['question_id']

            if features is not None:
                self.prefixes.append( embeddings[q]['image'] )

            self.question_ids.append( q )

            self.prompt_tokens.append(
                torch.tensor(
                    self.tokenizer(
                        prompt_text(generation_target, d, include_choices=prompt_with_choices)
                    )['input_ids'], dtype=torch.int64
                )
            )

            if self.eval is False:
                self.target_tokens.append([
                    torch.tensor(
                        self.tokenizer(t)['input_ids'],
                        dtype=torch.int64
                    ) for t in target_texts(generation_target, d)
                ])

                self.seq_lengths += [
                    self.prompt_tokens[-1].shape[0] + t.shape[0]
                    for t in self.target_tokens[-1]
                ]

        if self.eval is False:
            self.max_seq_len = max(self.seq_lengths)

    def __getitem__(self, i: int):
        prefix = self.prefixes[i]
        if self.normalize_prefix:
            prefix = prefix / prefix.norm(2, -1)

        prompt = self.prompt_tokens[i]
        prompt_len = prompt.shape[0]

        if self.eval:
            return prefix, prompt, prompt_len

        target = random.sample(self.target_tokens[i], 1)[0]
        target_len = target.shape[0]

        eos_token = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int64)

        padding = self.max_seq_len - (prompt.shape[0] + target.shape[0])
        padding = torch.zeros(padding, dtype=torch.int64)

        input_tokens = torch.cat([prompt, target, eos_token, padding])

        return prefix, input_tokens, prompt_len, target_len

    def __len__(self) -> int:
        return len(self.prefixes)


def prompt_text(generation_target, d, include_choices=False):
    return f"question: {d['question']} " + \
           (f"choices: {', '.join(d['choices'])}. " if include_choices else '') + \
           {'answer' : 'answer:', 'rationale' : 'context:'}[generation_target]


def target_texts(generation_target, d):
    if generation_target == 'answer':
        targets = [d['choices'][d['correct_choice_idx']]]
    elif generation_target == 'rationale':
        targets = d['rationales']
    return [f" {t}" for t in targets]
