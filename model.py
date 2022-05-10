import argparse

import torch

from ClipCap.train_clipcap import MappingType, ClipCaptionModel


def load_model(cfg: argparse.Namespace, model_path: str):
    prefix_dim = {'ViT-B/32' : 512, 'RN50x4' : 640}[cfg.clip_model_type]
    mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[cfg.mapping_type]

    model = ClipCapVQAModel(
        cfg.prefix_length, prefix_size=prefix_dim,
        num_layers=cfg.num_layers, mapping_type=mapping_type, finetune_gpt=cfg.finetune_gpt
    )

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model


class ClipCapVQAModel(ClipCaptionModel):

    def __init__(self, prefix_length: int, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, finetune_gpt: bool = False):
        super().__init__( prefix_length, clip_length=prefix_length, prefix_size=prefix_size,
                          num_layers=num_layers, mapping_type=mapping_type )
        self.finetune_gpt = finetune_gpt

    def forward(self, prefix: torch.Tensor, tokens: torch.Tensor):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        e = torch.cat(( prefix_projections, embedding_text ), dim=1)
        out = self.gpt(inputs_embeds=e)
        return out

    def parameters(self, recurse: bool = True):
        if self.finetune_gpt is False:
            return self.clip_project.parameters()
        return super().parameters(recurse)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.finetune_gpt is False:
            self.gpt.eval()
        return self
