from typing import List
import numpy
import math
from collections import OrderedDict

import torch
from torch.nn import init, Module, EmbeddingBag, Sequential, Linear

from ...nn import create_activation


class FCSettings(object):
    def __init__(self, features_size: int, act_type: str):
        self.features_size = features_size
        self.act_type = act_type


class DLRM(Module):
    def __init__(self, embed_features_size: int, embed_sizes: List[int], input_features_size: int,
                 input_mlp_settings: List[FCSettings], mlp_settings: List[FCSettings]):
        super().__init__()

        self.embeds = DLRM.create_embeddings(embed_features_size, embed_sizes)
        self.input_mlp = DLRM.create_mlp(input_mlp_settings, input_features_size)
        self.mlp = DLRM.create_mlp(mlp_settings, 0)

    @staticmethod
    def create_embeddings(embed_features_size: int, embed_sizes: List[int]) -> List[EmbeddingBag]:
        embeds = []  # type: List[EmbeddingBag]

        for embed_size in embed_sizes:
            embed = EmbeddingBag(embed_size, embed_features_size, mode='sum', sparse=True)
            init.uniform_(embed.weight, a=-1 * math.sqrt(1.0 / embed_size), b=math.sqrt(1.0 / embed_size))
            embeds.append(embed)

        return embeds

    @staticmethod
    def create_mlp(settings: List[FCSettings], in_features_size: int) -> Sequential:
        fcs = []

        for setting in settings:
            fcs.append(Sequential(OrderedDict([
                ('fc', Linear(in_features_size, setting.features_size, bias=True)),
                ('act', create_activation(setting.act_type, inplace=True, num_channels=setting.features_size))
            ])))
            in_features_size = setting.features_size

        return Sequential(*fcs)

