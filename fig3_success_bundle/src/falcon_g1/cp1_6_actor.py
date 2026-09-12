"""Exact PyTorch representation of the pinned dual-actor ONNX graph."""

from __future__ import annotations
import torch
from torch import nn


class ActorBranch(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__(); self.layers=nn.ModuleList([nn.Linear(575,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,output_dim)])

    def forward(self, value):
        for layer in self.layers: value=layer(value)
        return value


class FalconDualActor(nn.Module):
    def __init__(self):
        super().__init__(); self.lower_body=ActorBranch(15); self.upper_body=ActorBranch(14)

    def forward(self, observation): return torch.cat((self.lower_body(observation),self.upper_body(observation)),dim=-1)
