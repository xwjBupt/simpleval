import torch.nn as nn

from util import registry
from Model.builder import build_detector


class BaseEngine(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = build_detector(model)
