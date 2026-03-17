import torch.nn as nn

from abc import ABC, abstractmethod


class MultimodalAdapter(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def feature_fusion(self) -> nn.Module:
        pass

    @feature_fusion.setter
    @abstractmethod
    def feature_fusion(self, value: nn.Module) -> None:
        pass
