import torch.nn as nn


class AttributeSelector(nn.Module):
    def __init__(self, attrs):
        super().__init__()

        self.attrs = attrs

    def forward(self, sim: dict) -> dict:
        sim = {key: sim[key] for key in self.attrs if key in sim.keys()}
        return sim
