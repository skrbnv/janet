import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.linear_expand = nn.Linear(self.params['D_MODEL'],
                                       self.params['D_FFN_INNER'])
        self.linear_contract = nn.Linear(self.params['D_FFN_INNER'],
                                         self.params['D_MODEL'])

    def forward(self, x):
        x = self.linear_expand(x)
        x = F.relu(x)
        x = self.linear_contract(x)
        return x
