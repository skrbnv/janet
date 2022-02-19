import torch.nn as nn
import torch.nn.functional as F
from libs.models.attention.encoder import Encoder as BERT


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        params = {
            # ATTENTION
            'D_MODEL': 80,
            'D_FINAL': 128,
            'D_K': 32,
            'D_V': 32,
            'NUM_HEADS': 8,
            'ATTN_BATCH_SIZE': 64,
            'SEQUENCE_LENGTH': 200,
            'D_FFN_INNER': 2048,
            # DROPOUTS
            'POSITIONAL_ENCODER_DROPOUT': .1,
            'SCALED_DOT_PRODUCT_ATTENTION_DROPOUT': .1,
            'MULTI_HEAD_ATTENTION_DROPOUT': .1,
            'FEED_FORWARD_DROPOUT': .1,
            # DOT PRODUCT SCALE
            'DOT_PRODUCT_SCALE': 1024,  # D_K**2 ??
        }
        self.basemodel = BERT(params)

    def forward(self, p, n):
        p = self.innerModel(p)
        n = self.innerModel(n)
        return p, n

    def innerModel(self, x):
        return self.basemodel(x)
