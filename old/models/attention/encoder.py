import torch
import torch.nn as nn
import numpy as np
from libs.models.attention.encoding_block import EncodingBlock
from libs.models.attention.positional_encoding import PositionalEncoder
from libs.models.attention.masking import padding_mask


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoding_block = EncodingBlock(params)
        self.positional_encoder = PositionalEncoder(params)
        self.params = params
        self.postprocess_linear = nn.Linear(
            self.params['SEQUENCE_LENGTH'] * self.params['D_MODEL'],
            self.params['D_FINAL'])

    def preprocess(self, data):
        return data

    def forward(self, embeddings):
        ''' inputs = (phrase_length, D_MODEL) '''
        x = self.preprocess(embeddings)
        # generate padded mask
        mask = padding_mask(x)
        # add positional encoder, dropout included
        x = self.positional_encoder(x)
        # consequentially process encodings N = 6 times through encoding block
        for i in range(6):
            x = self.encoding_block(x, mask)
        x = x.view(embeddings.shape[0],
                   self.params['SEQUENCE_LENGTH'] * self.params['D_MODEL'])
        x = self.postprocess_linear(x)
        # to final sized embeddings -> scale -> softmax (???)
        return x
