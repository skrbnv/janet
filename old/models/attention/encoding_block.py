import torch
import torch.nn as nn
from libs.models.attention.multi_head_attention import MultiHeadAttention
from libs.models.attention.position_wise_feed_forward import PositionWiseFeedForward


class EncodingBlock(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.mhattention = MultiHeadAttention(self.params)
        self.norm = nn.LayerNorm(self.params['D_MODEL'],
                                 eps=1e-6)  # check eps value
        self.feedforward = PositionWiseFeedForward(self.params)
        self.dropout1 = nn.Dropout(
            p=self.params['MULTI_HEAD_ATTENTION_DROPOUT'])
        self.dropout2 = nn.Dropout(p=self.params['FEED_FORWARD_DROPOUT'])

    def forward(self, embeddings, mask=None):
        ''' input = embeddings with shape (ATTN_BATCH_SIZE, SEQUENCE_LENGTH, D_MODEL) '''
        ''' output have same shape '''
        # save inputs for residual connection
        residual = embeddings
        x, _ = self.mhattention(embeddings, embeddings, embeddings, mask)
        x = self.dropout1(x)
        x = self.norm(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.norm(x + residual)
        return x
