import torch.nn as nn
from libs.models.attention.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    ''' Multi Head Attention '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.linear_qs = nn.Linear(self.params['D_MODEL'],
                                   self.params['NUM_HEADS'] *
                                   self.params['D_K'],
                                   bias=False)
        self.linear_ks = nn.Linear(self.params['D_MODEL'],
                                   self.params['NUM_HEADS'] *
                                   self.params['D_K'],
                                   bias=False)
        self.linear_vs = nn.Linear(self.params['D_MODEL'],
                                   self.params['NUM_HEADS'] *
                                   self.params['D_V'],
                                   bias=False)

        self.attention = ScaledDotProductAttention(self.params)
        self.linear = nn.Linear(self.params['NUM_HEADS'] * self.params['D_V'],
                                self.params['D_MODEL'])

    def forward(self, input_q, input_k, input_v, mask=None):
        ''' input = 3 x vector sets with shape (ATTN_BATCH_SIZE, SEQUENCE_LENGTH, D_MODEL) '''
        len_q = input_q.shape[1]
        len_k = input_k.shape[1]
        len_v = input_v.shape[1]
        batch_size = input_q.shape[0]
        # Pass through preliminary linear transformation
        # vectors with len D_MODEL => len D_K * NUM_HEADS --- (D_V * NUM_HEADS)
        qs = self.linear_qs(input_q)
        ks = self.linear_ks(input_k)
        vs = self.linear_vs(input_v)
        # split each vector with len = num_heads*d_k(d_v)
        # into n = num_heads vectors, each with len = d_k (d_v)
        qs = qs.view(batch_size, len_q, self.params['NUM_HEADS'],
                     self.params['D_K'])
        ks = ks.view(batch_size, len_k, self.params['NUM_HEADS'],
                     self.params['D_K'])
        vs = vs.view(batch_size, len_v, self.params['NUM_HEADS'],
                     self.params['D_V'])
        # transpose, move heads one rank higher
        # (ATTN_BATCH_SIZE, NUM_HEADS, len, D)
        qs = qs.transpose(1, 2)
        ks = ks.transpose(1, 2)
        vs = vs.transpose(1, 2)
        # calculate scaled dot product attention (masked if neccesary)
        output, weights = self.attention(qs, ks, vs, mask=mask)
        # concat vectors back into one
        # transpose back to (ATTN_BATCH_SIZE, len, NUM_HEADS, D)
        output = output.transpose(1,
                                  2).contiguous().view(batch_size, len_q, -1)
        output = self.linear(output)
        return output, weights
