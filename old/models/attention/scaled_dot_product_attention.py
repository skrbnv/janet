import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.dropout = nn.Dropout(
            p=self.params['SCALED_DOT_PRODUCT_ATTENTION_DROPOUT'])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        '''
        input shape = (batch_size, NUM_HEADS, SEQUENCE_LENGTH, D_MODEL)
        '''
        batch_size = q.shape[0]
        attention_weights = torch.matmul(q, k.transpose(-1, -2))
        attention_weights /= self.params['DOT_PRODUCT_SCALE']
        # Here's an interesting topic for discussion
        # q=k, but we only mask padded values for keys
        # one reason I found mentions that softmax is calculated via rows,
        # so we can only mask columns in matmul(q,k.T)
        if mask is not None:
            assert mask.shape == (batch_size, self.params['SEQUENCE_LENGTH'],
                                  1), "Incorrect mask shape"
            mask = mask.unsqueeze(1)
            if torch.cuda.is_available() and mask.device.type == 'cpu':
                mask = mask.to(torch.device("cuda:0"))
            mask = mask.to(torch.bool)
            attention_weights = attention_weights.masked_fill(mask, value=-1e8)
        attention_weights = self.softmax(attention_weights)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
