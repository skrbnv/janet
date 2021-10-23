import torch
import torch.nn as nn
#import matplotlib.pyplot as plt


class PositionalEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.encoding_table = self.create_encoding_table(
            (self.params['SEQUENCE_LENGTH'], self.params['D_MODEL']))
        self.dropout = nn.Dropout(p=self.params['POSITIONAL_ENCODER_DROPOUT'])
        self.layer_norm = nn.LayerNorm(self.params['D_MODEL'], eps=1e-6)

    def create_encoding_table(self, shape):
        if torch.cuda.is_available():
            table = torch.empty(shape, device=torch.device('cuda:0'))
        else:
            table = torch.empty(shape)

        for i in range(shape[0]):
            for j in range(shape[1] // 2):
                table[i, 2 * j] = i / (10000**(2 * j / shape[1]))
        table[:, 0::2] = torch.sin(table[:, 0::2])
        table[:, 1::2] = torch.cos(table[:, 1::2])
        '''
        plt.figure(figsize=(51.2,20))
        plt.imshow(table.detach().numpy())
        plt.savefig('tmp.png')
        plt.close()
        '''
        return table

    def forward(self, x):
        x = self.dropout(x + self.encoding_table)
        return self.layer_norm(x)
