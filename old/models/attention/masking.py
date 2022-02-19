import numpy as np
import torch


def padding_mask(data):
    ''' Returns True-False mask.
        TLDR; True[1] - keep, False[0] - ignore (remove)
        False if elements in column are equal, this means this column was a result of zero padding
        BEFORE generating spectrogram.

        parameters:
            data: torch.Tensor with shape (batch_size, sequence_len, d_model)
            value: value to check for, default = 0
        output:
            np.array (batch_size, number of measurements, d_model size)
    '''
    assert len(data.shape) == 3, "Incorrect shape"
    mask = torch.empty(data.shape[0], data.shape[1], 1)
    for i, phrase in enumerate(data):
        mask[i] = ~torch.all(phrase.T == phrase.T[0, :], axis=0).unsqueeze(0).T
    return mask


#def nopeek_mask():
#    mask = 1-torch.tril(torch.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH)))
#    mask = mask.expand(ATTN_BATCH_SIZE*NUM_HEADS, -1, -1).to(torch.bool)
#    return mask

#def combined_mask(data):
#    pad_mask = padding_mask(data)
#    npk_mask = nopeek_mask()
#    return pad_mask | npk_mask #bitwise OR
