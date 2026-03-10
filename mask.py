import numpy as np


def create_causal_mask(seq_len):

    mask = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        for j in range(seq_len):

            if j > i:
                mask[i][j] = -np.inf

    return mask