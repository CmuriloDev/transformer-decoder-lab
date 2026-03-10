import numpy as np


def generate_next_token(current_sequence, encoder_out):

    vocab_size = 10000

    logits = np.random.randn(vocab_size)

    exp = np.exp(logits - np.max(logits))

    probs = exp / np.sum(exp)

    return probs