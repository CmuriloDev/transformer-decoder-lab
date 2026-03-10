import numpy as np


def generate_next_token(current_sequence, encoder_out):
    vocab_size = 10000

    logits = np.random.randn(vocab_size)

    # Encourage EOS after a few generated tokens
    if len(current_sequence) >= 5:
        logits[0] = 10.0

    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)

    return probs