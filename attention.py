import numpy as np


def softmax(x):

    x = x - np.max(x, axis=-1, keepdims=True)

    exp = np.exp(x)

    return exp / np.sum(exp, axis=-1, keepdims=True)


def cross_attention(encoder_out, decoder_state):

    d_model = encoder_out.shape[-1]

    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)

    Q = decoder_state @ Wq
    K = encoder_out @ Wk
    V = encoder_out @ Wv

    scores = Q @ np.transpose(K, (0, 2, 1))

    scores = scores / np.sqrt(d_model)

    weights = softmax(scores)

    output = weights @ V

    return output