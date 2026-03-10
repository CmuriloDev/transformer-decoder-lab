import numpy as np
from mask import create_causal_mask
from attention import cross_attention
from generation import generate_next_token


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def main():
    np.random.seed(42)

    print("TRANSFORMER DECODER LAB\n")

    encoder_output = np.random.randn(1, 10, 512)
    print("Encoder output shape:", encoder_output.shape)

    mask = create_causal_mask(5)
    print("\nCausal mask:")
    print(mask)

    Q_test = np.random.randn(1, 5, 4)
    K_test = np.random.randn(1, 5, 4)

    scores = Q_test @ np.transpose(K_test, (0, 2, 1))
    masked_scores = scores + mask
    masked_probs = softmax(masked_scores)

    print("\nMasked attention probabilities:")
    print(masked_probs[0])

    decoder_state = np.random.randn(1, 4, 512)
    attn_output = cross_attention(encoder_output, decoder_state)

    print("\nDecoder state shape:", decoder_state.shape)
    print("Cross attention output shape:", attn_output.shape)

    current_sequence = ["<START>"]

    print("\nGenerating sequence:")

    while True:
        probs = generate_next_token(current_sequence, encoder_output)
        next_token_id = np.argmax(probs)

        if next_token_id == 0:
            next_token = "<EOS>"
        else:
            next_token = f"token_{next_token_id}"

        current_sequence.append(next_token)

        if next_token == "<EOS>" or len(current_sequence) > 10:
            break

    print("Generated sequence:", current_sequence)


if __name__ == "__main__":
    main()