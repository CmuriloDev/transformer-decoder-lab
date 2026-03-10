import numpy as np
from mask import create_causal_mask
from attention import cross_attention
from generation import generate_next_token


def main():

    np.random.seed(42)

    print("=" * 50)
    print("TRANSFORMER DECODER LAB")
    print("=" * 50)

    # simulando saída do encoder
    encoder_output = np.random.randn(1, 10, 512)

    # sequência inicial gerada
    current_sequence = ["<START>"]

    print("\nEncoder output shape:", encoder_output.shape)

    # teste da máscara
    mask = create_causal_mask(5)

    print("\nCausal Mask Example:")
    print(mask)

    # teste cross attention
    decoder_state = np.random.randn(1, 4, 512)

    attn_output = cross_attention(encoder_output, decoder_state)

    print("\nCross attention output shape:", attn_output.shape)

    # teste geração
    print("\nGenerating sequence:")

    while True:

        probs = generate_next_token(current_sequence, encoder_output)

        next_token = np.argmax(probs)

        token = f"token_{next_token}"

        if token == "token_0":
            token = "<EOS>"

        current_sequence.append(token)

        if token == "<EOS>" or len(current_sequence) > 10:
            break

    print("Generated:", current_sequence)


if __name__ == "__main__":
    main()