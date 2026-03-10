# Transformer Decoder Lab

Academic project for the course **Artificial Intelligence Topics**  
Professor: **Dimmy Magalhães**  
Institution: **Faculdade iCEV**

## Description

This project simulates the core mathematical components of the Transformer Decoder architecture using **Python and NumPy only**.

The implementation focuses on three main elements:

- Causal Mask (Look-Ahead Mask)
- Cross Attention between Encoder and Decoder
- Autoregressive generation loop

The goal is to understand how modern language models generate text step by step.

## Project Structure
main.py → runs the simulation
mask.py → causal masking
attention.py → cross attention implementation
generation.py → token generation loop

## How to run
python main.py

## Example Output
Encoder output shape: (1, 10, 512)

Causal mask example:
[[ 0. -inf -inf -inf -inf]
[ 0. 0. -inf -inf -inf]
[ 0. 0. 0. -inf -inf]
[ 0. 0. 0. 0. -inf]
[ 0. 0. 0. 0. 0.]]

Generated sequence:
['<START>', 'token_43', 'token_92', '<EOS>']

## Notes
This is a simplified educational implementation of the Transformer Decoder based on the paper **"Attention Is All You Need" (Vaswani et al., 2017)**.