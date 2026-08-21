# Chapter 3: The Transformer Architecture (GPT)

This module integrates the foundational components into a complete **Generative Pre-trained Transformer (GPT)** architecture.

The original Transformer (Vaswani et al.) utilized an Encoder-Decoder architecture. However, modern LLMs like GPT are **Decoder-only**. They discard the Encoder and rely entirely on masked self-attention to auto-regressively predict the next token.

A GPT model encompasses:
1. **Token Embeddings:** Maps discrete token IDs to continuous dense vectors.
2. **Positional Embeddings:** Injects sequence order information (since attention operations are permutation invariant).
3. **Transformer Blocks:** A sequence of residual layers containing:
   - Masked Multi-Head Self-Attention
   - Feed-Forward Neural Networks (MLP)
   - Layer Normalization
4. **Language Modeling Head:** A linear projection from the hidden dimension back to the vocabulary space to output logits.

## Usage

Execute the script to observe a forward pass through the complete GPT architecture:
```bash
python gpt_model.py
```
