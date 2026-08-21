# Chapter 2: Attention Mechanism

The computational core of the Transformer architecture is the **Multi-Head Self-Attention Mechanism**.

Unlike Recurrent Neural Networks (RNNs) that process tokens sequentially, Self-Attention computes a representation of a sequence by relating different positions of the sequence simultaneously. This is achieved by projecting input embeddings into Query ($Q$), Key ($K$), and Value ($V$) matrices.

In this module, we implement:
1. **Scaled Dot-Product Attention:** The fundamental mathematical operation: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$. The scaling factor $\frac{1}{\sqrt{d_k}}$ is critical to prevent gradients from vanishing in the softmax function when dimensions are large.
2. **Multi-Head Attention:** Partitioning the queries, keys, and values into $h$ parallel "heads". This allows the model to jointly attend to information from different representation subspaces at different positions.

## Usage

Execute the script to observe the tensor transformations during the attention forward pass:
```bash
python attention.py
```
