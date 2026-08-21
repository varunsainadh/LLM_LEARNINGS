# Chapter 4: Training Loop & Optimization

A model's architecture defines its capacity, but the **training loop** dictates its behavior by optimizing the network parameters against a target objective.

In this module, we implement an auto-regressive training loop to minimize the Cross-Entropy loss for next-token prediction.

The workflow encompasses:
1. **Context Windowing:** Generating fixed-length sequence inputs ($X$) and shifted target sequences ($Y$).
2. **Forward Pass:** Computing logits across the vocabulary for each token in the sequence.
3. **Loss Computation:** Flattening the logits and targets to evaluate the negative log-likelihood (Cross-Entropy).
4. **Backpropagation:** Utilizing `loss.backward()` to compute gradients and stepping the **AdamW** optimizer to update network weights while applying weight decay for regularization.

## Usage

Execute the script to observe the gradient descent process minimizing the training loss:
```bash
python train.py
```
